"""Ball Don't Lie API client with tier-safe cached fetchers.

RACE uses this client for pregame-safe context enrichment. Every fetcher is
defensive by design: missing key, unavailable paid endpoint, empty responses,
or rate limits should not crash the app. Instead, methods return empty
DataFrames and emit a warning.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .constants import BDL_CACHE_DIR

# ── Field-name variants ───────────────────────────────────────────────────────
# BDL mirrors the NBA stats API, which uses slightly different names across
# endpoints.  We try each candidate in order and take the first match.
_PACE_KEYS = ("pace", "pace_per48")
_DEF_RTG_KEYS = ("def_rtg", "def_rating", "defensive_rating")
_OFF_RTG_KEYS = ("off_rtg", "off_rating", "offensive_rating")


def _warn(msg: str) -> None:
    # Deduplicate identical warnings within a single process so noisy retry
    # paths (e.g. 429 rate limits) don't spam Streamlit Cloud logs.
    if msg in _WARN_SEEN:
        return
    _WARN_SEEN.add(msg)
    warnings.warn(f"BDLClient: {msg}", RuntimeWarning, stacklevel=2)


_WARN_SEEN: set[str] = set()


# ── Key loading ───────────────────────────────────────────────────────────────
def _load_bdl_key(env_path: Path | str = ".env") -> str:
    """Read BDL_API_KEY from .env file → environment, then return it."""
    env_file = Path(env_path)
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
    return os.getenv("BDL_API_KEY", "").strip()


# ── Column-rename helper ──────────────────────────────────────────────────────
def _pick(df: pd.DataFrame, *candidates: str, rename: str) -> pd.DataFrame:
    """Rename the first matching column from *candidates* to *rename*."""
    for col in candidates:
        if col in df.columns:
            return df.rename(columns={col: rename})
    return df  # column not present — caller handles gracefully


# ── Client ────────────────────────────────────────────────────────────────────
class BDLClient:
    """Tier-safe HTTP client for api.balldontlie.io.

    Parameters
    ----------
    api_key
        GOAT-tier key.  Falls back to ``BDL_API_KEY`` env var / ``.env``.
    cache_dir
        Directory for per-season Parquet caches.
    pause
        Seconds between paginated requests (GOAT = 600 req/min, 0.1 s is safe).
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = BDL_CACHE_DIR,
        pause: float = 0.1,
    ) -> None:
        self.api_key = api_key or _load_bdl_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["Authorization"] = self.api_key
        self.enabled = bool(self.api_key)
        if not self.enabled:
            _warn("BDL_API_KEY not set; context endpoints will return empty frames.")

    # ── HTTP helpers ──────────────────────────────────────────────────────────
    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code == 401:
            _warn("401 Unauthorized or endpoint not available for this tier.")
            return None
        if resp.status_code == 429:
            _warn("429 rate limit exceeded; returning empty response.")
            return None
        if not resp.ok:
            _warn(f"HTTP {resp.status_code} from {url}; returning empty response.")
            return None
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Collect all pages from a cursor-paginated endpoint."""
        if not self.enabled:
            return []
        params = dict(params or {})
        params.setdefault("per_page", 100)
        results: list[dict[str, Any]] = []
        while True:
            data = self._get(url, params)
            if data is None:
                return []
            results.extend(data.get("data", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            params["cursor"] = cursor
            time.sleep(self.pause)
        return results

    def _cache_json_path(self, slug: str, season: int | None = None) -> Path:
        suffix = f"_{season}" if season is not None else ""
        return self.cache_dir / f"{slug}{suffix}.json"

    def _cache_parquet_path(self, slug: str, season: int | None = None) -> Path:
        suffix = f"_{season}" if season is not None else ""
        return self.cache_dir / f"{slug}{suffix}.parquet"

    def _load_cached_json(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_cached_json(self, path: Path, data: list[dict[str, Any]]) -> None:
        try:
            path.write_text(json.dumps(data))
        except Exception:
            pass

    # ── Team season averages ──────────────────────────────────────────────────
    def fetch_team_season_averages(
        self,
        season: int,
        stat_type: str = "advanced",
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return team season averages for all 30 teams in one season.

        Parameters
        ----------
        season
            BDL integer year (2024 = the 2024-25 season).
        stat_type
            One of ``"advanced"`` (pace, def_rtg, …) or ``"base"`` (stl, blk, …).

        Results are cached as Parquet; subsequent calls are instant.
        """
        cache_path = self._cache_parquet_path(f"team_avgs_general_{stat_type}", season)
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        rows = self._paginate(
            "https://api.balldontlie.io/nba/v1/team_season_averages/general",
            {"season": season, "season_type": "regular", "type": stat_type},
        )
        if not rows:
            _warn(f"No team season averages for {season} ({stat_type}).")
            return pd.DataFrame()

        records = []
        for r in rows:
            rec = {
                "bdl_season": int(r.get("season", season)),
                "abbreviation": r["team"]["abbreviation"],
            }
            rec.update(r.get("stats", {}))
            records.append(rec)

        df = pd.DataFrame(records)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df

    def fetch_advanced_stats_v2(self, season: int, *, force_refresh: bool = False) -> pd.DataFrame:
        """Attempt advanced-stats v2 endpoint; return empty DataFrame if unavailable."""
        cache_path = self._cache_parquet_path("advanced_stats_v2", season)
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        rows = self._paginate(
            "https://api.balldontlie.io/nba/v1/team_season_averages/advanced",
            {"season": season, "season_type": "regular"},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if not df.empty:
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
        return df

    def fetch_games_metadata(self, season: int, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch and cache basic game metadata (date/home/away/status)."""
        cache_path = self._cache_parquet_path("games", season)
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        rows = self._paginate(
            "https://api.balldontlie.io/nba/v1/games",
            {"seasons[]": season, "postseason": "false"},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "home_team" in df.columns:
            df["home_abbr"] = df["home_team"].apply(
                lambda t: t.get("abbreviation") if isinstance(t, dict) else None
            )
        if "visitor_team" in df.columns:
            df["away_abbr"] = df["visitor_team"].apply(
                lambda t: t.get("abbreviation") if isinstance(t, dict) else None
            )
        if "date" in df.columns:
            df["game_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df

    def fetch_lineups(self, season: int, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch lineup data if the endpoint/tier is available.

        The lineups API requires ``game_ids[]``. We therefore fetch season games
        first, then query lineups in game-id batches.

        Returns an empty DataFrame when lineups are unavailable for the key tier
        or no game IDs are available.
        """
        # BDL lineup data starts with the 2025 season. Treat earlier seasons as
        # expected-empty to avoid noisy warning output during mixed-season runs.
        if season < 2025:
            return pd.DataFrame()

        cache_path = self._cache_json_path("lineups", season)
        if cache_path.exists() and not force_refresh:
            rows = self._load_cached_json(cache_path)
            if rows:
                return pd.DataFrame(rows)

        games = self.fetch_games_metadata(season, force_refresh=force_refresh)
        if games.empty or "id" not in games.columns:
            _warn("No games metadata available; cannot request lineups by game_id.")
            return pd.DataFrame()

        game_ids = (
            pd.to_numeric(games["id"], errors="coerce")
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        if not game_ids:
            _warn("No game IDs found for lineup fetch.")
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        batch_size = 25
        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i : i + batch_size]
            part = self._paginate(
                "https://api.balldontlie.io/nba/v1/lineups",
                {"game_ids[]": batch},
            )
            if part:
                rows.extend(part)

        if not rows:
            _warn("Lineups endpoint unavailable or empty for this tier.")
            return pd.DataFrame()
        self._save_cached_json(cache_path, rows)
        return pd.DataFrame(rows)

    def fetch_player_injuries(self, season: int | None = None, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch player injuries if available on the current tier."""
        cache_path = self._cache_json_path("injuries", season)
        if cache_path.exists() and not force_refresh:
            rows = self._load_cached_json(cache_path)
            if rows:
                return pd.DataFrame(rows)

        # Docs endpoint: /v1/player_injuries. No season filter is documented.
        rows = self._paginate("https://api.balldontlie.io/nba/v1/player_injuries", {})
        if not rows:
            _warn("Injuries endpoint unavailable or empty for this tier.")
            return pd.DataFrame()
        self._save_cached_json(cache_path, rows)
        return pd.DataFrame(rows)

    def fetch_player_stats(self, season: int, *, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch basic per-game player stats for ID/name reconciliation."""
        cache_path = self._cache_parquet_path("player_stats", season)
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        rows = self._paginate(
            "https://api.balldontlie.io/nba/v1/stats",
            {"seasons[]": season, "postseason": False},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return df

    # ── Opponent context lookup ───────────────────────────────────────────────
    def build_opponent_lookup(self, seasons: list[int]) -> pd.DataFrame:
        """Build a per-team per-season opponent-context DataFrame.

        Combines ``general/advanced`` (pace, def_rtg) with ``general/base``
        (stl, blk per game) for every team across *seasons*.  Each row
        represents what a player faces when playing *against* that team.

        Returned columns
        ----------------
        ``bdl_season``, ``abbreviation``,
        ``opp_pace``, ``opp_def_rtg``, ``opp_off_rtg``,
        ``opp_stl_pg``, ``opp_blk_pg``
        """
        if not seasons:
            return pd.DataFrame()

        adv_parts: list[pd.DataFrame] = []
        base_parts: list[pd.DataFrame] = []

        for season in seasons:
            adv  = self.fetch_team_season_averages(season, "advanced")
            base = self.fetch_team_season_averages(season, "base")
            if not adv.empty:
                adv_parts.append(adv)
            if not base.empty:
                base_parts.append(base)

        if not adv_parts:
            _warn("Could not build opponent lookup: advanced team context unavailable.")
            return pd.DataFrame()

        # ── Advanced: pace + ratings ──────────────────────────────────────────
        adv_all = pd.concat(adv_parts, ignore_index=True)
        adv_all = _pick(adv_all, *_PACE_KEYS,    rename="opp_pace")
        adv_all = _pick(adv_all, *_DEF_RTG_KEYS, rename="opp_def_rtg")
        adv_all = _pick(adv_all, *_OFF_RTG_KEYS, rename="opp_off_rtg")

        keep_adv = ["bdl_season", "abbreviation"]
        for col in ("opp_pace", "opp_def_rtg", "opp_off_rtg"):
            if col in adv_all.columns:
                keep_adv.append(col)
        result = adv_all[keep_adv].copy()

        # ── Base: steals + blocks per game ────────────────────────────────────
        if base_parts:
            base_all = pd.concat(base_parts, ignore_index=True)
            rename_map = {}
            for raw, new in (("stl", "opp_stl_pg"), ("blk", "opp_blk_pg")):
                if raw in base_all.columns:
                    rename_map[raw] = new
            base_all = base_all.rename(columns=rename_map)

            merge_cols = ["bdl_season", "abbreviation"] + [
                v for v in rename_map.values() if v in base_all.columns
            ]
            result = result.merge(
                base_all[merge_cols],
                on=["bdl_season", "abbreviation"],
                how="left",
            )

        return result
