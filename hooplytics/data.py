"""Player game-log fetching, caching, and pregame-safe feature engineering."""
from __future__ import annotations

import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    CACHE_DIR,
    MODEL_SPECS,
    NBA_RENAME,
    ROLL_BASE_STATS,
    ROLL_WINDOWS,
)
from .features_context import build_context_features
from .features_market import build_market_features
from .features_role import build_role_features
from .fantasy import fantasy


class NBADataUnavailable(RuntimeError):
    """Raised when the NBA stats API can't be reached or returns no data.

    ``kind`` discriminates the failure mode so the UI can render an actionable
    message instead of always blaming the host:

    - ``"blocked"``  - HTTP 4xx / explicit block (typical on cloud datacenter IPs)
    - ``"timeout"``  - request timed out / connection error after retries
    - ``"empty"``    - API responded but returned no rows for the player/seasons
    - ``"unknown"``  - any other failure (parsing, library mismatch, etc.)
    """

    def __init__(self, message: str, *, kind: str = "unknown", player: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.player = player


def _configure_nba_stats_headers() -> None:
    """Augment nba_api's default headers with stats.nba.com-specific tokens.

    The library ships a Chrome User-Agent but omits the ``x-nba-stats-*``
    headers that ``stats.nba.com`` increasingly requires when traffic looks
    automated. Adding them here is a no-op when nba_api isn't installed and
    is safe to run multiple times.
    """
    try:
        from nba_api.stats.library.http import NBAStatsHTTP  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency at import time
        return
    NBAStatsHTTP.headers.setdefault("x-nba-stats-origin", "stats")
    NBAStatsHTTP.headers.setdefault("x-nba-stats-token", "true")
    NBAStatsHTTP.headers.setdefault("Sec-Fetch-Mode", "cors")
    NBAStatsHTTP.headers.setdefault("Sec-Fetch-Site", "same-site")


_configure_nba_stats_headers()


def nba_seasons(start: int, end: int) -> list[str]:
    """Season strings from ``start`` (inclusive) to ``end`` (exclusive on year, inclusive on season).

    ``nba_seasons(2024, 2026) -> ['2024-25', '2025-26']``.
    """
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start, end)]


def add_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append pregame-safe rolling features (computed from PRIOR games only).

    Per-player, sorted by ``game_date``, then ``.shift(1)`` before ``.rolling(N)``
    so the current game's value is never used as one of its own predictors.
    """
    df = df.sort_values(["player", "game_date"]).copy()
    g = df.groupby("player", group_keys=False)

    def _roll(col: str, window: int) -> pd.Series:
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 3)).mean()
        )

    for stat in ROLL_BASE_STATS:
        if stat not in df.columns:
            continue
        for window in ROLL_WINDOWS:
            df[f"{stat}_l{window}"] = _roll(stat, window)

    if "min_l30" in df.columns:
        min30 = df["min_l30"].replace(0, np.nan)
        for stat in ("ast", "stl", "blk", "tov", "fga"):
            src = f"{stat}_l30"
            if src in df.columns:
                df[f"{stat}_per36_l30"] = df[src] * 36.0 / min30

    if "min_l10" in df.columns:
        min10 = df["min_l10"].replace(0, np.nan)
        for stat in ("ast", "stl", "blk", "tov", "fga"):
            src = f"{stat}_l10"
            if src in df.columns:
                df[f"{stat}_per36_l10"] = df[src] * 36.0 / min10

    if {"fga", "fta", "tov", "min"}.issubset(df.columns):
        usg = (df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["min"].replace(0, np.nan)
        df["_usg_proxy_raw"] = usg
        df["usg_proxy_l30"] = df.groupby("player", group_keys=False)["_usg_proxy_raw"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=10).mean()
        )
        df = df.drop(columns=["_usg_proxy_raw"])

    # Rolling std dev (consistency signal — high variance = harder to predict)
    def _roll_std(col: str, window: int) -> pd.Series:
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 3)).std()
        )

    for stat in ("ast", "stl", "blk", "tov"):
        if stat not in df.columns:
            continue
        df[f"{stat}_std_l10"] = _roll_std(stat, 10)
        df[f"{stat}_std_l30"] = _roll_std(stat, 30)

    # Trend delta: recent form vs baseline (positive = heating up, negative = cooling off)
    for stat in ("ast", "stl", "blk", "tov"):
        l3 = f"{stat}_l3"
        l10 = f"{stat}_l10"
        l30 = f"{stat}_l30"
        if l3 in df.columns and l10 in df.columns:
            df[f"{stat}_trend_s"] = df[l3] - df[l10]   # short-term trend
        if l10 in df.columns and l30 in df.columns:
            df[f"{stat}_trend_l"] = df[l10] - df[l30]  # long-term trend

    # Days rest (fatigue/freshness — affects defensive effort stats most)
    if "game_date" in df.columns:
        df["days_rest"] = (
            df.groupby("player")["game_date"]
            .transform(lambda s: s.diff().dt.days.shift(1).clip(upper=14))
            .fillna(3)  # assume 3-day rest if no prior game
        )

    # Home/away indicator (derived from MATCHUP: "TM @ OPP" = away, "TM vs. OPP" = home)
    if "MATCHUP" in df.columns:
        df["is_home"] = (~df["MATCHUP"].str.contains("@", na=False)).astype(int)

    # Per-player deviation: recent form minus own season baseline.
    # Captures "above/below your own mean" without leaking the current game.
    # This improves ridge's ability to differentiate high from low scorers.
    for stat in ("pts", "reb", "ast", "fg3a", "tov", "stl", "blk"):
        l10 = f"{stat}_l10"
        l30 = f"{stat}_l30"
        if l10 in df.columns and l30 in df.columns:
            df[f"{stat}_dev_s"] = df[l10] - df[l30]   # recent vs season avg

    return df


# Per-season-row staleness threshold. When a cached season's most recent
# game is more than this many days behind today, we treat the cache as
# stale for that season and refetch — otherwise warmed-pre-playoffs caches
# silently serve incomplete data and break downstream joins.
_FRESHNESS_DAYS: int = 3


def _stale_seasons(cached: pd.DataFrame, seasons: list[str]) -> list[str]:
    """Return the subset of ``seasons`` whose cached coverage is stale.

    A season is "stale" when the cache holds rows for it AND the latest
    cached game date is more than :data:`_FRESHNESS_DAYS` behind today AND
    today still falls inside that season's NBA window (October → late June).
    Off-season seasons stay valid forever once cached.

    Tolerant to either the raw nba_api column name (``GAME_DATE``) or the
    post-rename name (``game_date``) — :func:`fetch_player_seasons` runs
    before the modeling-frame rename, so the raw form is what we actually
    see at this layer.
    """
    if cached.empty or "season" not in cached.columns:
        return []
    date_col = next(
        (c for c in ("game_date", "GAME_DATE") if c in cached.columns),
        None,
    )
    if date_col is None:
        return []
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=_FRESHNESS_DAYS)
    stale: list[str] = []
    for season in seasons:
        sub = cached[cached["season"] == season]
        if sub.empty:
            continue
        latest = pd.to_datetime(sub[date_col], errors="coerce").dropna()
        if latest.empty:
            continue
        if latest.max().normalize() >= cutoff:
            continue  # fresh enough
        # Only refetch when today is plausibly inside the season window.
        # NBA seasons run Oct → late June; outside that window we leave
        # the cache alone since no new games are happening.
        try:
            start_year = int(season.split("-")[0])
        except (ValueError, IndexError):
            continue
        season_start = pd.Timestamp(year=start_year, month=10, day=1)
        season_end = pd.Timestamp(year=start_year + 1, month=7, day=15)
        if season_start <= today <= season_end:
            stale.append(season)
    return stale


class PlayerStore:
    """Fetch and cache NBA player game logs as Parquet under ``cache_dir``.

    Parameters
    ----------
    cache_dir
        Directory for per-player Parquet files. Defaults to ``data/cache``.
    pause
        Sleep between successive ``nba_api`` calls (seconds).
    """

    def __init__(self, cache_dir: Path | str = CACHE_DIR, pause: float = 0.25) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause

    # ── Player resolution ────────────────────────────────────────────────────
    @staticmethod
    def resolve_player_id(name: str) -> int:
        from nba_api.stats.static import players as nba_players

        matches = nba_players.find_players_by_full_name(name)
        if not matches:
            raise ValueError(f"No NBA player matches '{name}'")
        return int(matches[0]["id"])

    @staticmethod
    def resolve_player_name(query: str, *, active_only: bool = True) -> str | None:
        """Fuzzy-resolve a partial/typo'd name to a canonical NBA player name.

        Returns ``None`` if no plausible match exists. Tries exact match,
        case-insensitive substring, then a simple character-overlap score.
        """
        from nba_api.stats.static import players as nba_players

        all_players = nba_players.get_players()
        if active_only:
            pool = [p["full_name"] for p in all_players if p.get("is_active")]
        else:
            pool = [p["full_name"] for p in all_players]

        q = query.strip()
        if not q:
            return None
        # Exact (case-insensitive)
        for name in pool:
            if name.lower() == q.lower():
                return name
        # Substring
        ql = q.lower()
        subs = [n for n in pool if ql in n.lower()]
        if subs:
            return min(subs, key=len)
        # rapidfuzz if available, else fallback to simple ratio
        try:
            from rapidfuzz import process

            best = process.extractOne(q, pool, score_cutoff=65)
            if best:
                return best[0]
        except ImportError:
            pass
        return None

    # ── Cached game-log fetch ────────────────────────────────────────────────
    @staticmethod
    def _cache_filename(name: str) -> str:
        """Filesystem-safe ASCII filename for a player parquet cache.

        nba_api returns canonical names with diacritics (e.g. ``"Nikola Jokić"``,
        ``"Nikola Vučević"``) but the shipped seed cache files are ASCII
        (``Nikola_Jokic.parquet``). Stripping diacritics keeps lookups stable
        across platforms and ensures the seed cache hydrates new players
        instantly instead of falling through to a slow live ``nba_api`` fetch
        that can hang the UI for 30+ seconds on first add.
        """
        ascii_name = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        return f"{ascii_name.replace(' ', '_')}.parquet"

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / self._cache_filename(name)

    def _seed_cache_path(self, name: str) -> Path:
        """Read-only seed cache shipped with the repo for fast cold starts."""
        return (
            Path(__file__).resolve().parent.parent
            / "data"
            / "seed_cache"
            / self._cache_filename(name)
        )

    def fetch_player_seasons(self, name: str, seasons: list[str]) -> pd.DataFrame:
        """Return raw game logs for ``name`` across ``seasons``, using disk cache.

        Cached players are returned without an API call when the cache covers
        every requested season; otherwise the missing seasons are fetched and
        merged into the cache.
        """
        cache_path = self._cache_path(name)
        # Hydrate from the shipped seed cache on first run if no per-user cache yet.
        if not cache_path.exists():
            seed = self._seed_cache_path(name)
            if seed.exists():
                try:
                    seed_df = pd.read_parquet(seed)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    seed_df.to_parquet(cache_path, index=False)
                except Exception:
                    pass
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                if set(seasons).issubset(cached["season"].unique()):
                    # Staleness guard: when the user requests an in-progress
                    # season but the cache's most recent game for that season
                    # is more than ``_FRESHNESS_DAYS`` behind today, the
                    # cache pre-dates the playoffs (or recent games) and
                    # would silently serve incomplete data — falsely making
                    # the historical-lines join fail because the player has
                    # odds for dates that aren't in their game logs. Drop
                    # the affected seasons from the cache so the fetch path
                    # below repopulates them.
                    stale_seasons = _stale_seasons(cached, seasons)
                    if stale_seasons:
                        cached = cached[~cached["season"].isin(stale_seasons)]
                    if not stale_seasons:
                        out = cached[cached["season"].isin(seasons)].copy()
                        # Seed parquets ship ASCII player names (e.g. 'Nikola Jokic')
                        # but nba_api resolves to the diacritic form ('Nikola Jokić')
                        # which is what the roster keys on. Normalize to the
                        # requested name so downstream filters match.
                        out["player"] = name
                        return out
            except Exception:
                # Corrupt or zero-byte cache file; remove and rebuild.
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
                cached = pd.DataFrame()
        else:
            cached = pd.DataFrame()

        pid = self.resolve_player_id(name)
        needed = [s for s in seasons if cached.empty or s not in cached["season"].unique()]
        frames: list[pd.DataFrame] = [cached] if not cached.empty else []

        # Parallelize the per-(season, season_type) game-log fetches. Serially
        # this stacks 4 HTTP calls (2 seasons × Regular Season + Playoffs) with
        # ~12s timeouts each, which can pin the UI for a full minute on a slow
        # network when a brand-new player has no seed cache. Threading drops
        # wall time to ~one slow call regardless of how many seasons are
        # requested. Each job retries once on transient network errors with a
        # short backoff — stats.nba.com cold calls routinely take 6–10s, so a
        # single 8s attempt (the previous behavior) was failing too eagerly.
        from concurrent.futures import ThreadPoolExecutor
        from nba_api.stats.endpoints import playergamelog as _pgl

        def _classify(exc: Exception) -> str:
            # Lazily import requests so missing optional deps don't crash here.
            try:
                from requests import exceptions as rexc  # type: ignore
            except Exception:  # noqa: BLE001
                rexc = None  # type: ignore[assignment]
            if rexc is not None:
                if isinstance(exc, (rexc.ReadTimeout, rexc.ConnectTimeout, rexc.Timeout)):
                    return "timeout"
                if isinstance(exc, rexc.ConnectionError):
                    return "timeout"
                if isinstance(exc, rexc.HTTPError):
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (401, 403, 429):
                        return "blocked"
                    return "unknown"
            # Fallback: inspect the message for the common timeout/block markers.
            msg = str(exc).lower()
            if "timeout" in msg or "timed out" in msg or "connection" in msg:
                return "timeout"
            if "403" in msg or "forbidden" in msg or "blocked" in msg:
                return "blocked"
            return "unknown"

        def _fetch_one(season: str, season_type: str) -> tuple[pd.DataFrame, Exception | None, str | None]:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    gl = _pgl.PlayerGameLog(
                        player_id=pid, season=season,
                        season_type_all_star=season_type, timeout=12,
                    ).get_data_frames()[0]
                except Exception as exc:  # noqa: BLE001 - classified below
                    last_exc = exc
                    kind = _classify(exc)
                    if kind == "blocked":
                        # Datacenter IP blocks won't recover on retry; fail fast.
                        return pd.DataFrame(), exc, kind
                    if attempt == 0:
                        time.sleep(0.75)
                        continue
                    return pd.DataFrame(), exc, kind
                if gl.empty:
                    return gl, None, None
                return gl.assign(player=name, season=season, season_type=season_type), None, None
            return pd.DataFrame(), last_exc, _classify(last_exc) if last_exc else None

        jobs = [
            (s, t) for s in needed for t in ("Regular Season", "Playoffs")
        ]
        errors: list[tuple[Exception, str]] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                results = list(pool.map(lambda args: _fetch_one(*args), jobs))
            for gl, exc, kind in results:
                if exc is not None and kind is not None:
                    errors.append((exc, kind))
                if not gl.empty:
                    frames.append(gl)

        if not frames:
            # Nothing came back. If every job failed with an exception, surface
            # a typed error that the UI can translate into the correct message
            # instead of unconditionally blaming the host.
            if errors and len(errors) == len(jobs):
                # Prefer the most actionable failure mode.
                priority = {"blocked": 0, "timeout": 1, "unknown": 2}
                errors.sort(key=lambda e: priority.get(e[1], 99))
                exc, kind = errors[0]
                raise NBADataUnavailable(
                    f"Failed to fetch game logs for {name!r}: {exc}",
                    kind=kind,
                    player=name,
                ) from exc
            # API responded cleanly but had no rows for these seasons.
            raise NBADataUnavailable(
                f"NBA stats API returned no game logs for {name!r} in {seasons}.",
                kind="empty",
                player=name,
            )
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Game_ID", "player"])
        # Normalize player to the requested (possibly diacritic) form before
        # persisting so future cache hits match roster keys exactly.
        out["player"] = name
        out.to_parquet(cache_path, index=False)
        return out[out["season"].isin(seasons)].copy()

    # ── Modeling-ready data ──────────────────────────────────────────────────
    def load_player_data(
        self,
        roster: dict[str, list[str]] | dict[str, dict],
        *,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Return a tidy modeling-ready DataFrame for every player in ``roster``.

        ``roster`` accepts either ``{name: [season, ...]}`` or the notebook's
        ``{name: {"seasons": [...], "proj": {...}}}`` shape.
        """
        parts: list[pd.DataFrame] = []
        for name, entry in roster.items():
            seasons = entry["seasons"] if isinstance(entry, dict) and "seasons" in entry else entry
            if verbose:
                print(f"Loading {name}  ({', '.join(seasons)}) …")
            parts.append(self.fetch_player_seasons(name, list(seasons)))

        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if raw.empty:
            return raw

        df = raw.rename(columns=NBA_RENAME)
        df["game_date"] = pd.to_datetime(df["game_date"], format="%b %d, %Y", errors="coerce")
        df = df.sort_values(["player", "game_date"]).reset_index(drop=True)
        df["pra"] = df["pts"] + df["reb"] + df["ast"]
        df["stl_blk"] = df["stl"] + df["blk"]
        df["fantasy_score"] = fantasy(df)
        df = add_pregame_features(df)

        # ── RACE context + role features (all pregame-safe) ───────────────────
        df = build_context_features(df)
        df = build_role_features(df)

        # ── Market features: join pregame consensus lines (NaN-safe) ─────────
        df = build_market_features(df)

        return df

    def modeling_frame(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Subset ``player_data`` to modeling columns.

        Optional context columns are allowed to remain NaN; model pipelines use
        imputers and target-specific feature selection.
        """
        meta = [c for c in ("game_date", "MATCHUP") if c in player_data.columns]
        cols = [c for c in player_data.columns if c not in {"player", *meta}]
        target_cols = [spec["target"] for spec in MODEL_SPECS.values() if spec["target"] in player_data.columns]
        return (
            player_data[["player", *meta, *cols]]
            .dropna(subset=target_cols)
            .reset_index(drop=True)
        )

    def player_modeling_rows(
        self,
        player: str,
        last_n: int,
        *,
        seasons: Iterable[str] | None = None,
        modeling_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the last ``last_n`` modeling-ready rows for ``player``.

        If the player isn't in ``modeling_df`` (or it's None), game logs are
        fetched on demand. ``seasons`` defaults to the current + previous season.
        """
        if modeling_df is not None:
            rows = modeling_df[modeling_df["player"] == player].tail(last_n)
            if not rows.empty:
                return rows

        if seasons is None:
            from datetime import date

            today = date.today()
            if today.month >= 10:
                start_year = today.year - 1
                end_exclusive = today.year + 1
            else:
                start_year = today.year - 2
                end_exclusive = today.year
            seasons = nba_seasons(start_year, end_exclusive)

        single = self.load_player_data({player: list(seasons)})
        if single.empty:
            raise ValueError(f"No NBA game logs found for '{player}'")
        rows = self.modeling_frame(single).tail(last_n)
        if rows.empty:
            raise ValueError(f"Fetched data for '{player}' but no complete rows after dropna.")
        return rows
