"""Player game-log fetching, caching, and pregame-safe feature engineering."""
from __future__ import annotations

import time
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


def _join_opponent_context(df: pd.DataFrame, opp_df: pd.DataFrame) -> pd.DataFrame:
    """Join opponent-context columns from ``opp_df`` into ``df``.

    Parses the opponent team abbreviation from the MATCHUP column
    (``"OKC @ LAL"`` → ``"LAL"``, ``"OKC vs. LAL"`` → ``"LAL"``)
    and maps the nba_api season string to a BDL integer year
    (``"2024-25"`` → ``2024``).

    Rows whose opponent is not found in ``opp_df`` receive NaN for
    the context columns (handled gracefully by RandomForest).
    """
    if opp_df.empty:
        return df
    if "MATCHUP" not in df.columns or "season" not in df.columns:
        return df

    opp_cols = [c for c in opp_df.columns if c not in ("bdl_season", "abbreviation")]
    if not opp_cols:
        return df

    work = df.copy()
    work["_opp_abbr"]   = work["MATCHUP"].str.strip().str.split().str[-1].str.upper()
    work["_bdl_season"] = (
        work["season"].astype(str).str.split("-").str[0].replace("", np.nan).astype(float).astype("Int64")
    )

    lookup = opp_df.rename(columns={"bdl_season": "_bdl_season", "abbreviation": "_opp_abbr"})
    merged = work.merge(lookup, on=["_bdl_season", "_opp_abbr"], how="left")

    for col in opp_cols:
        if col in merged.columns:
            work[col] = merged[col].values

    return work.drop(columns=["_opp_abbr", "_bdl_season"])


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


class PlayerStore:
    """Fetch and cache NBA player game logs as Parquet under ``cache_dir``.

    Parameters
    ----------
    cache_dir
        Directory for per-player Parquet files. Defaults to ``data/cache``.
    pause
        Sleep between successive ``nba_api`` calls (seconds).
    """

    def __init__(self, cache_dir: Path | str = CACHE_DIR, pause: float = 0.25, bdl_client: object | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.bdl_client = bdl_client  # optional BDLClient for opponent context

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
    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name.replace(' ', '_')}.parquet"

    def _seed_cache_path(self, name: str) -> Path:
        """Read-only seed cache shipped with the repo for fast cold starts."""
        return Path(__file__).resolve().parent.parent / "data" / "seed_cache" / f"{name.replace(' ', '_')}.parquet"

    def fetch_player_seasons(self, name: str, seasons: list[str]) -> pd.DataFrame:
        """Return raw game logs for ``name`` across ``seasons``, using disk cache.

        Cached players are returned without an API call when the cache covers
        every requested season; otherwise the missing seasons are fetched and
        merged into the cache.
        """
        from nba_api.stats.endpoints import playergamelog

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
                    return cached[cached["season"].isin(seasons)].copy()
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
        # Tighter timeouts + smaller backoff: nba_api occasionally hangs the
        # full default 30s window per attempt, which previously stacked up to
        # ~3min for a single brand-new player across 2 seasons * 2 types * 3
        # retries. 12s + 2s/4s backoff fails fast and keeps the UI responsive.
        for season in needed:
            season_frames: list[pd.DataFrame] = []
            for season_type in ("Regular Season", "Playoffs"):
                gl = pd.DataFrame()
                for attempt in range(2):
                    try:
                        gl = playergamelog.PlayerGameLog(
                            player_id=pid, season=season,
                            season_type_all_star=season_type, timeout=12
                        ).get_data_frames()[0]
                        break
                    except Exception:  # noqa: BLE001 — transient network errors
                        if attempt == 1:
                            break
                        time.sleep(1.5)
                if not gl.empty:
                    gl = gl.assign(player=name, season=season, season_type=season_type)
                    season_frames.append(gl)
                # Brief politeness pause only when an actual network hop happened.
                if not gl.empty or attempt > 0:
                    time.sleep(self.pause)
            if season_frames:
                frames.extend(season_frames)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Game_ID", "player"])
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
        df = build_context_features(df, bdl_client=self.bdl_client)
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
