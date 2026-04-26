"""Market feature engineering: join pregame betting lines as training features.

Pregame consensus lines from The Odds API historical data are joined as
model features so the RACE ensemble can learn from market signal.

All joins are pregame-safe: lines are fetched at noon ET on the game date,
before any NBA tip-offs. Rows with no market data receive NaN and are handled
gracefully by the imputer steps in each model pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ODDS_HIST_CACHE_DIR

# ── Market feature column names ──────────────────────────────────────────────
MARKET_FEATURE_COLS: list[str] = [
    "market_points_line",
    "market_rebounds_line",
    "market_assists_line",
    "market_threepm_line",
    "market_books_count",
]

# Internal mapping from ODDS_MARKETS model names to market feature columns.
_MODEL_TO_COL: dict[str, str] = {
    "points":   "market_points_line",
    "rebounds": "market_rebounds_line",
    "assists":  "market_assists_line",
    "threepm":  "market_threepm_line",
}


def _canon(s: str) -> str:
    """Lowercase, strip non-alpha — for fuzzy player name matching."""
    return re.sub(r"[^a-z]", "", s.lower())


def build_market_features(
    df: pd.DataFrame,
    *,
    odds_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Left-join pregame market lines onto a player game-log DataFrame.

    Reads all cached historical odds from ``ODDS_HIST_CACHE_DIR`` (no API
    calls made here).  Matches players via canonicalized name comparison
    to handle minor punctuation differences.

    Columns added to ``df``
    -----------------------
    market_points_line
        Consensus Over/Under line for points.
    market_rebounds_line
        Consensus Over/Under line for rebounds.
    market_assists_line
        Consensus Over/Under line for assists.
    market_threepm_line
        Consensus Over/Under line for 3-pointers made.
    market_books_count
        Maximum number of bookmakers that quoted any market for this player-date.
    """
    from .odds import load_cached_historical_odds

    # Initialize all market columns with NaN
    df = df.copy()
    for col in MARKET_FEATURE_COLS:
        df[col] = np.nan

    if "game_date" not in df.columns or "player" not in df.columns:
        return df

    odds_df = load_cached_historical_odds(odds_cache_dir)
    if odds_df.empty:
        return df

    # ── Pivot odds to wide: one row per (game_date, player) ──────────────
    pivot = odds_df.pivot_table(
        index=["game_date", "player"],
        columns="model",
        values="line",
        aggfunc="median",
    ).reset_index()
    pivot.columns.name = None

    for model_name, col_name in _MODEL_TO_COL.items():
        if model_name in pivot.columns:
            pivot = pivot.rename(columns={model_name: col_name})
        else:
            pivot[col_name] = np.nan

    # Max bookmaker count per player-date
    books_max = (
        odds_df.groupby(["game_date", "player"])["books"]
        .max()
        .reset_index()
        .rename(columns={"books": "market_books_count"})
    )
    pivot = pivot.merge(books_max, on=["game_date", "player"], how="left")

    # ── Canonicalize player names for fuzzy matching ──────────────────────
    odds_players = pivot["player"].unique()
    canon_to_odds: dict[str, str] = {_canon(p): p for p in odds_players}

    # Normalize df's game_date to date-only (no time component)
    df_dates = pd.to_datetime(df["game_date"]).dt.normalize()
    df_odds_player = df["player"].map(_canon).map(canon_to_odds)

    # ── Build join keys ───────────────────────────────────────────────────
    join_keys = pd.DataFrame(
        {
            "game_date": df_dates.values,
            "player": df_odds_player.values,
            "_row": np.arange(len(df)),
        }
    ).dropna(subset=["player"])

    if join_keys.empty:
        return df

    merged = join_keys.merge(pivot, on=["game_date", "player"], how="left")

    for col in MARKET_FEATURE_COLS:
        if col not in merged.columns:
            continue
        filled = merged.set_index("_row")[col].reindex(range(len(df)))
        df[col] = filled.values

    return df
