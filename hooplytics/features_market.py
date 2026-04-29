"""Market feature engineering: join pregame betting lines as training features.

Pregame consensus lines from The Odds API historical data are joined as
model features so the RACE ensemble can learn from market signal.

Two families of features are exposed:

* **Raw market lines** (one column per stat) plus the maximum book-count proxy
  for market depth. Available whenever the historical cache has at least one
  row for a (player, date).
* **Derived signal** computed from the per-row payload:
    - de-vigged implied Over probability (``market_<stat>_over_prob``)
    - per-book line dispersion (``market_<stat>_line_std``)
    - line-vs-recent-form delta (``market_<stat>_vs_l5`` / ``_vs_l10``)
    - average overround across markets (``market_overround``) as a market
      quality / book-confidence proxy.

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
# Raw consensus line + market depth (always populated when cache has rows).
MARKET_LINE_COLS: list[str] = [
    "market_points_line",
    "market_rebounds_line",
    "market_assists_line",
    "market_threepm_line",
    "market_turnovers_line",
    "market_pra_line",
    "market_steals_line",
    "market_blocks_line",
    "market_books_count",
]

# Derived signal computed from the cached payload (over/under prices, per-book
# spread, rolling form). Populated only when the source columns are present in
# the cache; otherwise NaN.
MARKET_DERIVED_COLS: list[str] = [
    # De-vigged implied probability the player goes Over the consensus line.
    "market_points_over_prob",
    "market_rebounds_over_prob",
    "market_assists_over_prob",
    "market_threepm_over_prob",
    "market_turnovers_over_prob",
    "market_pra_over_prob",
    "market_steals_over_prob",
    "market_blocks_over_prob",
    # Per-book line dispersion — high std == soft / disagreeing market.
    "market_points_line_std",
    "market_rebounds_line_std",
    "market_assists_line_std",
    "market_threepm_line_std",
    "market_turnovers_line_std",
    "market_pra_line_std",
    "market_steals_line_std",
    "market_blocks_line_std",
    # Average overround across the markets (proxy for vig / liquidity).
    "market_overround",
    # Market-vs-recent-form deltas (line minus rolling rate).
    "market_points_vs_l5",
    "market_points_vs_l10",
    "market_rebounds_vs_l5",
    "market_rebounds_vs_l10",
    "market_assists_vs_l5",
    "market_assists_vs_l10",
    "market_turnovers_vs_l5",
    "market_turnovers_vs_l10",
    "market_pra_vs_l5",
    "market_pra_vs_l10",
    "market_steals_vs_l5",
    "market_steals_vs_l10",
    "market_blocks_vs_l5",
    "market_blocks_vs_l10",
]

MARKET_FEATURE_COLS: list[str] = MARKET_LINE_COLS + MARKET_DERIVED_COLS

# Mapping from Odds API model name to (game-log stat, rolling form basis).
# threepm has no fg3m_l5/l10 baseline so it is intentionally excluded from the
# vs-form delta features.
_MODEL_TO_COL: dict[str, str] = {
    "points":    "market_points_line",
    "rebounds":  "market_rebounds_line",
    "assists":   "market_assists_line",
    "threepm":   "market_threepm_line",
    "turnovers": "market_turnovers_line",
    "pra":       "market_pra_line",
    "steals":    "market_steals_line",
    "blocks":    "market_blocks_line",
}
_MODEL_TO_STAT: dict[str, str] = {
    "points":    "pts",
    "rebounds":  "reb",
    "assists":   "ast",
    "turnovers": "tov",
    "pra":       "pra",
    "steals":    "stl",
    "blocks":    "blk",
}


def _canon(s: str) -> str:
    """Lowercase, strip non-alpha — for fuzzy player name matching."""
    return re.sub(r"[^a-z]", "", s.lower())


def _devig_over_prob(over_price: pd.Series, under_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert decimal Over/Under prices into de-vigged Over probability + overround.

    Returns ``(over_prob, overround)``. NaN where either price is missing or
    invalid (price <= 1.0). De-vig uses simple normalization:
    ``over_prob = (1/over_price) / (1/over_price + 1/under_price)``.
    """
    op = pd.to_numeric(over_price, errors="coerce")
    up = pd.to_numeric(under_price, errors="coerce")
    op = op.where(op > 1.0)
    up = up.where(up > 1.0)
    over_imp = 1.0 / op
    under_imp = 1.0 / up
    total = over_imp + under_imp
    over_prob = over_imp / total
    overround = total - 1.0  # vig (e.g. 0.04 == 4% juice)
    return over_prob, overround


def build_market_features(
    df: pd.DataFrame,
    *,
    odds_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Left-join pregame market lines and derived signal onto a player game-log frame.

    Reads all cached historical odds from ``ODDS_HIST_CACHE_DIR`` (no API
    calls made here).  Matches players via canonicalized name comparison
    to handle minor punctuation differences.

    Columns added to ``df``
    -----------------------
    See :data:`MARKET_FEATURE_COLS` for the full list. Includes raw consensus
    lines, de-vigged Over probabilities, per-book dispersion, average
    overround, and line-vs-recent-form deltas.
    """
    from .odds import load_cached_historical_odds

    df = df.copy()
    for col in MARKET_FEATURE_COLS:
        df[col] = np.nan

    if "game_date" not in df.columns or "player" not in df.columns:
        return df

    odds_df = load_cached_historical_odds(odds_cache_dir)
    if odds_df.empty:
        return df

    # Restrict the historical frame to only the players present in ``df``.
    # The cache holds 400+ players across the league, but per-player feature
    # builds (the hot path on roster mutation) pass in a single player. Without
    # this filter the pivots below run over the full league frame every call.
    df_canon_names = set(df["player"].map(_canon).dropna().unique())
    if df_canon_names:
        odds_df = odds_df[odds_df["player"].map(_canon).isin(df_canon_names)]
    if odds_df.empty:
        return df

    # ── Step 1: derive per-row Over prob + overround if prices are cached ──
    odds_df = odds_df.copy()
    if "over_price" in odds_df.columns and "under_price" in odds_df.columns:
        over_prob, overround = _devig_over_prob(
            odds_df["over_price"], odds_df["under_price"]
        )
        odds_df["over_prob"] = over_prob
        odds_df["overround"] = overround
    else:
        odds_df["over_prob"] = np.nan
        odds_df["overround"] = np.nan
    if "line_std" not in odds_df.columns:
        odds_df["line_std"] = np.nan

    # ── Step 2: pivot lines (long → wide) ────────────────────────────────
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

    # ── Step 3: pivot derived per-market columns ─────────────────────────
    def _pivot_extra(value_col: str, suffix: str) -> pd.DataFrame:
        if value_col not in odds_df.columns:
            return pd.DataFrame(columns=["game_date", "player"])
        wide = odds_df.pivot_table(
            index=["game_date", "player"],
            columns="model",
            values=value_col,
            aggfunc="median",
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(columns={
            m: f"market_{m}_{suffix}" for m in _MODEL_TO_COL.keys() if m in wide.columns
        })
        return wide

    over_prob_wide = _pivot_extra("over_prob", "over_prob")
    line_std_wide = _pivot_extra("line_std", "line_std")

    # Average overround across the four markets per (game_date, player).
    if "overround" in odds_df.columns:
        ovrr_avg = (
            odds_df.groupby(["game_date", "player"])["overround"]
            .mean()
            .reset_index()
            .rename(columns={"overround": "market_overround"})
        )
    else:
        ovrr_avg = pd.DataFrame(columns=["game_date", "player", "market_overround"])

    # Max bookmaker count per player-date (market depth proxy).
    books_max = (
        odds_df.groupby(["game_date", "player"])["books"]
        .max()
        .reset_index()
        .rename(columns={"books": "market_books_count"})
    )

    for extra in (over_prob_wide, line_std_wide, ovrr_avg, books_max):
        if not extra.empty:
            pivot = pivot.merge(extra, on=["game_date", "player"], how="left")

    # ── Step 4: canonicalize names + build join keys ─────────────────────
    odds_players = pivot["player"].unique()
    canon_to_odds: dict[str, str] = {_canon(p): p for p in odds_players}

    df_dates = pd.to_datetime(df["game_date"]).dt.normalize()
    df_odds_player = df["player"].map(_canon).map(canon_to_odds)

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

    # Write back per-column to df, indexing by original row position.
    for col in MARKET_FEATURE_COLS:
        if col not in merged.columns:
            continue
        filled = merged.set_index("_row")[col].reindex(range(len(df)))
        df[col] = filled.values

    # ── Step 5: line-vs-form deltas (computed against in-frame rolling cols)
    for model_name, stat in _MODEL_TO_STAT.items():
        line_col = f"market_{model_name}_line"
        if line_col not in df.columns:
            continue
        line = pd.to_numeric(df[line_col], errors="coerce")
        for window in (5, 10):
            roll_col = f"{stat}_l{window}"
            delta_col = f"market_{model_name}_vs_l{window}"
            if delta_col not in df.columns:
                continue
            if roll_col in df.columns:
                df[delta_col] = line - pd.to_numeric(df[roll_col], errors="coerce")
            # else: leave NaN (column was pre-allocated)

    return df
