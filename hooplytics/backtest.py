"""Projection accuracy backtest: retro-evaluate the RACE models against historical actuals.

For each evaluated game the model's prediction is reconstructed using only the
data that would have been available *before* that game (pregame-safe rolling
features are already baked into the modeling frame via shift(1) before rolling).

Line source priority per game:
  1. Cached historical odds (ingest-odds) — real pregame sportsbook consensus.
  2. Prior L5 average — fallback when no cached line exists for that date.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import MODEL_TO_COL
from .data import nba_seasons

if TYPE_CHECKING:
    from .data import PlayerStore
    from .models import ModelBundle


def _default_seasons() -> list[str]:
    today = date.today()
    if today.month >= 10:
        start_year = today.year - 1
        end_exclusive = today.year + 1
    else:
        start_year = today.year - 2
        end_exclusive = today.year
    return nba_seasons(start_year, end_exclusive)


def _load_player_odds(player: str, stat: str) -> pd.DataFrame:
    """Return cached historical odds rows for ``player`` / ``stat``, or empty."""
    try:
        from .odds import _canon_name as _canon, load_cached_historical_odds
        odds = load_cached_historical_odds()
        if odds.empty:
            return pd.DataFrame()
        odds = odds[odds["model"] == stat].copy()
        if odds.empty:
            return pd.DataFrame()
        canon = _canon(player)
        odds["_canon"] = odds["player"].map(_canon)
        odds = odds[odds["_canon"] == canon].copy()
        if odds.empty:
            return pd.DataFrame()
        odds["_date_norm"] = pd.to_datetime(odds["game_date"], errors="coerce").dt.normalize()
        return odds[["_date_norm", "line"]].dropna()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _lookup_line(odds: pd.DataFrame, game_date: pd.Timestamp) -> float | None:
    """Return the sportsbook line for ``game_date`` from the pre-filtered odds
    DataFrame, or ``None`` when no line is cached for that date."""
    if odds.empty:
        return None
    match = odds[odds["_date_norm"] == game_date.normalize()]
    if match.empty:
        return None
    return float(match.iloc[0]["line"])


def retro_projection_table(
    player: str,
    stat: str,
    *,
    store: PlayerStore,
    bundle: ModelBundle,
    n_games: int = 20,
    seasons: list[str] | None = None,
    min_prior_games: int = 5,
    last_n: int = 10,
) -> pd.DataFrame:
    """Reconstruct what the RACE model would have projected before each of the
    last ``n_games`` games, then compare to the actual result.

    The rolling features in the modeling frame are already pregame-safe
    (each feature is computed from prior games only via shift(1) before
    rolling), so filtering the frame to rows strictly before game N and
    running inference on that slice is a valid retro-evaluation.

    Line source priority (per game):
      1. Cached sportsbook line (from ``ingest-odds``) — when available this
         is used as the threshold for both the call and the result so the
         evaluation is directly comparable to real betting outcomes.
      2. Prior L5 average — fallback when no historical line is cached.

    Returns a DataFrame with columns:
        game_date, matchup, line, line_source, projected, actual,
        error, abs_error, hooplytics_projection, result

    Returns an empty DataFrame when there is insufficient data.
    """
    from .predict import project_next_game

    if stat not in MODEL_TO_COL:
        raise ValueError(f"Unknown stat '{stat}'. Valid: {', '.join(MODEL_TO_COL)}")

    target_col = MODEL_TO_COL[stat]
    seasons = seasons or _default_seasons()

    raw = store.load_player_data({player: list(seasons)})
    if raw.empty:
        return pd.DataFrame()

    full_df = store.modeling_frame(raw)
    player_rows = (
        full_df[full_df["player"] == player]
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    if target_col not in player_rows.columns:
        return pd.DataFrame()

    n_total = len(player_rows)
    if n_total < min_prior_games + 1:
        return pd.DataFrame()

    # Load cached historical odds for this player/stat once — avoid re-reading
    # the parquet on every iteration of the loop below.
    player_odds = _load_player_odds(player, stat)

    # The earliest evaluable position requires min_prior_games rows before it.
    # We walk ALL positions (not just the last n_games of them) so that
    # historical games with cached odds aren't silently dropped when they fall
    # outside an arbitrary calendar window. The n_games trim is applied AFTER
    # the cached-line filter so "n_games=15" means "the 15 most-recent
    # evaluable games", not "the 15 most-recent calendar positions, of which
    # most have no cached line and silently disappear."
    eval_positions = list(range(min_prior_games, n_total))

    records: list[dict] = []
    missing_line_dates: list[str] = []  # game dates skipped due to no cached sportsbook line

    for pos in eval_positions:
        # Rows strictly before this game — the "pregame" view the model would
        # have had before tip-off.
        prior = player_rows.iloc[:pos].copy()
        if len(prior) < min_prior_games:
            continue

        game_date_ts = pd.Timestamp(player_rows.iloc[pos]["game_date"])
        game_date_str = game_date_ts.strftime("%Y-%m-%d") if pd.notna(game_date_ts) else ""

        # Only include games where a real sportsbook line is cached.
        cached_line = _lookup_line(player_odds, game_date_ts)
        if cached_line is None:
            if game_date_str:
                missing_line_dates.append(game_date_str)
            continue
        threshold = cached_line

        try:
            proj_df = project_next_game(
                player,
                bundle=bundle,
                store=store,
                last_n=last_n,
                modeling_df=prior,
            )
        except Exception:  # noqa: BLE001
            continue

        if proj_df.empty:
            continue

        stat_rows = proj_df[proj_df["model"] == stat]
        if stat_rows.empty:
            continue

        projected = float(stat_rows.iloc[0]["prediction"])
        actual_raw = player_rows.iloc[pos][target_col]
        if pd.isna(actual_raw):
            continue
        actual = float(actual_raw)

        matchup = player_rows.iloc[pos].get("MATCHUP", "")
        error = actual - projected

        records.append({
            "game_date": game_date_str,
            "matchup": str(matchup) if pd.notna(matchup) else "",
            "line": round(threshold, 2),
            "projected": round(projected, 2),
            "actual": round(actual, 2),
            "error": round(error, 2),
            "abs_error": round(abs(error), 2),
            "hooplytics_projection": "MORE" if projected > threshold else "LESS",
            "result": "Over" if actual > threshold else ("Under" if actual < threshold else "Push"),
        })

    df = pd.DataFrame(records)
    # Trim to the most recent n_games of *evaluable* rows (after cached-line
    # filter). The DataFrame is already in chronological order because we
    # iterated eval_positions ascending, so .tail() takes the latest games.
    if not df.empty and len(df) > n_games:
        df = df.tail(n_games).reset_index(drop=True)
    # Store skipped dates so callers (e.g. Streamlit) can offer to fetch them.
    # Keep dates that are within the trimmed window — anything older is simply
    # outside what the user asked to see.
    if df.empty or "game_date" not in df.columns:
        df.attrs["missing_line_dates"] = missing_line_dates
    else:
        oldest_shown = df["game_date"].min()
        df.attrs["missing_line_dates"] = [
            d for d in missing_line_dates if d >= oldest_shown
        ]
    return df


def backtest_summary(table: pd.DataFrame) -> dict:
    """Compute accuracy metrics from a ``retro_projection_table`` result.

    Returns a dict with ``n_games``, ``mae``, ``rmse``, ``bias``,
    ``directional_accuracy``, and ``median_error``.
    """
    if table.empty or "error" not in table.columns:
        return {
            "n_games": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "directional_accuracy": None,
            "median_error": None,
        }

    errors = table["error"].dropna()
    n = len(errors)
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    bias = float(errors.mean())
    median_error = float(errors.median())

    # Directional accuracy: fraction of games where the call matched the result
    # (both vs the same threshold — sportsbook line or L5 avg).
    if "hooplytics_projection" in table.columns and "result" in table.columns:
        call_correct = (
            ((table["hooplytics_projection"] == "MORE") & (table["result"] == "Over")) |
            ((table["hooplytics_projection"] == "LESS") & (table["result"] == "Under"))
        )
        directional_accuracy = float(call_correct.mean())
    else:
        directional_accuracy = None

    return {
        "n_games": n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "bias": round(bias, 3),
        "directional_accuracy": round(directional_accuracy, 3) if directional_accuracy is not None else None,
        "median_error": round(median_error, 3),
    }
