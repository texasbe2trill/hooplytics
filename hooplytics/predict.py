"""Prediction & decision engine: scenario, project, custom_prop, fantasy_decisions."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import MODEL_TO_COL
from .data import PlayerStore
from .models import ModelBundle


# ── Scenario / projection ────────────────────────────────────────────────────
def predict_scenario(scenario: dict, bundle: ModelBundle) -> pd.DataFrame:
    """Run any model whose required features are all present in ``scenario``."""
    rows = []
    for name, est in bundle.estimators.items():
        spec = bundle.specs[name]
        feats = spec["features"]
        if not set(feats).issubset(scenario):
            continue
        X = pd.DataFrame([{k: scenario[k] for k in feats}])
        pred = float(est.predict(X)[0])
        rows.append({
            "model": name,
            "target": spec["target"],
            "uses": ", ".join(feats),
            "prediction": round(pred, 2),
        })
    return pd.DataFrame(rows)


def project_next_game(
    player: str,
    *,
    bundle: ModelBundle,
    store: PlayerStore,
    last_n: int = 10,
    seasons: Iterable[str] | None = None,
    modeling_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project ``player``'s next game from the median of their last ``last_n`` rows.

    Players not in ``modeling_df`` are auto-fetched via ``store``.
    """
    rows = store.player_modeling_rows(
        player, last_n, seasons=seasons, modeling_df=modeling_df
    )
    drop_cols = [c for c in ("player", "season", "game_date", "MATCHUP") if c in rows.columns]
    feats_all = rows.drop(columns=drop_cols).median(numeric_only=True).to_dict()
    proj = predict_scenario(feats_all, bundle)
    proj.insert(0, "player", player)
    return proj


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _recent_average(rows: pd.DataFrame, model_name: str) -> float:
    col = MODEL_TO_COL[model_name]
    if col not in rows.columns or rows.empty:
        return float("nan")
    return float(rows[col].mean())


# ── Single-prop decision ─────────────────────────────────────────────────────
def custom_prop(
    player: str,
    model_name: str,
    line: float | None = None,
    *,
    bundle: ModelBundle,
    store: PlayerStore,
    odds_api_key: str | None = None,
    last_n: int = 5,
    last_5_avg: float | None = None,
    seasons: Iterable[str] | None = None,
    modeling_df: pd.DataFrame | None = None,
    confidence_margin: float = 0.10,
    weight_model: float = 0.5,
    weight_5_game: float = 0.2,
) -> dict:
    """Run a single prop bet through the decision engine.

    If ``line`` is omitted and ``odds_api_key`` is provided, the live consensus
    line is fetched from The Odds API. Raises ``ValueError`` if no line is
    available from any source.
    """
    if model_name not in bundle.estimators:
        raise KeyError(f"Unknown model '{model_name}'. Pick from {bundle.names}")

    line_source = "manual"
    if line is None:
        if not odds_api_key:
            raise ValueError(
                f"No 'line' supplied and no odds_api_key. "
                f"Pass line=<number> or set ODDS_API_KEY."
            )
        from .odds import fetch_live_player_lines

        ll = fetch_live_player_lines(odds_api_key, [player])
        mask = (ll["player"].str.lower() == player.lower()) & (ll["model"] == model_name)
        if not mask.any():
            raise ValueError(
                f"No live line found for '{player}' / '{model_name}' today. "
                f"Pass line=<number> to use a specific value."
            )
        row = ll.loc[mask].iloc[0]
        line = float(row["line"])
        line_source = f"live ({int(row['books'])} books)"

    rows = store.player_modeling_rows(
        player, last_n, seasons=seasons, modeling_df=modeling_df
    )
    spec = bundle.specs[model_name]
    feats = rows.reindex(columns=spec["features"]).median(numeric_only=True).to_frame().T
    pred = float(bundle.estimators[model_name].predict(feats)[0])

    if last_5_avg is None:
        target_col = MODEL_TO_COL[model_name]
        last_5_avg = float(rows[target_col].mean()) if target_col in rows.columns else line

    weighted = weight_5_game * last_5_avg + (1 - weight_5_game) * line
    final_threshold = line + weight_model * (weighted - line)
    adjusted = final_threshold * (1 + confidence_margin)
    edge = pred - adjusted

    return {
        "player": player,
        "model": model_name,
        "line source": line_source,
        "model prediction": round(pred, 2),
        "posted line": round(line, 2),
        "5-game avg": round(last_5_avg, 2),
        "adj. threshold": round(adjusted, 2),
        "edge": round(edge, 2),
        "call": "MORE ✅" if edge > 0 else "LESS ❌",
    }


# ── Full per-player decision table ───────────────────────────────────────────
def fantasy_decisions(
    player: str,
    *,
    bundle: ModelBundle,
    store: PlayerStore,
    last_n: int = 5,
    confidence_margin: float = 0.10,
    weight_model: float = 0.5,
    weight_5_game: float = 0.2,
    live_lines: dict[str, float] | None = None,
    widget_projections: dict[str, float] | None = None,
    modeling_df: pd.DataFrame | None = None,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Generate MORE/LESS decisions for every model in ``bundle`` for ``player``.

    Line source priority (per stat):
      1. ``live_lines[model]``        — full ``confidence_margin`` applied (vig).
      2. ``widget_projections[model]`` — user baseline, no margin.
      3. season average               — fallback, no margin.
    """
    rows = store.player_modeling_rows(
        player, max(last_n, 30), seasons=seasons, modeling_df=modeling_df
    )
    season_rows = rows
    last_rows = rows.tail(last_n)

    # Per-model predictions using median of feature rows.
    predictions: dict[str, float] = {}
    for name, est in bundle.estimators.items():
        spec = bundle.specs[name]
        feats = last_rows.reindex(columns=spec["features"]).median(numeric_only=True).to_frame().T
        predictions[name] = float(est.predict(feats)[0])

    widget_projections = widget_projections or {}
    live_lines = live_lines or {}

    out = []
    for model_name, prediction in predictions.items():
        if model_name in live_lines:
            line = float(live_lines[model_name])
            margin = confidence_margin
            source = "live"
        elif model_name in widget_projections:
            line = float(widget_projections[model_name])
            margin = 0.0
            source = "widget"
        else:
            season_avg = _recent_average(season_rows, model_name)
            line = float(season_avg) if not np.isnan(season_avg) else prediction
            margin = 0.0
            source = "season avg"

        five = _recent_average(last_rows, model_name)
        weighted = (weight_5_game * five + (1 - weight_5_game) * line) if not np.isnan(five) else line
        final_threshold = float(line + weight_model * (weighted - line))
        adjusted = float(final_threshold * (1 + margin))
        decision = "MORE ✅" if prediction > adjusted else "LESS ❌"

        out.append({
            "model": model_name,
            "prediction": round(prediction, 2),
            "line": round(line, 2),
            "source": source,
            "5-game avg": round(five, 2) if not np.isnan(five) else None,
            "threshold": round(final_threshold, 2),
            "adj. threshold": round(adjusted, 2),
            "decision": decision,
        })
    return pd.DataFrame(out)
