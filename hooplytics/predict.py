"""Prediction & decision engine: scenario, project, custom_prop, fantasy_decisions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .calibration import (
    DEFAULT_CALIBRATION_PATH,
    Calibration,
    apply_calibration,
    load_calibration,
)
from .constants import MODEL_TO_COL
from .data import PlayerStore
from .models import ModelBundle, load_models


# ── Lazy default-artifact loaders ────────────────────────────────────────────
_DEFAULT_CALIB: Calibration | None | object = object()  # sentinel: not yet loaded
_DEFAULT_PLAYOFF_BUNDLE: ModelBundle | None | object = object()


def _resolve_default_calibration() -> Calibration | None:
    """Return the repo-shipped calibration once and memoize the result."""
    global _DEFAULT_CALIB
    if _DEFAULT_CALIB is not object():
        return _DEFAULT_CALIB  # type: ignore[return-value]
    env_path = os.getenv("HOOPLYTICS_CALIBRATION", "").strip()
    candidate = Path(env_path) if env_path else DEFAULT_CALIBRATION_PATH
    _DEFAULT_CALIB = load_calibration(candidate)
    return _DEFAULT_CALIB  # type: ignore[return-value]


def _resolve_default_playoff_bundle() -> ModelBundle | None:
    """Return the repo-shipped playoff bundle once and memoize the result."""
    global _DEFAULT_PLAYOFF_BUNDLE
    if _DEFAULT_PLAYOFF_BUNDLE is not object():
        return _DEFAULT_PLAYOFF_BUNDLE  # type: ignore[return-value]
    env_path = os.getenv("HOOPLYTICS_PLAYOFFS_BUNDLE", "").strip()
    if env_path and Path(env_path).exists():
        candidate = Path(env_path)
    else:
        candidate = Path("bundles/race_playoffs.joblib")
    if not candidate.exists():
        _DEFAULT_PLAYOFF_BUNDLE = None
        return None
    try:
        _DEFAULT_PLAYOFF_BUNDLE = load_models(candidate)
    except Exception:  # noqa: BLE001 — corrupt artifact shouldn't break inference
        _DEFAULT_PLAYOFF_BUNDLE = None
    return _DEFAULT_PLAYOFF_BUNDLE  # type: ignore[return-value]


def _bundle_for_context(
    default_bundle: ModelBundle,
    *,
    is_playoff: bool,
    playoff_bundle: ModelBundle | None,
) -> ModelBundle:
    """Pick the playoff bundle when the recent rows look post-season and one
    is available; otherwise stick with the regular-season bundle."""
    if is_playoff:
        candidate = playoff_bundle if playoff_bundle is not None else _resolve_default_playoff_bundle()
        if candidate is not None and candidate.estimators:
            return candidate
    return default_bundle


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
    feature_halflife: float = 2.5,
    playoffs_feature_halflife: float = 1.5,
    playoffs_blend: float = 0.4,
    playoff_bundle: ModelBundle | None = None,
    calibration: Calibration | None = None,
    line_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Project ``player``'s next game from a recency-weighted view of their last
    ``last_n`` modeling rows.

    Players not in ``modeling_df`` are auto-fetched via ``store``.

    Behaviour
    ---------
    * Features are an EWMA across the last ``last_n`` games (weighted toward
      the most recent), so the latest game dominates.
    * When the recent rows are predominantly Playoffs games, predictions
      switch to ``playoff_bundle`` (or the repo-shipped one auto-loaded from
      ``bundles/race_playoffs.joblib``) when available, and are blended with
      the player's playoff-only recent average via ``playoffs_blend``.
    * When ``calibration`` (or the repo-shipped one) is available **and**
      ``line_lookup`` provides a market line for a stat, the prediction is
      blended with the calibrated market baseline. Pass ``line_lookup={}`` to
      disable calibration even when an artifact is present.
    """
    rows = store.player_modeling_rows(
        player, last_n, seasons=seasons, modeling_df=modeling_df
    )
    is_playoff = _is_playoff_context(rows)
    halflife = playoffs_feature_halflife if is_playoff else feature_halflife
    active = _bundle_for_context(bundle, is_playoff=is_playoff, playoff_bundle=playoff_bundle)
    calib = calibration if calibration is not None else _resolve_default_calibration()

    proj_rows: list[dict] = []
    for name, est in active.estimators.items():
        spec = active.specs[name]
        feats = _ewma_feature_row(rows, spec["features"], halflife)
        if not set(spec["features"]).issubset(feats.columns):
            continue
        pred = float(est.predict(feats)[0])
        if is_playoff:
            po_avg = _playoff_recent_average(rows, name)
            if not np.isnan(po_avg):
                pred = (1.0 - playoffs_blend) * pred + playoffs_blend * po_avg
        line = line_lookup.get(name) if line_lookup else None
        if line is not None:
            pred = apply_calibration(
                pred, model=name, player=player, line=float(line), calib=calib,
            )
        proj_rows.append({
            "model": name,
            "target": spec["target"],
            "uses": ", ".join(spec["features"]),
            "prediction": round(pred, 2),
        })
    proj = pd.DataFrame(proj_rows)
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


def _is_playoff_context(rows: pd.DataFrame, threshold: float = 0.5) -> bool:
    """True when the recent rows are predominantly Playoffs games.

    Uses the ``season_type`` column attached upstream by ``PlayerStore`` (values
    are typically ``"Regular Season"`` or ``"Playoffs"``). Falls back to False
    when the column is missing so behaviour stays identical for callers that
    pass synthetic data without the column.
    """
    if rows.empty or "season_type" not in rows.columns:
        return False
    st_norm = rows["season_type"].astype(str).str.lower()
    share = st_norm.str.contains("playoff").mean()
    return bool(share >= threshold)


def _ewma_feature_row(
    rows: pd.DataFrame, feature_cols: list[str], halflife: float
) -> pd.DataFrame:
    """Single-row DataFrame of features weighted toward the most recent games.

    ``halflife`` is in rows. Smaller → more weight on the latest game. Falling
    back to the median preserves prior behaviour when EWMA can't be computed.
    """
    feats = rows.reindex(columns=feature_cols)
    if feats.empty:
        return feats.median(numeric_only=True).to_frame().T
    try:
        ewm = feats.ewm(halflife=halflife, adjust=True).mean().iloc[[-1]]
        # ewm() leaves NaNs only where every value in a column is NaN — fall
        # back to the column median in that rare case.
        med = feats.median(numeric_only=True)
        return ewm.fillna(med)
    except Exception:
        return feats.median(numeric_only=True).to_frame().T


def _playoff_recent_average(rows: pd.DataFrame, model_name: str) -> float:
    """Mean of ``model_name`` target across only the Playoffs rows in ``rows``."""
    col = MODEL_TO_COL[model_name]
    if col not in rows.columns or rows.empty or "season_type" not in rows.columns:
        return float("nan")
    mask = rows["season_type"].astype(str).str.lower().str.contains("playoff")
    if not mask.any():
        return float("nan")
    return float(rows.loc[mask, col].mean())


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
    confidence_margin: float = 0.05,
    weight_model: float = 0.5,
    weight_5_game: float = 0.35,
    playoffs_weight_5_game: float = 0.6,
    playoffs_blend: float = 0.4,
    feature_halflife: float = 2.5,
    playoffs_feature_halflife: float = 1.5,
    playoff_bundle: ModelBundle | None = None,
    calibration: Calibration | None = None,
) -> dict:
    """Run a single prop bet through the decision engine.

    If ``line`` is omitted and ``odds_api_key`` is provided, the live consensus
    line is fetched from The Odds API. Raises ``ValueError`` if no line is
    available from any source.

    Decision logic is symmetric: the threshold is the recent-form-blended
    line and the ``confidence_margin`` only flags low-confidence calls (it no
    longer biases every prop toward LESS). When the player's recent rows are
    predominantly Playoffs games, recency dominates: ``weight_5_game`` is
    raised to ``playoffs_weight_5_game``, the model prediction is blended
    with the playoff-only recent average using ``playoffs_blend``, and a
    dedicated ``playoff_bundle`` (when provided or auto-loaded from
    ``bundles/race_playoffs.joblib``) is used in place of the regular bundle.

    When a calibration artifact is available (passed in or auto-loaded from
    ``bundles/calibration_v1.json``), the model's prediction is additionally
    blended with the market-anchored baseline ``a + b * line`` to remove
    systematic bias surfaced by the historical-odds backtest.
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
    is_playoff = _is_playoff_context(rows)
    active = _bundle_for_context(
        bundle, is_playoff=is_playoff, playoff_bundle=playoff_bundle,
    )
    if model_name not in active.estimators:
        # Fallback: playoff bundle may not cover every market — use regular bundle.
        active = bundle
    spec = active.specs[model_name]
    halflife = playoffs_feature_halflife if is_playoff else feature_halflife
    feats = _ewma_feature_row(rows, spec["features"], halflife)
    pred = float(active.estimators[model_name].predict(feats)[0])

    # Anchor the prediction toward playoff-only recent production so the
    # regular-season-trained model can't anchor on the wrong distribution.
    if is_playoff:
        po_avg = _playoff_recent_average(rows, model_name)
        if not np.isnan(po_avg):
            pred = (1.0 - playoffs_blend) * pred + playoffs_blend * po_avg

    # Market-anchored calibration (no-op when no artifact / no fit for market).
    calib = calibration if calibration is not None else _resolve_default_calibration()
    pred = apply_calibration(
        pred, model=model_name, player=player, line=float(line), calib=calib,
    )

    if last_5_avg is None:
        target_col = MODEL_TO_COL[model_name]
        last_5_avg = float(rows[target_col].mean()) if target_col in rows.columns else line

    w5 = playoffs_weight_5_game if is_playoff else weight_5_game
    weighted = w5 * last_5_avg + (1 - w5) * line
    final_threshold = line + weight_model * (weighted - line)

    # Symmetric decision: prediction vs the recent-form-blended threshold.
    # ``confidence_margin`` only tags whether the gap clears the noise floor.
    edge = float(pred - final_threshold)
    noise = float(confidence_margin * max(abs(line), 1.0))
    confidence = "HIGH" if abs(edge) >= noise else "LOW"
    call = "MORE ✅" if edge > 0 else "LESS ❌"

    return {
        "player": player,
        "model": model_name,
        "line source": line_source,
        "model prediction": round(pred, 2),
        "posted line": round(line, 2),
        "5-game avg": round(last_5_avg, 2),
        "adj. threshold": round(final_threshold, 2),
        "edge": round(edge, 2),
        "confidence": confidence,
        "playoffs": is_playoff,
        "calibrated": calib is not None and model_name in calib.per_market,
        "bundle": "playoffs" if (active is not bundle and is_playoff) else "regular",
        "call": call,
    }


# ── Full per-player decision table ───────────────────────────────────────────
def fantasy_decisions(
    player: str,
    *,
    bundle: ModelBundle,
    store: PlayerStore,
    last_n: int = 5,
    confidence_margin: float = 0.05,
    weight_model: float = 0.5,
    weight_5_game: float = 0.35,
    playoffs_weight_5_game: float = 0.6,
    playoffs_blend: float = 0.4,
    feature_halflife: float = 2.5,
    playoffs_feature_halflife: float = 1.5,
    live_lines: dict[str, float] | None = None,
    widget_projections: dict[str, float] | None = None,
    modeling_df: pd.DataFrame | None = None,
    seasons: Iterable[str] | None = None,
    playoff_bundle: ModelBundle | None = None,
    calibration: Calibration | None = None,
) -> pd.DataFrame:
    """Generate MORE/LESS decisions for every model in ``bundle`` for ``player``.

    Line source priority (per stat):
      1. ``live_lines[model]``        — full ``confidence_margin`` applied (vig).
      2. ``widget_projections[model]`` — user baseline, no margin.
      3. season average               — fallback, no margin.

    Decision rule is **symmetric**: prediction is compared to a recent-form-blended
    threshold; ``confidence_margin`` only flags HIGH/LOW confidence rather than
    biasing the call toward LESS. When the player's recent rows are predominantly
    Playoffs games, recency dominates: ``weight_5_game`` is raised to
    ``playoffs_weight_5_game``, the rolling features are pulled toward the most
    recent game via a shorter EWMA half-life, and the model prediction is
    blended with the playoff-only recent average via ``playoffs_blend``.
    """
    rows = store.player_modeling_rows(
        player, max(last_n, 30), seasons=seasons, modeling_df=modeling_df
    )
    season_rows = rows
    last_rows = rows.tail(last_n)
    is_playoff = _is_playoff_context(last_rows)
    halflife = playoffs_feature_halflife if is_playoff else feature_halflife
    w5 = playoffs_weight_5_game if is_playoff else weight_5_game
    active = _bundle_for_context(
        bundle, is_playoff=is_playoff, playoff_bundle=playoff_bundle,
    )
    calib = calibration if calibration is not None else _resolve_default_calibration()

    # Per-model predictions using EWMA-weighted recent feature rows so the
    # most recent game dominates the projection (median previously diluted it).
    # Iterate over the *regular* bundle so every market the report expects is
    # always emitted; fall back to it when the playoff bundle doesn't cover a
    # particular market.
    predictions: dict[str, float] = {}
    for name in bundle.estimators.keys():
        est_bundle = active if name in active.estimators else bundle
        est = est_bundle.estimators[name]
        spec = est_bundle.specs[name]
        feats = _ewma_feature_row(last_rows, spec["features"], halflife)
        pred = float(est.predict(feats)[0])
        if is_playoff:
            po_avg = _playoff_recent_average(last_rows, name)
            if not np.isnan(po_avg):
                pred = (1.0 - playoffs_blend) * pred + playoffs_blend * po_avg
        predictions[name] = pred

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

        # Market-anchored calibration: only meaningful when the line is a real
        # market consensus (live source). Widget/season-avg lines are
        # user-supplied baselines, not market truth, so we skip calibration
        # there to avoid double-counting.
        if source == "live":
            prediction = apply_calibration(
                prediction, model=model_name, player=player,
                line=float(line), calib=calib,
            )

        five = _recent_average(last_rows, model_name)
        weighted = (w5 * five + (1 - w5) * line) if not np.isnan(five) else line
        final_threshold = float(line + weight_model * (weighted - line))

        # Symmetric decision band — no one-sided LESS bias.
        edge = float(prediction - final_threshold)
        noise = float(margin * max(abs(line), 1.0))
        confidence = "HIGH" if abs(edge) >= noise else "LOW"
        decision = "MORE ✅" if edge > 0 else "LESS ❌"

        out.append({
            "model": model_name,
            "prediction": round(prediction, 2),
            "line": round(line, 2),
            "source": source,
            "5-game avg": round(five, 2) if not np.isnan(five) else None,
            "threshold": round(final_threshold, 2),
            "adj. threshold": round(final_threshold, 2),
            "confidence": confidence,
            "playoffs": is_playoff,
            "calibrated": (
                source == "live"
                and calib is not None
                and model_name in calib.per_market
            ),
            "bundle": "playoffs" if active is not bundle and is_playoff else "regular",
            "decision": decision,
        })
    return pd.DataFrame(out)
