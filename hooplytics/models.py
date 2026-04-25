"""Trained model bundle, training, persistence, and lookup."""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ALL_COLS, MODEL_CACHE_DIR, MODEL_SPECS


@dataclass
class ModelBundle:
    """Container for trained sklearn pipelines + their evaluation metrics."""

    estimators: dict[str, Pipeline]
    specs: dict[str, dict[str, Any]] = field(default_factory=lambda: MODEL_SPECS)
    metrics: pd.DataFrame | None = None
    cv_rmse: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, dict] = field(default_factory=dict)
    trained_at: str = ""
    train_players: list[str] = field(default_factory=list)
    train_seasons: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0

    def predict(self, model_name: str, features: pd.DataFrame) -> float:
        if model_name not in self.estimators:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {list(self.estimators)}")
        return float(self.estimators[model_name].predict(features)[0])

    @property
    def names(self) -> list[str]:
        return list(self.estimators)


# ── Estimator factory ────────────────────────────────────────────────────────
def build_estimator(kind: str) -> tuple[Pipeline, dict]:
    if kind == "knn":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(weights="distance")),
        ])
        return pipe, {"model__n_neighbors": [3, 5, 7, 10, 15, 20]}
    if kind == "rf":
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", RandomForestRegressor(random_state=123, n_jobs=-1)),
        ])
        return pipe, {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        }
    if kind == "ridge":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(random_state=123)),
        ])
        return pipe, {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    raise ValueError(f"Unknown estimator kind '{kind}'")


# ── Training ─────────────────────────────────────────────────────────────────
def train_models(
    player_data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 123,
    verbose: bool = False,
) -> ModelBundle:
    """Train all 8 models on ``player_data`` and return a ``ModelBundle``.

    ``player_data`` must already contain pregame-safe rolling features
    (use ``PlayerStore.load_player_data``).
    """
    cols = [c for c in ALL_COLS if c in player_data.columns]
    meta = [c for c in ("game_date", "MATCHUP") if c in player_data.columns]
    modeling_df = (
        player_data[["player", *meta, *cols]]
        .dropna(subset=cols)
        .reset_index(drop=True)
    )
    train_df, test_df = train_test_split(
        modeling_df, test_size=test_size, random_state=random_state
    )

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    estimators: dict[str, Pipeline] = {}
    cv_rmse: dict[str, float] = {}
    best_params: dict[str, dict] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, spec in MODEL_SPECS.items():
            X = train_df[spec["features"]]
            y = train_df[spec["target"]]
            pipe, grid = build_estimator(spec["kind"])
            gs = GridSearchCV(
                pipe, grid, cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1, refit=True,
            )
            gs.fit(X, y)
            estimators[name] = gs.best_estimator_
            cv_rmse[name] = float(-gs.best_score_)
            best_params[name] = dict(gs.best_params_)
            if verbose:
                print(f"  {name:14s}  cv RMSE = {-gs.best_score_:.3f}   best = {gs.best_params_}")

    # Test-set metrics
    metric_rows = []
    for name, est in estimators.items():
        spec = MODEL_SPECS[name]
        Xt = test_df[spec["features"]]
        yt = test_df[spec["target"]].to_numpy()
        yhat = est.predict(Xt)
        metric_rows.append({
            "model": name,
            "target": spec["target"],
            "kind": spec["kind"],
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yhat))), 3),
            "MAE": round(float(mean_absolute_error(yt, yhat)), 3),
            "R²": round(float(r2_score(yt, yhat)), 3),
        })
    metrics = pd.DataFrame(metric_rows).sort_values("model").reset_index(drop=True)

    return ModelBundle(
        estimators=estimators,
        metrics=metrics,
        cv_rmse=cv_rmse,
        best_params=best_params,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_players=sorted(player_data["player"].unique().tolist()),
        train_seasons=sorted(player_data["season"].unique().tolist())
            if "season" in player_data.columns else [],
        n_train=len(train_df),
        n_test=len(test_df),
    )


# ── Persistence ──────────────────────────────────────────────────────────────
def _bundle_hash(players: list[str], seasons: list[str], spec_version: str = "v1") -> str:
    payload = "|".join(sorted(players)) + "::" + "|".join(sorted(seasons)) + "::" + spec_version
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def save_models(bundle: ModelBundle, path: Path | str) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_models(path: Path | str) -> ModelBundle:
    import joblib

    return joblib.load(Path(path))


def ensure_models(
    player_data: pd.DataFrame,
    *,
    cache_dir: Path | str = MODEL_CACHE_DIR,
    force: bool = False,
    verbose: bool = False,
) -> ModelBundle:
    """Load a cached ``ModelBundle`` matching ``player_data``, or train + save one.

    The cache key hashes the sorted player + season list and a feature-spec
    version tag — when those change, a fresh bundle is trained automatically.
    """
    players = sorted(player_data["player"].unique().tolist())
    seasons = (
        sorted(player_data["season"].unique().tolist())
        if "season" in player_data.columns else []
    )
    key = _bundle_hash(players, seasons)
    cache_path = Path(cache_dir) / f"models_{key}.joblib"

    if cache_path.exists() and not force:
        try:
            bundle = load_models(cache_path)
            if verbose:
                print(f"  loaded cached models from {cache_path}")
            return bundle
        except Exception as exc:  # noqa: BLE001 — corrupt cache, retrain
            if verbose:
                print(f"  ! cache unreadable ({exc}); retraining")

    if verbose:
        print(f"  training models for {len(players)} player(s) — first run takes ~30s …")
    bundle = train_models(player_data, verbose=verbose)
    try:
        save_models(bundle, cache_path)
        if verbose:
            print(f"  saved model bundle → {cache_path}")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  ! could not persist models: {exc}")
    return bundle
