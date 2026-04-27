"""RACE model training, persistence, and lookup.

RACE = Role-Adjusted Context Ensemble
- Target-specific feature sets
- Time-aware validation by default
- Baseline/context/role/full variants per target
- Defensive fallback behavior when context columns are missing
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import MODEL_CACHE_DIR, MODEL_SPECS

# Targets that are non-negative count-like distributions where Poisson loss
# typically beats MSE (heteroscedastic, lower bound at zero, often skewed).
COUNT_LIKE_TARGETS: frozenset[str] = frozenset({"stl_blk", "turnovers", "threepm"})

# Half-life (in days) for the exponential recency weight applied to training
# rows. Recent games carry more signal because rotations, role, and shot
# diet drift; ~120d ≈ a third of a season.
RECENCY_HALF_LIFE_DAYS: float = 120.0

# ── RACE feature groups ─────────────────────────────────────────────────────
SCHEDULE_CONTEXT_FEATURES = [
    "rest_days",
    "is_home",
    "is_back_to_back",
    "games_in_last_4_days",
]

OPPONENT_CONTEXT_FEATURES = [
    "opp_pace",
    "opp_def_rtg",
    "opp_off_rtg",
    "opp_stl_pg",
    "opp_blk_pg",
]

AVAILABILITY_CONTEXT_FEATURES = [
    "team_injury_count",
    "opp_injury_count",
    "teammate_usage_missing_proxy",
]

LINEUP_CONTEXT_FEATURES = [
    "lineup_stability_score",
    "expected_starter",
]

ROLE_BASE_FEATURES = [
    "role_cluster",
    "role_creator_flag",
    "role_defense_flag",
    "role_expected_minutes",
    "role_usage_proxy",
    "role_assist_rate",
    "role_turnover_rate",
    "role_threepa_rate",
    "role_rebound_rate",
    "role_stocks_rate",
    "role_starter_proxy",
]

MARKET_FEATURES = [
    "market_points_line",
    "market_rebounds_line",
    "market_assists_line",
    "market_threepm_line",
    "market_books_count",
]

# Derived market signal — populated when the historical cache has Over/Under
# prices, per-book line lists, and matching rolling form columns. Models
# tolerate NaN via their imputer step, so older caches without these columns
# keep working.
MARKET_DERIVED_POINTS = [
    "market_points_over_prob",
    "market_points_line_std",
    "market_points_vs_l5",
    "market_points_vs_l10",
]
MARKET_DERIVED_REBOUNDS = [
    "market_rebounds_over_prob",
    "market_rebounds_line_std",
    "market_rebounds_vs_l5",
    "market_rebounds_vs_l10",
]
MARKET_DERIVED_ASSISTS = [
    "market_assists_over_prob",
    "market_assists_line_std",
    "market_assists_vs_l5",
    "market_assists_vs_l10",
]
MARKET_DERIVED_THREEPM = [
    "market_threepm_over_prob",
    "market_threepm_line_std",
]
MARKET_DERIVED_AGGREGATE = ["market_overround"]

TARGET_FEATURE_GROUPS: dict[str, list[str]] = {
    "points": [
        *MODEL_SPECS["points"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        "market_points_line",
        "market_books_count",
        *MARKET_DERIVED_POINTS,
        *MARKET_DERIVED_AGGREGATE,
    ],
    "rebounds": [
        *MODEL_SPECS["rebounds"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *LINEUP_CONTEXT_FEATURES,
        "market_rebounds_line",
        "market_books_count",
        *MARKET_DERIVED_REBOUNDS,
        *MARKET_DERIVED_AGGREGATE,
    ],
    "assists": [
        *MODEL_SPECS["assists"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        *AVAILABILITY_CONTEXT_FEATURES,
        *LINEUP_CONTEXT_FEATURES,
        *ROLE_BASE_FEATURES,
        "assist_opportunity_score",
        "market_assists_line",
        "market_books_count",
        *MARKET_DERIVED_ASSISTS,
        *MARKET_DERIVED_AGGREGATE,
    ],
    "turnovers": [
        *MODEL_SPECS["turnovers"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        *AVAILABILITY_CONTEXT_FEATURES,
        *ROLE_BASE_FEATURES,
        "turnover_pressure_score",
        "market_books_count",
        *MARKET_DERIVED_AGGREGATE,
    ],
    "stl_blk": [
        *MODEL_SPECS["stl_blk"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        *AVAILABILITY_CONTEXT_FEATURES,
        *LINEUP_CONTEXT_FEATURES,
        *ROLE_BASE_FEATURES,
        "stocks_matchup_score",
        "market_books_count",
        *MARKET_DERIVED_AGGREGATE,
    ],
    "threepm": [
        *MODEL_SPECS["threepm"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        "market_threepm_line",
        "market_books_count",
        *MARKET_DERIVED_THREEPM,
        *MARKET_DERIVED_AGGREGATE,
    ],
    "pra": [
        *MODEL_SPECS["pra"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        *LINEUP_CONTEXT_FEATURES,
        "market_points_line",
        "market_rebounds_line",
        "market_assists_line",
        "market_books_count",
        *MARKET_DERIVED_POINTS,
        *MARKET_DERIVED_REBOUNDS,
        *MARKET_DERIVED_ASSISTS,
        *MARKET_DERIVED_AGGREGATE,
    ],
    "fantasy_score": [
        *MODEL_SPECS["fantasy_score"]["features"],
        *SCHEDULE_CONTEXT_FEATURES,
        *OPPONENT_CONTEXT_FEATURES,
        *AVAILABILITY_CONTEXT_FEATURES,
        *LINEUP_CONTEXT_FEATURES,
        *ROLE_BASE_FEATURES,
        *MARKET_FEATURES,
        *MARKET_DERIVED_POINTS,
        *MARKET_DERIVED_REBOUNDS,
        *MARKET_DERIVED_ASSISTS,
        *MARKET_DERIVED_THREEPM,
        *MARKET_DERIVED_AGGREGATE,
    ],
}


@dataclass
class ModelBundle:
    """Container for trained pipelines + metadata."""

    estimators: dict[str, Any]
    specs: dict[str, dict[str, Any]] = field(default_factory=lambda: MODEL_SPECS)
    metrics: pd.DataFrame | None = None
    cv_rmse: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, dict] = field(default_factory=dict)
    trained_at: str = ""
    train_players: list[str] = field(default_factory=list)
    train_seasons: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    validation_mode: str = "chronological"
    uplift_report: pd.DataFrame | None = None
    target_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def predict(self, model_name: str, features: pd.DataFrame) -> float:
        if model_name not in self.estimators:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {list(self.estimators)}")
        return float(self.estimators[model_name].predict(features)[0])

    @property
    def names(self) -> list[str]:
        return list(self.estimators)


class RACERegressor:
    """Simple weighted ensemble wrapper with per-component feature subsets."""

    def __init__(
        self,
        components: list[tuple[Any, list[str], float]],
        model_family: str,
    ) -> None:
        self.components = components
        self.model_family = model_family

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds: list[np.ndarray] = []
        weights: list[float] = []
        for est, feats, w in self.components:
            Xi = X.reindex(columns=feats)
            preds.append(np.asarray(est.predict(Xi), dtype=float))
            weights.append(float(w))
        if not preds:
            return np.zeros(len(X), dtype=float)
        W = np.asarray(weights, dtype=float)
        W = W / np.sum(W)
        stacked = np.vstack(preds)
        return np.average(stacked, axis=0, weights=W)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))


def _prediction_array(prediction: Any) -> np.ndarray:
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return np.asarray(prediction, dtype=float)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": _rmse(y, yhat),
        "R²": float(r2_score(y, yhat)) if len(np.unique(y)) > 1 else 0.0,
    }


def _make_model(family: str, random_state: int) -> tuple[Pipeline, list[dict[str, Any]]]:
    if family == "ridge":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(random_state=random_state)),
        ])
        grid = [{"model__alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]
        return pipe, grid

    if family == "rf":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=random_state)),
        ])
        grid = [
            {"model__n_estimators": 200, "model__max_depth": None, "model__min_samples_leaf": 1},
            {"model__n_estimators": 300, "model__max_depth": 16, "model__min_samples_leaf": 2},
            {"model__n_estimators": 300, "model__max_depth": 12, "model__min_samples_leaf": 4},
        ]
        return pipe, grid

    if family == "hgb":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )),
        ])
        grid = [
            {"model__max_depth": 4, "model__learning_rate": 0.05, "model__max_iter": 400},
            {"model__max_depth": 6, "model__learning_rate": 0.05, "model__max_iter": 500},
            {"model__max_depth": 8, "model__learning_rate": 0.03, "model__max_iter": 600},
        ]
        return pipe, grid

    if family == "hgb_poisson":
        # Poisson loss for non-negative count targets (stl_blk, turnovers,
        # threepm). Better likelihood fit than squared error on skewed,
        # zero-floored distributions.
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                loss="poisson",
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )),
        ])
        grid = [
            {"model__max_depth": 4, "model__learning_rate": 0.05, "model__max_iter": 400},
            {"model__max_depth": 6, "model__learning_rate": 0.04, "model__max_iter": 500},
            {"model__max_depth": 8, "model__learning_rate": 0.03, "model__max_iter": 600},
        ]
        return pipe, grid

    if family == "knn":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(weights="distance")),
        ])
        grid = [{"model__n_neighbors": k} for k in (5, 7, 10, 15, 20)]
        return pipe, grid

    raise ValueError(f"Unknown family: {family}")


def _fit_best_family(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    family: str,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    # Drop columns that have no observed values in the training data so that
    # SimpleImputer never receives fully-NaN columns (suppresses sklearn warnings
    # regardless of which call site invokes this function).
    observed_cols = [c for c in X_train.columns if X_train[c].notna().any()]
    X_train = X_train[observed_cols]
    X_val = X_val.reindex(columns=observed_cols)

    pipe, grid = _make_model(family, random_state=random_state)

    # Use a 3-fold expanding-window TimeSeriesSplit for hyperparameter
    # selection when we have enough data — averaging across folds gives a
    # far more stable signal than the single train/val split, which was
    # the main source of noise in the previous RACE bake-off. Falls back
    # to the original single-split scoring on tiny datasets.
    use_cv = len(X_train) >= 120
    n_splits = 3 if use_cv else 0
    tscv = TimeSeriesSplit(n_splits=n_splits) if use_cv else None

    # KNN doesn't accept sample_weight via the pipeline reliably; skip
    # weighting for that family. (Currently unused after pruning, kept for
    # safety in case it is reintroduced.)
    weights_supported = family != "knn"
    sw = sample_weight if (weights_supported and sample_weight is not None) else None

    best_est = None
    best_params: dict[str, Any] = {}
    best_m = {"MAE": np.inf, "RMSE": np.inf, "R²": -np.inf}
    best_score: tuple[float, float] = (np.inf, np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in grid:
            est = pipe.set_params(**params)

            if tscv is not None:
                fold_maes: list[float] = []
                cv_ok = True
                for tr_idx, te_idx in tscv.split(X_train):
                    Xtr = X_train.iloc[tr_idx]
                    ytr = y_train[tr_idx]
                    Xte = X_train.iloc[te_idx]
                    yte = y_train[te_idx]
                    fit_kwargs: dict[str, Any] = {}
                    if sw is not None:
                        fit_kwargs["model__sample_weight"] = sw[tr_idx]
                    try:
                        est.fit(Xtr, ytr, **fit_kwargs)
                    except Exception:
                        cv_ok = False
                        break
                    yhat_fold = _prediction_array(est.predict(Xte))
                    fold_maes.append(float(mean_absolute_error(yte, yhat_fold)))
                if not cv_ok or not fold_maes:
                    continue
                cv_mae = float(np.mean(fold_maes))

                # Refit on all of X_train so the held-out val metrics are
                # comparable to the previous behavior and we can score the
                # selected variant downstream.
                fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
                est.fit(X_train, y_train, **fit_kwargs)
                yhat = _prediction_array(est.predict(X_val))
                m = _metrics(y_val, yhat)
                m["CV_MAE"] = cv_mae
                score = (cv_mae, m["RMSE"])
            else:
                fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
                try:
                    est.fit(X_train, y_train, **fit_kwargs)
                except Exception:
                    continue
                yhat = _prediction_array(est.predict(X_val))
                m = _metrics(y_val, yhat)
                score = (m["MAE"], m["RMSE"])

            if score < best_score:
                best_est = est
                best_params = dict(params)
                best_m = m
                best_score = score

    if best_est is None:
        dum = DummyRegressor(strategy="mean").fit(X_train, y_train)
        yhat = _prediction_array(dum.predict(X_val))
        return dum, {"family": "dummy"}, _metrics(y_val, yhat)

    return best_est, best_params, best_m


def _observed_feature_subset(
    X: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return X restricted to features with at least one observed value."""
    if X.empty or not feature_names:
        return X.reindex(columns=[]), []
    Xf = X.reindex(columns=feature_names)
    observed = [c for c in Xf.columns if Xf[c].notna().any()]
    return Xf.reindex(columns=observed), observed


def _role_dynamic_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c.startswith("role_cluster_") or c.startswith("role_dist_"):
            cols.append(c)
    return sorted(cols)


def _split_chronological(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    time_col: str = "game_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    if time_col in df.columns:
        s = df.copy()
        s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
        s = s.sort_values([time_col, "player"] if "player" in s.columns else [time_col]).reset_index(drop=True)
    else:
        s = df.reset_index(drop=True)

    cut = max(1, int(len(s) * (1 - val_size)))
    cut = min(cut, len(s) - 1) if len(s) > 1 else 1
    return s.iloc[:cut].copy(), s.iloc[cut:].copy()


def _split_random(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= 1:
        return df.copy(), df.copy()
    n_val = max(1, int(len(df) * val_size))
    idx = np.random.default_rng(random_state).permutation(len(df))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns]


def _recency_weights(
    dates: pd.Series | None,
    *,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    n_rows: int,
) -> np.ndarray | None:
    """Exponential-decay sample weights so recent games dominate.

    Returns ``None`` if dates are missing/unusable so callers can skip
    weighting cleanly. Latest game gets weight 1.0; weight halves every
    ``half_life_days`` going back in time.
    """
    if dates is None or n_rows == 0:
        return None
    dt = pd.to_datetime(dates, errors="coerce")
    if dt.isna().all():
        return None
    latest = dt.max()
    age_days = (latest - dt).dt.days.astype("float64").to_numpy()
    age_days = np.where(np.isnan(age_days), float(np.nanmedian(age_days)) if not np.isnan(age_days).all() else 0.0, age_days)
    return np.power(0.5, age_days / max(half_life_days, 1.0))


def _families_for_target(name: str) -> list[str]:
    """Curated per-target family list.

    KNN was dropped from the bake-off — on tabular game logs it almost never
    beat ridge/HGB and inflated training time. Count-like targets get a
    Poisson-loss HGB candidate which usually outperforms MSE-loss boosters
    on zero-floored skewed distributions.
    """
    if name in COUNT_LIKE_TARGETS:
        return ["ridge", "hgb", "hgb_poisson"]
    return ["ridge", "hgb", "rf"]


def _fit_eligible_features(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target: str | None = None,
) -> list[str]:
    eligible = [f for f in features if f in train_df.columns and f != target]
    if not eligible:
        return []
    observed = train_df[eligible].notna().any(axis=0)
    return [f for f in eligible if bool(observed.get(f, False))]


def _target_variant_features(name: str, df: pd.DataFrame) -> dict[str, list[str]]:
    base = _available_features(df, MODEL_SPECS[name]["features"])
    role_dynamic = _role_dynamic_columns(df)

    context = _available_features(
        df,
        list(dict.fromkeys([
            *base,
            *SCHEDULE_CONTEXT_FEATURES,
            *OPPONENT_CONTEXT_FEATURES,
            *AVAILABILITY_CONTEXT_FEATURES,
            *LINEUP_CONTEXT_FEATURES,
        ])),
    )
    role = _available_features(
        df,
        list(dict.fromkeys([
            *base,
            *ROLE_BASE_FEATURES,
            *role_dynamic,
        ])),
    )

    full_group = list(dict.fromkeys([
        *TARGET_FEATURE_GROUPS.get(name, []),
        *role_dynamic,
    ]))
    full = _available_features(df, full_group)

    return {
        "baseline": base,
        "context": context,
        "role": role,
        "race": full,
    }


def train_models(
    player_data: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_state: int = 123,
    time_aware_validation: bool = True,
    verbose: bool = False,
    fast_mode: bool = False,
) -> ModelBundle:
    """Train RACE models for each target.

    By default, validation is chronological to avoid leaking adjacent game state
    across train/validation windows.

    When ``fast_mode`` is True, training is dramatically simplified for hosted
    deployments (e.g. Streamlit Cloud): only the baseline feature set and a
    single ridge family are evaluated, and ensemble blending is skipped. This
    reduces ~128 model fits to ~8 with similar baseline-quality predictions.
    """
    if player_data.empty:
        raise ValueError("player_data is empty; cannot train models")

    required_targets = [spec["target"] for spec in MODEL_SPECS.values()]
    keep_cols = list(dict.fromkeys([
        "player",
        "season",
        "game_date",
        "MATCHUP",
        *required_targets,
        *[c for cols in TARGET_FEATURE_GROUPS.values() for c in cols],
        *[c for c in player_data.columns if c.startswith("role_cluster_") or c.startswith("role_dist_")],
    ]))
    keep_cols = [c for c in keep_cols if c in player_data.columns]
    modeling_df = player_data[keep_cols].copy()

    estimators: dict[str, Any] = {}
    specs: dict[str, dict[str, Any]] = {}
    best_params: dict[str, dict] = {}
    cv_rmse: dict[str, float] = {}
    metric_rows: list[dict[str, Any]] = []
    uplift_rows: list[dict[str, Any]] = []
    target_metadata: dict[str, dict[str, Any]] = {}

    total_train_n = 0
    total_val_n = 0

    for name, spec in MODEL_SPECS.items():
        target = spec["target"]
        sub = modeling_df.dropna(subset=[target]).copy()
        if sub.empty:
            continue

        if time_aware_validation:
            train_df, val_df = _split_chronological(sub, val_size=validation_size)
        else:
            train_df, val_df = _split_random(
                sub,
                val_size=validation_size,
                random_state=random_state,
            )
        if len(train_df) < 10 or len(val_df) < 5:
            # tiny sample fallback
            full_feats = _fit_eligible_features(train_df, MODEL_SPECS[name]["features"], target=target)
            if not full_feats:
                full_feats = _fit_eligible_features(
                    train_df,
                    [c for c in sub.columns if c not in {"player", "season", "game_date", "MATCHUP", target}],
                    target=target,
                )
            Xf, full_feats = _observed_feature_subset(sub, full_feats)
            if not full_feats:
                continue
            yf = sub[target].to_numpy()
            dum = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ])
            dum.fit(Xf, yf)
            est = RACERegressor([(dum, full_feats, 1.0)], model_family="dummy")
            yhat = est.predict(Xf)
            m = _metrics(yf, yhat)
            selected_variant = "baseline"
            estimators[name] = est
            specs[name] = {"target": target, "features": full_feats, "kind": selected_variant}
            best_params[name] = {"family": "dummy", "variant": "baseline"}
            cv_rmse[name] = m["RMSE"]
            metric_rows.append({"model": name, "target": target, "kind": selected_variant, **{k: round(v, 3) for k, v in m.items()}})
            uplift_rows.append({
                "target": name,
                "selected_variant": selected_variant,
                "baseline_MAE": round(m["MAE"], 3),
                "baseline_RMSE": round(m["RMSE"], 3),
                "baseline_R²": round(m["R²"], 3),
                "selected_MAE": round(m["MAE"], 3),
                "selected_RMSE": round(m["RMSE"], 3),
                "selected_R²": round(m["R²"], 3),
                "R²_uplift": 0.0,
                "MAE_uplift": 0.0,
                "features_added": "fallback",
                "model_family": "dummy",
            })
            continue

        total_train_n += len(train_df)
        total_val_n += len(val_df)

        y_train = train_df[target].to_numpy()
        y_val = val_df[target].to_numpy()

        # Recency-weight training rows so recent games carry more signal.
        # Computed once per target — same weights flow through every variant
        # and every family that supports sample_weight.
        train_weights = _recency_weights(
            train_df.get("game_date"),
            n_rows=len(train_df),
        )
        # Poisson loss requires strictly non-negative targets — clip just
        # in case any synthetic/edge row sneaks in below zero.
        if (y_train < 0).any():
            y_train = np.clip(y_train, 0.0, None)

        variant_features = _target_variant_features(name, sub)
        if fast_mode:
            # Only evaluate the baseline (lagged) feature set in fast mode.
            variant_features = {k: v for k, v in variant_features.items() if k == "baseline"}
        variant_results: dict[str, dict[str, Any]] = {}

        for variant, feats in variant_features.items():
            feats = _fit_eligible_features(train_df, feats, target=target)
            if not feats:
                continue

            X_train = train_df.reindex(columns=feats)
            X_val = val_df.reindex(columns=feats)

            if fast_mode:
                families = ["ridge"]
            else:
                families = _families_for_target(name)

            family_candidates: list[dict[str, Any]] = []
            for fam in families:
                try:
                    est, params, m = _fit_best_family(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        family=fam,
                        random_state=random_state,
                        sample_weight=train_weights,
                    )
                    family_candidates.append(
                        {
                            "family": fam,
                            "estimator": est,
                            "params": params,
                            "metrics": m,
                            "features": feats,
                        }
                    )
                except Exception:
                    continue

            if not family_candidates:
                continue
            best = sorted(family_candidates, key=lambda r: (r["metrics"]["MAE"], r["metrics"]["RMSE"]))[0]
            variant_results[variant] = best

            if verbose:
                print(
                    f"  {name:14s} [{variant:8s}] {best['family']:>5s} "
                    f"MAE={best['metrics']['MAE']:.3f} RMSE={best['metrics']['RMSE']:.3f} "
                    f"R²={best['metrics']['R²']:.3f} feats={len(best['features'])}"
                )

        if not variant_results:
            continue

        # Pick best single variant first.
        best_variant_name, best_variant = sorted(
            variant_results.items(),
            key=lambda kv: (kv[1]["metrics"]["MAE"], kv[1]["metrics"]["RMSE"]),
        )[0]

        # Optional blend by inverse MAE over available variants.
        blend_parts = sorted(
            variant_results.items(),
            key=lambda kv: kv[1]["metrics"]["MAE"],
        )[:3]
        blend_use = False
        blend_estimator: Any = None
        blend_metrics = None
        if not fast_mode and len(blend_parts) >= 2:
            weights = []
            comps = []
            preds = []
            for _, info in blend_parts:
                mae = max(info["metrics"]["MAE"], 1e-6)
                w = 1.0 / mae
                weights.append(w)
                comps.append((info["estimator"], info["features"], w))
                p = info["estimator"].predict(val_df.reindex(columns=info["features"]))
                preds.append(np.asarray(p, dtype=float))
            W = np.asarray(weights, dtype=float)
            W = W / np.sum(W)
            y_blend = np.average(np.vstack(preds), axis=0, weights=W)
            blend_metrics = _metrics(y_val, y_blend)
            if blend_metrics["MAE"] < best_variant["metrics"]["MAE"] * 0.995:
                blend_use = True
                blend_estimator = RACERegressor(comps, model_family="blend")

        # Refit selected model(s) on full target data (train+val) for inference.
        X_full = sub
        y_full = sub[target].to_numpy()
        if (y_full < 0).any():
            y_full = np.clip(y_full, 0.0, None)
        full_weights = _recency_weights(
            sub.get("game_date"),
            n_rows=len(sub),
        )

        if blend_use and blend_estimator is not None and blend_metrics is not None:
            refit_components: list[tuple[Any, list[str], float]] = []
            for _, info in blend_parts:
                fam = info["family"]
                feats = info["features"]
                est_refit, params_refit, _ = _fit_best_family(
                    X_full.reindex(columns=feats),
                    y_full,
                    X_full.reindex(columns=feats),
                    y_full,
                    family=fam,
                    random_state=random_state,
                    sample_weight=full_weights,
                )
                w = 1.0 / max(info["metrics"]["MAE"], 1e-6)
                refit_components.append((est_refit, feats, w))
                info["params"] = params_refit

            final_est = RACERegressor(refit_components, model_family="blend")
            selected_metrics = blend_metrics
            selected_variant = "blend"
            selected_family = "blend"
            selected_params = {
                "variant": "blend",
                "components": [
                    {"family": info["family"], "features": len(info["features"]), "params": info["params"]}
                    for _, info in blend_parts
                ],
            }
            features_used = sorted({f for _, info in blend_parts for f in info["features"]})
        else:
            selected_variant = best_variant_name
            selected_family = best_variant["family"]
            selected_metrics = best_variant["metrics"]
            selected_params = {"variant": best_variant_name, **best_variant["params"]}
            feats = best_variant["features"]
            est_refit, params_refit, _ = _fit_best_family(
                X_full.reindex(columns=feats),
                y_full,
                X_full.reindex(columns=feats),
                y_full,
                family=selected_family,
                random_state=random_state,
                sample_weight=full_weights,
            )
            selected_params.update(params_refit)
            final_est = RACERegressor([(est_refit, feats, 1.0)], model_family=selected_family)
            features_used = feats

        estimators[name] = final_est
        specs[name] = {"target": target, "features": features_used, "kind": selected_variant}
        best_params[name] = selected_params
        cv_rmse[name] = float(selected_metrics["RMSE"])

        metric_rows.append(
            {
                "model": name,
                "target": target,
                "kind": selected_variant,
                "RMSE": round(float(selected_metrics["RMSE"]), 3),
                "MAE": round(float(selected_metrics["MAE"]), 3),
                "R²": round(float(selected_metrics["R²"]), 3),
            }
        )

        baseline_m = variant_results.get("baseline", {}).get("metrics", {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan})
        selected_m = selected_metrics
        features_added = sorted(set(features_used) - set(variant_features.get("baseline", [])))
        uplift_rows.append(
            {
                "target": name,
                "selected_variant": selected_variant,
                "baseline_MAE": round(float(baseline_m.get("MAE", np.nan)), 3),
                "baseline_RMSE": round(float(baseline_m.get("RMSE", np.nan)), 3),
                "baseline_R²": round(float(baseline_m.get("R²", np.nan)), 3),
                "selected_MAE": round(float(selected_m["MAE"]), 3),
                "selected_RMSE": round(float(selected_m["RMSE"]), 3),
                "selected_R²": round(float(selected_m["R²"]), 3),
                "R²_uplift": round(float(selected_m["R²"] - baseline_m.get("R²", np.nan)), 3),
                "MAE_uplift": round(float((baseline_m.get("MAE", np.nan) - selected_m["MAE"])), 3),
                "features_added": ", ".join(features_added[:20]) if features_added else "none",
                "model_family": selected_family,
            }
        )

        target_metadata[name] = {
            "target": target,
            "selected_variant": selected_variant,
            "features_used": features_used,
            "validation_metrics": selected_m,
            "model_family": selected_family,
            "context_enabled": any(f in features_used for f in OPPONENT_CONTEXT_FEATURES + AVAILABILITY_CONTEXT_FEATURES + LINEUP_CONTEXT_FEATURES + SCHEDULE_CONTEXT_FEATURES),
            "role_features_enabled": any(f in features_used for f in ROLE_BASE_FEATURES) or any(f.startswith("role_cluster_") for f in features_used),
            "variant_metrics": {k: v["metrics"] for k, v in variant_results.items()},
        }

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics = metrics.sort_values("model").reset_index(drop=True)

    uplift_report = pd.DataFrame(uplift_rows)
    if not uplift_report.empty:
        uplift_report = uplift_report.sort_values("target").reset_index(drop=True)

    return ModelBundle(
        estimators=estimators,
        specs=specs,
        metrics=metrics,
        cv_rmse=cv_rmse,
        best_params=best_params,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_players=sorted(player_data["player"].dropna().unique().tolist()) if "player" in player_data.columns else [],
        train_seasons=sorted(player_data["season"].dropna().unique().tolist()) if "season" in player_data.columns else [],
        n_train=total_train_n,
        n_test=total_val_n,
        validation_mode="chronological" if time_aware_validation else "random",
        uplift_report=uplift_report,
        target_metadata=target_metadata,
    )


def _bundle_hash(players: list[str], seasons: list[str], spec_version: str = "race-v1") -> str:
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
    time_aware_validation: bool = True,
    fast_mode: bool = False,
) -> ModelBundle:
    """Load a cached RACE bundle matching ``player_data``, or train + save one."""
    players = sorted(player_data["player"].dropna().unique().tolist()) if "player" in player_data.columns else []
    seasons = sorted(player_data["season"].dropna().unique().tolist()) if "season" in player_data.columns else []
    spec_version = "race-v2-fast" if fast_mode else "race-v2"
    key = _bundle_hash(players, seasons, spec_version=spec_version)
    cache_path = Path(cache_dir) / f"models_{key}.joblib"

    if cache_path.exists() and not force:
        try:
            bundle = load_models(cache_path)
            if verbose:
                print(f"  loaded cached models from {cache_path}")
            return bundle
        except Exception as exc:
            if verbose:
                print(f"  ! cache unreadable ({exc}); retraining")

    if verbose:
        print(f"  training RACE models for {len(players)} player(s)…")

    bundle = train_models(
        player_data,
        verbose=verbose,
        time_aware_validation=time_aware_validation,
        fast_mode=fast_mode,
    )

    try:
        save_models(bundle, cache_path)
        if verbose:
            print(f"  saved model bundle -> {cache_path}")
    except Exception as exc:
        if verbose:
            print(f"  ! could not persist models: {exc}")

    return bundle
