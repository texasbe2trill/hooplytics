"""Smoke tests for prediction & model persistence using a synthetic dataset.

We avoid network calls entirely by constructing a tiny game-log DataFrame
that satisfies the schema and feature engineering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hooplytics import predict_scenario, project_next_game
from hooplytics.constants import ALL_COLS, MODEL_SPECS
from hooplytics.data import PlayerStore, add_pregame_features
from hooplytics.fantasy import fantasy
from hooplytics.models import ensure_models, train_models


def _synthetic_player(name: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build a believable per-game log so feature engineering produces no NaNs after warmup."""
    dates = pd.date_range("2024-10-25", periods=n, freq="2D")
    minutes = rng.normal(34, 4, n).clip(15, 42)
    fga = rng.normal(18, 4, n).clip(5, 30)
    fg3a = rng.normal(7, 2, n).clip(1, 15)
    fta = rng.normal(5, 2, n).clip(0, 14)
    fgm = (fga * rng.uniform(0.42, 0.55, n)).round()
    fg3m = (fg3a * rng.uniform(0.30, 0.42, n)).round()
    ftm = (fta * rng.uniform(0.70, 0.92, n)).round()
    df = pd.DataFrame({
        "player": name,
        "season": "2024-25",
        "game_date": dates,
        "min": minutes,
        "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
        "fg_pct": fgm / fga.clip(1),
        "fg3_pct": np.where(fg3a > 0, fg3m / fg3a.clip(1), 0),
        "ft_pct": np.where(fta > 0, ftm / fta.clip(1), 0),
        "oreb": rng.integers(0, 4, n),
        "dreb": rng.integers(2, 11, n),
        "ast": rng.integers(2, 12, n),
        "stl": rng.integers(0, 4, n),
        "blk": rng.integers(0, 4, n),
        "tov": rng.integers(0, 6, n),
        "plus_minus": rng.normal(2, 8, n).round(),
    })
    df["pts"] = (2 * df["fgm"] + df["fg3m"] + df["ftm"]).astype(int)
    df["reb"] = (df["oreb"] + df["dreb"]).astype(int)
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["stl_blk"] = df["stl"] + df["blk"]
    df["fantasy_score"] = fantasy(df)
    return df


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    parts = [_synthetic_player(name, 60, rng) for name in ("Player A", "Player B", "Player C")]
    df = pd.concat(parts, ignore_index=True)
    return add_pregame_features(df)


def test_train_models_produces_eight_estimators(synthetic_data: pd.DataFrame) -> None:
    bundle = train_models(synthetic_data)
    assert set(bundle.estimators) == set(MODEL_SPECS)
    assert bundle.metrics is not None and len(bundle.metrics) == len(MODEL_SPECS)
    assert bundle.n_train > 0 and bundle.n_test > 0


def test_predict_scenario(synthetic_data: pd.DataFrame) -> None:
    bundle = train_models(synthetic_data)
    scenario = {c: 5.0 for c in ALL_COLS}
    scenario.update({"min": 32, "fg_pct": 0.5, "fg3_pct": 0.4, "ft_pct": 0.85})
    df = predict_scenario(scenario, bundle)
    assert not df.empty
    assert {"model", "prediction"}.issubset(df.columns)


def test_ensure_models_roundtrip(tmp_path: Path, synthetic_data: pd.DataFrame) -> None:
    cache = tmp_path / "models"
    b1 = ensure_models(synthetic_data, cache_dir=cache)
    b2 = ensure_models(synthetic_data, cache_dir=cache)  # should hit cache
    # Same hash → same file → identical estimator dict keys
    assert set(b1.estimators) == set(b2.estimators)
    files = list(cache.glob("models_*.joblib"))
    assert len(files) == 1


def test_project_next_game(synthetic_data: pd.DataFrame) -> None:
    bundle = train_models(synthetic_data)
    store = PlayerStore()
    modeling_df = store.modeling_frame(synthetic_data)
    proj = project_next_game(
        "Player A", bundle=bundle, store=store,
        last_n=10, modeling_df=modeling_df,
    )
    assert not proj.empty
    assert "prediction" in proj.columns
    assert all(proj["prediction"] > 0)
