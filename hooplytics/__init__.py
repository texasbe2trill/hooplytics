"""Hooplytics — NBA player projections + More/Less decision engine.

Public API:
    PlayerStore        — fetch + cache NBA game logs as Parquet
    ModelBundle        — trained sklearn pipelines (8 models)
    train_models       — fit pipelines on a player_data DataFrame
    ensure_models      — load from joblib cache or train + save
    project_next_game  — project a player's next game from recent form
    predict_scenario   — predict from a hypothetical box-score row
    custom_prop        — single-prop MORE/LESS decision (auto-fetches line)
    fantasy_decisions  — full 8-stat decision table for a player
    fetch_live_player_lines — pull consensus lines from The Odds API
    nba_seasons        — 'YYYY-YY' season strings
"""
from __future__ import annotations

from .constants import (
    DEFAULT_ROSTER,
    FANTASY_WEIGHTS,
    METRICS,
    MODEL_SPECS,
    MODEL_TO_COL,
    PROJ_STATS,
    ROLL_BASE_STATS,
    ROLL_WINDOWS,
)
from .data import PlayerStore, add_pregame_features, nba_seasons
from .fantasy import fantasy
from .models import ModelBundle, ensure_models, train_models
from .odds import fetch_live_player_lines, load_api_key
from .predict import (
    custom_prop,
    fantasy_decisions,
    predict_scenario,
    project_next_game,
)

__all__ = [
    "DEFAULT_ROSTER",
    "FANTASY_WEIGHTS",
    "METRICS",
    "MODEL_SPECS",
    "MODEL_TO_COL",
    "ModelBundle",
    "PROJ_STATS",
    "PlayerStore",
    "ROLL_BASE_STATS",
    "ROLL_WINDOWS",
    "add_pregame_features",
    "custom_prop",
    "ensure_models",
    "fantasy",
    "fantasy_decisions",
    "fetch_live_player_lines",
    "load_api_key",
    "nba_seasons",
    "predict_scenario",
    "project_next_game",
    "train_models",
]

__version__ = "0.1.0"
