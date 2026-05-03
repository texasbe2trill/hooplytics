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
from .features_context import (
    add_assist_opportunity_score,
    add_availability_context,
    add_lineup_context,
    add_opponent_context,
    add_schedule_context,
    add_stocks_matchup_score,
    add_turnover_pressure_score,
    build_context_features,
)
from .features_role import build_role_features
from .fantasy import fantasy
from .features_market import build_market_features
from .models import ModelBundle, TARGET_FEATURE_GROUPS, ensure_models, train_models
from .odds import fetch_live_player_lines, ingest_historical_odds, load_api_key, load_cached_historical_odds
from .predict import (
    custom_prop,
    fantasy_decisions,
    predict_scenario,
    project_next_game,
)
from .backtest import backtest_summary, retro_projection_table

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
    "backtest_summary",
    "retro_projection_table",
    "add_pregame_features",
    "add_schedule_context",
    "add_opponent_context",
    "add_availability_context",
    "add_lineup_context",
    "build_context_features",
    "build_market_features",
    "build_role_features",
    "add_assist_opportunity_score",
    "add_turnover_pressure_score",
    "add_stocks_matchup_score",
    "custom_prop",
    "ensure_models",
    "fantasy",
    "fantasy_decisions",
    "fetch_live_player_lines",
    "ingest_historical_odds",
    "load_api_key",
    "load_cached_historical_odds",
    "nba_seasons",
    "predict_scenario",
    "project_next_game",
    "TARGET_FEATURE_GROUPS",
    "train_models",
]

__version__ = "0.1.0"
