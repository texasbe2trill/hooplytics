"""Centralized configuration: model specs, fantasy weights, rolling windows, etc."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/cache")
ODDS_CACHE_DIR = CACHE_DIR / "odds"
MODEL_CACHE_DIR = Path(".hooplytics_cache/models")

# ── Fantasy scoring ───────────────────────────────────────────────────────────
FANTASY_WEIGHTS: dict[str, float] = dict(
    pts=1.0, reb=1.2, ast=1.5, stl=3.0, blk=3.0, tov=-1.0
)

# ── Stats the More/Less engine projects ──────────────────────────────────────
PROJ_STATS: list[str] = ["points", "fantasy_score", "pra", "threepm", "assists"]

METRICS: dict[str, str] = {
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
    "PRA": "pra",
    "3PM": "fg3m",
    "Stl+Blk": "stl_blk",
    "Turnovers": "tov",
    "Fantasy": "fantasy_score",
}

# ── Default starter roster ───────────────────────────────────────────────────
DEFAULT_ROSTER: dict[str, dict[str, float]] = {
    "LeBron James":            {"points": 21.5, "fantasy_score": 41.5, "pra": 34.0, "threepm": 1.5, "assists": 7.0},
    "Kevin Durant":            {"points": 26.0, "fantasy_score": 42.0, "pra": 36.0, "threepm": 2.5, "assists": 4.5},
    "Victor Wembanyama":       {"points": 25.0, "fantasy_score": 53.0, "pra": 39.5, "threepm": 1.5, "assists": 3.0},
    "Shai Gilgeous-Alexander": {"points": 31.0, "fantasy_score": 50.0, "pra": 41.0, "threepm": 2.0, "assists": 6.5},
    "Chet Holmgren":           {"points": 18.0, "fantasy_score": 38.0, "pra": 30.0, "threepm": 1.5, "assists": 2.5},
    "Ausar Thompson":          {"points": 14.5, "fantasy_score": 34.0, "pra": 26.0, "threepm": 0.5, "assists": 3.5},
}

# Veteran anchor cohort used to stabilize model training.
# The web app trains on these players plus the current display roster so models
# see deeper historical samples while still supporting roster-specific inference.
TRAINING_ANCHOR_PLAYERS: list[str] = [
    "LeBron James",
    "Kevin Durant",
    "Stephen Curry",
    "James Harden",
    "Damian Lillard",
    "Kyrie Irving",
    "DeMar DeRozan",
    "Jimmy Butler",
    "Paul George",
    "Chris Paul",
    "Nikola Jokic",
    "Giannis Antetokounmpo",
]

# ── Pregame-safe rolling features ────────────────────────────────────────────
ROLL_BASE_STATS: list[str] = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fga", "fg3a", "fta", "plus_minus",
]
ROLL_WINDOWS: tuple[int, ...] = (3, 5, 10, 30)

# ── Model specs ──────────────────────────────────────────────────────────────
MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Pregame-safe: all features are rolled over PRIOR games only (shift(1) before rolling).
    # Same-game component columns (fgm, oreb, etc.) were removed to eliminate data leakage.
    "points":        {"target": "pts",           "features": ["pts_l3", "pts_l5", "pts_l10", "pts_l30", "pts_dev_s", "fga_l10", "fga_l30", "fta_l10", "min_l3", "min_l10", "min_l30", "usg_proxy_l30", "days_rest", "is_home"], "kind": "rf"},
    "rebounds":      {"target": "reb",           "features": ["reb_l3", "reb_l5", "reb_l10", "reb_l30", "reb_dev_s", "min_l3", "min_l10", "min_l30", "days_rest", "is_home"], "kind": "rf"},
    "assists":       {"target": "ast",           "features": ["ast_l3", "ast_l5", "ast_l10", "ast_l30", "ast_dev_s", "ast_per36_l10", "ast_per36_l30", "ast_std_l10", "ast_trend_s", "ast_trend_l", "tov_l10", "fga_l10", "pts_l10", "min_l10", "min_l3", "days_rest", "is_home", "usg_proxy_l30", "opp_pace", "opp_def_rtg"], "kind": "rf"},
    "pra":           {"target": "pra",           "features": ["pts_l3", "pts_l5", "pts_l10", "pts_l30", "pts_dev_s", "reb_l3", "reb_l5", "reb_l10", "reb_l30", "reb_dev_s", "ast_l3", "ast_l5", "ast_l10", "ast_l30", "ast_dev_s", "min_l3", "min_l10", "plus_minus_l10", "usg_proxy_l30", "days_rest", "is_home"], "kind": "rf"},
    "threepm":       {"target": "fg3m",          "features": ["fg3a_l3", "fg3a_l5", "fg3a_l10", "fg3a_l30", "fg3a_dev_s", "fga_l10", "fga_l30", "min_l3", "min_l10", "days_rest", "is_home"], "kind": "rf"},
    "stl_blk":       {"target": "stl_blk",       "features": ["stl_l3", "blk_l3", "stl_l5", "blk_l5", "stl_l10", "blk_l10", "stl_l30", "blk_l30", "stl_dev_s", "blk_dev_s", "stl_std_l10", "blk_std_l10", "stl_trend_s", "blk_trend_s", "stl_trend_l", "blk_trend_l", "min_l3", "min_l10", "min_l30", "days_rest", "is_home", "opp_pace", "opp_stl_pg", "opp_blk_pg"], "kind": "rf"},
    "turnovers":     {"target": "tov",           "features": ["tov_l3", "tov_l5", "tov_l10", "tov_l30", "tov_dev_s", "tov_per36_l10", "tov_per36_l30", "tov_std_l10", "tov_trend_s", "tov_trend_l", "fga_l10", "fga_l30", "fg3a_l10", "fta_l10", "min_l3", "min_l10", "days_rest", "is_home", "usg_proxy_l30", "ast_l10", "opp_pace", "opp_def_rtg", "opp_stl_pg"], "kind": "rf"},
    "fantasy_score": {"target": "fantasy_score", "features": ["pts_l3", "pts_l5", "pts_l10", "pts_l30", "pts_dev_s", "reb_l3", "reb_l5", "reb_l10", "reb_l30", "reb_dev_s", "ast_l3", "ast_l5", "ast_l10", "ast_l30", "ast_dev_s", "stl_l3", "stl_l5", "stl_l10", "stl_dev_s", "blk_l3", "blk_l5", "blk_l10", "blk_dev_s", "tov_l3", "tov_l5", "tov_l10", "tov_dev_s", "min_l3", "min_l10", "min_l30", "plus_minus_l10", "usg_proxy_l30", "days_rest", "is_home"], "kind": "rf"},
}

MODEL_TO_COL: dict[str, str] = {name: spec["target"] for name, spec in MODEL_SPECS.items()}

# All columns required for modeling (features + targets)
ALL_COLS: list[str] = sorted({c for spec in MODEL_SPECS.values() for c in [*spec["features"], spec["target"]]})

# ── NBA stats column rename map ──────────────────────────────────────────────
NBA_RENAME: dict[str, str] = {
    "PTS": "pts", "REB": "reb", "OREB": "oreb", "DREB": "dreb",
    "AST": "ast", "STL": "stl", "BLK": "blk", "TOV": "tov",
    "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
    "FTM": "ftm", "FTA": "fta", "MIN": "min",
    "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
    "PLUS_MINUS": "plus_minus", "GAME_DATE": "game_date",
}

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_MARKETS: dict[str, str] = {
    "player_points":   "points",
    "player_rebounds": "rebounds",
    "player_assists":  "assists",
    "player_threes":   "threepm",
}
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_HISTORICAL_BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
ODDS_HIST_CACHE_DIR = ODDS_CACHE_DIR / "history"
# Player prop data is only available in the historical API from this date onward.
ODDS_PLAYER_PROPS_CUTOFF = "2023-05-03"
