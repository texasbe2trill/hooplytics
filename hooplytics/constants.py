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

# ── Pregame-safe rolling features ────────────────────────────────────────────
ROLL_BASE_STATS: list[str] = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fga", "fg3a", "fta", "plus_minus",
]
ROLL_WINDOWS: tuple[int, ...] = (5, 10, 30)

# ── Model specs ──────────────────────────────────────────────────────────────
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "points":        {"target": "pts",           "features": ["fgm", "fg3m", "ftm", "min", "fg_pct", "ft_pct"], "kind": "knn"},
    "rebounds":      {"target": "reb",           "features": ["oreb", "dreb", "min"], "kind": "knn"},
    "assists":       {"target": "ast",           "features": ["ast_l5", "ast_l10", "ast_l30", "ast_per36_l30", "min_l10", "usg_proxy_l30"], "kind": "ridge"},
    "pra":           {"target": "pra",           "features": ["pts", "reb", "ast", "min", "plus_minus"], "kind": "knn"},
    "threepm":       {"target": "fg3m",          "features": ["fg3a", "min", "fg3_pct"], "kind": "knn"},
    "stl_blk":       {"target": "stl_blk",       "features": ["stl_l10", "blk_l10", "stl_l30", "blk_l30", "stl_per36_l30", "blk_per36_l30", "min_l30"], "kind": "ridge"},
    "turnovers":     {"target": "tov",           "features": ["tov_l5", "tov_l10", "tov_l30", "tov_per36_l30", "min_l10", "usg_proxy_l30", "ast_l10", "fga_l10"], "kind": "ridge"},
    "fantasy_score": {"target": "fantasy_score", "features": ["pts", "reb", "ast", "stl", "blk", "tov", "min", "plus_minus"], "kind": "rf"},
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
