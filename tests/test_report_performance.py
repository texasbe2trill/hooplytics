"""Smoke tests for the Player Performance Analytics PDF report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hooplytics.report_performance import (
    build_player_performance_report,
    player_performance_summary,
)


def _synthetic_game_log(seed: int, n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-10-22", periods=n, freq="2D")
    fga = rng.integers(10, 22, n).astype(float)
    fgm = (fga * rng.uniform(0.4, 0.55, n)).round()
    fg3a = rng.integers(3, 9, n).astype(float)
    fg3m = (fg3a * rng.uniform(0.3, 0.45, n)).round()
    fta = rng.integers(2, 9, n).astype(float)
    ftm = (fta * rng.uniform(0.7, 0.9, n)).round()
    pts = (2 * (fgm - fg3m) + 3 * fg3m + ftm).astype(float)
    reb = rng.integers(3, 12, n).astype(float)
    ast = rng.integers(2, 11, n).astype(float)
    stl = rng.integers(0, 4, n).astype(float)
    blk = rng.integers(0, 3, n).astype(float)
    tov = rng.integers(0, 5, n).astype(float)
    minutes = rng.uniform(28, 38, n)
    return pd.DataFrame({
        "player": ["Test Player"] * n,
        "game_date": dates,
        "MATCHUP": ["TST vs. OPP"] * n,
        "pts": pts,
        "reb": reb,
        "ast": ast,
        "stl": stl,
        "blk": blk,
        "tov": tov,
        "fga": fga, "fgm": fgm,
        "fg3a": fg3a, "fg3m": fg3m,
        "fta": fta, "ftm": ftm,
        "min": minutes,
        "pra": pts + reb + ast,
        "fantasy_score": pts + 1.2 * reb + 1.5 * ast + 3 * stl + 3 * blk - tov,
        "plus_minus": rng.integers(-15, 15, n).astype(float),
    })


def test_player_performance_summary_basic() -> None:
    games = _synthetic_game_log(seed=1)
    summary = player_performance_summary(games, recent_n=10)
    assert summary["games_played"] == len(games)
    assert "kpis" in summary and "pts" in summary["kpis"]
    assert summary["kpis"]["pts"]["season_avg"] > 0
    assert "shooting" in summary
    assert 0.0 < summary["shooting"]["ts_pct"] < 1.0
    assert "streaks" in summary and "pts" in summary["streaks"]


def test_player_performance_summary_handles_empty() -> None:
    summary = player_performance_summary(pd.DataFrame())
    assert summary == {"games_played": 0}


def test_build_player_performance_report_returns_pdf_bytes() -> None:
    roster = {
        "Player One": ["2025-26"],
        "Player Two": ["2025-26"],
    }
    player_games = {
        "Player One": _synthetic_game_log(seed=1).assign(player="Player One"),
        "Player Two": _synthetic_game_log(seed=2).assign(player="Player Two"),
    }
    pdf = build_player_performance_report(
        roster=roster,
        player_games=player_games,
        bundle_metrics=None,
        ai_sections=None,
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    # A non-trivial multi-page report should weigh at least a few KB.
    assert len(pdf) > 8 * 1024


def test_build_player_performance_report_with_ai_sections() -> None:
    roster = {"Player One": ["2025-26"]}
    player_games = {
        "Player One": _synthetic_game_log(seed=3).assign(player="Player One"),
    }
    ai_sections = {
        "roster_overview": "Player One is trending up across primary scoring stats.",
        "players": {
            "Player One": {
                "strengths": "Efficient self-creation and clean shot diet.",
                "growth_areas": "Defensive closeouts and rim protection consistency.",
                "coaching_focus": "Rep weak-side rotations and short-roll reads.",
                "matchup_context": "Matchup unconfirmed",
            }
        },
    }
    pdf = build_player_performance_report(
        roster=roster,
        player_games=player_games,
        ai_sections=ai_sections,
    )
    assert pdf.startswith(b"%PDF")
    # AI prose path is a separate code branch — ensure it builds cleanly.
    assert len(pdf) > 8 * 1024
