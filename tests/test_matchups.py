"""Smoke tests for hooplytics.matchups: deterministic team rollups + win prob."""
from __future__ import annotations

import pandas as pd

from hooplytics.matchups import (
    attach_market_lines,
    build_slate_predictions,
    margin_to_home_win_prob,
    project_matchup,
    to_grounding_payload,
)


ABBR = {"DET": "Detroit Pistons", "ORL": "Orlando Magic"}


def _modeling_df() -> pd.DataFrame:
    rows = []
    for team, players in {
        "DET": [("Cade", 35, 26), ("Ivey", 30, 17), ("Harris", 30, 16),
                ("Duren", 28, 12), ("Thompson", 26, 11), ("Stewart", 22, 8),
                ("Sasser", 20, 8), ("Bogdanovic", 22, 10), ("Grimes", 18, 6)],
        "ORL": [("Banchero", 36, 24), ("Wagner", 34, 22), ("Suggs", 30, 14),
                ("Carter", 28, 12), ("Anthony", 24, 10), ("Isaac", 22, 9),
                ("Black", 18, 7), ("Bitadze", 16, 6), ("Houstan", 14, 4)],
    }.items():
        for p, mins, pts in players:
            for d in range(10):
                rows.append({
                    "player": p, "team_abbr": team,
                    "game_date": pd.Timestamp("2026-04-15") + pd.Timedelta(days=d),
                    "min": mins, "pts": pts,
                })
    return pd.DataFrame(rows)


def test_margin_to_home_win_prob_anchors() -> None:
    assert abs(margin_to_home_win_prob(0.0) - 0.5) < 1e-6
    assert margin_to_home_win_prob(20.0) > margin_to_home_win_prob(5.0) > 0.5
    assert margin_to_home_win_prob(-5.0) < 0.5


def test_project_matchup_uses_user_projection_when_available() -> None:
    modeling_df = _modeling_df()
    projections = {
        "Cade": pd.DataFrame([{"model": "points", "prediction": 30.0}]),
    }
    pred = project_matchup(
        home_team="Detroit Pistons", away_team="Orlando Magic",
        matchup="Orlando Magic @ Detroit Pistons", tipoff_iso="",
        abbr_to_full=ABBR, modeling_df=modeling_df, projections=projections,
        roster_players=["Cade"],
    )
    # Cade's L10 average is 26 but the model says 30 — team total should reflect
    # the +4 boost for the only player the user has projected.
    cade_entry = next(r for r in pred.rotation_players_home if r["player"] == "Cade")
    assert cade_entry["source"] == "model"
    assert cade_entry["pts_proj"] == 30.0
    assert "Cade" in pred.rostered_players_home
    assert pred.confidence == "high"


def test_project_matchup_handles_missing_team() -> None:
    pred = project_matchup(
        home_team="Detroit Pistons", away_team="Phoenix Suns",
        matchup="Phoenix Suns @ Detroit Pistons", tipoff_iso="",
        abbr_to_full=ABBR, modeling_df=_modeling_df(), projections=None,
        roster_players=None,
    )
    # Confidence is "thin" when one side has no coverage at all — callers
    # should use this to suppress the model rollup and fall back to market data.
    assert pred.confidence == "thin"
    assert pred.rotation_players_away == []
    assert pred.away_pts_proj == 0.0


def test_build_slate_predictions_roster_only_filters_unrelated_games() -> None:
    slate = [
        # Game with rostered players (synthetic modeling_df only has DET/ORL).
        {"home_team": "Detroit Pistons", "away_team": "Orlando Magic",
         "matchup": "ORL @ DET", "tipoff_iso": ""},
        # Game with no rostered players (Phoenix vs Lakers).
        {"home_team": "Phoenix Suns", "away_team": "Los Angeles Lakers",
         "matchup": "LAL @ PHX", "tipoff_iso": ""},
    ]
    projections = {
        "Cade": pd.DataFrame([{"model": "points", "prediction": 28.0}]),
    }
    preds = build_slate_predictions(
        slate=slate, abbr_to_full=ABBR, modeling_df=_modeling_df(),
        projections=projections, roster_players=["Cade"], roster_only=True,
    )
    assert len(preds) == 1
    assert preds[0].home_team == "Detroit Pistons"


def test_build_slate_predictions_skips_invalid_entries() -> None:
    slate = [
        {"home_team": "Detroit Pistons", "away_team": "Orlando Magic",
         "matchup": "ORL @ DET", "tipoff_iso": ""},
        {"home_team": "", "away_team": "Orlando Magic"},  # invalid
    ]
    preds = build_slate_predictions(
        slate=slate, abbr_to_full=ABBR, modeling_df=_modeling_df(),
        projections=None, roster_players=None,
    )
    assert len(preds) == 1


def test_attach_market_lines_sets_upset_flag_and_edges() -> None:
    modeling_df = _modeling_df()
    pred = project_matchup(
        home_team="Detroit Pistons", away_team="Orlando Magic",
        matchup="ORL @ DET", tipoff_iso="",
        abbr_to_full=ABBR, modeling_df=modeling_df, projections=None,
        roster_players=None,
    )
    # Force the model to favor Orlando (away), then give the market a Detroit favor.
    pred.model_spread = -3.0
    pred.model_total = 220.0
    pred.home_win_prob = 0.4
    attach_market_lines([pred], [{
        "home_team": "Detroit Pistons", "away_team": "Orlando Magic",
        "home_spread": -7.5, "total": 218.5,
        "home_ml": -300, "away_ml": 240, "source": "consensus",
    }])
    assert pred.market_spread == -7.5
    assert pred.upset_flag is True
    assert pred.spread_edge == 4.5
    assert pred.market_home_win_prob is not None and pred.market_home_win_prob > 0.5


def test_grounding_payload_truncates_rotations() -> None:
    preds = [project_matchup(
        home_team="Detroit Pistons", away_team="Orlando Magic",
        matchup="ORL @ DET", tipoff_iso="",
        abbr_to_full=ABBR, modeling_df=_modeling_df(), projections=None,
        roster_players=None,
    )]
    payload = to_grounding_payload(preds, max_rotation_per_team=3)
    assert len(payload) == 1
    assert len(payload[0]["top_contributors_home"]) == 3
    assert len(payload[0]["top_contributors_away"]) == 3
    assert "model_home_win_prob" in payload[0]
