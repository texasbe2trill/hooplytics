from __future__ import annotations

import numpy as np
import pandas as pd

from hooplytics.data import add_pregame_features
from hooplytics.features_context import build_context_features
from hooplytics.features_role import build_role_features
from hooplytics.models import TARGET_FEATURE_GROUPS, train_models


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    players = ["A", "B", "C"]
    rows = []
    for p in players:
        for i in range(45):
            dt = pd.Timestamp("2024-10-20") + pd.Timedelta(days=i * 2)
            fga = float(rng.integers(8, 24))
            fg3a = float(rng.integers(2, 10))
            fta = float(rng.integers(1, 8))
            fgm = float(min(fga, rng.integers(3, 14)))
            fg3m = float(min(fg3a, rng.integers(1, 6)))
            ftm = float(min(fta, rng.integers(1, 7)))
            pts = int(2 * fgm + fg3m + ftm)
            reb = int(rng.integers(2, 12))
            ast = int(rng.integers(1, 10))
            stl = int(rng.integers(0, 4))
            blk = int(rng.integers(0, 4))
            tov = int(rng.integers(0, 6))
            rows.append(
                {
                    "player": p,
                    "season": "2024-25",
                    "game_date": dt,
                    "MATCHUP": "LAL @ BOS" if i % 2 else "LAL vs. BOS",
                    "min": float(rng.integers(18, 39)),
                    "fga": fga,
                    "fgm": fgm,
                    "fg3a": fg3a,
                    "fg3m": fg3m,
                    "fta": fta,
                    "ftm": ftm,
                    "pts": pts,
                    "reb": reb,
                    "ast": ast,
                    "stl": stl,
                    "blk": blk,
                    "tov": tov,
                    "oreb": int(rng.integers(0, 4)),
                    "dreb": max(0, reb - int(rng.integers(0, 3))),
                    "plus_minus": float(rng.normal(0, 8)),
                    "fg_pct": fgm / max(fga, 1),
                    "fg3_pct": fg3m / max(fg3a, 1),
                    "ft_pct": ftm / max(fta, 1),
                }
            )
    df = pd.DataFrame(rows).sort_values(["player", "game_date"]).reset_index(drop=True)
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["stl_blk"] = df["stl"] + df["blk"]
    df["fantasy_score"] = (
        df["pts"]
        + 1.2 * df["reb"]
        + 1.5 * df["ast"]
        + 3.0 * df["stl"]
        + 3.0 * df["blk"]
        - 1.0 * df["tov"]
    )
    return df


def test_context_features_create_without_bdl() -> None:
    df = _toy_df()
    df = add_pregame_features(df)
    out = build_context_features(df)

    for col in (
        "rest_days",
        "is_home",
        "is_back_to_back",
        "games_in_last_4_days",
        "assist_opportunity_score",
        "turnover_pressure_score",
        "stocks_matchup_score",
    ):
        assert col in out.columns


def test_no_leakage_in_rolling_features() -> None:
    df = _toy_df()
    out = add_pregame_features(df)

    pa = out[out["player"] == "A"].sort_values("game_date").reset_index(drop=True)
    # First row has no prior-game information.
    assert pd.isna(pa.loc[0, "ast_l3"])

    # At row index 3, ast_l3 should reflect only rows [0,1,2].
    expected = pa.loc[0:2, "ast"].mean()
    got = pa.loc[3, "ast_l3"]
    assert np.isfinite(got)
    assert abs(float(got) - float(expected)) < 1e-6


def test_role_clustering_output_shape() -> None:
    df = _toy_df()
    df = add_pregame_features(df)
    out = build_role_features(df)

    assert "role_cluster" in out.columns
    role_onehots = [c for c in out.columns if c.startswith("role_cluster_")]
    assert role_onehots, "Expected one-hot role cluster columns"
    assert len(out) == len(df)


def test_target_specific_feature_selection_present() -> None:
    assert "assists" in TARGET_FEATURE_GROUPS
    assert "turnovers" in TARGET_FEATURE_GROUPS
    assert "stl_blk" in TARGET_FEATURE_GROUPS
    assert "assist_opportunity_score" in TARGET_FEATURE_GROUPS["assists"]
    assert "turnover_pressure_score" in TARGET_FEATURE_GROUPS["turnovers"]
    assert "stocks_matchup_score" in TARGET_FEATURE_GROUPS["stl_blk"]


def test_train_models_fallback_when_context_missing() -> None:
    df = _toy_df()
    df = add_pregame_features(df)
    # Deliberately do not call build_context_features; training should still run.
    bundle = train_models(df, time_aware_validation=True)

    assert bundle.metrics is not None
    assert not bundle.metrics.empty
    assert bundle.uplift_report is not None
    assert not bundle.uplift_report.empty
