"""Pregame-safe context feature engineering for RACE.

These features are derived from schedule metadata, prior games, and optional
Ball Don't Lie context endpoints. Missing context never crashes the pipeline;
when unavailable, conservative proxy values are emitted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _team_abbr_from_matchup(matchup: object, side: str) -> str | None:
    if not isinstance(matchup, str) or not matchup.strip():
        return None
    parts = matchup.strip().split()
    if len(parts) < 3:
        return None
    # nba_api format: "LAL @ BOS" or "LAL vs. BOS"
    if side == "team":
        return parts[0].upper()
    return parts[-1].upper()


def _robust_z(s: pd.Series, clip: float = 4.0) -> pd.Series:
    med = float(s.median()) if not s.dropna().empty else 0.0
    q1 = float(s.quantile(0.25)) if not s.dropna().empty else 0.0
    q3 = float(s.quantile(0.75)) if not s.dropna().empty else 1.0
    iqr = max(q3 - q1, 1e-6)
    out = (s - med) / iqr
    return out.clip(-clip, clip)


def add_schedule_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add schedule-derived pregame-safe features.

    Features:
    - rest_days
    - is_home
    - is_back_to_back
    - games_in_last_4_days
    """
    if df.empty:
        return df

    out = df.copy()
    if "game_date" not in out.columns:
        out["rest_days"] = np.nan
        out["is_home"] = 0
        out["is_back_to_back"] = 0
        out["games_in_last_4_days"] = np.nan
        return out

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values(["player", "game_date"]).reset_index(drop=True)

    if "MATCHUP" in out.columns:
        out["is_home"] = (~out["MATCHUP"].astype(str).str.contains("@", na=False)).astype(int)
        out["team_abbr"] = out["MATCHUP"].apply(lambda x: _team_abbr_from_matchup(x, "team"))
        out["opp_abbr"] = out["MATCHUP"].apply(lambda x: _team_abbr_from_matchup(x, "opp"))
    else:
        out["is_home"] = 0
        out["team_abbr"] = None
        out["opp_abbr"] = None

    out["rest_days"] = (
        out.groupby("player")["game_date"]
        .transform(lambda s: s.diff().dt.days.shift(1).clip(lower=0, upper=14))
        .fillna(3)
    )
    out["is_back_to_back"] = (out["rest_days"] <= 1).astype(int)

    def _prior_games_last_4_days(s: pd.Series) -> pd.Series:
        vals = s.astype("int64") // 10**9
        arr = vals.to_numpy()
        cnt = np.full(len(arr), np.nan)
        four_days = 4 * 24 * 3600
        for i in range(len(arr)):
            if i == 0:
                continue
            prior = arr[:i]
            cnt[i] = float(np.sum((arr[i] - prior) <= four_days))
        return pd.Series(cnt, index=s.index)

    out["games_in_last_4_days"] = out.groupby("player", group_keys=False)["game_date"].apply(_prior_games_last_4_days)
    return out


def add_opponent_context(df: pd.DataFrame, bdl_client: Any = None) -> pd.DataFrame:
    """Join opponent context from BDL lookup when available.

    Fallback behavior: emit NaN columns and continue.
    """
    if df.empty:
        return df

    out = df.copy()
    for col in ("opp_pace", "opp_def_rtg", "opp_off_rtg", "opp_stl_pg", "opp_blk_pg"):
        if col not in out.columns:
            out[col] = np.nan

    if bdl_client is None or "season" not in out.columns or "opp_abbr" not in out.columns:
        return out

    try:
        seasons = sorted({
            int(str(s).split("-")[0])
            for s in out["season"].dropna().unique()
            if isinstance(s, str) and "-" in s
        })
        lookup = bdl_client.build_opponent_lookup(seasons)
    except Exception:
        return out

    if lookup is None or lookup.empty:
        return out

    merge = out.copy()
    merge["bdl_season"] = merge["season"].astype(str).str.split("-").str[0]
    merge["bdl_season"] = pd.to_numeric(merge["bdl_season"], errors="coerce")

    keep = [c for c in ("bdl_season", "abbreviation", "opp_pace", "opp_def_rtg", "opp_off_rtg", "opp_stl_pg", "opp_blk_pg") if c in lookup.columns]
    joined = merge.merge(
        lookup[keep].rename(columns={"abbreviation": "opp_abbr"}),
        on=["bdl_season", "opp_abbr"],
        how="left",
        suffixes=("", "_ctx"),
    )

    for col in ("opp_pace", "opp_def_rtg", "opp_off_rtg", "opp_stl_pg", "opp_blk_pg"):
        src = f"{col}_ctx"
        if src in joined.columns:
            out[col] = joined[src].values
    return out


def add_availability_context(df: pd.DataFrame, bdl_client: Any = None) -> pd.DataFrame:
    """Add availability features.

    Uses conservative proxies when injury feed is unavailable:
    - team_injury_count proxy from prior team active-player count drift
    - opp_injury_count via opponent team's proxy on the same game date
    - teammate_usage_missing_proxy from team-level active-player contraction
    """
    if df.empty:
        return df

    out = df.copy()

    if "team_abbr" not in out.columns or "game_date" not in out.columns:
        out["teammate_usage_missing_proxy"] = 0.0
        out["team_injury_count"] = 0.0
        out["opp_injury_count"] = 0.0
        return out

    work = out[["team_abbr", "game_date", "player"]].copy()
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    team_day = (
        work.groupby(["team_abbr", "game_date"], dropna=False)["player"]
        .nunique()
        .rename("team_active_players")
        .reset_index()
        .sort_values(["team_abbr", "game_date"])
    )
    team_day["team_active_prev"] = team_day.groupby("team_abbr")["team_active_players"].shift(1)
    team_day["team_active_mean_l5"] = (
        team_day.groupby("team_abbr")["team_active_players"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    )
    team_day["teammate_usage_missing_proxy"] = (
        team_day["team_active_mean_l5"] - team_day["team_active_prev"]
    ).clip(lower=0).fillna(0)
    team_day["team_injury_count"] = team_day["teammate_usage_missing_proxy"].round(0)

    out = out.merge(
        team_day[["team_abbr", "game_date", "teammate_usage_missing_proxy", "team_injury_count"]],
        on=["team_abbr", "game_date"],
        how="left",
    )

    opp = team_day.rename(
        columns={
            "team_abbr": "opp_abbr",
            "team_injury_count": "opp_injury_count",
        }
    )[["opp_abbr", "game_date", "opp_injury_count"]]
    out = out.merge(opp, on=["opp_abbr", "game_date"], how="left")

    # Optional injury feed refinement.
    if bdl_client is not None:
        try:
            injuries = bdl_client.fetch_player_injuries()
        except Exception:
            injuries = pd.DataFrame()
        if injuries is not None and not injuries.empty:
            team_col = None
            for c in ("team", "team_abbr", "abbreviation"):
                if c in injuries.columns:
                    team_col = c
                    break
            status_col = "status" if "status" in injuries.columns else None
            if team_col is not None:
                inj = injuries.copy()
                inj["team_abbr"] = inj[team_col].astype(str).str.upper().str[-3:]
                if status_col is not None:
                    inj = inj[inj[status_col].astype(str).str.lower().isin(["out", "doubtful", "questionable"]) ]
                inj_counts = inj.groupby("team_abbr").size().rename("inj_cnt").reset_index()
                out = out.merge(inj_counts.rename(columns={"inj_cnt": "_team_inj_live"}), on="team_abbr", how="left")
                out = out.merge(
                    inj_counts.rename(columns={"team_abbr": "opp_abbr", "inj_cnt": "_opp_inj_live"}),
                    on="opp_abbr",
                    how="left",
                )
                out["team_injury_count"] = out["_team_inj_live"].fillna(out["team_injury_count"]).fillna(0)
                out["opp_injury_count"] = out["_opp_inj_live"].fillna(out["opp_injury_count"]).fillna(0)
                out = out.drop(columns=[c for c in ("_team_inj_live", "_opp_inj_live") if c in out.columns])

    for col in ("teammate_usage_missing_proxy", "team_injury_count", "opp_injury_count"):
        if col not in out.columns:
            out[col] = 0.0

    out["teammate_usage_missing_proxy"] = out["teammate_usage_missing_proxy"].fillna(0.0)
    out["team_injury_count"] = out["team_injury_count"].fillna(0.0)
    out["opp_injury_count"] = out["opp_injury_count"].fillna(0.0)
    return out


def add_lineup_context(df: pd.DataFrame, bdl_client: Any = None) -> pd.DataFrame:
    """Add lineup-stability and expected-starter context.

    When lineup endpoints are unavailable, this falls back to prior-games minutes
    stability proxies, which are pregame-safe.
    """
    if df.empty:
        return df

    out = df.copy()
    min_l10 = pd.to_numeric(out.get("min_l10", np.nan), errors="coerce")
    min_l5 = pd.to_numeric(out.get("min_l5", np.nan), errors="coerce")

    out["expected_starter"] = ((min_l10.fillna(0) >= 24) | (min_l5.fillna(0) >= 24)).astype(int)

    min_std = pd.to_numeric(
        out["min_std_l10"] if "min_std_l10" in out.columns else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    if min_std.isna().all() and "min" in out.columns:
        out = out.sort_values(["player", "game_date"]).reset_index(drop=True)
        out["min_std_l10"] = (
            out.groupby("player")["min"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
        )
    else:
        out["min_std_l10"] = min_std

    out["lineup_stability_score"] = 1.0 - (
        out["min_std_l10"].fillna(out["min_std_l10"].median()) / (min_l10.abs().fillna(20.0) + 1.0)
    )
    out["lineup_stability_score"] = out["lineup_stability_score"].clip(0.0, 1.0)

    if bdl_client is not None and "season" in out.columns:
        # Non-fatal enrich: if lineups endpoint exists, use lineup continuity per team-date.
        try:
            seasons = sorted({int(str(s).split("-")[0]) for s in out["season"].dropna().unique() if isinstance(s, str) and "-" in s})
            lineup_frames = [bdl_client.fetch_lineups(season) for season in seasons]
            lineups = pd.concat([d for d in lineup_frames if d is not None and not d.empty], ignore_index=True) if lineup_frames else pd.DataFrame()
        except Exception:
            lineups = pd.DataFrame()

        if not lineups.empty:
            tcol = "team_abbr" if "team_abbr" in lineups.columns else None
            dcol = "game_date" if "game_date" in lineups.columns else None
            lcol = "lineup_id" if "lineup_id" in lineups.columns else None
            if tcol and dcol and lcol:
                lu = lineups[[tcol, dcol, lcol]].copy()
                lu[dcol] = pd.to_datetime(lu[dcol], errors="coerce")
                lu = lu.sort_values([tcol, dcol])
                lu["lineup_stability_score_bdl"] = (
                    lu.groupby(tcol)[lcol]
                    .transform(lambda s: (s.shift(1) == s).astype(float))
                )
                merge_lu = lu.rename(columns={tcol: "team_abbr", dcol: "game_date"})[["team_abbr", "game_date", "lineup_stability_score_bdl"]]
                out = out.merge(merge_lu, on=["team_abbr", "game_date"], how="left")
                out["lineup_stability_score"] = out["lineup_stability_score_bdl"].fillna(out["lineup_stability_score"])
                out = out.drop(columns=["lineup_stability_score_bdl"])

    return out


def add_assist_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mins = pd.to_numeric(out.get("min_l10", out.get("min_l5", np.nan)), errors="coerce")
    ast_pm = pd.to_numeric(out.get("ast_per36_l10", np.nan), errors="coerce") / 36.0
    usage = pd.to_numeric(out.get("usg_proxy_l30", np.nan), errors="coerce")
    pace = pd.to_numeric(out.get("opp_pace", np.nan), errors="coerce")
    creator = pd.to_numeric(out.get("role_creator_flag", out.get("expected_starter", 0)), errors="coerce")

    score = (
        0.30 * _robust_z(mins.fillna(mins.median()))
        + 0.30 * _robust_z(ast_pm.fillna(ast_pm.median()))
        + 0.20 * _robust_z(usage.fillna(usage.median()))
        + 0.10 * _robust_z(pace.fillna(pace.median()))
        + 0.10 * creator.fillna(0)
    )
    out["assist_opportunity_score"] = score.clip(-4, 4)
    return out


def add_turnover_pressure_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mins = pd.to_numeric(out.get("min_l10", out.get("min_l5", np.nan)), errors="coerce")
    tov_pm = pd.to_numeric(out.get("tov_per36_l10", np.nan), errors="coerce") / 36.0
    usage = pd.to_numeric(out.get("usg_proxy_l30", np.nan), errors="coerce")
    opp_stl = pd.to_numeric(out.get("opp_stl_pg", np.nan), errors="coerce")
    pace = pd.to_numeric(out.get("opp_pace", np.nan), errors="coerce")
    creator = pd.to_numeric(out.get("role_creator_flag", out.get("expected_starter", 0)), errors="coerce")

    score = (
        0.25 * _robust_z(mins.fillna(mins.median()))
        + 0.30 * _robust_z(tov_pm.fillna(tov_pm.median()))
        + 0.20 * _robust_z(usage.fillna(usage.median()))
        + 0.15 * _robust_z(opp_stl.fillna(opp_stl.median()))
        + 0.05 * _robust_z(pace.fillna(pace.median()))
        + 0.05 * creator.fillna(0)
    )
    out["turnover_pressure_score"] = score.clip(-4, 4)
    return out


def add_stocks_matchup_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mins = pd.to_numeric(out.get("min_l10", out.get("min_l5", np.nan)), errors="coerce")
    stocks_pm = (
        pd.to_numeric(out.get("stl_per36_l10", np.nan), errors="coerce")
        + pd.to_numeric(out.get("blk_per36_l10", np.nan), errors="coerce")
    ) / 36.0
    pace = pd.to_numeric(out.get("opp_pace", np.nan), errors="coerce")
    opp_blk = pd.to_numeric(out.get("opp_blk_pg", np.nan), errors="coerce")
    opp_stl = pd.to_numeric(out.get("opp_stl_pg", np.nan), errors="coerce")
    def_role = pd.to_numeric(out.get("role_defense_flag", out.get("expected_starter", 0)), errors="coerce")

    score = (
        0.30 * _robust_z(mins.fillna(mins.median()))
        + 0.30 * _robust_z(stocks_pm.fillna(stocks_pm.median()))
        + 0.15 * _robust_z(pace.fillna(pace.median()))
        + 0.10 * _robust_z(opp_blk.fillna(opp_blk.median()))
        + 0.10 * _robust_z(opp_stl.fillna(opp_stl.median()))
        + 0.05 * def_role.fillna(0)
    )
    out["stocks_matchup_score"] = score.clip(-4, 4)
    return out


def build_context_features(df: pd.DataFrame, bdl_client: Any = None) -> pd.DataFrame:
    """Build all context features in a leakage-safe order."""
    out = add_schedule_context(df)
    out = add_opponent_context(out, bdl_client=bdl_client)
    out = add_availability_context(out, bdl_client=bdl_client)
    out = add_lineup_context(out, bdl_client=bdl_client)
    out = add_assist_opportunity_score(out)
    out = add_turnover_pressure_score(out)
    out = add_stocks_matchup_score(out)
    return out
