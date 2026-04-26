"""Role-state feature engineering for RACE.

Role fingerprints are generated from prior-game rolling rates only. Cluster
labels are intentionally heuristic; the numeric cluster features are what the
models consume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROLE_LABELS = [
    "primary_creator",
    "secondary_creator",
    "scoring_wing",
    "rim_big",
    "stretch_big",
    "defensive_specialist",
    "low_usage_bench",
]


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rolling_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _distance_features(X_scaled: np.ndarray, kmeans: KMeans, prefix: str = "role_dist") -> pd.DataFrame:
    dists = kmeans.transform(X_scaled)
    cols = [f"{prefix}_{i}" for i in range(dists.shape[1])]
    return pd.DataFrame(dists, columns=cols)


def build_role_features(
    df: pd.DataFrame,
    *,
    n_clusters: int = 7,
    random_state: int = 123,
    add_distance_features: bool = True,
) -> pd.DataFrame:
    """Add role cluster features from pregame-safe rolling role fingerprints."""
    if df.empty:
        return df

    out = df.copy().sort_values(["player", "game_date"]).reset_index(drop=True)

    min_l10 = _safe_num(out, "min_l10")
    min_l5 = _safe_num(out, "min_l5")
    usg = _safe_num(out, "usg_proxy_l30")

    out["role_expected_minutes"] = min_l10.fillna(min_l5)
    out["role_usage_proxy"] = usg

    out["role_assist_rate"] = _rolling_rate(_safe_num(out, "ast_l10"), min_l10)
    out["role_turnover_rate"] = _rolling_rate(_safe_num(out, "tov_l10"), min_l10)
    out["role_threepa_rate"] = _rolling_rate(_safe_num(out, "fg3a_l10"), min_l10)
    out["role_rebound_rate"] = _rolling_rate(_safe_num(out, "reb_l10"), min_l10)
    out["role_stocks_rate"] = _rolling_rate(_safe_num(out, "stl_l10") + _safe_num(out, "blk_l10"), min_l10)
    out["role_starter_proxy"] = ((min_l10.fillna(0) >= 24) | (min_l5.fillna(0) >= 24)).astype(int)

    if "POSITION" in out.columns:
        pos = out["POSITION"].astype(str).str.upper()
        out["role_is_guard"] = pos.str.contains("G", na=False).astype(int)
        out["role_is_forward"] = pos.str.contains("F", na=False).astype(int)
        out["role_is_center"] = pos.str.contains("C", na=False).astype(int)
    else:
        out["role_is_guard"] = 0
        out["role_is_forward"] = 0
        out["role_is_center"] = 0

    role_cols = [
        "role_expected_minutes",
        "role_usage_proxy",
        "role_assist_rate",
        "role_turnover_rate",
        "role_threepa_rate",
        "role_rebound_rate",
        "role_stocks_rate",
        "role_starter_proxy",
        "role_is_guard",
        "role_is_forward",
        "role_is_center",
    ]

    X = out[role_cols].copy()
    # Fill with robust medians to keep rows usable early in season windows.
    for c in role_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = float(X[c].median()) if not X[c].dropna().empty else 0.0
        X[c] = X[c].fillna(med)

    valid_rows = len(X)
    if valid_rows < max(20, n_clusters * 3):
        out["role_cluster"] = -1
        for i in range(n_clusters):
            out[f"role_cluster_{i}"] = 0
        out["role_creator_flag"] = out["role_starter_proxy"].astype(float)
        out["role_defense_flag"] = (_safe_num(out, "stl_l10") + _safe_num(out, "blk_l10") > 1.5).astype(float)
        return out

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    n_fit_clusters = min(n_clusters, max(2, len(X) // 5))
    kmeans = KMeans(n_clusters=n_fit_clusters, random_state=random_state, n_init=10)
    cluster = kmeans.fit_predict(X_scaled)

    out["role_cluster"] = cluster.astype(int)
    for i in range(n_fit_clusters):
        out[f"role_cluster_{i}"] = (out["role_cluster"] == i).astype(int)

    if add_distance_features:
        dist_df = _distance_features(X_scaled, kmeans)
        for c in dist_df.columns:
            out[c] = dist_df[c].values

    # Optional coarse labels for diagnostics/human readability.
    label_map = {i: ROLE_LABELS[i % len(ROLE_LABELS)] for i in range(n_fit_clusters)}
    out["role_label"] = out["role_cluster"].map(label_map).fillna("unknown")

    # Soft indicators used by engineered context scores.
    creator_labels = {"primary_creator", "secondary_creator"}
    defense_labels = {"defensive_specialist", "rim_big"}
    out["role_creator_flag"] = out["role_label"].isin(creator_labels).astype(float)
    out["role_defense_flag"] = out["role_label"].isin(defense_labels).astype(float)

    return out
