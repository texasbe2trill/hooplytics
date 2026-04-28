"""Market-anchored calibration of model predictions using historical odds.

Trains a per-market linear correction ``actual ~ a + b * line`` from the
cached Odds API history, plus per-(player, model) residual means. At
inference, the model's prediction is blended with the calibrated market
baseline; this prevents systematic over/under-projection drift while still
letting the model contribute its edge.

The artifact is a small JSON file (``bundles/calibration_v1.json``) so it
ships in the repo and can be regenerated without re-training the RACE bundle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import MODEL_TO_COL


# Per-market default blend weight: how much the *model* prediction contributes
# vs the calibrated market baseline. Low-signal markets (assists, stl_blk,
# turnovers) should lean more on the market because their R² is low; the
# model adds noise more often than edge there.
DEFAULT_BLEND_WEIGHTS: dict[str, float] = {
    "points":        0.55,
    "rebounds":      0.50,
    "assists":       0.30,
    "pra":           0.55,
    "threepm":       0.45,
    "stl_blk":       0.20,
    "turnovers":     0.20,
    "fantasy_score": 0.55,
}

# Targets the calibration covers directly (must match the historical odds
# market vocabulary in ``ODDS_MARKETS``).
DIRECT_MARKETS: tuple[str, ...] = ("points", "rebounds", "assists", "threepm")

DEFAULT_CALIBRATION_PATH = Path("bundles/calibration_v1.json")
CALIBRATION_VERSION = "calibration_v1"


@dataclass
class Calibration:
    """Per-market and per-player correction parameters."""

    per_market: dict[str, dict[str, float]] = field(default_factory=dict)
    per_player: dict[str, dict[str, float]] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BLEND_WEIGHTS))
    trained_at: str = ""
    version: str = CALIBRATION_VERSION

    def to_dict(self) -> dict:
        # Tuple keys (player, model) → "player||model" for JSON safety.
        per_player_serial = {
            f"{player}||{model}": stats for (player, model), stats in self.per_player.items()
        }
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "per_market": self.per_market,
            "per_player": per_player_serial,
            "blend_weights": self.blend_weights,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        per_player: dict[str, dict[str, float]] = {}
        for key, stats in (payload.get("per_player") or {}).items():
            if "||" not in key:
                continue
            player, model = key.split("||", 1)
            per_player[(player, model)] = stats  # type: ignore[index]
        return cls(
            per_market=dict(payload.get("per_market") or {}),
            per_player=per_player,
            blend_weights={**DEFAULT_BLEND_WEIGHTS, **dict(payload.get("blend_weights") or {})},
            trained_at=str(payload.get("trained_at", "")),
            version=str(payload.get("version", CALIBRATION_VERSION)),
        )


# ── Persistence ──────────────────────────────────────────────────────────────
def save_calibration(calib: Calibration, path: Path | str = DEFAULT_CALIBRATION_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib.to_dict(), indent=2))
    return out


def load_calibration(path: Path | str = DEFAULT_CALIBRATION_PATH) -> Calibration | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — corrupt file shouldn't crash predict
        return None
    return Calibration.from_dict(payload)


# ── Name canonicalisation ────────────────────────────────────────────────────
def _canon_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


# ── Build ────────────────────────────────────────────────────────────────────
def _huber_fit(line: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """Robust linear fit ``actual ≈ a + b * line`` using Huber loss.

    Falls back to OLS when scikit-learn's HuberRegressor is unavailable.
    Returns ``(a, b, rmse)``.
    """
    line = np.asarray(line, dtype=float).reshape(-1, 1)
    actual = np.asarray(actual, dtype=float)
    try:
        from sklearn.linear_model import HuberRegressor

        est = HuberRegressor(max_iter=200).fit(line, actual)
        a = float(est.intercept_)
        b = float(est.coef_[0])
    except Exception:  # noqa: BLE001
        cov = np.cov(line.ravel(), actual, ddof=0)
        var_l = float(np.var(line))
        b = float(cov[0, 1] / var_l) if var_l > 0 else 1.0
        a = float(np.mean(actual) - b * np.mean(line.ravel()))
    pred = a + b * line.ravel()
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return a, b, rmse


def _join_odds_with_actuals(
    player_data: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join historical odds rows to actual game outcomes.

    Returns columns ``[player, model, game_date, line, actual, books, season_type?]``.
    """
    if player_data.empty or odds_df.empty:
        return pd.DataFrame(columns=["player", "model", "game_date", "line", "actual", "books"])

    pd_df = player_data.copy()
    pd_df["game_date"] = pd.to_datetime(pd_df["game_date"], errors="coerce").dt.normalize()
    pd_df = pd_df.dropna(subset=["game_date", "player"])
    pd_df["_canon"] = pd_df["player"].map(_canon_name)

    od = odds_df.copy()
    od["game_date"] = pd.to_datetime(od["game_date"], errors="coerce").dt.normalize()
    od = od.dropna(subset=["game_date", "player", "model", "line"])
    od["_canon"] = od["player"].map(_canon_name)

    pieces: list[pd.DataFrame] = []
    for model_name in od["model"].dropna().unique().tolist():
        target_col = MODEL_TO_COL.get(str(model_name))
        if target_col is None or target_col not in pd_df.columns:
            continue
        od_sub = od[od["model"] == model_name][["_canon", "game_date", "line", "books"]]
        merge_cols = ["_canon", "game_date", target_col, "player"]
        if "season_type" in pd_df.columns:
            merge_cols.append("season_type")
        merged = od_sub.merge(
            pd_df[merge_cols],
            on=["_canon", "game_date"],
            how="inner",
        )
        if merged.empty:
            continue
        merged = merged.rename(columns={target_col: "actual"})
        merged["model"] = model_name
        keep = ["player", "model", "game_date", "line", "actual", "books"]
        if "season_type" in merged.columns:
            keep.append("season_type")
        pieces.append(merged[keep])

    if not pieces:
        return pd.DataFrame(columns=["player", "model", "game_date", "line", "actual", "books"])
    return pd.concat(pieces, ignore_index=True)


def build_calibration(
    player_data: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    min_market_samples: int = 200,
    min_player_samples: int = 25,
    blend_weights: dict[str, float] | None = None,
) -> Calibration:
    """Fit per-market and per-player corrections from historical odds.

    Parameters
    ----------
    player_data
        Modeling-ready player game logs with ``player``, ``game_date``, and
        the target columns (``pts``, ``reb``, ``ast``, ``fg3m``, …).
    odds_df
        Output of :func:`hooplytics.odds.load_cached_historical_odds`.
    min_market_samples
        Skip a market if fewer than this many joined samples are available
        (keeps noisy fits out of the artifact).
    min_player_samples
        Skip per-player residual correction unless enough samples exist.
    blend_weights
        Optional override for the per-market blend weight between model
        prediction and calibrated baseline. Falls back to
        :data:`DEFAULT_BLEND_WEIGHTS`.
    """
    joined = _join_odds_with_actuals(player_data, odds_df)
    per_market: dict[str, dict[str, float]] = {}
    per_player: dict[tuple[str, str], dict[str, float]] = {}

    if joined.empty:
        return Calibration(
            per_market={},
            per_player={},
            blend_weights={**DEFAULT_BLEND_WEIGHTS, **(blend_weights or {})},
            trained_at=date.today().isoformat(),
        )

    for model_name, sub in joined.groupby("model"):
        if len(sub) < min_market_samples:
            continue
        a, b, rmse = _huber_fit(sub["line"].to_numpy(), sub["actual"].to_numpy())
        per_market[str(model_name)] = {
            "a": round(a, 4),
            "b": round(b, 4),
            "n": int(len(sub)),
            "rmse": round(rmse, 4),
        }

        # Per-player residual on top of the calibrated baseline.
        baseline = a + b * sub["line"].to_numpy()
        residual = sub["actual"].to_numpy() - baseline
        sub_with_resid = sub.assign(_resid=residual)
        for player, prows in sub_with_resid.groupby("player"):
            if len(prows) < min_player_samples:
                continue
            mean_resid = float(prows["_resid"].mean())
            # Clip extreme residuals so a single hot stretch can't move the
            # baseline by more than ~20% of its scale.
            cap = max(abs(a) * 0.25 + abs(b) * float(prows["line"].mean()) * 0.15, 1.0)
            mean_resid = float(np.clip(mean_resid, -cap, cap))
            per_player[(str(player), str(model_name))] = {
                "residual_mean": round(mean_resid, 4),
                "n": int(len(prows)),
            }

    return Calibration(
        per_market=per_market,
        per_player=per_player,
        blend_weights={**DEFAULT_BLEND_WEIGHTS, **(blend_weights or {})},
        trained_at=date.today().isoformat(),
    )


# ── Apply ────────────────────────────────────────────────────────────────────
def calibrated_baseline(
    line: float,
    *,
    model: str,
    player: str | None = None,
    calib: Calibration,
) -> float | None:
    """Return ``a + b * line + per_player_residual`` if a fit exists, else None."""
    market = calib.per_market.get(model)
    if not market:
        return None
    base = float(market["a"]) + float(market["b"]) * float(line)
    if player is not None:
        bonus = calib.per_player.get((player, model), {}).get("residual_mean", 0.0)
        base += float(bonus)
    return base


def apply_calibration(
    model_prediction: float,
    *,
    model: str,
    player: str | None,
    line: float | None,
    calib: Calibration | None,
    weight_override: float | None = None,
) -> float:
    """Blend ``model_prediction`` with the calibrated market baseline.

    Returns the model prediction unchanged when no calibration is loaded,
    no line is available, or no fit exists for the market. ``weight_override``
    forces a specific model-vs-baseline weight (useful for the LOW-confidence
    fallback path).
    """
    if calib is None or line is None or not np.isfinite(line):
        return float(model_prediction)
    base = calibrated_baseline(float(line), model=model, player=player, calib=calib)
    if base is None or not np.isfinite(base):
        return float(model_prediction)
    w = (
        float(weight_override)
        if weight_override is not None
        else float(calib.blend_weights.get(model, 0.5))
    )
    w = max(0.0, min(1.0, w))
    return float(w * float(model_prediction) + (1.0 - w) * float(base))


def fit_from_cache(
    *,
    seasons: Iterable[str] | None = None,
    players: Iterable[str] | None = None,
    output_path: Path | str = DEFAULT_CALIBRATION_PATH,
    verbose: bool = False,
) -> Calibration:
    """Convenience entry point: load cached odds, fetch logs for the players
    that appear in them, fit, and persist.

    ``players`` defaults to every player in the historical odds cache that
    has at least one row matched to ``seasons``. ``seasons`` defaults to the
    last two NBA seasons.
    """
    from .data import PlayerStore, nba_seasons
    from .odds import load_cached_historical_odds

    odds_df = load_cached_historical_odds()
    if odds_df.empty:
        raise RuntimeError(
            "No historical odds cache found. Run `hooplytics-train-bundle` "
            "ingest first or pre-populate data/cache/odds/history/."
        )

    if seasons is None:
        today = date.today()
        if today.month >= 10:
            start, end = today.year - 2, today.year + 1
        else:
            start, end = today.year - 3, today.year
        seasons = nba_seasons(start, end)
    seasons = list(seasons)

    if players is None:
        players = sorted({str(p) for p in odds_df["player"].dropna().unique()})
    players = list(players)
    if verbose:
        print(f"Calibration target: {len(players)} players × {len(seasons)} seasons")

    store = PlayerStore()
    parts: list[pd.DataFrame] = []
    for i, player in enumerate(players, 1):
        try:
            df = store.fetch_player_seasons(player, seasons)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        parts.append(df.assign(player=player))
        if verbose and i % 25 == 0:
            print(f"  fetched {i}/{len(players)}")

    if not parts:
        raise RuntimeError("Failed to fetch any player game logs for calibration.")

    from .constants import NBA_RENAME
    from .fantasy import fantasy

    raw = pd.concat(parts, ignore_index=True).rename(columns=NBA_RENAME)
    raw["game_date"] = pd.to_datetime(raw["game_date"], format="%b %d, %Y", errors="coerce")
    raw["pra"] = raw.get("pts", 0) + raw.get("reb", 0) + raw.get("ast", 0)
    raw["stl_blk"] = raw.get("stl", 0) + raw.get("blk", 0)
    raw["fantasy_score"] = fantasy(raw)

    calib = build_calibration(raw, odds_df)
    save_calibration(calib, output_path)
    if verbose:
        print(
            f"Saved calibration → {output_path} "
            f"(markets={len(calib.per_market)}, players={len(calib.per_player)})"
        )
    return calib
