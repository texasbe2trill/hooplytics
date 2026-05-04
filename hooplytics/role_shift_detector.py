"""role_shift_detector.py
━━━━━━━━━━━━━━━━━━━━━━━
Pre-inference guard that catches in-series player role transitions before
they break RACE directional calls.

Checks 4 signals against rolling baselines:
    1. ASSISTS  — ast_l3 vs ast_l30,  std = ast_std_l10
    2. POINTS   — pts_l3 vs pts_l30,  std = pts_dev_s
    3. USAGE    — fga_l10 vs fga_l30, pct delta (10% ≈ 1σ)
    4. MINUTES  — min_l3 vs min_l30,  pct delta

Thresholds (constructor-configurable):
    |z| >= 1.5σ → WARN     (confidence -20%, recommend last_n=3)
    |z| >= 2.0σ → SUPPRESS (NO_CALL,         recommend last_n=2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────────────


class ShiftType(str, Enum):
    PLAYMAKER_SURGE = "PLAYMAKER_SURGE"
    PLAYMAKER_DROP = "PLAYMAKER_DROP"
    SCORER_SURGE = "SCORER_SURGE"
    SCORER_SUPPRESSED = "SCORER_SUPPRESSED"
    USAGE_COLLAPSE = "USAGE_COLLAPSE"
    MINUTES_SHIFT = "MINUTES_SHIFT"
    MULTI_ROLE_SHIFT = "MULTI_ROLE_SHIFT"


class Severity(str, Enum):
    NONE = "NONE"
    WARN = "WARN"
    SUPPRESS = "SUPPRESS"


# ── Signal result ─────────────────────────────────────────────────────────────


@dataclass
class SignalResult:
    stat: str
    recent: float
    baseline: float
    z_score: float
    action: str  # "OK" | "WARN" | "SUPPRESS"


# ── Stats suppressed by each shift type ──────────────────────────────────────
#
# These mappings are *empirically tuned* from a backtest of ~190 games across
# the default roster + anchor cohort (Feb–May 2026). The original spec
# suppressed broader stat lists, but validation showed those broader
# suppressions actively HURT directional accuracy on collateral stats — the
# detector was voiding good calls. See hooplytics/role_shift_attribute.py for
# the cross-table that justifies each entry.
#
# Rule of thumb: only suppress the stat whose model directly relies on the
# rolling feature that triggered the signal.
_SUPPRESSED_STAT_MAP: dict[ShiftType, list[str]] = {
    # Assists signal (ast_l3 vs ast_l30) — voids assists call only.
    # 350-game backtest: signal never fired at |z| ≥ 1.5σ in test window
    # (max |z| = 1.38). Scope kept narrow as canonical Barnes design.
    ShiftType.PLAYMAKER_SURGE:    ["assists"],
    ShiftType.PLAYMAKER_DROP:     ["assists"],
    # Points signal (pts_l3 vs pts_l30 / pts_dev_s) — strongest data-supported
    # suppression of any signal. 350-game backtest at |z| ≥ 1.5σ (n_fired=49):
    #   suppress points → +0.236 directional-acc lift
    #   suppress pra    → +0.310 lift
    #   suppress assists → +0.062 (mild, kept off to avoid collateral risk)
    # Guarded by min_std floor to prevent divide-by-near-zero blow-ups.
    ShiftType.SCORER_SURGE:       ["points", "pra"],
    ShiftType.SCORER_SUPPRESSED:  ["points", "pra"],
    # Usage (FGA) signal — only the points-side suppression is data-supported.
    # 350-game backtest at |z| ≥ 1.5σ (n_fired=36):
    #   suppress points  → +0.098 lift
    #   suppress assists → -0.322 lift (would void 83% accurate calls)
    #   suppress pra     → -0.046 (neutral)
    ShiftType.USAGE_COLLAPSE:     ["points"],
    # Minutes signal — only the pra-side suppression is data-supported.
    # 350-game backtest at |z| ≥ 1.5σ (n_fired=13, small but consistent):
    #   suppress pra     → +0.149 lift
    #   suppress points  → -0.170 lift
    #   suppress assists → -0.210 lift
    ShiftType.MINUTES_SHIFT:      ["pra"],
    # Multi-signal divergence is treated as a *label* — suppressed stats come
    # purely from the union of individual shift types. Adding stats here
    # caused collateral voiding in earlier experiments.
    ShiftType.MULTI_ROLE_SHIFT:   [],
}


# ── RoleShiftResult ───────────────────────────────────────────────────────────


@dataclass
class RoleShiftResult:
    player_name: str
    shift_detected: bool
    shift_types: list[ShiftType]
    signals: list[SignalResult]
    suppressed_stats: list[str]
    confidence_penalty: float
    recommended_last_n: int
    severity: Severity

    def summary(self) -> str:
        if not self.shift_detected:
            return f"{self.player_name}: No role shift detected."
        shift_names = ", ".join(s.value for s in self.shift_types)
        sup = ", ".join(self.suppressed_stats) if self.suppressed_stats else "none"
        return (
            f"{self.player_name}: {self.severity.value} — {shift_names}. "
            f"Suppressed stats: {sup}. "
            f"Confidence penalty: -{self.confidence_penalty:.0%}. "
            f"Recommended last_n={self.recommended_last_n}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "shift_detected": self.shift_detected,
            "shift_types": [s.value for s in self.shift_types],
            "signals": [
                {
                    "stat": sig.stat,
                    "recent": sig.recent,
                    "baseline": sig.baseline,
                    "z_score": round(sig.z_score, 3),
                    "action": sig.action,
                }
                for sig in self.signals
            ],
            "suppressed_stats": self.suppressed_stats,
            "confidence_penalty": self.confidence_penalty,
            "recommended_last_n": self.recommended_last_n,
            "severity": self.severity.value,
        }


# ── Detector ──────────────────────────────────────────────────────────────────


class RoleShiftDetector:
    """Stateless pre-inference guard for in-series player role transitions.

    All thresholds are constructor params so they can be tuned without
    touching the detection logic.
    """

    # Per-signal threshold defaults — empirically tuned from the 423-game
    # backtest. Stronger signals (points, usage) trip earlier because their
    # cross-table lifts at |z| ≥ 1.5σ remain strongly positive; weaker/sparser
    # signals (minutes, assists) keep the conservative 2σ default to avoid
    # over-firing on small samples.
    DEFAULT_SIGNAL_THRESHOLDS: dict[str, dict[str, float]] = {
        # Strong signal — 49 fires at 1.5σ in backtest, lifts +0.236 (points),
        # +0.310 (pra). Earlier suppress threshold is data-supported.
        "points":    {"warn": 1.0, "suppress": 1.5},
        # Strong signal — 36 fires at 1.5σ, +0.098 lift on points. Catching
        # more borderline cases improves the points-side suppression yield.
        "usage_fga": {"warn": 1.0, "suppress": 1.5},
        # Sparse signal — only 13 fires at 1.5σ, 3 at 2σ. Lift +0.149 on pra
        # is real but small-sample. Keep conservative.
        "minutes":   {"warn": 1.5, "suppress": 2.0},
        # Never fires at 1.5σ in backtest (max |z| = 1.38). Keep canonical
        # Barnes design at 2σ — when it does fire, it should be unambiguous.
        "assists":   {"warn": 1.5, "suppress": 2.0},
    }

    def __init__(
        self,
        warn_threshold: float = 1.5,
        suppress_threshold: float = 2.0,
        min_std: float = 1.0,
        signal_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        # Global fallback thresholds — used only for signals not present in
        # signal_thresholds (i.e. for backwards-compatible custom setups).
        self.warn_threshold = warn_threshold
        self.suppress_threshold = suppress_threshold
        # Min-std floor prevents divide-by-near-zero blow-ups when a player has
        # been remarkably consistent (e.g. pts_dev_s = 0.1 produces |z| = 100).
        # Backtest showed points_sig hitting σ=158 without this guard.
        self.min_std = min_std
        # Per-signal threshold overrides. Each entry: {"warn": x, "suppress": y}.
        # When None, uses class default (DEFAULT_SIGNAL_THRESHOLDS).
        self.signal_thresholds: dict[str, dict[str, float]] = (
            signal_thresholds
            if signal_thresholds is not None
            else dict(self.DEFAULT_SIGNAL_THRESHOLDS)
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _std_z(self, recent: float, baseline: float, std: float) -> float:
        """Standard z-score using an explicit std dev. Floored at self.min_std."""
        eff_std = max(std, self.min_std)
        if eff_std <= 0:
            return 0.0
        return (recent - baseline) / eff_std

    def _pct_z(
        self,
        recent: float,
        baseline: float,
        pct_per_sigma: float = 0.10,
    ) -> float:
        """Percentage-delta z-score.  10% change ≈ 1σ by default."""
        if baseline <= 0:
            return 0.0
        return ((recent - baseline) / baseline) / pct_per_sigma

    def _action_for(self, signal: str, z: float) -> str:
        """Per-signal severity classification (uses per-signal threshold map)."""
        az = abs(z)
        bounds = self.signal_thresholds.get(
            signal,
            {"warn": self.warn_threshold, "suppress": self.suppress_threshold},
        )
        if az >= bounds["suppress"]:
            return "SUPPRESS"
        if az >= bounds["warn"]:
            return "WARN"
        return "OK"

    def _action(self, z: float) -> str:
        """Legacy single-threshold action — kept for backward compatibility."""
        az = abs(z)
        if az >= self.suppress_threshold:
            return "SUPPRESS"
        if az >= self.warn_threshold:
            return "WARN"
        return "OK"

    # ── Public API ───────────────────────────────────────────────────────────

    def check(self, player_name: str, features: dict[str, float]) -> RoleShiftResult:
        """Run all 4 role-shift signals and return a RoleShiftResult."""
        signals: list[SignalResult] = []

        # 1. ASSISTS
        ast_l3 = features.get("ast_l3")
        ast_l30 = features.get("ast_l30")
        ast_std = features.get("ast_std_l10")
        if None not in (ast_l3, ast_l30, ast_std):
            z = self._std_z(ast_l3, ast_l30, ast_std)
            signals.append(SignalResult("assists", ast_l3, ast_l30, z, self._action_for("assists", z)))

        # 2. POINTS
        pts_l3 = features.get("pts_l3")
        pts_l30 = features.get("pts_l30")
        pts_std = features.get("pts_dev_s")
        if None not in (pts_l3, pts_l30, pts_std):
            z = self._std_z(pts_l3, pts_l30, pts_std)
            signals.append(SignalResult("points", pts_l3, pts_l30, z, self._action_for("points", z)))

        # 3. USAGE (field-goal attempts as usage proxy)
        fga_l10 = features.get("fga_l10")
        fga_l30 = features.get("fga_l30")
        if None not in (fga_l10, fga_l30):
            z = self._pct_z(fga_l10, fga_l30)
            signals.append(SignalResult("usage_fga", fga_l10, fga_l30, z, self._action_for("usage_fga", z)))

        # 4. MINUTES
        min_l3 = features.get("min_l3")
        min_l30 = features.get("min_l30")
        if None not in (min_l3, min_l30):
            z = self._pct_z(min_l3, min_l30)
            signals.append(SignalResult("minutes", min_l3, min_l30, z, self._action_for("minutes", z)))

        # ── Classify shift types ─────────────────────────────────────────────
        shift_types: list[ShiftType] = []
        max_severity = Severity.NONE

        for sig in signals:
            if sig.action == "OK":
                continue
            if sig.action == "SUPPRESS":
                max_severity = Severity.SUPPRESS
            elif sig.action == "WARN" and max_severity == Severity.NONE:
                max_severity = Severity.WARN

            if sig.stat == "assists":
                shift_types.append(
                    ShiftType.PLAYMAKER_SURGE if sig.z_score > 0 else ShiftType.PLAYMAKER_DROP
                )
            elif sig.stat == "points":
                shift_types.append(
                    ShiftType.SCORER_SURGE if sig.z_score > 0 else ShiftType.SCORER_SUPPRESSED
                )
            elif sig.stat == "usage_fga" and sig.z_score < 0:
                shift_types.append(ShiftType.USAGE_COLLAPSE)
            elif sig.stat == "minutes":
                shift_types.append(ShiftType.MINUTES_SHIFT)

        # Deduplicate while preserving order
        seen: set[ShiftType] = set()
        unique: list[ShiftType] = []
        for st in shift_types:
            if st not in seen:
                seen.add(st)
                unique.append(st)
        shift_types = unique

        # Multi-role shift when 2+ independent signal categories are flagged
        flagged_categories = {sig.stat for sig in signals if sig.action != "OK"}
        if len(flagged_categories) >= 2:
            shift_types.append(ShiftType.MULTI_ROLE_SHIFT)

        shift_detected = max_severity != Severity.NONE

        # ── Suppressed stats ─────────────────────────────────────────────────
        # Only SUPPRESS-severity voids calls. An earlier experiment promoted
        # WARN to soft-suppression based on a 185-game sample (WARN bucket
        # 45.5% directional acc); the 255-game backtest showed that was noise
        # — WARN games actually run ABOVE the SUPPRESS bucket (62.5% vs 58.3%).
        # WARN keeps its informational confidence_penalty role.
        suppressed: set[str] = set()
        if max_severity == Severity.SUPPRESS:
            for st in shift_types:
                suppressed.update(_SUPPRESSED_STAT_MAP.get(st, []))

        suppressed_stats = sorted(suppressed)

        # ── Confidence + window recommendation ───────────────────────────────
        if max_severity == Severity.SUPPRESS:
            confidence_penalty = 0.0   # full suppression; penalty concept doesn't apply
            recommended_last_n = 2
        elif max_severity == Severity.WARN:
            confidence_penalty = 0.20
            recommended_last_n = 3
        else:
            confidence_penalty = 0.0
            recommended_last_n = 5

        return RoleShiftResult(
            player_name=player_name,
            shift_detected=shift_detected,
            shift_types=shift_types,
            signals=signals,
            suppressed_stats=suppressed_stats,
            confidence_penalty=confidence_penalty,
            recommended_last_n=recommended_last_n,
            severity=max_severity,
        )


# ── Guard function ────────────────────────────────────────────────────────────


def apply_role_shift_guard(
    player_name: str,
    features: dict[str, float],
    model_outputs: dict[str, Any],
    detector: RoleShiftDetector,
) -> dict[str, Any]:
    """Annotate model_outputs with role-shift metadata and downgrade suppressed calls.

    - Always adds a "role_shift" key to model_outputs.
    - SUPPRESS severity: sets call="NO_CALL", call_valid=False for suppressed stats.
    - WARN severity: adds confidence_penalty=0.20 to every stat entry.
    """
    result = detector.check(player_name, features)
    model_outputs["role_shift"] = result.to_dict()

    if not result.shift_detected:
        return model_outputs

    if result.severity == Severity.SUPPRESS:
        reason = (
            f"Role shift detected ({', '.join(s.value for s in result.shift_types)}). "
            f"Signal exceeded {detector.suppress_threshold}σ threshold — call suppressed."
        )
        for stat in result.suppressed_stats:
            if stat in model_outputs and isinstance(model_outputs[stat], dict):
                model_outputs[stat]["call"] = "NO_CALL"
                model_outputs[stat]["call_valid"] = False
                model_outputs[stat]["call_reason"] = reason

    elif result.severity == Severity.WARN:
        for key, val in model_outputs.items():
            if key != "role_shift" and isinstance(val, dict):
                val["confidence_penalty"] = result.confidence_penalty

    return model_outputs


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = RoleShiftDetector()

    # Barnes problem that motivated this module
    barnes_features: dict[str, float] = {
        "ast_l3": 10.3,
        "ast_l30": 5.8,
        "ast_std_l10": 2.1,
        "pts_l3": 21.7,
        "pts_l30": 22.4,
        "pts_dev_s": 4.2,
        "fga_l10": 14.2,
        "fga_l30": 15.8,
        "min_l3": 36.8,
        "min_l30": 35.4,
    }

    result = detector.check("Scottie Barnes", barnes_features)

    print(result.summary())
    print()
    print(f"Severity:        {result.severity.value}")
    print(f"Shift types:     {[s.value for s in result.shift_types]}")
    print(f"Suppressed:      {result.suppressed_stats}")
    print(f"Recommended last_n: {result.recommended_last_n}")
    print()
    print("Signals:")
    for sig in result.signals:
        print(
            f"  {sig.stat:<12} L3={sig.recent:>6.1f}  L30={sig.baseline:>5.1f}"
            f"  σ={sig.z_score:>+6.2f}  → {sig.action}"
        )
