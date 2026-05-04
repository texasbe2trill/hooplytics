"""role_shift_attribute.py
━━━━━━━━━━━━━━━━━━━━━━━━
Attribution-aware validation: bucket accuracy by (triggering signal × predicted stat)
to learn the *correct* suppression map from data instead of guessing.

For every evaluable game:
  - Compute z-scores for all 4 signals (assists, points, usage, minutes).
  - Tag the game with which signals fired (|z| ≥ threshold).
  - For each predicted stat, compare projection vs actual vs cached line.
  - Aggregate: when signal X fires, what does directional accuracy on stat Y do?

This tells us, per (signal, stat) cell, whether suppression is justified.

Run:
    .venv/bin/python3 -m hooplytics.role_shift_attribute
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .backtest import _load_player_odds, _lookup_line
from .constants import MODEL_TO_COL
from .data import PlayerStore, nba_seasons
from .models import load_models
from .predict import project_next_game


_STATS = ["assists", "points", "pra"]


def _seasons() -> list[str]:
    today = date.today()
    if today.month >= 10:
        start, end_ex = today.year - 1, today.year + 1
    else:
        start, end_ex = today.year - 2, today.year
    return nba_seasons(start, end_ex)


def _signal_z(features: dict[str, float]) -> dict[str, float]:
    """Return z-scores for all 4 signals using the same math as the detector."""
    out: dict[str, float] = {}

    ast_l3, ast_l30, ast_std = (
        features.get("ast_l3"), features.get("ast_l30"), features.get("ast_std_l10"),
    )
    if all(v is not None for v in (ast_l3, ast_l30, ast_std)) and ast_std > 0:
        out["assists_sig"] = (ast_l3 - ast_l30) / ast_std

    pts_l3, pts_l30, pts_std = (
        features.get("pts_l3"), features.get("pts_l30"), features.get("pts_dev_s"),
    )
    if all(v is not None for v in (pts_l3, pts_l30, pts_std)) and pts_std > 0:
        out["points_sig"] = (pts_l3 - pts_l30) / pts_std

    fga_l10, fga_l30 = features.get("fga_l10"), features.get("fga_l30")
    if all(v is not None for v in (fga_l10, fga_l30)) and fga_l30 > 0:
        out["usage_sig"] = ((fga_l10 - fga_l30) / fga_l30) / 0.10

    min_l3, min_l30 = features.get("min_l3"), features.get("min_l30")
    if all(v is not None for v in (min_l3, min_l30)) and min_l30 > 0:
        out["minutes_sig"] = ((min_l3 - min_l30) / min_l30) / 0.10

    return out


def _extract_features(row) -> dict[str, float]:
    keys = [
        "ast_l3", "ast_l30", "ast_std_l10",
        "pts_l3", "pts_l30", "pts_dev_s",
        "fga_l10", "fga_l30",
        "min_l3", "min_l30",
    ]
    out: dict[str, float] = {}
    for k in keys:
        if k in row.index:
            v = row[k]
            if pd.notna(v):
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def collect(players: list[str]) -> pd.DataFrame:
    """Per (player, game, stat) row with signal z-scores + outcome."""
    bundle = load_models("bundles/race_fast.joblib")
    store = PlayerStore()

    records: list[dict] = []
    for player in players:
        raw = store.load_player_data({player: _seasons()})
        if raw.empty:
            continue
        full_df = store.modeling_frame(raw)
        rows = (
            full_df[full_df["player"] == player]
            .sort_values("game_date")
            .reset_index(drop=True)
        )
        if len(rows) < 11:
            continue

        # Pre-load odds for all stats
        odds_by_stat = {s: _load_player_odds(player, s) for s in _STATS}

        for pos in range(10, len(rows)):
            prior = rows.iloc[:pos]
            game_row = rows.iloc[pos]
            game_date_ts = pd.Timestamp(game_row["game_date"])

            features = _extract_features(prior.iloc[-1])
            if not features:
                continue
            zs = _signal_z(features)

            try:
                proj_df = project_next_game(
                    player, bundle=bundle, store=store,
                    last_n=10, modeling_df=prior,
                )
            except Exception:
                continue

            for stat in _STATS:
                target_col = MODEL_TO_COL.get(stat)
                if target_col is None or pd.isna(game_row.get(target_col)):
                    continue
                line = _lookup_line(odds_by_stat[stat], game_date_ts)
                if line is None:
                    continue
                sub = proj_df[proj_df["model"] == stat]
                if sub.empty:
                    continue

                projected = float(sub.iloc[0]["prediction"])
                actual = float(game_row[target_col])
                call = "MORE" if projected > line else "LESS"
                outcome = "Over" if actual > line else ("Under" if actual < line else "Push")
                correct = (call == "MORE" and outcome == "Over") or (
                    call == "LESS" and outcome == "Under"
                )
                rec = {
                    "player": player,
                    "game_date": game_date_ts.strftime("%Y-%m-%d"),
                    "stat": stat,
                    "abs_err": abs(actual - projected),
                    "correct": bool(correct),
                    **{k: round(v, 3) for k, v in zs.items()},
                }
                records.append(rec)
    return pd.DataFrame(records)


def cross_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """For each (signal, predicted-stat) cell, compute clean vs fired accuracy."""
    sig_cols = ["assists_sig", "points_sig", "usage_sig", "minutes_sig"]
    rows = []
    for sig in sig_cols:
        if sig not in df.columns:
            continue
        for stat in _STATS:
            sub = df[df["stat"] == stat].copy()
            sub = sub.dropna(subset=[sig])
            if sub.empty:
                continue
            fired = sub[sub[sig].abs() >= threshold]
            clean = sub[sub[sig].abs() < threshold]
            if fired.empty:
                rows.append({
                    "signal": sig, "stat": stat,
                    "n_fired": 0, "n_clean": len(clean),
                    "acc_fired": None,
                    "acc_clean": round(float(clean["correct"].mean()), 3) if len(clean) else None,
                    "lift": None,
                    "verdict": "no_fire",
                })
                continue
            acc_fired = float(fired["correct"].mean())
            acc_clean = float(clean["correct"].mean()) if len(clean) else None
            lift = acc_clean - acc_fired if acc_clean is not None else None
            verdict = (
                "SUPPRESS_HELPS" if (lift is not None and lift > 0.05)
                else "SUPPRESS_HURTS" if (lift is not None and lift < -0.05)
                else "neutral"
            )
            rows.append({
                "signal": sig, "stat": stat,
                "n_fired": len(fired), "n_clean": len(clean),
                "acc_fired": round(acc_fired, 3),
                "acc_clean": round(acc_clean, 3) if acc_clean is not None else None,
                "lift": round(lift, 3) if lift is not None else None,
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


def main() -> None:
    players = [
        "Jalen Brunson", "Donovan Mitchell", "James Harden",
        "LeBron James", "Kevin Durant",
        "Nikola Jokic", "Giannis Antetokounmpo",
    ]
    print("Collecting per-game records (slow step)…")
    df = collect(players)
    if df.empty:
        print("No evaluable games.")
        return

    print(f"\nTotal records: {len(df)} ({df['stat'].nunique()} stats × games)")
    print("\nSignal z-score ranges (across all games):")
    for sig in ["assists_sig", "points_sig", "usage_sig", "minutes_sig"]:
        if sig in df.columns:
            s = df[sig].dropna()
            print(f"  {sig:<14} min={s.min():+.2f}  max={s.max():+.2f}  "
                  f"|z|≥1.5: {(s.abs() >= 1.5).sum():>3}  |z|≥2.0: {(s.abs() >= 2.0).sum():>3}")

    for t in (1.5, 1.25, 1.0):
        print(f"\n{'='*70}")
        print(f"Cross-table at |z| ≥ {t}σ  (clean − fired = lift; positive ⇒ suppression helps)")
        print(f"{'='*70}")
        ct = cross_table(df, t)
        print(ct.to_string(index=False))


if __name__ == "__main__":
    main()
