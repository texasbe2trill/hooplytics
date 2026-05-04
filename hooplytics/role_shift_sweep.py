"""role_shift_sweep.py
━━━━━━━━━━━━━━━━━━━
Threshold sweep on the assists signal: find the σ cutoff that maximises the
gap between clean-bucket and suppress-bucket directional accuracy.

Reuses the per-game records produced by role_shift_validate to avoid
re-running RACE inference for every threshold value.

Run:
    .venv/bin/python3 -m hooplytics.role_shift_sweep
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .backtest import _load_player_odds, _lookup_line
from .constants import MODEL_TO_COL
from .data import PlayerStore, nba_seasons
from .models import load_models
from .predict import project_next_game


def _seasons() -> list[str]:
    today = date.today()
    if today.month >= 10:
        start, end_ex = today.year - 1, today.year + 1
    else:
        start, end_ex = today.year - 2, today.year
    return nba_seasons(start, end_ex)


def collect_assists_records(players: list[str]) -> pd.DataFrame:
    """One row per evaluable game with the raw assists z-score and outcome."""
    bundle = load_models("bundles/race_fast.joblib")
    store = PlayerStore()
    target_col = MODEL_TO_COL["assists"]

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
        if target_col not in rows.columns or len(rows) < 11:
            continue
        odds = _load_player_odds(player, "assists")
        if odds.empty:
            continue

        for pos in range(10, len(rows)):
            prior = rows.iloc[:pos]
            game_row = rows.iloc[pos]
            game_date_ts = pd.Timestamp(game_row["game_date"])
            line = _lookup_line(odds, game_date_ts)
            if line is None or pd.isna(game_row[target_col]):
                continue

            latest = prior.iloc[-1]
            ast_l3 = latest.get("ast_l3")
            ast_l30 = latest.get("ast_l30")
            ast_std = latest.get("ast_std_l10")
            if pd.isna(ast_l3) or pd.isna(ast_l30) or pd.isna(ast_std) or ast_std <= 0:
                continue
            z = float((ast_l3 - ast_l30) / ast_std)

            try:
                proj_df = project_next_game(
                    player, bundle=bundle, store=store,
                    last_n=10, modeling_df=prior,
                )
            except Exception:
                continue
            sub = proj_df[proj_df["model"] == "assists"]
            if sub.empty:
                continue
            projected = float(sub.iloc[0]["prediction"])
            actual = float(game_row[target_col])

            call = "MORE" if projected > line else "LESS"
            outcome = "Over" if actual > line else ("Under" if actual < line else "Push")
            correct = (call == "MORE" and outcome == "Over") or (
                call == "LESS" and outcome == "Under"
            )
            records.append({
                "player": player,
                "game_date": game_date_ts.strftime("%Y-%m-%d"),
                "z": z,
                "abs_z": abs(z),
                "abs_err": abs(actual - projected),
                "correct": bool(correct),
            })
    return pd.DataFrame(records)


def sweep(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """For each threshold, split into would-suppress / clean buckets and report."""
    rows: list[dict] = []
    n_total = len(df)
    overall_acc = float(df["correct"].mean()) if n_total else 0.0
    for t in thresholds:
        sup = df[df["abs_z"] >= t]
        clean = df[df["abs_z"] < t]
        if sup.empty or clean.empty:
            rows.append({
                "threshold": t,
                "n_suppress": len(sup),
                "n_clean": len(clean),
                "acc_suppress": None,
                "acc_clean": None,
                "lift": None,
                "voided_pct": round(len(sup) / n_total, 3) if n_total else None,
            })
            continue
        acc_sup = float(sup["correct"].mean())
        acc_clean = float(clean["correct"].mean())
        rows.append({
            "threshold": t,
            "n_suppress": len(sup),
            "n_clean": len(clean),
            "acc_suppress": round(acc_sup, 3),
            "acc_clean": round(acc_clean, 3),
            "lift": round(acc_clean - acc_sup, 3),
            "voided_pct": round(len(sup) / n_total, 3),
        })
    out = pd.DataFrame(rows)
    out.attrs["overall_acc"] = round(overall_acc, 3)
    out.attrs["n_total"] = n_total
    return out


def main() -> None:
    players = [
        "Jalen Brunson", "Donovan Mitchell", "James Harden",
        "LeBron James", "Kevin Durant",
        "Nikola Jokic", "Giannis Antetokounmpo",
    ]
    print("Collecting per-game assists records (this is the slow step)…")
    df = collect_assists_records(players)
    if df.empty:
        print("No evaluable games found.")
        return

    print(f"\nTotal evaluable assists games: {len(df)}")
    print(f"Overall directional accuracy:  {df['correct'].mean():.3f}")
    print(f"Z-score range: [{df['z'].min():.2f}, {df['z'].max():.2f}], "
          f"|z| mean={df['abs_z'].mean():.2f}\n")

    thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    table = sweep(df, thresholds)
    print("Threshold sweep — assists signal:")
    print(table.to_string(index=False))
    print()

    # Best by lift (clean − suppress accuracy gap)
    valid = table.dropna(subset=["lift"])
    if valid.empty:
        return
    best = valid.sort_values("lift", ascending=False).iloc[0]
    print(f"Best threshold by lift: {best['threshold']:.2f}σ  "
          f"(clean={best['acc_clean']:.3f}, suppress={best['acc_suppress']:.3f}, "
          f"lift=+{best['lift']:.3f}, voids {best['voided_pct']:.0%} of games)")


if __name__ == "__main__":
    main()
