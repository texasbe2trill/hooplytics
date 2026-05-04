"""role_shift_validate.py
━━━━━━━━━━━━━━━━━━━━━━
Empirical validation: do RoleShiftDetector NO_CALLs actually correlate with
reduced RACE directional accuracy?

For each (player, stat) pair, walk the historical game log:
  1. Reconstruct the pregame feature row (rows strictly before game N).
  2. Run the RACE projection from that pregame view.
  3. Run RoleShiftDetector on the same pregame features.
  4. Compare the model's MORE/LESS call to the actual result vs the cached
     sportsbook line.
  5. Bucket the game by detector severity (NONE / WARN / SUPPRESS) and
     report per-bucket directional accuracy + MAE.

Run:
    .venv/bin/python3 -m hooplytics.role_shift_validate
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .backtest import _load_player_odds, _lookup_line
from .constants import MODEL_TO_COL
from .data import PlayerStore, nba_seasons
from .models import load_models
from .predict import project_next_game
from .role_shift_detector import RoleShiftDetector

_FEATURE_KEYS = [
    "ast_l3", "ast_l30", "ast_std_l10",
    "pts_l3", "pts_l30", "pts_dev_s",
    "fga_l10", "fga_l30",
    "min_l3", "min_l30",
]


def _seasons() -> list[str]:
    today = date.today()
    if today.month >= 10:
        start, end_ex = today.year - 1, today.year + 1
    else:
        start, end_ex = today.year - 2, today.year
    return nba_seasons(start, end_ex)


def _extract_features(row) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _FEATURE_KEYS:
        if k in row.index:
            v = row[k]
            if pd.notna(v):
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def validate_player_stat(
    player: str,
    stat: str,
    *,
    store: PlayerStore,
    bundle,
    detector: RoleShiftDetector,
    min_prior_games: int = 10,
    last_n: int = 10,
) -> pd.DataFrame:
    """Walk every game with a cached line and tag it with detector severity + outcome."""
    if stat not in MODEL_TO_COL:
        raise ValueError(f"Unknown stat '{stat}'")
    target_col = MODEL_TO_COL[stat]

    raw = store.load_player_data({player: _seasons()})
    if raw.empty:
        return pd.DataFrame()
    full_df = store.modeling_frame(raw)
    player_rows = (
        full_df[full_df["player"] == player]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    if target_col not in player_rows.columns:
        return pd.DataFrame()
    if len(player_rows) < min_prior_games + 1:
        return pd.DataFrame()

    odds = _load_player_odds(player, stat)
    if odds.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for pos in range(min_prior_games, len(player_rows)):
        prior = player_rows.iloc[:pos]
        game_row = player_rows.iloc[pos]
        game_date_ts = pd.Timestamp(game_row["game_date"])

        line = _lookup_line(odds, game_date_ts)
        if line is None:
            continue
        actual_raw = game_row[target_col]
        if pd.isna(actual_raw):
            continue
        actual = float(actual_raw)

        try:
            proj_df = project_next_game(
                player, bundle=bundle, store=store,
                last_n=last_n, modeling_df=prior,
            )
        except Exception:
            continue
        stat_rows = proj_df[proj_df["model"] == stat]
        if stat_rows.empty:
            continue
        projected = float(stat_rows.iloc[0]["prediction"])

        # Detector runs on the same pregame view that fed the projection.
        features = _extract_features(prior.iloc[-1])
        if not features:
            continue
        result = detector.check(player, features)

        call = "MORE" if projected > line else "LESS"
        outcome = "Over" if actual > line else ("Under" if actual < line else "Push")
        correct = (call == "MORE" and outcome == "Over") or (
            call == "LESS" and outcome == "Under"
        )

        records.append({
            "player": player,
            "stat": stat,
            "game_date": game_date_ts.strftime("%Y-%m-%d"),
            "severity": result.severity.value,
            "in_suppressed_list": stat in result.suppressed_stats,
            "line": line,
            "projected": round(projected, 2),
            "actual": actual,
            "abs_err": round(abs(actual - projected), 2),
            "call": call,
            "outcome": outcome,
            "correct": bool(correct),
        })
    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-severity-bucket directional accuracy + MAE."""
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("severity")
        .agg(
            n=("correct", "size"),
            directional_acc=("correct", "mean"),
            mae=("abs_err", "mean"),
        )
        .reset_index()
        .sort_values("severity")
    )
    out["directional_acc"] = out["directional_acc"].round(3)
    out["mae"] = out["mae"].round(2)
    return out


def summarise_suppressed_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Among games where THIS stat was in the suppressed list, what's accuracy?

    This is the tighter test — even when severity=SUPPRESS the detector only
    voids specific stats.  We compare:
      - games where the stat would have been suppressed
      - games where it would NOT have been suppressed
    """
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("in_suppressed_list")
        .agg(
            n=("correct", "size"),
            directional_acc=("correct", "mean"),
            mae=("abs_err", "mean"),
        )
        .reset_index()
        .rename(columns={"in_suppressed_list": "would_suppress"})
    )
    out["directional_acc"] = out["directional_acc"].round(3)
    out["mae"] = out["mae"].round(2)
    return out


def main() -> None:
    from pathlib import Path
    bundle_path = Path("bundles/race_fast.joblib")
    if not bundle_path.exists():
        raise SystemExit(f"Prebuilt bundle not found at {bundle_path}")

    print(f"Loading bundle: {bundle_path}")
    bundle = load_models(bundle_path)
    store = PlayerStore()
    detector = RoleShiftDetector()

    # Anchor cohort with deep history + the new default roster
    players = [
        "Jalen Brunson", "Donovan Mitchell", "James Harden",
        "LeBron James", "Kevin Durant", "Stephen Curry",
        "Damian Lillard", "Nikola Jokic", "Giannis Antetokounmpo",
    ]
    stats = ["assists", "points", "pra"]

    all_rows: list[pd.DataFrame] = []
    for p in players:
        for s in stats:
            try:
                print(f"  • {p:<26} {s:<8}", end=" ", flush=True)
                df = validate_player_stat(
                    p, s, store=store, bundle=bundle, detector=detector
                )
                print(f"{len(df):>4} games")
                if not df.empty:
                    all_rows.append(df)
            except Exception as exc:
                print(f"failed: {exc}")

    if not all_rows:
        print("\nNo evaluable games found. Run `hooplytics ingest-odds` first.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Total evaluable games: {len(full)}")
    print(f"Players × stats:       {full[['player','stat']].drop_duplicates().shape[0]}")
    print(f"Date range:            {full['game_date'].min()} → {full['game_date'].max()}")
    print(f"{'='*60}\n")

    print("Per-severity bucket (any flagged stat in the result):")
    print(summarise(full).to_string(index=False))
    print()

    print("Per-stat: did the detector help on the stat it actually suppressed?")
    print("(would_suppress=True means the detector marked THIS stat as NO_CALL)")
    print(summarise_suppressed_subset(full).to_string(index=False))
    print()

    # Per-stat breakdown
    print("Breakdown by stat:")
    for s in stats:
        sub = full[full["stat"] == s]
        if sub.empty:
            continue
        print(f"\n  ── {s} ({len(sub)} games) ──")
        print(summarise(sub).to_string(index=False))
        sup_view = summarise_suppressed_subset(sub)
        if not sup_view.empty and len(sup_view) > 1:
            ok = sup_view[~sup_view["would_suppress"]]
            sp = sup_view[sup_view["would_suppress"]]
            if not ok.empty and not sp.empty:
                lift = float(ok["directional_acc"].iloc[0]) - float(sp["directional_acc"].iloc[0])
                print(f"  detector lift on {s}: "
                      f"clean={float(ok['directional_acc'].iloc[0]):.3f} − "
                      f"suppressed={float(sp['directional_acc'].iloc[0]):.3f} = {lift:+.3f}")


if __name__ == "__main__":
    main()
