"""Rebuild data/odds/historical_props.parquet from the local JSON odds cache.

Run this after ingesting new historical odds with ``hooplytics ingest-odds``
to refresh the committed bundle that Streamlit Cloud uses for backtests.

Usage:
    .venv/bin/python3 scripts/build_odds_bundle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402


def main() -> None:
    from hooplytics.constants import ODDS_HIST_CACHE_DIR
    from hooplytics.odds import _load_cached_historical_odds_impl

    hist_dir: Path = ODDS_HIST_CACHE_DIR

    if not hist_dir.exists() or not list(hist_dir.glob("nba_player_props_*.json")):
        print("No odds cache files found. Run `hooplytics ingest-odds` first.")
        sys.exit(1)

    # Use the canonical loader (handles both history and live snapshot formats).
    df = _load_cached_historical_odds_impl(str(hist_dir), ("build_bundle",))

    if df.empty:
        print("Cache loaded but produced an empty DataFrame — check JSON files.")
        sys.exit(1)

    out = Path(__file__).parent.parent / "data" / "odds" / "historical_props.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Written: {out}")
    print(f"  rows   : {len(df):,}")
    print(f"  players: {df['player'].nunique():,}")
    print(f"  stats  : {sorted(df['model'].unique())}")
    print(f"  dates  : {df['game_date'].min().date()} → {df['game_date'].max().date()}")
    print(f"  size   : {out.stat().st_size / 1024:.0f} KB")
    print()
    print("Commit the updated bundle:")
    print("  git add data/odds/historical_props.parquet && git commit -m 'refresh odds bundle'")


if __name__ == "__main__":
    main()
