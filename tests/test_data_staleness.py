"""Cache-staleness guard for the per-player game-log Parquet cache.

Once a player's cache is warmed during the regular season, the original
"does it cover all requested seasons?" check returned the cached frame
forever — even after playoffs began and the cache stopped reflecting
recent games. ``_stale_seasons`` flags those caches so the fetcher
repopulates them.
"""
from __future__ import annotations

import pandas as pd

from hooplytics.data import _FRESHNESS_DAYS, _stale_seasons


def _today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


def test_in_season_cache_more_than_threshold_old_is_stale() -> None:
    cached = pd.DataFrame([
        {"season": "2025-26", "game_date": _today() - pd.Timedelta(days=_FRESHNESS_DAYS + 18)},
    ])
    assert _stale_seasons(cached, ["2025-26"]) == ["2025-26"]


def test_fresh_in_season_cache_is_not_stale() -> None:
    cached = pd.DataFrame([
        {"season": "2025-26", "game_date": _today() - pd.Timedelta(days=1)},
    ])
    assert _stale_seasons(cached, ["2025-26"]) == []


def test_completed_prior_season_is_never_stale() -> None:
    # 2022-23 ended in June 2023 — it should never trigger a refetch
    # regardless of how old the cache is.
    cached = pd.DataFrame([
        {"season": "2022-23", "game_date": pd.Timestamp("2023-06-12")},
    ])
    assert _stale_seasons(cached, ["2022-23"]) == []


def test_empty_cache_returns_empty_list() -> None:
    assert _stale_seasons(pd.DataFrame(), ["2025-26"]) == []


def test_cache_missing_required_columns_is_skipped() -> None:
    cached = pd.DataFrame([{"player": "X"}])
    assert _stale_seasons(cached, ["2025-26"]) == []


def test_handles_raw_uppercase_game_date_column() -> None:
    # The per-player Parquet cache stores the raw nba_api column name
    # (``GAME_DATE``) — the rename to ``game_date`` happens later in the
    # modeling-frame pipeline. The staleness check runs at the raw layer
    # so it must accept either spelling.
    cached = pd.DataFrame([
        {"season": "2025-26",
         "GAME_DATE": _today() - pd.Timedelta(days=_FRESHNESS_DAYS + 30)},
    ])
    assert _stale_seasons(cached, ["2025-26"]) == ["2025-26"]
