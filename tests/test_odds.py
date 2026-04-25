"""Smoke test: parse the bundled odds JSON fixture without network access."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hooplytics.odds import _canon_name, fetch_live_player_lines

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "cache" / "odds" / "nba_player_props_2026-04-24.json"


def test_canon_name_handles_punctuation() -> None:
    assert _canon_name("Shai Gilgeous-Alexander") == _canon_name("Shai Gilgeous Alexander")


def test_fetch_live_player_lines_empty_inputs() -> None:
    assert fetch_live_player_lines("", ["LeBron James"]).empty
    assert fetch_live_player_lines("fake-key", []).empty


def test_odds_fixture_loads_as_json() -> None:
    if not FIXTURE.exists():
        return  # fixture is gitignored; skip silently
    payload = json.loads(FIXTURE.read_text())
    assert isinstance(payload, list)
    if payload:
        assert "props" in payload[0] or "matchup" in payload[0]
