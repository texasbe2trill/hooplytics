"""The Odds API integration: fetch consensus player prop lines."""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .constants import ODDS_BASE, ODDS_CACHE_DIR, ODDS_MARKETS


def load_api_key(env_path: Path | str = ".env") -> str:
    """Resolve ODDS_API_KEY from .env file → shell env. Returns '' if not set."""
    env_file = Path(env_path)
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
    return os.getenv("ODDS_API_KEY", "").strip()


def _canon_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _odds_cache_path(date_str: str) -> Path:
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ODDS_CACHE_DIR / f"nba_player_props_{date_str}.json"


def _fetch_odds_payload(api_key: str, *, force_refresh: bool = False) -> list[dict]:
    """Fetch (or load from disk cache) today's full per-event player-prop payload."""
    cache_path = _odds_cache_path(date.today().isoformat())

    if cache_path.exists() and not force_refresh:
        try:
            with cache_path.open("r") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass

    events_resp = requests.get(
        f"{ODDS_BASE}/events", params={"apiKey": api_key}, timeout=15
    )
    if events_resp.status_code == 401:
        raise RuntimeError("Odds API 401 Unauthorized — invalid or revoked key.")
    if events_resp.status_code == 429:
        raise RuntimeError("Odds API 429 — monthly quota exhausted.")
    if not events_resp.ok:
        raise RuntimeError(f"Odds API /events returned HTTP {events_resp.status_code}")
    events = events_resp.json()
    if not isinstance(events, list):
        raise RuntimeError("Odds API error — check key, quota, and network.")

    if not events:
        with cache_path.open("w") as f:
            json.dump([], f)
        return []

    payload: list[dict] = []
    for ev in events:
        matchup = f"{ev.get('away_team', '?')} @ {ev.get('home_team', '?')}"
        try:
            resp = requests.get(
                f"{ODDS_BASE}/events/{ev['id']}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": ",".join(ODDS_MARKETS),
                    "oddsFormat": "american",
                },
                timeout=15,
            )
            if resp.status_code == 401:
                # Player props are a paid tier — bail out gracefully.
                return []
            if resp.status_code == 429:
                return payload  # quota exhausted mid-loop
            resp.raise_for_status()
            payload.append({
                "matchup": matchup,
                "away_team": ev.get("away_team"),
                "home_team": ev.get("home_team"),
                "props": resp.json(),
            })
        except Exception:  # noqa: BLE001 — skip flaky events
            continue

    try:
        with cache_path.open("w") as f:
            json.dump(payload, f)
    except Exception:  # noqa: BLE001
        pass
    return payload


def fetch_live_player_lines(
    api_key: str,
    roster_players: list[str],
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return one row per (player, model_name, line) using consensus median across books.

    Empty DataFrame on off-days, no-key, or no-matches. Player matching is
    case- and punctuation-insensitive (handles 'Shai Gilgeous-Alexander' vs
    'Shai Gilgeous Alexander').
    """
    if not api_key or not roster_players:
        return pd.DataFrame(columns=["player", "model", "line", "books", "matchup"])

    payload = _fetch_odds_payload(api_key, force_refresh=force_refresh)
    if not payload:
        return pd.DataFrame(columns=["player", "model", "line", "books", "matchup"])

    canon_to_name: dict[str, str] = {_canon_name(p): p for p in roster_players}
    rows: list[dict] = []

    for ev_entry in payload:
        matchup = ev_entry.get("matchup", "?")
        props = ev_entry.get("props", {})
        bucket: dict[tuple[str, str], list[float]] = {}
        for bm in props.get("bookmakers", []):
            for market in bm.get("markets", []):
                model_name = ODDS_MARKETS.get(market["key"])
                if model_name is None:
                    continue
                for o in market.get("outcomes", []):
                    if o.get("name") != "Over":
                        continue
                    api_name = o.get("description", "")
                    if not api_name:
                        continue
                    roster_name = canon_to_name.get(_canon_name(api_name))
                    if roster_name is None:
                        continue
                    bucket.setdefault((roster_name, model_name), []).append(float(o["point"]))

        for (roster_name, model_name), points in bucket.items():
            rows.append({
                "player": roster_name,
                "model": model_name,
                "line": float(np.median(points)),
                "books": len(points),
                "matchup": matchup,
            })

    return pd.DataFrame(rows)
