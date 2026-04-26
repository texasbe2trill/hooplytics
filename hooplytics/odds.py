"""The Odds API integration: fetch consensus player prop lines."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .constants import (
    ODDS_BASE,
    ODDS_CACHE_DIR,
    ODDS_HIST_CACHE_DIR,
    ODDS_HISTORICAL_BASE,
    ODDS_MARKETS,
    ODDS_PLAYER_PROPS_CUTOFF,
)


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


# ── Historical odds ──────────────────────────────────────────────────────────

def _hist_cache_path(date_str: str) -> Path:
    ODDS_HIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ODDS_HIST_CACHE_DIR / f"nba_player_props_{date_str}.json"


def fetch_historical_odds_for_date(
    api_key: str,
    date_str: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch historical player prop lines for a specific NBA game date.

    Queries the Odds API historical event odds endpoint at noon ET on the
    requested date to capture pregame lines. Results are cached to disk.

    Parameters
    ----------
    api_key
        The Odds API key (must be on a paid plan for historical access).
    date_str
        ISO date string, e.g. ``"2024-01-15"``.
    force_refresh
        Re-fetch even if a cache file exists.

    Returns
    -------
    DataFrame with columns ``[game_date, player, model, line, books]``.
    Empty on error, no API key, or date before the player-props cutoff.

    Notes
    -----
    * Player props are only available from ``ODDS_PLAYER_PROPS_CUTOFF`` onward.
    * Historical endpoints cost 10× the normal quota per market per region.
      All 4 markets in one region = 40 credits per event.
    """
    empty = pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])
    if not api_key:
        return empty
    if date_str < ODDS_PLAYER_PROPS_CUTOFF:
        return empty

    cache_path = _hist_cache_path(date_str)
    if cache_path.exists() and not force_refresh:
        try:
            cached = pd.read_json(cache_path, orient="records")
            # An empty list is a valid cached result (no games that day).
            return cached if not cached.empty else empty
        except Exception:  # noqa: BLE001
            pass

    # Query at 18:00 UTC (1 pm ET) — before any NBA tip-off.
    query_ts = f"{date_str}T18:00:00Z"

    # ── Step 1: historical events list for this date ──────────────────────
    try:
        ev_resp = requests.get(
            f"{ODDS_HISTORICAL_BASE}/events",
            params={
                "apiKey": api_key,
                "date": query_ts,
                "commenceTimeFrom": f"{date_str}T00:00:00Z",
                "commenceTimeTo": f"{date_str}T23:59:59Z",
            },
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        return empty

    if ev_resp.status_code == 401:
        raise RuntimeError(
            "Odds API 401 — historical endpoints require a paid plan."
        )
    if ev_resp.status_code == 422:
        # Date out of range or invalid; cache empty result.
        empty.to_json(cache_path, orient="records")
        return empty
    if ev_resp.status_code == 429:
        raise RuntimeError("Odds API 429 — quota exhausted.")
    if not ev_resp.ok:
        return empty

    ev_payload = ev_resp.json()
    events = ev_payload.get("data", []) if isinstance(ev_payload, dict) else []
    if not events:
        empty.to_json(cache_path, orient="records")
        return empty

    # ── Step 2: player prop odds for each event ───────────────────────────
    rows: list[dict] = []
    for ev in events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        try:
            odds_resp = requests.get(
                f"{ODDS_HISTORICAL_BASE}/events/{ev_id}/odds",
                params={
                    "apiKey": api_key,
                    "date": query_ts,
                    "regions": "us",
                    "markets": ",".join(ODDS_MARKETS),
                    "oddsFormat": "decimal",
                },
                timeout=20,
            )
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
            continue

        if odds_resp.status_code in (401, 429):
            break  # stop early; don't burn quota
        if not odds_resp.ok:
            time.sleep(0.3)
            continue

        payload = odds_resp.json()
        event_data = payload.get("data", {}) if isinstance(payload, dict) else {}

        bucket: dict[tuple[str, str], list[float]] = {}
        for bm in event_data.get("bookmakers", []):
            for market in bm.get("markets", []):
                model_name = ODDS_MARKETS.get(market["key"])
                if model_name is None:
                    continue
                for o in market.get("outcomes", []):
                    if o.get("name") != "Over":
                        continue
                    player_name = o.get("description", "")
                    if not player_name:
                        continue
                    bucket.setdefault((player_name, model_name), []).append(
                        float(o["point"])
                    )

        for (player_name, model_name), pts in bucket.items():
            rows.append({
                "game_date": date_str,
                "player": player_name,
                "model": model_name,
                "line": float(np.median(pts)),
                "books": len(pts),
            })

        time.sleep(0.25)  # be kind to the API

    result = pd.DataFrame(rows) if rows else empty
    try:
        result.to_json(cache_path, orient="records")
    except Exception:  # noqa: BLE001
        pass
    return result


def ingest_historical_odds(
    api_key: str,
    dates: list[str],
    *,
    force_refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch and cache historical odds for a list of ISO date strings.

    Skips dates already cached (unless ``force_refresh=True``) and dates
    before ``ODDS_PLAYER_PROPS_CUTOFF``.  Returns the combined DataFrame.

    Parameters
    ----------
    api_key
        The Odds API key (paid plan required).
    dates
        List of ISO date strings like ``["2024-01-15", "2024-01-16"]``.
    force_refresh
        Re-fetch cached dates.
    verbose
        Print progress.
    """
    frames: list[pd.DataFrame] = []
    for d in dates:
        if d < ODDS_PLAYER_PROPS_CUTOFF:
            continue
        cache_path = _hist_cache_path(d)
        if cache_path.exists() and not force_refresh:
            try:
                cached = pd.read_json(cache_path, orient="records")
                if not cached.empty:
                    frames.append(cached)
                    continue
            except Exception:  # noqa: BLE001
                pass
        if verbose:
            print(f"  Fetching historical odds for {d} …")
        try:
            df = fetch_historical_odds_for_date(api_key, d, force_refresh=force_refresh)
            if not df.empty:
                frames.append(df)
        except RuntimeError as exc:
            if verbose:
                print(f"  [warn] {exc} — stopping ingestion.")
            break

    if not frames:
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])
    return pd.concat(frames, ignore_index=True)


def load_cached_historical_odds(
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all cached historical player prop lines into one DataFrame.

    Scans ``ODDS_HIST_CACHE_DIR`` (or ``cache_dir``) for
    ``nba_player_props_*.json`` files and concatenates them.
    Returns an empty DataFrame if no cache files are found.
    """
    hist_dir = Path(cache_dir) if cache_dir else ODDS_HIST_CACHE_DIR
    if not hist_dir.exists():
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])

    frames: list[pd.DataFrame] = []
    for path in sorted(hist_dir.glob("nba_player_props_*.json")):
        try:
            df = pd.read_json(path, orient="records")
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: BLE001
            continue

    if not frames:
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])

    result = pd.concat(frames, ignore_index=True)
    result["game_date"] = pd.to_datetime(result["game_date"]).dt.normalize()
    return result
