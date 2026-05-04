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
    NA_BOOKMAKER_TITLES,
    NA_BOOKMAKERS,
    ODDS_BASE,
    ODDS_CACHE_DIR,
    ODDS_HIST_CACHE_DIR,
    ODDS_HISTORICAL_BASE,
    ODDS_HISTORICAL_REGIONS,
    ODDS_MARKETS,
    ODDS_PLAYER_PROPS_CUTOFF,
    ODDS_REGIONS,
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


# Generational suffixes that show up inconsistently across data sources —
# e.g. NBA Stats reports "Jabari Smith Jr." while The Odds API lists him as
# "Jabari Smith". Stripping them before canonicalizing keeps the join from
# silently dropping a player when one source carries the suffix and the
# other doesn't.
_NAME_SUFFIX_TOKENS: frozenset[str] = frozenset({
    "jr", "sr", "ii", "iii", "iv", "v",
})


def _canon_name(s: str) -> str:
    """Canonicalize a player name for tolerant cross-source matching.

    Lowercases, strips a trailing generational suffix (Jr./Sr./II–V), then
    removes every non-letter so "Jabari Smith Jr." and "Jabari Smith" both
    canonicalize to ``"jabarismith"``.
    """
    text = str(s or "").lower().strip()
    if not text:
        return ""
    tokens = text.split()
    if tokens and re.sub(r"[^a-z]", "", tokens[-1]) in _NAME_SUFFIX_TOKENS:
        tokens = tokens[:-1]
    return re.sub(r"[^a-z]", "", " ".join(tokens))


def _odds_cache_path(date_str: str) -> Path:
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ODDS_CACHE_DIR / f"nba_player_props_{date_str}.json"


def _cache_age_minutes(path: Path) -> float:
    """Return how many minutes ago ``path`` was last written. Inf if missing."""
    try:
        return (time.time() - path.stat().st_mtime) / 60
    except OSError:
        return float("inf")


# Minimum minutes between live API fetches even when force_refresh=True.
# Prevents rapid roster/season-change events from each burning credits.
_MIN_REFRESH_MINUTES: int = 30


def _fetch_odds_payload(
    api_key: str,
    *,
    force_refresh: bool = False,
    min_refresh_minutes: int = _MIN_REFRESH_MINUTES,
) -> list[dict]:
    """Fetch (or load from disk cache) today's full per-event player-prop payload.

    ``force_refresh`` is honoured only when the cache file is at least
    ``min_refresh_minutes`` old (default 30 min). This prevents rapid
    roster/season-change events from each firing a fresh API request and
    burning quota unnecessarily.
    """
    cache_path = _odds_cache_path(date.today().isoformat())
    age = _cache_age_minutes(cache_path)

    # Use cache if: explicitly not refreshing, OR cache is too fresh to re-fetch.
    if cache_path.exists() and (not force_refresh or age < min_refresh_minutes):
        try:
            with cache_path.open("r") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass

    events_resp = requests.get(
        f"{ODDS_BASE}/events", params={"apiKey": api_key}, timeout=8
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

    # Per-event timeout is intentionally tight: a slate of 10+ games at the old
    # 15s timeout could pin the UI for 2+ minutes if any bookmaker endpoint
    # was slow. We also enforce an overall wall-clock budget so the function
    # always returns control to Streamlit promptly even on a degraded API.
    payload: list[dict] = []
    deadline = time.monotonic() + 25.0  # overall budget (seconds)
    for ev in events:
        if time.monotonic() >= deadline:
            break
        matchup = f"{ev.get('away_team', '?')} @ {ev.get('home_team', '?')}"
        try:
            resp = requests.get(
                f"{ODDS_BASE}/events/{ev['id']}/odds",
                params={
                    "apiKey": api_key,
                    "regions": ODDS_REGIONS,
                    "markets": ",".join(ODDS_MARKETS),
                    "bookmakers": ",".join(NA_BOOKMAKERS),
                    "oddsFormat": "american",
                },
                timeout=6,
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

    # Only persist when we actually got something — otherwise a partial budget
    # exhaustion would overwrite a good cache with an empty payload.
    if payload:
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
    min_refresh_minutes: int = _MIN_REFRESH_MINUTES,
) -> pd.DataFrame:
    """Return one row per (player, model_name) using consensus median across high-quality NA books.

    Only books in :data:`NA_BOOKMAKERS` (DraftKings, FanDuel, BetMGM, Caesars,
    BetRivers, ESPN BET, Hard Rock Bet, Fanatics, Bovada) contribute to the
    consensus, which removes noise from low-liquidity offshore books.

    Each row exposes both an aggregate count (``books``) and a per-book
    breakdown (``book_names`` as a comma-joined string and ``book_lines`` as a
    ``{book_title: line}`` dict) so the UI can show which sportsbooks contributed.

    Empty DataFrame on off-days, no-key, or no-matches. Player matching is
    case- and punctuation-insensitive (handles 'Shai Gilgeous-Alexander' vs
    'Shai Gilgeous Alexander').
    """
    empty_cols = ["player", "model", "line", "books", "book_names", "book_lines", "matchup"]
    if not api_key or not roster_players:
        return pd.DataFrame(columns=empty_cols)

    payload = _fetch_odds_payload(
        api_key,
        force_refresh=force_refresh,
        min_refresh_minutes=min_refresh_minutes,
    )
    if not payload:
        return pd.DataFrame(columns=empty_cols)

    canon_to_name: dict[str, str] = {_canon_name(p): p for p in roster_players}
    allowed_books = set(NA_BOOKMAKERS)
    rows: list[dict] = []

    for ev_entry in payload:
        matchup = ev_entry.get("matchup", "?")
        props = ev_entry.get("props", {})
        # bucket: (roster_name, model_name) -> {book_key: line}
        bucket: dict[tuple[str, str], dict[str, float]] = {}
        for bm in props.get("bookmakers", []):
            book_key = bm.get("key", "")
            if book_key not in allowed_books:
                continue
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
                    bucket.setdefault((roster_name, model_name), {})[book_key] = float(o["point"])

        for (roster_name, model_name), books_map in bucket.items():
            titles = {NA_BOOKMAKER_TITLES.get(k, k.title()): v for k, v in books_map.items()}
            rows.append({
                "player": roster_name,
                "model": model_name,
                "line": float(np.median(list(books_map.values()))),
                "books": len(books_map),
                "book_names": ", ".join(sorted(titles.keys())),
                "book_lines": titles,
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
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch historical player prop lines for a specific NBA game date.

    Queries the Odds API historical event odds endpoint at noon ET on the
    requested date to capture pregame lines. Results are cached to disk and
    merged across runs so a partial fetch (e.g. only ``points`` + ``rebounds``)
    can be filled in later by re-running with the missing markets.

    Parameters
    ----------
    api_key
        The Odds API key (must be on a paid plan for historical access).
    date_str
        ISO date string, e.g. ``"2024-01-15"``.
    force_refresh
        Re-fetch even if a cache file exists.
    markets
        Optional subset of model names (``"points"``, ``"rebounds"``,
        ``"assists"``, ``"threepm"``) to fetch. Defaults to all four.
        Each market costs 10 credits per event, so a 2-market subset
        cuts API cost in half versus the default.

    Returns
    -------
    DataFrame with columns ``[game_date, player, model, line, books, …]``.
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

    # Resolve which markets to request (model-name values).
    all_models = list(ODDS_MARKETS.values())
    requested_models = [m for m in (markets or all_models) if m in all_models]
    if not requested_models:
        return empty

    cache_path = _hist_cache_path(date_str)
    existing_df: pd.DataFrame | None = None
    if cache_path.exists() and not force_refresh:
        try:
            cached = pd.read_json(cache_path, orient="records")
        except Exception:  # noqa: BLE001
            cached = pd.DataFrame()
        # An empty cached file is a valid "no games this day" sentinel.
        if cached.empty:
            return empty
        existing_df = cached
        existing_models = set(cached["model"].unique()) if "model" in cached.columns else set()
        missing_models = [m for m in requested_models if m not in existing_models]
        if not missing_models:
            return cached
        # Only fetch the markets we don't yet have for this date.
        requested_models = missing_models

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
    # Map model names back to API market keys for the subset request.
    model_to_api = {v: k for k, v in ODDS_MARKETS.items()}
    api_markets = [model_to_api[m] for m in requested_models if m in model_to_api]
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
                    "regions": ODDS_HISTORICAL_REGIONS,
                    "markets": ",".join(api_markets),
                    "bookmakers": ",".join(NA_BOOKMAKERS),
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

        # bucket: (player, model) -> {"lines": [...], "over_prices": [...], "under_prices": [...]}
        bucket: dict[tuple[str, str], dict[str, list[float]]] = {}
        allowed_books = set(NA_BOOKMAKERS)
        for bm in event_data.get("bookmakers", []):
            if bm.get("key", "") not in allowed_books:
                continue
            for market in bm.get("markets", []):
                model_name = ODDS_MARKETS.get(market["key"])
                if model_name is None:
                    continue
                # Pair Over/Under outcomes per player so we capture both prices.
                outcomes = market.get("outcomes", [])
                pair: dict[str, dict[str, dict[str, float]]] = {}
                for o in outcomes:
                    side = o.get("name")
                    if side not in ("Over", "Under"):
                        continue
                    player_name = o.get("description", "")
                    if not player_name:
                        continue
                    pair.setdefault(player_name, {})[side] = {
                        "point": float(o["point"]),
                        "price": float(o.get("price", 0.0)) or float("nan"),
                    }
                for player_name, sides in pair.items():
                    over = sides.get("Over")
                    if over is None:
                        continue  # need at least the Over to anchor the line
                    slot = bucket.setdefault((player_name, model_name), {
                        "lines": [], "over_prices": [], "under_prices": [],
                    })
                    slot["lines"].append(over["point"])
                    slot["over_prices"].append(over["price"])
                    under = sides.get("Under")
                    if under is not None:
                        slot["under_prices"].append(under["price"])

        for (player_name, model_name), agg in bucket.items():
            lines = agg["lines"]
            over_prices = [p for p in agg["over_prices"] if p == p and p > 1.0]
            under_prices = [p for p in agg["under_prices"] if p == p and p > 1.0]
            row = {
                "game_date": date_str,
                "player": player_name,
                "model": model_name,
                "line": float(np.median(lines)),
                "books": len(lines),
                "line_std": float(np.std(lines)) if len(lines) > 1 else 0.0,
                "over_price": float(np.median(over_prices)) if over_prices else float("nan"),
                "under_price": float(np.median(under_prices)) if under_prices else float("nan"),
            }
            rows.append(row)

        time.sleep(0.25)  # be kind to the API

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    # Merge with any pre-existing cached rows (different markets fetched earlier).
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df
        combined = combined.drop_duplicates(subset=["game_date", "player", "model"], keep="last")
    else:
        combined = new_df
    result = combined if not combined.empty else empty
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
    markets: list[str] | None = None,
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
    all_models = list(ODDS_MARKETS.values())
    requested_models = [m for m in (markets or all_models) if m in all_models] or all_models
    full_set = set(requested_models) == set(all_models)
    for d in dates:
        if d < ODDS_PLAYER_PROPS_CUTOFF:
            continue
        cache_path = _hist_cache_path(d)
        if cache_path.exists() and not force_refresh:
            try:
                cached = pd.read_json(cache_path, orient="records")
            except Exception:  # noqa: BLE001
                cached = pd.DataFrame()
            if cached.empty:
                # Known no-games day — nothing to fetch.
                continue
            cached_models = set(cached["model"].unique()) if "model" in cached.columns else set()
            missing = [m for m in requested_models if m not in cached_models]
            if not missing:
                # Already have everything requested for this date.
                frames.append(cached)
                continue
            # Fall through to incremental fetch of the missing markets.
        if verbose:
            label = "all markets" if full_set else ",".join(requested_models)
            print(f"  Fetching historical odds for {d} [{label}] …")
        try:
            df = fetch_historical_odds_for_date(
                api_key, d, force_refresh=force_refresh, markets=requested_models,
            )
            if not df.empty:
                frames.append(df)
        except RuntimeError as exc:
            if verbose:
                print(f"  [warn] {exc} — stopping ingestion.")
            break

    if not frames:
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])
    return pd.concat(frames, ignore_index=True)


def _live_props_filename_to_date(name: str) -> str | None:
    """Pull the ``YYYY-MM-DD`` segment from a live-cache filename."""
    m = re.search(r"nba_player_props_(\d{4}-\d{2}-\d{2})\.json$", name)
    return m.group(1) if m else None


def _flatten_live_props_payload(
    payload: list[dict], date_str: str,
) -> pd.DataFrame:
    """Flatten a live-shape props cache (events with bookmakers) into the
    same row layout as historical odds (``game_date, player, model, line,
    books, line_std, over_price, under_price``).

    The live cache is the per-event payload written by ``_fetch_odds_payload``.
    Historical files are already flat. This converter is what lets the report's
    per-player history table populate from live snapshots before any backfill.
    """
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()

    allowed_books = set(NA_BOOKMAKERS)
    # bucket: (player, model) -> {"lines": [...], "over_prices": [...], "under_prices": [...]}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}

    for ev_entry in payload:
        if not isinstance(ev_entry, dict):
            continue
        props = ev_entry.get("props") or {}
        for bm in props.get("bookmakers", []) or []:
            if bm.get("key") not in allowed_books:
                continue
            for market in bm.get("markets", []) or []:
                model_name = ODDS_MARKETS.get(market.get("key"))
                if model_name is None:
                    continue
                # Pair Over/Under for each player so we can capture both prices.
                pair: dict[str, dict[str, dict[str, float]]] = {}
                for o in market.get("outcomes", []) or []:
                    side = o.get("name")
                    if side not in ("Over", "Under"):
                        continue
                    name = o.get("description", "")
                    if not name:
                        continue
                    try:
                        point = float(o["point"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    try:
                        price = float(o.get("price", 0.0))
                    except (TypeError, ValueError):
                        price = float("nan")
                    pair.setdefault(name, {})[side] = {"point": point, "price": price}
                for player_name, sides in pair.items():
                    over = sides.get("Over")
                    if over is None:
                        continue
                    slot = buckets.setdefault(
                        (player_name, model_name),
                        {"lines": [], "over_prices": [], "under_prices": []},
                    )
                    slot["lines"].append(over["point"])
                    if over["price"] == over["price"]:  # not NaN
                        slot["over_prices"].append(over["price"])
                    under = sides.get("Under")
                    if under is not None and under["price"] == under["price"]:
                        slot["under_prices"].append(under["price"])

    if not buckets:
        return pd.DataFrame()

    rows: list[dict] = []
    for (player_name, model_name), agg in buckets.items():
        lines = agg["lines"]
        if not lines:
            continue
        rows.append({
            "game_date": date_str,
            "player": player_name,
            "model": model_name,
            "line": float(np.median(lines)),
            "books": len(lines),
            "line_std": float(np.std(lines)) if len(lines) > 1 else 0.0,
            "over_price": float(np.median(agg["over_prices"])) if agg["over_prices"] else float("nan"),
            "under_price": float(np.median(agg["under_prices"])) if agg["under_prices"] else float("nan"),
        })
    return pd.DataFrame(rows)


def _load_cached_historical_odds_impl(hist_dir_str: str, _fingerprint: tuple) -> pd.DataFrame:
    hist_dir = Path(hist_dir_str)
    frames: list[pd.DataFrame] = []

    if hist_dir.exists():
        for path in sorted(hist_dir.glob("nba_player_props_*.json")):
            try:
                df = pd.read_json(path, orient="records")
                if not df.empty:
                    frames.append(df)
            except Exception:  # noqa: BLE001
                continue

    # Fallback: also flatten live snapshot files sitting in the parent
    # ``data/cache/odds/`` directory. These are written by every roster
    # refresh and become historical the moment the game is played.
    # Without this, fresh installs (or users who haven't run the historical
    # backfill CLI) would see an empty per-player history table.
    seen_dates: set[str] = set()
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        if "game_date" in merged.columns:
            seen_dates = set(
                pd.to_datetime(merged["game_date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .dropna()
                .tolist()
            )
    live_dir = hist_dir.parent
    if live_dir.exists() and live_dir != hist_dir:
        for path in sorted(live_dir.glob("nba_player_props_*.json")):
            date_str = _live_props_filename_to_date(path.name)
            if date_str is None or date_str in seen_dates:
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            # Live cache files are nested lists of events; historical files
            # are flat row records. Detect by inspecting the first element.
            if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "props" in payload[0]:
                df = _flatten_live_props_payload(payload, date_str)
            else:
                try:
                    df = pd.DataFrame(payload)
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
            if not df.empty:
                frames.append(df)
                seen_dates.add(date_str)

    if not frames:
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])

    result = pd.concat(frames, ignore_index=True)
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["game_date"])
    return result.reset_index(drop=True)


# Process-level memo so per-player feature builds reuse the parsed frame.
# Keyed on (cache_dir, file_count, latest_mtime) so new cache writes invalidate.
_HIST_ODDS_CACHE: dict[tuple, pd.DataFrame] = {}


def _bundled_odds_path() -> Path:
    """Return the path to the committed historical-odds parquet bundle.

    The bundle lives at ``data/odds/historical_props.parquet`` relative to the
    repository root (two levels up from this file).  It is generated locally
    with ``hooplytics-build-odds-bundle`` and committed so that Streamlit Cloud
    and other environments without a local odds cache can still run backtests
    and role-shift validation.
    """
    return Path(__file__).parent.parent / "data" / "odds" / "historical_props.parquet"


def _load_bundled_odds() -> pd.DataFrame:
    """Read the committed parquet bundle, or return empty on failure.

    Always normalizes ``game_date`` to a Timestamp so the result can be
    safely concatenated with the JSON-cache loader output.
    """
    bundle = _bundled_odds_path()
    if not bundle.exists():
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])
    try:
        df = pd.read_parquet(bundle)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["game_date", "player", "model", "line", "books"])
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["game_date"])
    return df.reset_index(drop=True)


def load_cached_historical_odds(
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all cached historical player prop lines into one DataFrame.

    Three sources are merged, with later sources taking precedence on
    overlapping (game_date, player, model) keys:

      1. ``data/odds/historical_props.parquet`` — committed bundle that ships
         with the repo so Streamlit Cloud and other no-cache environments can
         still run backtests / role-shift validation without a local ingest.
      2. ``ODDS_HIST_CACHE_DIR`` — JSON files written by ``hooplytics
         ingest-odds`` (the historical endpoint).
      3. Live snapshots in ``ODDS_CACHE_DIR`` — written every time the app
         pulls today's slate. These are the freshest source and override
         older bundle/historical entries for the same date.

    The merge is critical on Streamlit Cloud: the bundle alone isn't enough
    once the app writes a single live snapshot to disk, because a naive
    "fallback only when empty" check would then ignore the bundle entirely.

    The parsed frame is memoized at module level keyed on (file count,
    latest mtime) so repeat calls return instantly.
    """
    hist_dir = Path(cache_dir) if cache_dir else ODDS_HIST_CACHE_DIR
    live_dir = hist_dir.parent

    files = list(hist_dir.glob("nba_player_props_*.json")) if hist_dir.exists() else []
    if live_dir.exists() and live_dir != hist_dir:
        files += list(live_dir.glob("nba_player_props_*.json"))

    bundle_df = _load_bundled_odds()

    if not files:
        # No local JSON cache — return the bundle as-is.
        return bundle_df

    fingerprint = (
        str(hist_dir),
        len(files),
        max((f.stat().st_mtime for f in files), default=0.0),
        bool(not bundle_df.empty),
    )
    cached = _HIST_ODDS_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    json_df = _load_cached_historical_odds_impl(str(hist_dir), fingerprint)

    # Merge: bundle first (oldest), JSON cache second (overrides on conflict).
    if bundle_df.empty:
        result = json_df
    elif json_df.empty:
        result = bundle_df
    else:
        merged = pd.concat([bundle_df, json_df], ignore_index=True)
        # Newer entries (later in concat order) win on duplicate keys.
        result = merged.drop_duplicates(
            subset=["game_date", "player", "model"], keep="last"
        ).reset_index(drop=True)

    # Keep only the latest fingerprint to avoid unbounded growth as the
    # cache directory churns.
    _HIST_ODDS_CACHE.clear()
    _HIST_ODDS_CACHE[fingerprint] = result
    return result


# ── Scores ───────────────────────────────────────────────────────────────────

def fetch_recent_scores(
    api_key: str,
    *,
    days_from: int = 1,
    timeout: float = 8.0,
) -> pd.DataFrame:
    """Fetch live, upcoming, and recently completed NBA games with scores.

    Wraps the Odds API ``/scores`` endpoint. Returns one row per game with
    home/away teams, scores when available, and completion status. Useful for
    automated actuals-vs-prediction tracking without an additional NBA stats
    API call.

    Parameters
    ----------
    api_key
        The Odds API key.
    days_from
        Include completed games from up to this many days ago (1-3). Costs
        2 quota credits when set; 1 credit for live/upcoming only.
    timeout
        HTTP timeout in seconds.

    Returns
    -------
    DataFrame with columns ``[id, commence_time, home_team, away_team,
    home_score, away_score, completed, last_update]``. Empty on no key,
    quota exhaustion, or network error.
    """
    cols = [
        "id", "commence_time", "home_team", "away_team",
        "home_score", "away_score", "completed", "last_update",
    ]
    if not api_key:
        return pd.DataFrame(columns=cols)

    params: dict[str, str | int] = {"apiKey": api_key, "dateFormat": "iso"}
    if days_from:
        params["daysFrom"] = max(1, min(3, int(days_from)))

    try:
        resp = requests.get(f"{ODDS_BASE}/scores", params=params, timeout=timeout)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)

    if resp.status_code in (401, 429) or not resp.ok:
        return pd.DataFrame(columns=cols)

    payload = resp.json()
    if not isinstance(payload, list):
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for ev in payload:
        scores = {s.get("name"): s.get("score") for s in (ev.get("scores") or [])}
        home = ev.get("home_team")
        away = ev.get("away_team")
        rows.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": home,
            "away_team": away,
            "home_score": _safe_int(scores.get(home)),
            "away_score": _safe_int(scores.get(away)),
            "completed": bool(ev.get("completed", False)),
            "last_update": ev.get("last_update"),
        })
    return pd.DataFrame(rows, columns=cols)


def _safe_int(value: object) -> int | float:
    """Coerce a score string/number to int; NaN if missing or unparseable."""
    if value is None or value == "":
        return float("nan")
    try:
        return int(value)
    except (TypeError, ValueError):
        return float("nan")


# ── Game lines (spread / total / moneyline) ─────────────────────────────────

def _game_lines_cache_path(date_str: str) -> Path:
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ODDS_CACHE_DIR / f"nba_game_lines_{date_str}.json"


def fetch_game_lines(
    api_key: str,
    *,
    force_refresh: bool = False,
    min_refresh_minutes: int = _MIN_REFRESH_MINUTES,
    timeout: float = 8.0,
) -> list[dict]:
    """Fetch consensus spread / total / moneyline for tonight's NBA slate.

    Returns a list of ``{home_team, away_team, home_spread, total, home_ml,
    away_ml, source, books}`` rows — one per game on the slate. Lines are the
    median across the same NA bookmaker whitelist used for player props
    (:data:`NA_BOOKMAKERS`). Empty list on off-days, no-key, or quota
    exhaustion.

    The h2h (moneyline), spreads, and totals markets are part of the standard
    region package on The Odds API and do **not** carry the player-props price
    multiplier — fetching them is one quota credit per market per region.
    """
    cache_path = _game_lines_cache_path(date.today().isoformat())
    age = _cache_age_minutes(cache_path)
    if cache_path.exists() and (not force_refresh or age < min_refresh_minutes):
        try:
            with cache_path.open("r") as f:
                cached = json.load(f)
            if isinstance(cached, list):
                return cached
        except Exception:  # noqa: BLE001
            pass

    if not api_key:
        return []

    try:
        resp = requests.get(
            f"{ODDS_BASE}/odds",
            params={
                "apiKey": api_key,
                "regions": ODDS_REGIONS,
                "markets": "h2h,spreads,totals",
                "bookmakers": ",".join(NA_BOOKMAKERS),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        return []

    if resp.status_code in (401, 429) or not resp.ok:
        return []

    try:
        events = resp.json()
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(events, list):
        return []

    allowed_books = set(NA_BOOKMAKERS)
    rows: list[dict] = []
    for ev in events:
        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not (home and away):
            continue

        spreads_home: list[float] = []
        totals: list[float] = []
        home_mls: list[float] = []
        away_mls: list[float] = []
        contributing_books: set[str] = set()

        for bm in ev.get("bookmakers") or []:
            if bm.get("key") not in allowed_books:
                continue
            for market in bm.get("markets") or []:
                key = market.get("key")
                outcomes = market.get("outcomes") or []
                if key == "spreads":
                    for o in outcomes:
                        if o.get("name") == home:
                            try:
                                spreads_home.append(float(o.get("point")))
                                contributing_books.add(bm.get("key", ""))
                            except (TypeError, ValueError):
                                continue
                elif key == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            try:
                                totals.append(float(o.get("point")))
                                contributing_books.add(bm.get("key", ""))
                            except (TypeError, ValueError):
                                continue
                elif key == "h2h":
                    for o in outcomes:
                        try:
                            price = float(o.get("price"))
                        except (TypeError, ValueError):
                            continue
                        if o.get("name") == home:
                            home_mls.append(price)
                            contributing_books.add(bm.get("key", ""))
                        elif o.get("name") == away:
                            away_mls.append(price)
                            contributing_books.add(bm.get("key", ""))

        rows.append({
            "home_team": home,
            "away_team": away,
            "commence_time": str(ev.get("commence_time") or ""),
            "home_spread": float(np.median(spreads_home)) if spreads_home else None,
            "total": float(np.median(totals)) if totals else None,
            "home_ml": float(np.median(home_mls)) if home_mls else None,
            "away_ml": float(np.median(away_mls)) if away_mls else None,
            "books": len(contributing_books),
            "source": "consensus",
        })

    if rows:
        try:
            with cache_path.open("w") as f:
                json.dump(rows, f)
        except Exception:  # noqa: BLE001
            pass
    return rows


def load_cached_game_lines(date_str: str | None = None) -> list[dict]:
    """Load cached game lines for ``date_str`` (today by default).

    Returns an empty list when no cache exists. Useful when you want the
    market lines but don't want to risk burning a fresh API credit — the
    Roster Report pulls these passively for grounding only.
    """
    iso = date_str or date.today().isoformat()
    path = _game_lines_cache_path(iso)
    if not path.exists():
        return []
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return []
    return data if isinstance(data, list) else []


# ── Market discovery ─────────────────────────────────────────────────────────

def fetch_event_available_markets(
    api_key: str,
    event_id: str,
    *,
    timeout: float = 8.0,
) -> dict[str, list[str]]:
    """Discover which player-prop markets each book has open for an event.

    Wraps the Odds API ``/events/{eventId}/markets`` endpoint (1 quota credit
    regardless of market count). Use this before calling the per-event odds
    endpoint to skip events/books that don't yet have the markets you want,
    avoiding wasted quota on the more expensive odds call.

    Returns a mapping of ``{bookmaker_key: [market_key, …]}``. Empty dict on
    no key, error, or unsupported event.
    """
    if not api_key or not event_id:
        return {}

    try:
        resp = requests.get(
            f"{ODDS_BASE}/events/{event_id}/markets",
            params={"apiKey": api_key, "regions": ODDS_REGIONS},
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        return {}

    if resp.status_code in (401, 429) or not resp.ok:
        return {}

    payload = resp.json()
    if not isinstance(payload, dict):
        return {}

    result: dict[str, list[str]] = {}
    for bm in payload.get("bookmakers", []) or []:
        key = bm.get("key")
        if not key:
            continue
        result[key] = [m.get("key", "") for m in (bm.get("markets") or []) if m.get("key")]
    return result
