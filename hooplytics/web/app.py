"""Streamlit UI for Hooplytics — premium dashboard for the projection engine."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Cloud may keep an old site-packages wheel around; force imports to
# resolve from the mounted repo source first, and evict any pre-cached
# `hooplytics` submodules that may have been imported before this shim.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
elif sys.path.index(_REPO_ROOT) != 0:
    sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

# Drop any already-loaded hooplytics modules so the next imports below resolve
# from the live repo source instead of any stale editable/site-packages copy.
for _mod in [m for m in list(sys.modules) if m == "hooplytics" or m.startswith("hooplytics.")]:
    # Don't evict ourselves (hooplytics.web.app) — we're currently executing.
    if _mod == "hooplytics.web.app":
        continue
    sys.modules.pop(_mod, None)

from hooplytics.constants import (
    DEFAULT_ROSTER,
    MODEL_SPECS,
    MODEL_TO_COL,
)

# Veteran anchors augment the training corpus so models see deeper samples.
# Defined here rather than constants to avoid stale-install import issues.
_TRAINING_ANCHOR_PLAYERS: list[str] = [
    "LeBron James",
    "Kevin Durant",
    "Stephen Curry",
    "James Harden",
    "Damian Lillard",
    "Kyrie Irving",
    "DeMar DeRozan",
    "Jimmy Butler",
    "Paul George",
    "Chris Paul",
    "Nikola Jokic",
    "Giannis Antetokounmpo",
]
from hooplytics.data import NBADataUnavailable, PlayerStore, nba_seasons
from hooplytics.models import ModelBundle, ensure_models, load_models
from hooplytics.odds import fetch_live_player_lines, load_cached_historical_odds
from hooplytics.openai_agent import (
    OpenAIConnection,
    auto_select_model,
    build_grounding_payload,
    chat_complete,
    connect as openai_connect,
    evidence_chips,
    filter_chat_models,
    generate_report_sections,
    parse_chart_blocks,
)
from hooplytics.predict import (
    fantasy_decisions,
    predict_scenario,
    project_next_game,
)
from hooplytics.report import build_pdf_report
from hooplytics.web import charts
from hooplytics.web.styles import (
    chip,
    disclaimer,
    divider,
    empty_state,
    inject_css,
    insight_card,
    kpi_grid,
    meta_row,
    mini_kpi,
    page_hero,
    pill,
    player_color_map,
    register_template,
    section,
)


# Page setup
st.set_page_config(
    page_title="Hooplytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
register_template()
inject_css()


# Caching ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24, max_entries=1)
def _all_active_players() -> list[str]:
    """Return sorted list of all active NBA player names (cached for the session)."""
    try:
        from nba_api.stats.static import players as nba_players  # type: ignore
        return sorted(p["full_name"] for p in nba_players.get_players() if p.get("is_active"))
    except Exception:
        return []


def _deployment_odds_api_key() -> str:
    """Resolve a deployment-configured Odds API key without exposing it in the UI.

    Order: ODDS_API_KEY env var, then Streamlit secrets["ODDS_API_KEY"].
    """
    key = os.getenv("ODDS_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("ODDS_API_KEY", "")).strip()
    except Exception:
        return ""


def _deployment_openai_api_key() -> str:
    """Resolve a deployment-configured OpenAI API key without exposing it.

    Order: OPENAI_API_KEY env var, then Streamlit secrets["OPENAI_API_KEY"].
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def _store() -> PlayerStore:
    return PlayerStore()


@st.cache_data(show_spinner="Fetching season game logs…", ttl=60 * 60 * 6, max_entries=3)
def _player_data(roster_key: str) -> pd.DataFrame:
    roster = json.loads(roster_key)
    return _store().load_player_data(roster)


def _season_start_year(season: str) -> int | None:
    try:
        return int(str(season).split("-")[0])
    except Exception:
        return None


def _parse_seasons_input(value: str) -> list[str]:
    """Parse season input into canonical nba season strings.

    Accepted token forms (comma-separated):
    - `2024-25` (season string)
    - `2024` (single start year -> `2024-25`)
    - `2021-2024` (year range -> `2021-22`..`2024-25`)
    """
    tokens = [t.strip() for t in str(value).split(",") if t.strip()]
    out: list[str] = []
    for t in tokens:
        m_season = re.fullmatch(r"(\d{4})-(\d{2})", t)
        if m_season:
            y = int(m_season.group(1))
            out.append(f"{y}-{str(y + 1)[-2:]}")
            continue

        m_year = re.fullmatch(r"\d{4}", t)
        if m_year:
            y = int(t)
            out.append(f"{y}-{str(y + 1)[-2:]}")
            continue

        m_range = re.fullmatch(r"(\d{4})\s*[-:]\s*(\d{4})", t)
        if m_range:
            y0 = int(m_range.group(1))
            y1 = int(m_range.group(2))
            if y1 < y0:
                y0, y1 = y1, y0
            out.extend(nba_seasons(y0, y1 + 1))
            continue

    # preserve order, remove duplicates
    return list(dict.fromkeys(out))


def _training_roster(
    display_roster: dict[str, list[str]],
    include_display_players: bool = False,
) -> dict[str, list[str]]:
    # Fast default: train on veteran anchors only. Users can opt in to include
    # displayed players for more personalization at higher compute cost.
    all_seasons = [s for seasons in display_roster.values() for s in seasons]
    train_seasons = sorted({s for s in all_seasons if isinstance(s, str) and s.strip()})
    if not train_seasons:
        # Fallback to current default window if roster is malformed.
        train_seasons = _default_seasons()
    train_players = list(_TRAINING_ANCHOR_PLAYERS)
    if include_display_players:
        train_players = list(dict.fromkeys([*train_players, *display_roster.keys()]))
    return {name: list(train_seasons) for name in train_players}


def _training_roster_key(roster_key: str, include_display_players: bool) -> str:
    """Stable cache key for the *resolved* training roster.

    When ``include_display_players`` is False (the default) the training roster
    is just the veteran anchors plus whatever seasons the user has selected, so
    adding/removing a sidebar player must NOT invalidate the trained bundle.
    """
    display_roster = json.loads(roster_key)
    resolved = _training_roster(
        display_roster, include_display_players=include_display_players
    )
    # sort for determinism so equivalent rosters hit the same cache slot
    return json.dumps(
        {name: sorted(seasons) for name, seasons in sorted(resolved.items())},
        separators=(",", ":"),
    )


@st.cache_data(show_spinner="Building training corpus…", ttl=60 * 60 * 6, max_entries=6)
def _training_data(training_key: str) -> pd.DataFrame:
    training_roster = json.loads(training_key)
    return _store().load_player_data(training_roster)


@st.cache_resource(show_spinner=False, max_entries=2)
def _prebuilt_bundle(prebuilt_path: str) -> ModelBundle:
    """Load a prebuilt RACE bundle from disk. Cached for the whole session."""
    return load_models(prebuilt_path)


@st.cache_resource(show_spinner="Training models…", max_entries=2)
def _trained_bundle(training_key: str, fast_mode: bool) -> ModelBundle:
    return ensure_models(_training_data(training_key), fast_mode=fast_mode)


def _bundle(
    roster_key: str,
    use_prebuilt: bool,
    prebuilt_path: str,
    include_display_players: bool,
    fast_mode: bool,
) -> ModelBundle:
    """Resolve the active bundle: prefer prebuilt (cached on path only)."""
    if use_prebuilt and prebuilt_path:
        return _prebuilt_bundle(prebuilt_path)
    training_key = _training_roster_key(roster_key, include_display_players)
    return _trained_bundle(training_key, fast_mode)


def _default_prebuilt_bundle_path() -> str:
    # Prefer the canonical repo-shipped RACE fast bundle when present — this
    # is the high-accuracy production blend and should win over stale env
    # paths or older cache artifacts. Env override is still honored if it
    # points to a different existing file.
    repo_bundle = Path(__file__).resolve().parents[2] / "bundles" / "race_fast.joblib"
    if repo_bundle.exists():
        return str(repo_bundle)
    env_path = os.getenv("HOOPLYTICS_PRETRAINED_BUNDLE", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    model_dir = Path(".hooplytics_cache/models")
    if not model_dir.exists():
        return ""
    candidates = sorted(model_dir.glob("models_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


@st.cache_data(show_spinner=False, ttl=60 * 5)
def _live_lines(api_key: str, players_key: str, _bust: int = 0,
                _force_refresh: bool = False) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["player", "model", "line", "books", "matchup"])
    return fetch_live_player_lines(
        api_key, json.loads(players_key), force_refresh=_force_refresh
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24, max_entries=1)
def _seed_modeling_frame() -> pd.DataFrame:
    """Load the shipped per-default-player modeling parquet (if present)."""
    seed = Path(__file__).resolve().parents[2] / "data" / "seed_cache" / "_modeling_default.parquet"
    if seed.exists():
        try:
            return pd.read_parquet(seed)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner="Building player features…", ttl=60 * 60 * 6, max_entries=64)
def _player_modeling_rows(player: str, seasons_key: str) -> pd.DataFrame:
    """Compute modeling-ready rows for one player. Cached per (player, seasons).

    Per-player caching means adding or removing a sidebar player only triggers
    pipeline work for that single player — every other roster member's
    modeling rows are reused from cache (or from the shipped seed parquet).
    """
    seasons = list(json.loads(seasons_key))
    raw = _store().load_player_data({player: seasons})
    if raw.empty:
        return raw
    return _store().modeling_frame(raw)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6, max_entries=12)
def _modeling_frame(roster_key: str) -> pd.DataFrame:
    """Compose the modeling frame by concatenating per-player cached chunks.

    Default-roster players are sliced directly from the shipped seed parquet
    (no pipeline cost); non-default players are pulled from the per-player
    cache, which only invokes the full data + feature pipeline once per
    (player, seasons) tuple per session.
    """
    try:
        roster: dict[str, list[str]] = json.loads(roster_key)
    except Exception:
        roster = {}
    if not roster:
        return pd.DataFrame()

    seed = _seed_modeling_frame()
    seed_players = set(seed["player"].unique()) if not seed.empty else set()

    chunks: list[pd.DataFrame] = []
    for player, seasons in roster.items():
        season_list = [s for s in (seasons or []) if isinstance(s, str)]
        if player in seed_players and season_list:
            sub = seed[seed["player"] == player]
            if "season" in sub.columns:
                seed_seasons = set(sub["season"].dropna().astype(str).unique())
                requested = set(season_list)
                # Only short-circuit to the seed slice when it covers EVERY
                # requested season. Otherwise fall through so newly-selected
                # seasons (e.g. the current in-progress season) are pulled via
                # the live per-player pipeline.
                if requested.issubset(seed_seasons):
                    sub = sub[sub["season"].isin(season_list)]
                    if not sub.empty:
                        chunks.append(sub)
                        continue
        seasons_key = json.dumps(sorted(season_list))
        try:
            rows = _player_modeling_rows(player, seasons_key)
        except NBADataUnavailable:
            # An individual player's data may be unreachable (cold cache + a
            # stats.nba.com hiccup). Skip them here so the rest of the roster
            # still renders; the explicit add-to-roster flow surfaces the
            # typed error directly to the user.
            continue
        if not rows.empty:
            chunks.append(rows)

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True, sort=False)
    return _store().modeling_frame(df)


@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=32)
def _player_games(roster_key: str, player: str) -> pd.DataFrame:
    df = _modeling_frame(roster_key)
    if df.empty or "player" not in df.columns:
        return pd.DataFrame()
    sub = df[df["player"] == player]
    if "game_date" in sub.columns:
        sub = sub.sort_values("game_date")
    return sub.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=32)
def _project_next_game_cached(
    roster_key: str,
    player: str,
    last_n: int,
    use_prebuilt: bool,
    prebuilt_path: str,
    include_display_players: bool,
    fast_mode: bool,
) -> pd.DataFrame:
    """Cache projection by (roster, player, window, bundle-config).

    Bundle config is passed in so the cached result is invalidated only when
    the underlying bundle would change \u2014 not on every Streamlit rerun.
    """
    bundle = _bundle(roster_key, use_prebuilt, prebuilt_path,
                     include_display_players, fast_mode)
    modeling_df = _modeling_frame(roster_key)
    return project_next_game(
        player,
        bundle=bundle,
        store=_store(),
        last_n=last_n,
        modeling_df=modeling_df,
    )


# State ───────────────────────────────────────────────────────────────────────
def _default_seasons() -> list[str]:
    """Return the active + previous NBA season(s).

    NBA seasons start in October. Before October we're in the back half of the
    prior season, so the "current" season ends at year-1. Including a season
    that hasn't started yet causes nba_api fetches that bypass the seed cache
    and stall the Streamlit app.
    """
    today = datetime.now().date()
    if today.month >= 10:
        # In-season: current season is today.year–(year+1); include the prior season too.
        start = today.year - 1
        end_exclusive = today.year + 1
    else:
        # Off-season / first half of calendar year: current season ended this spring.
        start = today.year - 2
        end_exclusive = today.year
    return nba_seasons(start, end_exclusive)


def _season_dropdown_options() -> list[str]:
    today = datetime.now().date()
    # Don't offer a season that hasn't started yet (NBA seasons start in October).
    end_exclusive = today.year + 1 if today.month >= 10 else today.year
    return nba_seasons(2015, end_exclusive)


def _init_state() -> None:
    if "roster" not in st.session_state:
        seasons = _default_seasons()
        st.session_state.roster = {p: list(seasons) for p in DEFAULT_ROSTER}
    if "live_bust" not in st.session_state:
        st.session_state.live_bust = 0
    if "session_odds_api_key" not in st.session_state:
        st.session_state.session_odds_api_key = ""
    if "session_odds_api_key_input" not in st.session_state:
        st.session_state.session_odds_api_key_input = st.session_state.session_odds_api_key
    if "sidebar_seasons_input" not in st.session_state:
        st.session_state.sidebar_seasons_input = "2024-25,2025-26"
    if "sidebar_seasons_select" not in st.session_state:
        cur = sorted({s for seasons in st.session_state.roster.values() for s in seasons})
        st.session_state.sidebar_seasons_select = cur or _default_seasons()
    if "prebuilt_bundle_path" not in st.session_state:
        st.session_state.prebuilt_bundle_path = _default_prebuilt_bundle_path()
    if "use_prebuilt_bundle" not in st.session_state:
        st.session_state.use_prebuilt_bundle = bool(st.session_state.prebuilt_bundle_path)
    if "train_on_display_roster" not in st.session_state:
        st.session_state.train_on_display_roster = False
    if "fast_training_mode" not in st.session_state:
        st.session_state.fast_training_mode = True
    if "odds_api_status" not in st.session_state:
        st.session_state.odds_api_status = ""  # "", "ok", "error"
    if "odds_api_error" not in st.session_state:
        st.session_state.odds_api_error = ""
    # OpenAI / chatbot session state
    if "session_openai_api_key" not in st.session_state:
        st.session_state.session_openai_api_key = ""
    if "session_openai_api_key_input" not in st.session_state:
        st.session_state.session_openai_api_key_input = (
            st.session_state.session_openai_api_key
        )
    if "openai_models" not in st.session_state:
        st.session_state.openai_models = []
    if "openai_selected_model" not in st.session_state:
        st.session_state.openai_selected_model = ""
    if "openai_connect_status" not in st.session_state:
        st.session_state.openai_connect_status = ""  # "", "ok", "error"
    if "openai_connect_error" not in st.session_state:
        st.session_state.openai_connect_error = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[{role, content, evidence?}]
    if "chat_strict_grounded" not in st.session_state:
        st.session_state.chat_strict_grounded = False
    if "chat_pending" not in st.session_state:
        st.session_state.chat_pending = ""


def _sync_session_odds_api_key() -> None:
    st.session_state.session_odds_api_key = (
        st.session_state.session_odds_api_key_input.strip()
    )


def _sync_session_openai_api_key() -> None:
    new_key = st.session_state.session_openai_api_key_input.strip()
    if new_key != st.session_state.session_openai_api_key:
        # Key changed → drop cached models so the user can re-list with the new key.
        st.session_state.openai_models = []
        st.session_state.openai_selected_model = ""
        st.session_state.openai_connect_status = ""
        st.session_state.openai_connect_error = ""
    st.session_state.session_openai_api_key = new_key


@st.cache_resource(show_spinner=False)
def _openai_connect_cached(api_key: str) -> OpenAIConnection:
    """Cache one OpenAIConnection per key for the session (key never logged)."""
    return openai_connect(api_key)


def _resolve_openai_connection(api_key: str) -> OpenAIConnection | None:
    """Return a cached OpenAIConnection, refreshing session model state on success."""
    if not api_key:
        return None
    try:
        conn = _openai_connect_cached(api_key)
    except Exception as exc:
        st.session_state.openai_connect_status = "error"
        st.session_state.openai_connect_error = str(exc)
        return None
    st.session_state.openai_connect_status = "ok"
    st.session_state.openai_connect_error = ""
    # Always sync with filtered chat-capable models from the latest connection
    # so stale session values cannot keep non-chat options in the selector.
    st.session_state.openai_models = list(conn.models)
    if not st.session_state.openai_selected_model:
        st.session_state.openai_selected_model = (
            conn.default_model or (conn.models[0] if conn.models else "")
        )
    elif conn.models and st.session_state.openai_selected_model not in conn.models:
        st.session_state.openai_selected_model = (
            conn.default_model or conn.models[0]
        )
    return conn


def _apply_sidebar_seasons_to_roster() -> None:
    """Apply sidebar season input to all rostered players.

    This runs on text-input change so editing the year/season range immediately
    refreshes the training/data cache keys used by the app.
    """
    roster = st.session_state.get("roster", {})
    if not roster:
        return
    seasons_all = _parse_seasons_input(st.session_state.get("sidebar_seasons_input", ""))
    if not seasons_all:
        return
    changed = False
    for p in list(roster):
        if list(roster[p]) != list(seasons_all):
            roster[p] = list(seasons_all)
            changed = True
    if changed:
        st.session_state.live_bust += 1


def _apply_sidebar_season_select_to_roster() -> None:
    roster = st.session_state.get("roster", {})
    if not roster:
        return
    seasons_all = [s for s in st.session_state.get("sidebar_seasons_select", []) if s]
    if not seasons_all:
        return
    changed = False
    for p in list(roster):
        if list(roster[p]) != list(seasons_all):
            roster[p] = list(seasons_all)
            changed = True
    if changed:
        # Bumping live_bust invalidates the streamlit cache for _live_lines and
        # _build_edge_board_cached, so the new roster shape is reflected on the
        # next render against the already-cached daily odds payload. We do NOT
        # force a fresh Odds API fetch here — that path can stack multiple 15s
        # HTTP timeouts (one /events + one per game) and was the source of the
        # "add to roster spins forever" hang. The explicit sidebar Refresh
        # button is the only place that should force a re-fetch.
        st.session_state.live_bust += 1
        st.rerun()


def _roster_key() -> str:
    roster = st.session_state.roster
    return json.dumps({p: list(s) for p, s in sorted(roster.items())}, sort_keys=True)


def _active_training_seasons() -> list[str]:
    roster = st.session_state.get("roster", {})
    if not roster:
        return []
    train = _training_roster(
        {p: list(s) for p, s in roster.items()},
        include_display_players=bool(st.session_state.get("train_on_display_roster", False)),
    )
    if not train:
        return []
    return list(next(iter(train.values()), []))


def _bundle_for_ui() -> ModelBundle:
    return _bundle(
        _roster_key(),
        bool(st.session_state.get("use_prebuilt_bundle", False)),
        str(st.session_state.get("prebuilt_bundle_path", "")),
        bool(st.session_state.get("train_on_display_roster", False)),
        bool(st.session_state.get("fast_training_mode", True)),
    )


# Edge board builder ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 5, max_entries=4)
def _build_edge_board_cached(roster_key: str, api_key: str, _bust: int,
                             _force_refresh: bool = False) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    roster = json.loads(roster_key)
    players = list(roster)
    live = _live_lines(api_key, json.dumps(players), _bust, _force_refresh)
    if live.empty:
        return pd.DataFrame()

    bundle = _bundle_for_ui()
    store = _store()
    modeling_df = _modeling_frame(roster_key)

    out: list[dict] = []
    for player in players:
        plr_lines = live[live["player"] == player]
        if plr_lines.empty:
            continue
        live_map = dict(zip(plr_lines["model"], plr_lines["line"]))
        try:
            decisions = fantasy_decisions(
                player, bundle=bundle, store=store,
                live_lines=live_map, modeling_df=modeling_df,
            )
        except Exception:
            continue
        for _, r in decisions.iterrows():
            if r["source"] != "live":
                continue
            books = int(plr_lines.loc[plr_lines["model"] == r["model"], "books"].iloc[0])
            matchup = plr_lines.loc[plr_lines["model"] == r["model"], "matchup"].iloc[0]
            book_names_val = ""
            book_lines_val: dict = {}
            if "book_names" in plr_lines.columns:
                book_names_val = str(plr_lines.loc[plr_lines["model"] == r["model"], "book_names"].iloc[0])
            if "book_lines" in plr_lines.columns:
                bl_raw = plr_lines.loc[plr_lines["model"] == r["model"], "book_lines"].iloc[0]
                if isinstance(bl_raw, dict):
                    book_lines_val = bl_raw
            out.append({
                "player": player,
                "model": r["model"],
                "posted line": r["line"],
                "model prediction": r["prediction"],
                "5-game avg": r["5-game avg"],
                "adj. threshold": r["adj. threshold"],
                "edge": r["prediction"] - r["adj. threshold"],
                "call": r["decision"].split()[0],
                "matchup": matchup,
                "books": books,
                "book_names": book_names_val,
                "book_lines": book_lines_val,
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["abs_edge"] = df["edge"].abs()
    df["side"] = np.where(df["edge"] > 0, "MORE", "LESS")
    return df.sort_values("abs_edge", ascending=False).reset_index(drop=True)


def _build_edge_board(roster: dict, api_key: str) -> pd.DataFrame:
    # Consume + reset the one-shot force-refresh flag so the upstream daily
    # odds JSON is re-fetched on this run, then return to disk-cache reads.
    force = bool(st.session_state.pop("force_refresh_odds", False))
    if not api_key:
        st.session_state.odds_api_status = ""
        st.session_state.odds_api_error = ""
        return pd.DataFrame()
    try:
        out = _build_edge_board_cached(
            json.dumps({p: list(s) for p, s in sorted(roster.items())}, sort_keys=True),
            api_key,
            int(st.session_state.get("live_bust", 0)),
            force,
        )
    except Exception as exc:
        # Degrade gracefully when deployment/session keys are invalid/revoked.
        st.session_state.odds_api_status = "error"
        st.session_state.odds_api_error = str(exc)
        return pd.DataFrame()
    st.session_state.odds_api_status = "ok"
    st.session_state.odds_api_error = ""
    # Record how old the odds cache is so the UI can surface it.
    from hooplytics.odds import _cache_age_minutes, _odds_cache_path  # noqa: PLC0415
    from datetime import date as _date  # noqa: PLC0415
    st.session_state["odds_cache_age_minutes"] = _cache_age_minutes(
        _odds_cache_path(_date.today().isoformat())
    )
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60, max_entries=2)
def _diagnostics_panels(roster_key: str, color_map: dict) -> list[dict]:
    """Rebuild predicted-vs-actual / residual panels once per (roster, bundle)."""
    from sklearn.model_selection import train_test_split

    bundle = _bundle_for_ui()
    modeling_df = _modeling_frame(roster_key)
    metrics = bundle.metrics
    panels: list[dict] = []
    if modeling_df.empty or metrics is None or metrics.empty:
        return panels
    _, test = train_test_split(modeling_df, test_size=0.2, random_state=123)
    for name, est in bundle.estimators.items():
        spec = bundle.specs[name]
        X = test.reindex(columns=spec["features"])
        y = test[spec["target"]].to_numpy()
        yhat = est.predict(X)
        r2 = float(metrics.loc[metrics["model"] == name, "R²"].iloc[0])
        points = [
            {
                "player": p, "date": str(d.date()) if pd.notna(d) else "",
                "matchup": m or "", "actual": float(a), "pred": float(yh),
                "color": color_map.get(p, "#888"),
            }
            for p, d, m, a, yh in zip(
                test["player"],
                test.get("game_date", pd.Series([pd.NaT] * len(test))),
                test.get("MATCHUP", pd.Series([""] * len(test))),
                y, yhat,
            )
        ]
        panels.append({"metric": name, "r2": r2, "points": points})
    return panels


# Sidebar ─────────────────────────────────────────────────────────────────────
def _render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown(
            '<div class="hl-brand" style="margin: 0.2rem 0 1.4rem 0;">'
            '<span class="hl-brand-mark"></span>HOOPLYTICS'
            '</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        divider()

        # Roster manager
        st.markdown('<p class="hl-section">Roster</p>', unsafe_allow_html=True)
        roster = st.session_state.roster
        for player in list(roster):
            cols = st.columns([6, 1], gap="small", vertical_alignment="center")
            cols[0].markdown(
                f'<div class="hl-roster-row">{player}</div>',
                unsafe_allow_html=True,
            )
            if cols[1].button("×", key=f"rm_{player}", help=f"Remove {player}"):
                roster.pop(player)
                # See _apply_sidebar_season_select_to_roster — live_bust alone
                # invalidates the cached edge board; forcing an Odds API
                # re-fetch here is what caused the add/remove spinner to hang.
                st.session_state.live_bust += 1
                st.rerun()

        seasons_selected = st.multiselect(
            "Seasons",
            options=_season_dropdown_options(),
            key="sidebar_seasons_select",
            on_change=_apply_sidebar_season_select_to_roster,
            help="Active seasons for the whole roster. Changing this applies to all rostered players immediately.",
        )
        active_seasons = _active_training_seasons()
        if active_seasons:
            st.caption(f"Active: {', '.join(active_seasons)}")
        if st.button(
            "Update all players\u2019 seasons",
            width="stretch",
            key="apply_seasons_btn",
            help="Forces every rostered player to use the seasons selected above.",
        ):
            if not seasons_selected:
                st.error("Select at least one season first.")
            else:
                for p in list(roster):
                    roster[p] = list(seasons_selected)
                st.session_state.live_bust += 1
                st.rerun()

        st.markdown(
            '<p class="hl-section-sub" style="margin-top:1rem;">Add a player</p>',
            unsafe_allow_html=True,
        )
        all_names = _all_active_players()
        new_player = st.selectbox(
            "Search for a player",
            options=[""] + all_names,
            index=0,
            label_visibility="collapsed",
            placeholder="Search by name\u2026",
        )
        if st.button("Add to roster", width="stretch", key="add_player_btn") and new_player and new_player.strip():
            seasons = [s for s in seasons_selected if s]
            if not seasons:
                st.error("Select at least one season first.")
                st.stop()
            try:
                resolved = PlayerStore.resolve_player_name(new_player) or new_player
                # Eagerly fetch this player's data so we can surface a clear
                # error if the NBA stats API is unreachable (common on hosted
                # environments where data-center IPs are blocked). Without
                # this guard the user sees a long spinner followed by a
                # silent "0 game rows" card with no explanation.
                with st.spinner(f"Loading game logs for {resolved}\u2026"):
                    seasons_key = json.dumps(sorted(seasons))
                    try:
                        rows = _player_modeling_rows(resolved, seasons_key)
                    except NBADataUnavailable as nba_exc:
                        kind = nba_exc.kind
                        if kind == "blocked":
                            st.error(
                                f"NBA stats API blocked the request for **{resolved}** "
                                "(HTTP 403/429). This is expected on Streamlit Cloud and "
                                "other datacenter hosts \u2014 stats.nba.com filters those IPs. "
                                "Try again from a local install."
                            )
                        elif kind == "timeout":
                            st.error(
                                f"Timed out reaching the NBA stats API while loading "
                                f"**{resolved}**. The endpoint can be slow on cold calls; "
                                "try again in a few seconds."
                            )
                        elif kind == "empty":
                            st.error(
                                f"NBA stats API returned no game logs for **{resolved}** "
                                "in the selected seasons. Try a different season range."
                            )
                        else:
                            st.error(
                                f"Couldn\u2019t load game logs for **{resolved}**: {nba_exc}"
                            )
                        st.stop()
                if rows.empty:
                    st.error(
                        f"Couldn\u2019t load game logs for **{resolved}**. "
                        "The NBA stats API returned data but no complete modeling rows "
                        "after feature construction. Try a wider season range."
                    )
                    st.stop()
                roster[resolved] = seasons
                # live_bust invalidates the streamlit cache so the new player is
                # filtered into the edge board on the next render, reusing the
                # already-cached daily Odds API payload. Forcing a refresh here
                # was stacking 15s HTTP timeouts and causing the spinner to
                # hang on add-to-roster. The sidebar Refresh button still
                # fires a real re-fetch when the user wants fresh lines.
                st.session_state.live_bust += 1
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.checkbox(
            "Use prebuilt model bundle",
            key="use_prebuilt_bundle",
            help="When enabled, load the prebuilt bundle path below instead of retraining in-app.",
        )
        st.checkbox(
            "Fast training mode (recommended)",
            key="fast_training_mode",
            help="Trains a single ridge baseline per target. Much faster on Streamlit Cloud.",
        )
        st.checkbox(
            "Include roster players in training (slower)",
            key="train_on_display_roster",
            help="Off by default for speed. Enable to personalize training to displayed players.",
        )
        st.caption("Prebuilt bundle: " + (st.session_state.prebuilt_bundle_path or "not found"))

        divider()

        # API key
        st.markdown('<p class="hl-section">Live odds</p>', unsafe_allow_html=True)
        deployment_key = _deployment_odds_api_key()
        if deployment_key:
            st.caption(
                "A deployment-configured Odds API key is active. Paste your own "
                "key below to override it for this session \u2014 the deployment "
                "key is never displayed."
            )
        else:
            st.caption(
                "Bring your own key from The Odds API for this session. The Streamlit app does "
                "not load a deployment key into the UI, and your pasted key is not "
                "written to the repository."
            )
        if (
            st.session_state.session_odds_api_key_input
            != st.session_state.session_odds_api_key
        ):
            st.session_state.session_odds_api_key_input = (
                st.session_state.session_odds_api_key
            )
        st.text_input(
            "The Odds API key",
            type="password",
            key="session_odds_api_key_input",
            on_change=_sync_session_odds_api_key,
            label_visibility="collapsed",
            placeholder="Paste your key from The Odds API",
        )
        api_key = st.session_state.session_odds_api_key.strip() or deployment_key
        cols = st.columns(3)
        if cols[0].button("Refresh", width="stretch", key="odds_refresh_btn"):
            st.session_state.live_bust += 1
            st.session_state.force_refresh_odds = True
            st.session_state.odds_api_status = ""
            st.session_state.odds_api_error = ""
            st.rerun()
        if cols[1].button("Clear key", width="stretch", key="odds_clear_key_btn"):
            st.session_state.session_odds_api_key = ""
            st.session_state.session_odds_api_key_input = ""
            st.session_state.odds_api_status = ""
            st.session_state.odds_api_error = ""
            st.rerun()
        odds_status_pill = (
            pill("LIVE", "live")
            if st.session_state.get("odds_api_status") == "ok"
            else pill("ERROR", "warn")
            if st.session_state.get("odds_api_status") == "error"
            else pill("LIVE", "live")
            if api_key
            else pill("OFFLINE", "warn")
        )
        cols[2].markdown(odds_status_pill, unsafe_allow_html=True)
        age_min = st.session_state.get("odds_cache_age_minutes")
        if age_min is not None and age_min < float("inf"):
            if age_min < 1:
                age_label = "updated just now"
            elif age_min < 60:
                age_label = f"updated {int(age_min)}m ago"
            else:
                age_label = f"updated {int(age_min // 60)}h {int(age_min % 60)}m ago"
            cols[2].caption(age_label)
        if st.session_state.get("odds_api_status") == "error":
            st.caption(
                "Odds API unavailable (invalid/revoked key or quota/network issue). "
                "Continuing in offline mode."
            )

        divider()

        # OpenAI key + model picker (powers the Hooplytics Scout tab)
        st.markdown('<p class="hl-section">Hooplytics Scout</p>', unsafe_allow_html=True)
        deployment_openai_key = _deployment_openai_api_key()
        if deployment_openai_key:
            st.caption(
                "A deployment-configured OpenAI key is active. Paste your own key "
                "below to override it for this session \u2014 the deployment key "
                "is never displayed."
            )
        else:
            st.caption(
                "Bring your own OpenAI key to enable the Hooplytics Scout tab. The key "
                "stays in session memory and is never written to disk or logs."
            )
        st.text_input(
            "OpenAI API key",
            type="password",
            key="session_openai_api_key_input",
            on_change=_sync_session_openai_api_key,
            label_visibility="collapsed",
            placeholder="Paste your OpenAI API key (sk-...)",
        )
        openai_api_key = (
            st.session_state.session_openai_api_key.strip() or deployment_openai_key
        )

        # Auto-connect exactly once per key. Status is set to "ok"/"error" by
        # _resolve_openai_connection so this branch cannot re-enter and cause a
        # render loop on subsequent reruns.
        if openai_api_key and st.session_state.openai_connect_status == "":
            with st.spinner("Connecting to OpenAI\u2026"):
                _resolve_openai_connection(openai_api_key)
            st.rerun()

        oa_cols = st.columns(3)
        if oa_cols[0].button(
            "Reconnect",
            width="stretch",
            disabled=not openai_api_key,
            key="openai_connect_btn",
        ):
            # Force a fresh discovery on explicit reconnect.
            try:
                _openai_connect_cached.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            st.session_state.openai_models = []
            st.session_state.openai_selected_model = ""
            st.session_state.openai_connect_status = ""
            st.session_state.openai_connect_error = ""
            _resolve_openai_connection(openai_api_key)
            st.rerun()
        if oa_cols[1].button("Clear key", width="stretch", key="openai_clear_key_btn"):
            st.session_state.session_openai_api_key = ""
            st.session_state.openai_models = []
            st.session_state.openai_selected_model = ""
            st.session_state.openai_connect_status = ""
            st.session_state.openai_connect_error = ""
            # Safely reset widget-bound state.
            for _wk in ("session_openai_api_key_input", "openai_model_select"):
                st.session_state.pop(_wk, None)
            try:
                _openai_connect_cached.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            st.rerun()
        status_pill = (
            pill("CONNECTED", "live")
            if st.session_state.openai_connect_status == "ok"
            else pill("ERROR", "warn")
            if st.session_state.openai_connect_status == "error"
            else pill("OFFLINE", "warn")
        )
        oa_cols[2].markdown(status_pill, unsafe_allow_html=True)

        if st.session_state.openai_connect_status == "error":
            st.caption(
                f"OpenAI: {st.session_state.openai_connect_error or 'connection failed'}"
            )

        models = filter_chat_models(st.session_state.openai_models or [])
        if models != list(st.session_state.openai_models or []):
            st.session_state.openai_models = list(models)
        if models:
            current = st.session_state.openai_selected_model or (
                auto_select_model(models) or models[0]
            )
            if current not in models:
                current = models[0]
            picked = st.selectbox(
                "Model",
                options=models,
                index=models.index(current),
                key="openai_model_select",
                help="Auto-selects the best available GPT-style model on connect.",
            )
            st.session_state.openai_selected_model = picked

        st.checkbox(
            "Strict grounded mode (no general reasoning)",
            key="chat_strict_grounded",
            help="When on, the chatbot will only answer from local Hooplytics data.",
        )

    return page, api_key, openai_api_key


# ── Pages ────────────────────────────────────────────────────────────────────


def page_home(roster: dict, api_key: str) -> None:
    if not roster:
        empty_state("No players in roster", "Add at least one player using the sidebar to get started.")
        return
    bundle = _bundle_for_ui()
    modeling_df = _modeling_frame(_roster_key())

    # Home hero
    last_refresh = datetime.now().strftime("%b %d · %H:%M")
    status = (
        f'<span class="hl-status-dot"></span>LIVE LINES · UPDATED {last_refresh}'
        if api_key else
        f'<span class="hl-status-dot" style="background:#f5b041;box-shadow:0 0 0 3px rgba(245,176,65,0.18)"></span>'
        f'OFFLINE · ADD THE ODDS API KEY FOR LIVE LINES'
    )
    st.markdown(
        '<div class="hl-hero-wrap">'
        '<h1 class="hl-tagline">Today\'s slate, analyzed.</h1>'
        '<p class="hl-tagline-sub">A precision view of player form, projection gaps, '
        'model confidence, and matchup context.</p>'
        f'<div class="hl-home-status">{status}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPI strip
    median_r2 = float(bundle.metrics["R²"].median()) if bundle.metrics is not None else float("nan")
    cols = st.columns(4)
    cols[0].metric("Players tracked", len(roster))
    cols[1].metric("Game rows", f"{len(modeling_df):,}")
    cols[2].metric("Models trained", len(bundle.estimators))
    cols[3].metric("Median R²", f"{median_r2:.2f}" if not np.isnan(median_r2) else "—")

    # Main grid: edges (wider) + model quality
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="hl-card">', unsafe_allow_html=True)
        st.markdown('<p class="hl-card-title">Strongest model signals</p>',
                    unsafe_allow_html=True)
        edge_df = _build_edge_board(roster, api_key)
        if edge_df.empty:
            st.markdown(
                '<p class="hl-subtle">No live player lines available right now. '
                'Add an Odds API key in the sidebar to surface today\'s largest '
                'projection gaps here.</p>',
                unsafe_allow_html=True,
            )
            if st.session_state.get("odds_api_status") == "error":
                st.caption(
                    "Live lines request failed. Check/replace your Odds API key in the sidebar."
                )
        else:
            top = edge_df.head(6)
            rows_html = []
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                kind = "more" if row["side"] == "MORE" else "less"
                direction = "Above line" if row["side"] == "MORE" else "Below line"
                rank_cls = "hl-edge-rank top" if i <= 3 else "hl-edge-rank"
                rows_html.append(
                    f'<div class="hl-edge {kind}">'
                    f'<div class="{rank_cls}">{i}</div>'
                    f'<div class="hl-edge-main">'
                    f'<div class="hl-edge-player">{row["player"]}</div>'
                    f'<div class="hl-edge-meta"><strong>{row["model"]}</strong>'
                    f' · line {row["posted line"]:g} · projection {row["model prediction"]:.1f}'
                    f' · {row["matchup"]}</div>'
                    f'</div>'
                    f'<div class="hl-edge-num">'
                    f'<div class="hl-edge-edge hl-mono">{row["edge"]:+.2f}</div>'
                    f'<div class="hl-edge-side">{direction}</div>'
                    f'</div>'
                    f'</div>'
                )
            st.markdown("".join(rows_html), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hl-card">', unsafe_allow_html=True)
        st.markdown('<p class="hl-card-title">Model quality</p>',
                    unsafe_allow_html=True)
        if bundle.metrics is not None:
            m = bundle.metrics.sort_values("R²", ascending=False)
            rows_html = [
                _r2_row(row["model"], float(row["R²"]))
                for _, row in m.iterrows()
            ]
            st.markdown("".join(rows_html), unsafe_allow_html=True)
        else:
            st.markdown('<p class="hl-subtle">Metrics not available.</p>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Training meta
        trained = bundle.trained_at[:10] if bundle.trained_at else "—"
        st.markdown(
            f'<div class="hl-card" style="margin-top:0.8rem">'
            f'<p class="hl-card-title">Training</p>'
            f'<div class="hl-edge-meta" style="line-height:1.7">'
            f'<strong>Trained</strong> · {trained}<br>'
            f'<strong>Train rows</strong> · {bundle.n_train:,}<br>'
            f'<strong>Test rows</strong> · {bundle.n_test:,}<br>'
            f'<strong>Players</strong> · {len(bundle.train_players)}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Roster grid
    section("Roster", "Your tracked players and the seasons in their training corpus.")
    cmap = player_color_map(list(roster))
    cards = []
    missing_players: list[str] = []
    for player, seasons in roster.items():
        games = (modeling_df["player"] == player).sum() if "player" in modeling_df.columns else 0
        color = cmap[player]
        if games == 0:
            missing_players.append(player)
        cards.append(
            f'<div class="hl-roster-card">'
            f'<div class="hl-roster-name">'
            f'<span class="hl-roster-dot" style="background:{color}"></span>{player}'
            f'</div>'
            f'<div class="hl-roster-meta">{", ".join(seasons)}</div>'
            f'<div class="hl-roster-meta hl-mono">{games} game rows</div>'
            f'</div>'
        )
    st.markdown(f'<div class="hl-roster-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True)
    if missing_players:
        st.warning(
            "No game rows loaded for: **" + ", ".join(missing_players) + "**. "
            "The NBA stats API may be unreachable from this host "
            "(common on cloud deployments). Try again later or run locally."
        )


def _r2_row(name: str, r2: float) -> str:
    pct = max(0.0, min(1.0, float(r2))) * 100
    cls = "hl-r2-fill weak" if r2 < 0.3 else "hl-r2-fill"
    return (
        f'<div class="hl-r2-row">'
        f'<div class="hl-r2-name">{name}</div>'
        f'<div class="hl-r2-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="hl-r2-val">{r2:.2f}</div>'
        f'</div>'
    )


def _player_summary_kpis(player: str, games: pd.DataFrame) -> None:
    """Render a per-player summary KPI grid above the projection workspace."""
    if games.empty:
        empty_state(
            "No games available",
            f"No game logs found for {player} in the current training corpus.",
        )
        return

    date_col = "game_date" if "game_date" in games.columns else None
    latest = ""
    if date_col:
        ts = pd.to_datetime(games[date_col], errors="coerce").dropna()
        if not ts.empty:
            latest = ts.max().strftime("%b %d, %Y")

    def _avg(col: str, n: int | None = None) -> float | None:
        if col not in games.columns:
            return None
        s = pd.to_numeric(games[col], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.tail(n).mean()) if n else float(s.mean())

    def _fmt(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "—"

    def _trend(col: str) -> tuple[str | None, str | None]:
        full = _avg(col)
        last5 = _avg(col, 5)
        if full is None or last5 is None:
            return None, None
        delta = last5 - full
        if abs(delta) < 0.05:
            return f"L5 flat vs season", None
        kind = "up" if delta > 0 else "down"
        sign = "+" if delta > 0 else ""
        return f"L5 {sign}{delta:.1f} vs season", kind

    pts_sub, pts_dir = _trend("pts")
    reb_sub, reb_dir = _trend("reb")
    ast_sub, ast_dir = _trend("ast")
    pra_sub, pra_dir = _trend("pra")
    fan_sub, fan_dir = _trend("fantasy_score")

    tiles = [
        mini_kpi("Games", f"{len(games):,}", sub=f"latest {latest}" if latest else None),
        mini_kpi("PTS", _fmt(_avg("pts")), sub=pts_sub, trend=pts_dir),
        mini_kpi("REB", _fmt(_avg("reb")), sub=reb_sub, trend=reb_dir),
        mini_kpi("AST", _fmt(_avg("ast")), sub=ast_sub, trend=ast_dir),
        mini_kpi("PRA", _fmt(_avg("pra")), sub=pra_sub, trend=pra_dir),
        mini_kpi("FAN", _fmt(_avg("fantasy_score")), sub=fan_sub, trend=fan_dir),
        mini_kpi("MIN", _fmt(_avg("min"))),
    ]
    st.markdown(
        f'<div class="hl-card">'
        f'<p class="hl-card-title">{player} — recent form snapshot</p>'
        f'<div class="hl-kpi-grid">{"".join(tiles)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _projection_kpis(proj: pd.DataFrame) -> None:
    """Render a KPI tile per model from the projection table."""
    if proj is None or proj.empty:
        return
    cols_needed = {"model", "prediction"}
    if not cols_needed.issubset(proj.columns):
        return
    tiles = []
    for _, row in proj.iterrows():
        model = str(row["model"])
        pred = float(row["prediction"])
        baseline = row.get("5-game avg", None)
        sub = None
        trend = None
        if baseline is not None and pd.notna(baseline):
            delta = pred - float(baseline)
            sign = "+" if delta >= 0 else ""
            trend = "up" if delta > 0.05 else ("down" if delta < -0.05 else None)
            sub = f"{sign}{delta:.1f} vs L5 avg"
        tiles.append(mini_kpi(model, f"{pred:.1f}", sub=sub, trend=trend))
    st.markdown(
        f'<div class="hl-card">'
        f'<p class="hl-card-title">Model projections — next game</p>'
        f'<div class="hl-kpi-grid">{"".join(tiles)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def page_projection(roster: dict, api_key: str) -> None:
    page_hero(
        "Player projection",
        "Project a player's next game across every model, with recent form, "
        "distribution, and skill-profile context.",
    )
    if not roster:
        empty_state("No players in roster",
                    "Add a player from the sidebar to start projecting games.")
        return

    cols = st.columns([2, 1, 1])
    player = cols[0].selectbox("Player", list(roster))
    last_n = cols[1].slider("Recent-form window", 5, 30, 10)
    run = cols[2].button("Project", width="stretch", type="primary")

    games = _player_games(_roster_key(), player)
    _player_summary_kpis(player, games)

    if run or "last_proj_player" in st.session_state:
        if run:
            st.session_state.last_proj_player = player
            st.session_state.last_proj_n = last_n
        try:
            with st.spinner("Projecting next game…"):
                proj = _project_next_game_cached(
                    _roster_key(),
                    st.session_state.last_proj_player,
                    int(st.session_state.last_proj_n),
                    bool(st.session_state.get("use_prebuilt_bundle", False)),
                    str(st.session_state.get("prebuilt_bundle_path", "")),
                    bool(st.session_state.get("train_on_display_roster", False)),
                    bool(st.session_state.get("fast_training_mode", True)),
                )
        except Exception as exc:
            empty_state("Projection failed", f"{exc}")
            proj = pd.DataFrame()
        if not proj.empty:
            _projection_kpis(proj)
            with st.expander("Full projection table", expanded=False):
                st.dataframe(proj, width="stretch", hide_index=True)

    if games.empty:
        return

    tabs = st.tabs(["Overview", "Trends", "Profile", "Game log"])

    with tabs[0]:
        insight_card(
            "Recent form",
            "How this player has performed across the last several games. The dashed "
            "line marks the season average; bars above it indicate above-average outings.",
            icon="i",
        )
        c1, c2 = st.columns(2)
        metric_overview = c1.selectbox(
            "Metric", ["pts", "reb", "ast", "pra", "fantasy_score", "fg3m", "stl_blk", "tov"],
            key="proj_overview_metric",
        )
        n_games = c2.slider("Games shown", 5, 25, 10, key="proj_overview_n")
        color = player_color_map([player])[player]
        st.plotly_chart(
            charts.last_n_games_bar_chart(games, metric_overview, n=n_games, color=color),
            width="stretch",
        )
        st.plotly_chart(
            charts.metric_distribution_chart(
                games, metric_overview, player_label=player, color=color),
            width="stretch",
        )

    with tabs[1]:
        insight_card(
            "Rolling trends",
            "Smoothed rolling averages reveal sustained shifts in form rather than "
            "single-game variance.",
            icon="i",
        )
        metric = st.selectbox(
            "Metric",
            ["pts", "reb", "ast", "pra", "stl_blk", "fg3m", "tov", "fantasy_score"],
            key="rf_metric",
        )
        window = st.slider("Rolling window", 3, 15, 10, key="rf_window")
        avg = float(games[metric].mean()) if metric in games.columns else None
        fig = charts.rolling_form_chart(
            games, metric, window=window, player_label=player,
            color=player_color_map([player])[player],
            season_avg=avg,
        )
        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        insight_card(
            "Skill profile",
            "Each axis is normalized against the roster maximum, so longer spokes "
            "indicate where this player rates well relative to peers tracked here.",
            icon="i",
        )
        metrics = ["pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "fg_pct"]
        present = [m for m in metrics if m in games.columns]
        if not present:
            empty_state("No profile data", "Required metrics are missing for this player.")
        else:
            # Use the modeling frame (per-player cached) for roster-wide maxima
            # so we don't trigger a separate full-roster pipeline run.
            mdf = _modeling_frame(_roster_key())
            roster_max = {m: max(1e-9, float(mdf[m].max())) for m in present
                          if m in mdf.columns}
            profile = {
                player: {m: float(games[m].mean() / roster_max[m])
                         for m in present if m in roster_max}
            }
            st.plotly_chart(
                charts.normalized_radar(profile, title=f"{player} — skill profile"),
                width="stretch",
            )

    with tabs[3]:
        cols_show = [c for c in ("game_date", "MATCHUP", "pts", "reb", "ast", "fg3m",
                                 "stl", "blk", "tov", "min", "fantasy_score")
                     if c in games.columns]
        st.dataframe(games[cols_show].tail(40).iloc[::-1],
                     width="stretch", hide_index=True)


# ── Analytics Dashboard helpers ─────────────────────────────────────────────

_METRIC_PRETTY = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists",
    "fg3m": "3-Pointers Made", "stl_blk": "Stocks (STL+BLK)",
    "stl": "Steals", "blk": "Blocks", "tov": "Turnovers",
    "pra": "Points + Rebounds + Assists",
    "fantasy_score": "Fantasy Score",
}


def _pretty_metric(name: str) -> str:
    return _METRIC_PRETTY.get(str(name), str(name).replace("_", " ").title())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prepare_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an edge-board frame into a stable signal schema.

    Required output columns: player, metric, line, projection, gap, abs_gap,
    direction, gap_pct, signal_strength. Optional: books, matchup.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "player", "metric", "line", "projection", "gap", "abs_gap",
            "direction", "gap_pct", "signal_strength",
        ])

    player_col = _find_col(df, ["player", "player_name", "name"])
    metric_col = _find_col(df, ["model", "metric", "market", "stat", "category", "target"])
    line_col = _find_col(df, ["posted line", "line", "market_line", "point", "points", "current_line"])
    proj_col = _find_col(df, ["model prediction", "projection", "prediction", "pred",
                              "model_projection", "predicted"])
    gap_col = _find_col(df, ["edge", "gap", "projection_gap", "diff", "delta"])

    if not all([player_col, metric_col, line_col, proj_col]):
        return pd.DataFrame()

    out = pd.DataFrame({
        "player": df[player_col].astype(str),
        "metric": df[metric_col].astype(str),
        "line": pd.to_numeric(df[line_col], errors="coerce"),
        "projection": pd.to_numeric(df[proj_col], errors="coerce"),
    })
    if gap_col is not None:
        out["gap"] = pd.to_numeric(df[gap_col], errors="coerce")
    else:
        out["gap"] = out["projection"] - out["line"]

    out = out.dropna(subset=["line", "projection", "gap"]).reset_index(drop=True)
    if out.empty:
        return out

    out["abs_gap"] = out["gap"].abs()
    out["direction"] = np.where(out["gap"] >= 0, "Above line", "Below line")
    out["gap_pct"] = np.where(out["line"].abs() > 1e-9,
                              out["gap"] / out["line"].abs() * 100.0,
                              np.nan)

    # Signal strength buckets (quantile-based)
    p25, p75, p90 = (float(out["abs_gap"].quantile(q)) for q in (0.25, 0.75, 0.90))

    def _bucket(v: float) -> str:
        if v >= p90: return "Extreme"
        if v >= p75: return "High"
        if v >= p25: return "Medium"
        return "Low"

    out["signal_strength"] = out["abs_gap"].apply(_bucket)

    # Optional passthroughs
    for opt_src, opt_dst in (
        (_find_col(df, ["books", "book", "sportsbook", "book_count"]), "books"),
        (_find_col(df, ["matchup", "game", "opponent"]), "matchup"),
        (_find_col(df, ["book_names", "book_list"]), "book_names"),
        (_find_col(df, ["book_lines"]), "book_lines"),
    ):
        if opt_src is not None:
            out[opt_dst] = df[opt_src].values

    return out.sort_values("abs_gap", ascending=False).reset_index(drop=True)


def _fmt_num(value, digits: int = 1) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def _section_header(eyebrow: str, title: str, copy: str | None = None) -> None:
    body = f'<p class="hl-section-copy">{copy}</p>' if copy else ""
    st.markdown(
        f'<div class="hl-section-header">'
        f'<span class="hl-section-eyebrow">{eyebrow}</span>'
        f'<h2 class="hl-section-title">{title}</h2>'
        f'{body}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_dashboard_hero(signal_df: pd.DataFrame, *, has_api_key: bool,
                          last_refresh: str) -> None:
    if signal_df is None or signal_df.empty:
        if not has_api_key:
            copy = ("Hooplytics is currently <strong>offline</strong> for live line "
                    "intelligence. Add your Odds API key in the sidebar to begin "
                    "comparing model projections against the market.")
        else:
            copy = ("Hooplytics is online but no player lines were returned for the "
                    "current slate. Check back later or refresh from the sidebar.")
    else:
        n_lines = len(signal_df)
        n_players = signal_df["player"].nunique()
        n_metrics = signal_df["metric"].nunique()
        med = float(signal_df["abs_gap"].median())
        big = float(signal_df["abs_gap"].max())
        books_part = ""
        if "books" in signal_df.columns and not signal_df["books"].dropna().empty:
            try:
                bmax = int(pd.to_numeric(signal_df["books"], errors="coerce").max())
                books_part = (f" Market depth peaks at <strong>{bmax}</strong> "
                              f"sportsbooks for the most-covered lines.")
            except Exception:
                pass
        copy = (
            f"Hooplytics is currently analyzing <strong>{n_lines}</strong> player "
            f"lines across <strong>{n_players}</strong> players and "
            f"<strong>{n_metrics}</strong> statistical categories. "
            f"The median absolute projection gap is <strong>{med:.2f}</strong>, "
            f"with the largest single divergence reaching <strong>{big:.2f}</strong>."
            f"{books_part} The dashboard surfaces where model expectations diverge "
            f"from market lines, then layers in trend, coverage, and signal-quality "
            f"context for disciplined investigation."
        )

    status = (
        f'<span class="hl-status-dot"></span>LIVE LINES · UPDATED {last_refresh}'
        if has_api_key else
        f'<span class="hl-status-dot" style="background:#f5b041;'
        f'box-shadow:0 0 0 3px rgba(245,176,65,0.18)"></span>OFFLINE'
    )
    st.markdown(
        f'<div class="hl-dashboard-hero">'
        f'<span class="hl-hero-eyebrow">Analytics Dashboard</span>'
        f'<h1 class="hl-hero-title">Market Intelligence</h1>'
        f'<p class="hl-hero-subtitle">Model projections, player form, and live '
        f'line context in one command center.</p>'
        f'<p class="hl-hero-copy">{copy}</p>'
        f'<div class="hl-status" style="margin-top:1rem">{status}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, caption: str = "") -> str:
    cap = f'<span class="hl-kpi-caption">{caption}</span>' if caption else ""
    return (
        f'<div class="hl-kpi-card">'
        f'<span class="hl-kpi-label">{label}</span>'
        f'<span class="hl-kpi-value">{value}</span>'
        f'{cap}'
        f'</div>'
    )


def _render_kpi_strip(signal_df: pd.DataFrame, *, last_refresh: str) -> None:
    if signal_df is None or signal_df.empty:
        return
    n_lines = len(signal_df)
    n_players = signal_df["player"].nunique()
    n_metrics = signal_df["metric"].nunique()
    med = float(signal_df["abs_gap"].median())
    big = float(signal_df["abs_gap"].max())
    avg_proj = float(signal_df["projection"].mean())

    # Render the timestamp as a thin meta row above the strip rather than a
    # KPI card — it isn't a metric and the long value wraps awkwardly when
    # squeezed into an 8-card grid.
    meta_row([
        f'<span class="hl-mono">UPDATED · {last_refresh.upper()}</span>',
        f'<span>Refresh from the sidebar to pull new lines</span>',
    ])

    cards = [
        _kpi_card("Live lines analyzed", f"{n_lines:,}",
                  "projections compared against current market lines"),
        _kpi_card("Players covered", f"{n_players:,}",
                  "unique players with at least one available line"),
        _kpi_card("Metrics covered", f"{n_metrics}",
                  "statistical categories represented"),
        _kpi_card("Median projection gap", f"{med:.2f}",
                  "typical divergence between model and market"),
        _kpi_card("Largest projection gap", f"{big:.2f}",
                  "strongest single statistical signal"),
        _kpi_card("Avg model projection", f"{avg_proj:.1f}",
                  "mean of all model projections in view"),
    ]
    if "books" in signal_df.columns and not signal_df["books"].dropna().empty:
        try:
            bmax = int(pd.to_numeric(signal_df["books"], errors="coerce").max())
            cards.append(_kpi_card("Sportsbooks (max)", f"{bmax}",
                                   "deepest market coverage on a single line"))
        except Exception:
            pass

    st.markdown(
        f'<div class="hl-kpi-strip">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _signal_insight_sentence(row: pd.Series) -> str:
    metric = _pretty_metric(row["metric"])
    gap = float(row["gap"])
    abs_gap = float(row["abs_gap"])
    direction = row["direction"].lower()
    strength = str(row.get("signal_strength", "")).lower()
    qualifier = {
        "extreme": "an extreme", "high": "a wider-than-typical",
        "medium": "a moderate", "low": "a narrow",
    }.get(strength, "a")
    base = (
        f"The model projects {row['player']} {abs_gap:.1f} {metric.lower()} "
        f"{direction}, {qualifier} divergence from the current market line."
    )
    tail = (" Validate against recent form and matchup context before treating it "
            "as a stronger analytical signal.")
    return base + tail if abs(gap) > 0 else base


def _render_signal_cards(signal_df: pd.DataFrame, *, top_n: int = 6) -> None:
    if signal_df is None or signal_df.empty:
        empty_state("No signals to display",
                    "Live model-vs-market signals will appear here once data is available.")
        return
    top = signal_df.head(top_n)
    cards_html = []
    for _, row in top.iterrows():
        is_above = row["direction"] == "Above line"
        cls = "above" if is_above else "below"
        gap_cls = "gap-pos" if is_above else "gap-neg"
        meta_bits = [_pretty_metric(row["metric"])]
        if "matchup" in row and isinstance(row["matchup"], str) and row["matchup"]:
            meta_bits.append(row["matchup"])
        if "books" in row and pd.notna(row.get("books")):
            try:
                meta_bits.append(f"{int(row['books'])} books")
            except Exception:
                pass
        strength = str(row.get("signal_strength", "Medium"))
        chip_cls = f"hl-chip-{strength.lower()}"
        insight = _signal_insight_sentence(row)
        cards_html.append(
            f'<div class="hl-signal-card {cls}">'
            f'<div class="hl-signal-top">'
            f'<div>'
            f'<div class="hl-signal-player">{row["player"]}</div>'
            f'<div class="hl-signal-meta">{" · ".join(meta_bits)}</div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.3rem">'
            f'<span class="hl-signal-direction {cls}">{row["direction"]}</span>'
            f'<span class="{chip_cls}">{strength}</span>'
            f'</div>'
            f'</div>'
            f'<div class="hl-signal-value-row">'
            f'<div class="hl-signal-cell">'
            f'<span class="hl-signal-cell-label">Current line</span>'
            f'<span class="hl-signal-cell-value">{_fmt_num(row["line"], 1)}</span>'
            f'</div>'
            f'<div class="hl-signal-cell">'
            f'<span class="hl-signal-cell-label">Model projection</span>'
            f'<span class="hl-signal-cell-value">{_fmt_num(row["projection"], 1)}</span>'
            f'</div>'
            f'<div class="hl-signal-cell">'
            f'<span class="hl-signal-cell-label">Projection gap</span>'
            f'<span class="hl-signal-cell-value {gap_cls}">{row["gap"]:+.2f}</span>'
            f'</div>'
            f'</div>'
            f'<div class="hl-signal-insight">{insight}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="hl-signal-grid">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


def _render_dashboard_explorer(signal_df: pd.DataFrame) -> None:
    if signal_df is None or signal_df.empty:
        empty_state("Nothing to explore yet",
                    "The explorer will populate once signals are available.")
        return

    cols = st.columns([2, 2, 2, 1, 1])
    search = cols[0].text_input("Player search", value="",
                                placeholder="Filter by player name")
    metrics = cols[1].multiselect(
        "Metric", sorted(signal_df["metric"].unique()),
        default=sorted(signal_df["metric"].unique()),
    )
    directions = cols[2].multiselect(
        "Direction", ["Above line", "Below line"],
        default=["Above line", "Below line"],
    )
    gap_max = float(signal_df["abs_gap"].max())
    min_gap = cols[3].slider("Min |gap|", 0.0, max(gap_max, 0.5),
                             0.0, step=0.1)
    top_n = cols[4].slider("Top N", 5, max(50, len(signal_df)),
                           min(40, len(signal_df)))

    f = signal_df[
        signal_df["metric"].isin(metrics)
        & signal_df["direction"].isin(directions)
        & (signal_df["abs_gap"] >= min_gap)
    ].copy()
    if search.strip():
        f = f[f["player"].str.contains(search.strip(), case=False, na=False)]
    f = f.head(top_n)

    if f.empty:
        empty_state("No signals match the current filters",
                    "Try widening the metric, direction, or minimum gap filters.")
        return

    display = f.copy()
    display["metric"] = display["metric"].apply(_pretty_metric)
    display["line"] = display["line"].round(2)
    display["projection"] = display["projection"].round(2)
    display["gap"] = display["gap"].round(2)
    display["abs_gap"] = display["abs_gap"].round(2)
    display["gap_pct"] = display["gap_pct"].round(1)

    rename = {
        "player": "Player", "metric": "Metric",
        "line": "Current line", "projection": "Model projection",
        "gap": "Gap", "abs_gap": "|Gap|", "gap_pct": "Gap %",
        "direction": "Direction", "signal_strength": "Signal",
        "books": "Books", "book_names": "Sportsbooks", "matchup": "Matchup",
    }
    display = display.rename(columns=rename)
    keep = [c for c in (
        "Player", "Metric", "Direction", "Signal",
        "Current line", "Model projection", "Gap", "|Gap|", "Gap %",
        "Books", "Sportsbooks", "Matchup",
    ) if c in display.columns]
    st.dataframe(display[keep], width="stretch", hide_index=True)

    # Per-book line breakdown (expandable). Uses raw `book_lines` dict from
    # the filtered signal frame so users can see exactly which sportsbook
    # posted which line for the highest-edge plays.
    if "book_lines" in f.columns:
        rows: list[dict] = []
        for _, row in f.iterrows():
            bl = row.get("book_lines")
            if not isinstance(bl, dict) or not bl:
                continue
            for book, line in sorted(bl.items()):
                rows.append({
                    "Player": row.get("player", ""),
                    "Metric": _pretty_metric(row.get("metric", "")),
                    "Sportsbook": book,
                    "Line": round(float(line), 2),
                })
        if rows:
            with st.expander(f"Per-book line breakdown ({len(rows)} entries)", expanded=False):
                st.caption(
                    "Consensus line is the median across the high-quality NA "
                    "books listed below. Lines that diverge from the consensus "
                    "indicate book-specific edges."
                )
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                )


def page_edge_board(roster: dict, api_key: str) -> None:
    last_refresh = datetime.now().strftime("%b %d · %H:%M")

    edge_df = _build_edge_board(roster, api_key) if api_key else pd.DataFrame()
    signal_df = _prepare_signal_frame(edge_df)

    # 1. Hero / narrative summary
    _render_dashboard_hero(signal_df, has_api_key=bool(api_key),
                           last_refresh=last_refresh)

    if not api_key:
        empty_state(
            "Live line context is unavailable",
            "Add your Odds API key from the sidebar to unlock model-vs-market "
            "analysis, signal cards, coverage telemetry, and the explorer.",
        )
        return

    if signal_df.empty:
        empty_state(
            "No active player lines",
            "No NBA player lines were returned for tonight's slate. Try refreshing "
            "from the sidebar, or check that your roster has lines posted.",
        )
        return

    # 2. KPI strip
    _render_kpi_strip(signal_df, last_refresh=last_refresh)

    # 3. Strongest model signals
    _section_header(
        "Section 01", "Strongest model signals",
        "The largest absolute projection gaps on the slate, packaged with line, "
        "projection, and signal-strength context. Treat each card as a starting "
        "point for investigation, not a conclusion.",
    )
    n_signal_cards = st.slider("Cards", 4, 12, 6, key="dash_signal_n",
                               label_visibility="collapsed")
    _render_signal_cards(signal_df, top_n=n_signal_cards)

    # 4. Projection gap analysis
    _section_header(
        "Section 02", "Projection gap analysis",
        "Projection gaps compare the model’s expected outcome against the current "
        "available line. The largest gaps are not automatic decisions — they are "
        "the strongest places to investigate first.",
    )
    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        st.plotly_chart(
            charts.projection_gap_bar_chart(signal_df, top_n=15),
            width="stretch",
        )
        st.plotly_chart(
            charts.projection_gap_scatter(signal_df),
            width="stretch",
        )
    with g2:
        st.plotly_chart(
            charts.direction_split_chart(signal_df),
            width="stretch",
        )
        st.plotly_chart(
            charts.projection_gap_distribution_chart(signal_df),
            width="stretch",
        )
        st.plotly_chart(
            charts.metric_signal_heatmap(signal_df),
            width="stretch",
        )

    # 5. Player and metric coverage
    _section_header(
        "Section 03", "Coverage telemetry",
        "Coverage telemetry shows where the app has the most market context. "
        "Broader coverage usually makes comparison easier, while thin coverage "
        "should be treated with more caution.",
    )
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.plotly_chart(charts.player_coverage_bar(signal_df), width="stretch")
    with c2:
        st.plotly_chart(charts.metric_coverage_bar(signal_df), width="stretch")
    with c3:
        st.plotly_chart(charts.avg_gap_by_metric_bar(signal_df), width="stretch")

    # 6. Signal quality / volatility
    _section_header(
        "Section 04", "Signal quality",
        "Signal strength is a ranking tool, not a guarantee. Hooplytics uses "
        "projection gap size to prioritize attention, then encourages users to "
        "validate the signal against player role, minutes, matchup, and recent form.",
    )
    sq1, sq2 = st.columns([2, 3], gap="large")
    with sq1:
        st.plotly_chart(
            charts.signal_strength_distribution_chart(signal_df),
            width="stretch",
        )
        # Strength legend
        st.markdown(
            '<div class="hl-chip-row">'
            '<span class="hl-chip-low">Low</span>'
            '<span class="hl-chip-medium">Medium</span>'
            '<span class="hl-chip-high">High</span>'
            '<span class="hl-chip-extreme">Extreme</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with sq2:
        extreme = signal_df[signal_df["signal_strength"].isin(["Extreme", "High"])].head(8)
        if extreme.empty:
            empty_state("No high-strength signals yet",
                        "Once larger projection gaps emerge they will surface here.")
        else:
            _render_signal_cards(extreme, top_n=len(extreme))

    # 7. Detailed explorer
    _section_header(
        "Section 05", "Explorer",
        "Use the explorer to filter, sort, and inspect the underlying "
        "player-line data behind the dashboard.",
    )
    _render_dashboard_explorer(signal_df)

    st.markdown(
        '<div class="hl-disclaimer-card">'
        '<strong>For statistical analysis and entertainment only.</strong> '
        'Hooplytics compares model projections against publicly posted player '
        'lines from The Odds API to highlight where performance expectations '
        'differ from recent statistical evidence. Output is not gambling advice.'
        '</div>',
        unsafe_allow_html=True,
    )


def page_compare(roster: dict, api_key: str) -> None:
    page_hero(
        "Compare players",
        "Side-by-side recent form, distributions, profiles, and trends across selected players.",
    )
    if len(roster) < 2:
        empty_state("Need at least two players",
                    "Add another player from the sidebar to enable comparisons.")
        return

    selected = st.multiselect("Players", list(roster), default=list(roster)[:4])
    if len(selected) < 2:
        empty_state("Select at least two players", "Choose two or more above to compare.")
        return

    games_by_player = {p: _player_games(_roster_key(), p) for p in selected}
    metrics = ["pts", "reb", "ast", "stl_blk", "fg3m", "tov", "min", "fantasy_score"]
    metrics = [m for m in metrics if any(m in g.columns for g in games_by_player.values())]

    tabs = st.tabs(["Summary", "Trends", "Distributions", "Shape", "Profile", "Game log"])

    # ── Summary ─────────────────────────────────────────────────────────────
    with tabs[0]:
        insight_card(
            "Recent form snapshot",
            "Per-player averages across the full corpus, plus a leaderboard for "
            "any single metric.",
            icon="i",
        )
        rows = []
        for p, g in games_by_player.items():
            row = {"player": p, "Games": len(g)}
            for col, label in (("pts", "PPG"), ("reb", "RPG"), ("ast", "APG"),
                               ("pra", "PRA"), ("fantasy_score", "FAN"),
                               ("min", "MPG")):
                if col in g.columns:
                    s = pd.to_numeric(g[col], errors="coerce").dropna()
                    row[label] = round(float(s.mean()), 1) if not s.empty else None
            rows.append(row)
        summary_df = pd.DataFrame(rows).set_index("player")
        st.dataframe(summary_df, width="stretch")

        leaderboard_metric_options = [c for c in ("PPG", "RPG", "APG", "PRA", "FAN", "MPG")
                                      if c in summary_df.columns]
        if leaderboard_metric_options:
            lb_metric = st.selectbox("Leaderboard metric", leaderboard_metric_options,
                                     key="cmp_lb_metric")
            st.plotly_chart(
                charts.player_summary_bar(summary_df, lb_metric),
                width="stretch",
            )

    # ── Trends ──────────────────────────────────────────────────────────────
    with tabs[1]:
        insight_card(
            "Rolling metric comparison",
            "Each line is the rolling per-game average for the chosen metric, "
            "letting you compare how players' recent form has evolved together.",
            icon="i",
        )
        trend_metric = st.selectbox("Metric", metrics, key="cmp_trend_metric")
        window = st.slider("Rolling window", 3, 15, 5, key="cmp_trend_window")
        st.plotly_chart(
            charts.rolling_metric_compare(games_by_player, trend_metric, window=window),
            width="stretch",
        )

    # ── Distributions ───────────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(charts.distribution_facets(games_by_player, metrics),
                        width="stretch")

    # ── Shape ───────────────────────────────────────────────────────────────
    with tabs[3]:
        st.plotly_chart(charts.violin_grid(games_by_player, metrics),
                        width="stretch")

    # ── Profile ─────────────────────────────────────────────────────────────
    with tabs[4]:
        present = [m for m in ("pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "fg_pct")
                   if all(m in g.columns for g in games_by_player.values())]
        if not present:
            empty_state("No shared profile metrics",
                        "These players don't share enough common metrics for a profile chart.")
        else:
            roster_max = {m: max(1e-9, max(g[m].max() for g in games_by_player.values()))
                          for m in present}
            profiles = {
                p: {m: float(games_by_player[p][m].mean() / roster_max[m]) for m in present}
                for p in selected
            }
            st.plotly_chart(charts.normalized_radar(profiles), width="stretch")

    # ── Game log ────────────────────────────────────────────────────────────
    with tabs[5]:
        for p in selected:
            section(p)
            g = games_by_player[p]
            cols_show = [c for c in ("game_date", "MATCHUP", "pts", "reb", "ast",
                                     "fg3m", "stl", "blk", "tov", "min", "fantasy_score")
                         if c in g.columns]
            st.dataframe(g[cols_show].tail(15).iloc[::-1],
                         width="stretch", hide_index=True)


_SCENARIO_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Scoring",     ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta")),
    ("Efficiency",  ("fg_pct", "fg3_pct", "ft_pct", "fga_l10")),
    ("Rebounding", ("reb", "oreb", "dreb")),
    ("Playmaking", ("ast", "ast_l5", "ast_l10", "ast_l30", "ast_per36_l30",
                    "usg_proxy_l30")),
    ("Defense",    ("stl", "blk", "stl_l10", "blk_l10", "stl_l30", "blk_l30",
                    "stl_per36_l30", "blk_per36_l30")),
    ("Turnovers",  ("tov", "tov_l5", "tov_l10", "tov_l30", "tov_per36_l30")),
    ("Minutes & impact", ("min", "min_l10", "min_l30", "plus_minus")),
]


def _classify_scenario_features(features: list[str]) -> list[tuple[str, list[str]]]:
    """Group features by analytics category, with an Other bucket for the rest."""
    grouped: list[tuple[str, list[str]]] = []
    used: set[str] = set()
    for label, members in _SCENARIO_GROUPS:
        bucket = [f for f in features if f in members]
        if bucket:
            grouped.append((label, bucket))
            used.update(bucket)
    leftover = [f for f in features if f not in used]
    if leftover:
        grouped.append(("Other", leftover))
    return grouped


# ── Player Line Lab helpers ─────────────────────────────────────────────────
_LAB_DISCLAIMER = (
    "Hooplytics is for statistical analysis and entertainment. Lines are used as "
    "context for comparing player performance, model projections, and historical "
    "outcomes. This is not financial or betting advice."
)


def _get_player_games(modeling_df: pd.DataFrame, player: str) -> pd.DataFrame:
    if modeling_df is None or modeling_df.empty or "player" not in modeling_df.columns:
        return pd.DataFrame()
    df = modeling_df[modeling_df["player"] == player].copy()
    if "game_date" in df.columns:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


def _calc_recent_averages(games: pd.DataFrame, col: str) -> dict[str, float]:
    out = {"season": float("nan"), "last5": float("nan"),
           "last10": float("nan"), "last15": float("nan"),
           "std": float("nan"), "min_avg": float("nan")}
    if games is None or games.empty or col not in games.columns:
        return out
    s = pd.to_numeric(games[col], errors="coerce").dropna()
    if s.empty:
        return out
    out["season"] = float(s.mean())
    out["std"]    = float(s.std()) if len(s) > 1 else 0.0
    if len(s) >= 1:  out["last5"]  = float(s.tail(5).mean())
    if len(s) >= 1:  out["last10"] = float(s.tail(10).mean())
    if len(s) >= 1:  out["last15"] = float(s.tail(15).mean())
    if "min" in games.columns:
        m = pd.to_numeric(games["min"], errors="coerce").dropna()
        if not m.empty:
            out["min_avg"] = float(m.tail(10).mean())
    return out


def _classify_recent_trend(last5: float, last10: float, season: float) -> tuple[str, str]:
    """Return (label, css_class)."""
    if any(pd.isna(x) for x in (last5, last10, season)):
        return ("Insufficient data", "hl-snap-trend-flat")
    diff = last5 - season
    rel  = (diff / season) if season else 0.0
    if rel > 0.06 and last5 >= last10:
        return ("Heating up", "hl-snap-trend-up")
    if rel < -0.06 and last5 <= last10:
        return ("Cooling off", "hl-snap-trend-down")
    return ("Stable", "hl-snap-trend-flat")


def _calc_line_outcomes(games: pd.DataFrame, col: str, line: float) -> dict[str, float]:
    out = {
        "n": 0, "above": 0, "below": 0, "push": 0,
        "above_rate": float("nan"), "below_rate": float("nan"),
        "avg_margin": float("nan"), "med_margin": float("nan"),
        "best": float("nan"), "worst": float("nan"), "std": float("nan"),
    }
    if games is None or games.empty or col not in games.columns:
        return out
    vals = pd.to_numeric(games[col], errors="coerce").dropna()
    if vals.empty:
        return out
    line = float(line)
    margins = vals - line
    out["n"]          = int(len(vals))
    out["above"]      = int((vals > line).sum())
    out["below"]      = int((vals < line).sum())
    out["push"]       = int((vals == line).sum())
    out["above_rate"] = float(out["above"] / out["n"]) if out["n"] else float("nan")
    out["below_rate"] = float(out["below"] / out["n"]) if out["n"] else float("nan")
    out["avg_margin"] = float(margins.mean())
    out["med_margin"] = float(margins.median())
    out["best"]       = float(vals.max())
    out["worst"]      = float(vals.min())
    out["std"]        = float(vals.std()) if len(vals) > 1 else 0.0
    return out


def _model_projection_for(player: str, model_name: str, *, bundle, modeling_df) -> float:
    try:
        proj = project_next_game(player, bundle=bundle, store=_store(),
                                 last_n=10, modeling_df=modeling_df)
    except Exception:
        return float("nan")
    if proj is None or proj.empty or "model" not in proj.columns:
        return float("nan")
    row = proj[proj["model"] == model_name]
    if row.empty:
        return float("nan")
    return float(row["prediction"].iloc[0])


def _line_for(live_df: pd.DataFrame, player: str, model_name: str) -> tuple[float | None, dict]:
    """Return (consensus line, context dict with books, matchup)."""
    if live_df is None or live_df.empty:
        return (None, {})
    rows = live_df[(live_df["player"] == player) & (live_df["model"] == model_name)]
    if rows.empty:
        return (None, {})
    r = rows.iloc[0]
    return (float(r["line"]), {
        "books": int(r.get("books", 0) or 0),
        "matchup": str(r.get("matchup", "") or ""),
    })


def _snap_card(label: str, value: str, caption: str = "", trend_class: str = "") -> str:
    cap = f'<div class="hl-snap-caption {trend_class}">{caption}</div>' if caption else ""
    return (f'<div class="hl-snap-card">'
            f'<div class="hl-snap-label">{label}</div>'
            f'<div class="hl-snap-value">{value}</div>{cap}</div>')


def _outcome_card(label: str, value: str) -> str:
    return (f'<div class="hl-outcome-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


def _fmt(v, digits: int = 1, suffix: str = "") -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _generate_player_line_lab_summary(
    *, player: str, metric: str, line: float, projection: float,
    averages: dict, outcomes: dict, trend_label: str,
) -> str:
    last10 = averages.get("last10", float("nan"))
    season = averages.get("season", float("nan"))
    above_rate = outcomes.get("above_rate", float("nan"))
    n = outcomes.get("n", 0)

    def above_or_below(val: float) -> str | None:
        if pd.isna(val):
            return None
        if val > line + 0.01: return "above"
        if val < line - 0.01: return "below"
        return "at"

    proj_pos = above_or_below(projection)
    l10_pos  = above_or_below(last10)

    parts: list[str] = []
    if proj_pos == "above" and l10_pos == "above":
        parts.append("The data leans above the current line.")
    elif proj_pos == "below" and l10_pos == "below":
        parts.append("The data leans below the current line.")
    elif proj_pos and l10_pos and proj_pos != l10_pos:
        parts.append("The evidence is mixed — model projection and recent form disagree.")
    else:
        parts.append("The signal is directionally neutral around the current threshold.")

    if not pd.isna(projection):
        gap = projection - line
        sign = "+" if gap >= 0 else ""
        parts.append(f"Model projects {projection:.1f} ({sign}{gap:.1f} vs the {line:.1f} line).")
    if not pd.isna(last10) and not pd.isna(season):
        parts.append(f"Last-10 average is {last10:.1f} compared with a season mean of {season:.1f}.")
    if n and not pd.isna(above_rate):
        parts.append(f"Historically, {player} has finished above this threshold in "
                     f"{int(round(above_rate * 100))}% of {n} available games.")
    if trend_label and trend_label != "Stable":
        parts.append(f"Recent trend is **{trend_label.lower()}**, which adds context to the read.")
    parts.append("This is a directional analytics view — not a recommendation.")
    return " ".join(parts)


def page_scenario(roster: dict, api_key: str) -> None:
    """Player Line Lab — analytics workbench around a player + metric + line."""
    st.markdown(
        '<div class="hl-lab-hero">'
        '<h1>Player Line Lab</h1>'
        '<p>Compare player form, model projections, and current line context in one '
        'analytical workspace. Pick a player and metric, then explore how the current '
        'threshold sits against historical performance and projected output.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not roster:
        empty_state("No players in roster",
                    "Add a player from the sidebar to launch the Player Line Lab.")
        return

    bundle = _bundle_for_ui()
    modeling_df = _modeling_frame(_roster_key())

    # Pretty model labels (display only)
    pretty = {
        "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
        "pra": "PRA (pts+reb+ast)", "threepm": "3-pointers made",
        "stl_blk": "Stl + Blk", "turnovers": "Turnovers",
        "fantasy_score": "Fantasy score",
    }
    model_names = list(MODEL_SPECS.keys())

    # ── Section 1: Lab Controls ────────────────────────────────────────────
    st.markdown('<div class="hl-lab-controls">'
                '<p class="hl-controls-eyebrow">Lab controls</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1])
    player = c1.selectbox("Player", list(roster), key="lab_player")
    metric_label = c2.selectbox(
        "Metric", [pretty.get(m, m) for m in model_names], key="lab_metric")
    model_name = model_names[[pretty.get(m, m) for m in model_names].index(metric_label)]
    metric_col = MODEL_TO_COL.get(model_name, model_name)

    window_label = c3.selectbox("Recent form window",
                                ["Last 5", "Last 10", "Last 15", "Full season"],
                                index=1, key="lab_window")
    tolerance = c4.select_slider("Similar-line tolerance",
                                 options=[0.5, 1.0, 1.5, 2.0, 3.0],
                                 value=1.0, key="lab_tol")

    # Live line context
    _force_lab = bool(st.session_state.pop("force_refresh_odds", False))
    live_df = _live_lines(api_key, json.dumps(list(roster)),
                          st.session_state.live_bust,
                          _force_lab) if api_key else pd.DataFrame()
    live_line, line_ctx = _line_for(live_df, player, model_name)

    c5, c6, c7 = st.columns([1, 1, 1])
    use_manual = c5.checkbox("Override line manually",
                             value=(live_line is None), key="lab_manual_toggle")
    default_line = float(live_line) if live_line is not None else 0.0
    manual_line = c6.number_input(
        "Manual line", value=round(default_line, 1), step=0.5,
        key="lab_manual_line", disabled=not use_manual,
    )
    line_value = float(manual_line) if use_manual else float(live_line) if live_line is not None else float(manual_line)
    c7.markdown(
        f'<div style="padding-top:1.7rem;color:var(--hl-ink-muted);font-size:0.82rem">'
        f'Active line: <span class="hl-threshold-chip">{line_value:.1f}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    games = _get_player_games(modeling_df, player)
    if games.empty or metric_col not in games.columns:
        empty_state(
            f"No data for {player} · {metric_label}",
            "This metric has no historical games available for the selected player. "
            "Try a different player or metric, or add seasons in the sidebar.",
        )
        return

    # ── Section 2: Player Snapshot ─────────────────────────────────────────
    section("Player snapshot",
            "Recent form, season baseline, and trend direction for the active player.")
    averages = _calc_recent_averages(games, metric_col)
    trend_label, trend_class = _classify_recent_trend(
        averages["last5"], averages["last10"], averages["season"])
    last_date = ""
    if "game_date" in games.columns:
        try:
            last_date = pd.to_datetime(games["game_date"].iloc[-1]).strftime("%b %d, %Y")
        except Exception:
            last_date = ""

    snap_cards = [
        _snap_card("Player", player, f"{len(games)} games · last {last_date}" if last_date else f"{len(games)} games"),
        _snap_card(f"{metric_label} · season avg", _fmt(averages["season"])),
        _snap_card("Last 5 avg", _fmt(averages["last5"])),
        _snap_card("Last 10 avg", _fmt(averages["last10"])),
        _snap_card("Volatility (σ)", _fmt(averages["std"])),
        _snap_card("Recent trend", trend_label, "", trend_class),
    ]
    if not pd.isna(averages["min_avg"]):
        snap_cards.append(_snap_card("Minutes (L10)", _fmt(averages["min_avg"])))
    st.markdown(f'<div class="hl-player-snapshot">{"".join(snap_cards)}</div>',
                unsafe_allow_html=True)

    # Snapshot prose
    if not pd.isna(averages["last10"]) and not pd.isna(averages["season"]):
        diff = averages["last10"] - averages["season"]
        direction = "above" if diff >= 0 else "below"
        st.markdown(
            f'<div class="hl-analyst-note">'
            f'<p class="eyebrow">Snapshot</p>'
            f'<p>Over the selected window, <strong>{player}</strong> is averaging '
            f'<strong>{averages["last10"]:.1f}</strong> {metric_label.lower()} across the last 10 games '
            f'compared with a season average of <strong>{averages["season"]:.1f}</strong>. '
            f'Recent form is trending <strong>{direction} baseline</strong>, '
            f'with game-to-game volatility of σ={averages["std"]:.1f}.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Section 3: Line Context ────────────────────────────────────────────
    section("Line context",
            "Current line summary from The Odds API. Used as analytical context only.")
    if live_line is None:
        st.markdown(
            '<div class="hl-warning-soft">No live line context is available for this '
            'player and metric. Use the manual line control to run a historical line study.</div>',
            unsafe_allow_html=True,
        )
    else:
        ctx_cols = st.columns(4)
        ctx_cols[0].markdown(_outcome_card("Consensus line", _fmt(live_line)),
                             unsafe_allow_html=True)
        ctx_cols[1].markdown(_outcome_card("Books contributing",
                                           str(line_ctx.get("books", "—"))),
                             unsafe_allow_html=True)
        ctx_cols[2].markdown(_outcome_card("Matchup",
                                           line_ctx.get("matchup", "—") or "—"),
                             unsafe_allow_html=True)
        ctx_cols[3].markdown(_outcome_card("Active threshold", _fmt(line_value)),
                             unsafe_allow_html=True)
        # Single-bar book chart (consensus only — Hooplytics aggregates pre-storage)
        st.plotly_chart(
            charts.book_line_comparison_chart(
                pd.DataFrame({"book": ["Consensus"], "line": [live_line]}),
                player=player, metric=metric_label,
            ),
            width="stretch",
        )

    # ── Section 4: Projection vs Line ──────────────────────────────────────
    section("Projection vs line",
            "Where the model projection and recent averages sit against the active threshold.")
    projection = _model_projection_for(player, model_name,
                                        bundle=bundle, modeling_df=modeling_df)
    historical_outcomes = _calc_line_outcomes(games, metric_col, line_value)
    gap = projection - line_value if not pd.isna(projection) else float("nan")

    pcols = st.columns(6)
    pcols[0].markdown(_outcome_card("Active line", _fmt(line_value)),       unsafe_allow_html=True)
    pcols[1].markdown(_outcome_card("Model projection", _fmt(projection)),  unsafe_allow_html=True)
    gap_str = (f'+{gap:.1f}' if not pd.isna(gap) and gap >= 0
               else (f'{gap:.1f}' if not pd.isna(gap) else "—"))
    pcols[2].markdown(_outcome_card("Projection gap", gap_str),             unsafe_allow_html=True)
    pcols[3].markdown(_outcome_card("Last 5 avg", _fmt(averages["last5"])), unsafe_allow_html=True)
    pcols[4].markdown(_outcome_card("Season avg", _fmt(averages["season"])),unsafe_allow_html=True)
    pcols[5].markdown(_outcome_card("Above-line rate (hist.)",
                                    _fmt_pct(historical_outcomes["above_rate"])),
                      unsafe_allow_html=True)

    bullet_values = {
        "line":       line_value,
        "projection": projection,
        "last5":      averages["last5"],
        "last10":     averages["last10"],
        "season":     averages["season"],
    }
    st.plotly_chart(charts.projection_vs_line_bullet_chart(bullet_values),
                    width="stretch")

    # ── Section 5: Historical Outcome Study ────────────────────────────────
    section("Historical outcome study",
            "How the player has actually performed against this threshold across available games.")
    o = historical_outcomes
    ocols = st.columns(6)
    ocols[0].markdown(_outcome_card("Games", str(o["n"])),                unsafe_allow_html=True)
    ocols[1].markdown(_outcome_card("Above line", str(o["above"])),       unsafe_allow_html=True)
    ocols[2].markdown(_outcome_card("Below line", str(o["below"])),       unsafe_allow_html=True)
    ocols[3].markdown(_outcome_card("Above-line rate", _fmt_pct(o["above_rate"])), unsafe_allow_html=True)
    ocols[4].markdown(_outcome_card("Avg margin",  _fmt(o["avg_margin"])),unsafe_allow_html=True)
    ocols[5].markdown(_outcome_card("Median margin", _fmt(o["med_margin"])),unsafe_allow_html=True)

    if o["n"] > 0:
        margin_word = "above" if o["avg_margin"] >= 0 else "below"
        st.markdown(
            f'<div class="hl-analyst-note">'
            f'<p class="eyebrow">Outcome read</p>'
            f'<p>In <strong>{o["n"]}</strong> historical games, this player finished '
            f'above <strong>{line_value:.1f}</strong> {metric_label.lower()} '
            f'<strong>{o["above"]}</strong> times, below it <strong>{o["below"]}</strong> '
            f'times, and exactly on it <strong>{o["push"]}</strong>. The average margin against '
            f'the line was <strong>{o["avg_margin"]:+.1f}</strong>, suggesting the threshold sits '
            f'<strong>{margin_word}</strong> the player\'s typical output across this sample.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    h1, h2 = st.columns(2)
    h1.plotly_chart(
        charts.historical_outcome_distribution_chart(games, metric_col, line_value),
        width="stretch")
    h2.plotly_chart(
        charts.game_by_game_margin_chart(games, metric_col, line_value, n=20),
        width="stretch")
    st.plotly_chart(
        charts.above_below_timeline_chart(games, metric_col, line_value, n=20),
        width="stretch")

    # ── Section 6: Similar-Line Analysis ───────────────────────────────────
    section("Similar-line analysis",
            "Outcomes near the current threshold, weighted by your selected tolerance band.")
    st.markdown(
        '<div class="hl-methodology-card">'
        '<p>Historical market lines are not stored locally, so this view compares the '
        'current line against historical player outcomes inside a ±tolerance band, rather '
        'than past book prices.</p></div>',
        unsafe_allow_html=True,
    )
    vals = pd.to_numeric(games[metric_col], errors="coerce").dropna()
    in_band = vals[(vals >= line_value - tolerance) & (vals <= line_value + tolerance)]
    near_above = float((vals >= line_value).mean()) if len(vals) else float("nan")

    sims = st.columns(4)
    sims[0].markdown(_outcome_card("Tolerance", f"±{tolerance:.1f}"),                unsafe_allow_html=True)
    sims[1].markdown(_outcome_card("Games in band", str(int(len(in_band)))),         unsafe_allow_html=True)
    sims[2].markdown(_outcome_card("Above-line rate (full)", _fmt_pct(near_above)),  unsafe_allow_html=True)
    sims[3].markdown(_outcome_card("Volatility (band σ)",
                                   _fmt(float(in_band.std()) if len(in_band) > 1 else float("nan"))),
                     unsafe_allow_html=True)

    st.plotly_chart(
        charts.similar_threshold_outcome_chart(games, metric_col, line_value, tolerance),
        width="stretch",
    )

    # ── Section 7: Sensitivity Lab ─────────────────────────────────────────
    section("Sensitivity lab",
            "How fragile or stable the historical signal is as the threshold moves.")
    deltas = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
    sens_rows = []
    last5_vals = vals.tail(5)
    last10_vals = vals.tail(10)
    for d in deltas:
        t = line_value + d
        sens_rows.append({
            "Threshold": round(t, 2),
            "Δ vs current": f"{d:+.1f}",
            "Above-line rate": float((vals >= t).mean()) if len(vals) else float("nan"),
            "Avg margin":      float((vals - t).mean()) if len(vals) else float("nan"),
            "Last 5 above":    float((last5_vals >= t).mean()) if len(last5_vals) else float("nan"),
            "Last 10 above":   float((last10_vals >= t).mean()) if len(last10_vals) else float("nan"),
        })
    sens_df = pd.DataFrame(sens_rows)
    st.dataframe(
        sens_df, hide_index=True, width="stretch",
        column_config={
            "Above-line rate": st.column_config.NumberColumn(format="%.0f%%"),
            "Last 5 above":    st.column_config.NumberColumn(format="%.0f%%"),
            "Last 10 above":   st.column_config.NumberColumn(format="%.0f%%"),
            "Avg margin":      st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.plotly_chart(
        charts.line_sensitivity_curve_chart(games, metric_col, line_value,
                                            projection=projection),
        width="stretch",
    )
    st.markdown(
        '<div class="hl-methodology-card">'
        '<p>Sensitivity analysis shows how quickly the historical outcome rate changes as '
        'the threshold moves. A steep curve means the signal is fragile; a flatter curve '
        'means production has been more stable around this range.</p></div>',
        unsafe_allow_html=True,
    )

    # ── Section 8: Analyst Notes ───────────────────────────────────────────
    section("Analyst notes",
            "Synthesis of projection, recent form, and historical outcomes.")
    summary = _generate_player_line_lab_summary(
        player=player, metric=metric_label, line=line_value, projection=projection,
        averages=averages, outcomes=historical_outcomes, trend_label=trend_label,
    )
    st.markdown(
        f'<div class="hl-signal-summary">'
        f'<p class="eyebrow">Synthesis</p>'
        f'<p>{summary}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Section 9: Detailed Game Log ───────────────────────────────────────
    section("Detailed game log",
            "Per-game performance vs the active line for the selected metric.")
    log = games.copy()
    cols_keep = ["game_date"]
    if "MATCHUP" in log.columns:
        cols_keep.append("MATCHUP")
    if "min" in log.columns:
        cols_keep.append("min")
    cols_keep.append(metric_col)

    log = log[[c for c in cols_keep if c in log.columns]].copy()
    log["Line"]   = float(line_value)
    log["Margin"] = pd.to_numeric(log[metric_col], errors="coerce") - float(line_value)
    log["Outcome"] = log["Margin"].apply(
        lambda v: "Above" if pd.notna(v) and v > 0 else
                  ("Below" if pd.notna(v) and v < 0 else "Push")
    )
    rename = {"game_date": "Date", "MATCHUP": "Matchup", "min": "Minutes",
              metric_col: metric_label}
    log = log.rename(columns=rename)
    if "Date" in log.columns:
        log = log.sort_values("Date", ascending=False)
    st.dataframe(
        log, hide_index=True, width="stretch",
        column_config={
            "Margin": st.column_config.NumberColumn(format="%+.1f"),
            "Line":   st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # Disclaimer
    st.markdown(
        f'<div class="hl-disclaimer-card"><strong>Note:</strong> {_LAB_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def page_diagnostics(roster: dict, api_key: str) -> None:
    page_hero(
        "Model diagnostics",
        "How well the trained models actually fit the held-out data — accuracy, "
        "ranking, residuals, and feature drivers.",
    )
    if not roster:
        empty_state("No players in roster",
                    "Add a player from the sidebar to run model diagnostics.")
        return
    bundle = _bundle_for_ui()
    modeling_df = _modeling_frame(_roster_key())

    if bundle.metrics is None or bundle.metrics.empty:
        empty_state("No metrics available",
                    "Train the model bundle first by loading some players in the sidebar.")
        return

    # Health summary
    metrics = bundle.metrics
    best_row = metrics.sort_values("R²", ascending=False).iloc[0]
    median_r2 = float(metrics["R²"].median())
    cols = st.columns(5)
    cols[0].metric("Best model (R²)", str(best_row["model"]),
                   delta=f"{float(best_row['R²']):.2f}")
    cols[1].metric("Median R²", f"{median_r2:.2f}")
    cols[2].metric("Models", len(bundle.estimators))
    cols[3].metric("Train rows", f"{bundle.n_train:,}")
    cols[4].metric("Test rows", f"{bundle.n_test:,}")

    tabs = st.tabs(["Overview", "Predicted vs actual", "Residuals",
                    "Feature importance", "Hyperparameters"])

    # ── Overview ────────────────────────────────────────────────────────────
    with tabs[0]:
        insight_card(
            "Test-set accuracy",
            "Higher R² and lower MAE/RMSE indicate a better-fit model. R² of 1.0 is "
            "perfect; values near 0 mean the model barely beats predicting the mean.",
            icon="i",
        )
        cols = st.columns([2, 3])
        with cols[0]:
            section("Test-set metrics")
            st.dataframe(metrics, width="stretch", hide_index=True)
        with cols[1]:
            section("R² by model")
            sorted_metrics = metrics.sort_values("R²", ascending=False)
            rows = "".join(
                _r2_row(row["model"], float(row["R²"]))
                for _, row in sorted_metrics.iterrows()
            )
            st.markdown(f'<div class="hl-card">{rows}</div>', unsafe_allow_html=True)

        section("Model ranking")
        rank_metric_options = [c for c in ("R²", "MAE", "RMSE") if c in metrics.columns]
        rank_metric = st.selectbox("Rank by", rank_metric_options, key="diag_rank_metric")
        st.plotly_chart(
            charts.model_metric_ranking_chart(metrics, metric=rank_metric),
            width="stretch",
        )

        section(
            "Context Feature Uplift",
            "This compares the baseline feature set against the best validated variant "
            "chosen from baseline, context, role, and blended RACE candidates. "
            "Positive uplift means the richer context actually helped on held-out data.",
        )
        uplift = bundle.uplift_report if hasattr(bundle, "uplift_report") else None
        if uplift is None or uplift.empty:
            empty_state(
                "No uplift report available",
                "Train the latest model bundle to compare baseline and selected-variant performance.",
            )
        else:
            show = uplift.copy()
            show = show.rename(
                columns={
                    "target": "target",
                    "selected_variant": "selected_variant",
                    "baseline_R²": "baseline_r2",
                    "selected_R²": "selected_r2",
                    "R²_uplift": "r2_uplift",
                    "baseline_MAE": "baseline_mae",
                    "selected_MAE": "selected_mae",
                    "features_added": "features_added",
                    "model_family": "model_family",
                }
            )
            keep = [
                "target",
                "selected_variant",
                "baseline_r2",
                "selected_r2",
                "r2_uplift",
                "baseline_mae",
                "selected_mae",
                "features_added",
                "model_family",
            ]
            keep = [c for c in keep if c in show.columns]
            st.dataframe(show[keep], width="stretch", hide_index=True)

    # ── Predicted vs actual + residuals share the same panel construction ──
    panels: list[dict] = []
    try:
        cmap = player_color_map(list(roster))
        panels = _diagnostics_panels(_roster_key(), cmap)
    except Exception as exc:
        st.warning(f"Could not rebuild prediction panels: {exc}")

    with tabs[1]:
        insight_card(
            "Predicted vs actual",
            "Each point is one held-out game. Points hugging the diagonal indicate "
            "well-calibrated projections; vertical scatter indicates noise.",
            icon="i",
        )
        if panels:
            st.plotly_chart(charts.predicted_vs_actual_grid(panels), width="stretch")
        else:
            empty_state("No prediction panels", "Could not reconstruct predictions for the test set.")

    with tabs[2]:
        insight_card(
            "Residual distributions",
            "Residual = actual − predicted. A distribution centered on zero with "
            "low spread suggests the model is unbiased and tight; long tails or "
            "shifts away from zero point to systematic error.",
            icon="i",
        )
        if panels:
            st.plotly_chart(charts.residual_distribution_chart(panels), width="stretch")
        else:
            empty_state("No residuals to display", "Prediction panels are unavailable.")

    with tabs[3]:
        insight_card(
            "Feature importance",
            "For tree-based models this shows feature_importances_ from the trained "
            "estimator. For linear models (ridge) the absolute coefficient magnitude "
            "is shown as a proxy — features are standardized during training so "
            "magnitudes are directly comparable.",
            icon="i",
        )

        def _extract_importance(model: Any, feats: list[str]) -> pd.Series | None:
            """Return a Series of feature importances for tree- or linear-based models."""
            if hasattr(model, "feature_importances_"):
                vals = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_, dtype=float)
                # Ridge can output (n_targets, n_features); collapse to per-feature magnitude.
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                vals = np.abs(coef)
            else:
                return None
            if len(vals) != len(feats):
                return None
            return pd.Series(vals, index=feats)

        def _importance_for(name: str) -> tuple[pd.Series | None, str]:
            """Resolve importance for the active estimator (Pipeline or RACE)."""
            est = bundle.estimators[name]
            # Direct sklearn Pipeline case
            if hasattr(est, "named_steps"):
                model = est.named_steps.get("model", None)
                series = _extract_importance(model, bundle.specs[name]["features"])
                if series is not None:
                    kind = "importance" if hasattr(model, "feature_importances_") else "|coef|"
                    return series, kind
            # RACE wrapper case — aggregate across components weighted by their weights.
            if hasattr(est, "components") and est.components:
                aggregated: dict[str, float] = {}
                kinds: set[str] = set()
                total_weight = 0.0
                for component in est.components:
                    inner_est, feats, weight = component[0], component[1], component[2]
                    inner_model = (
                        inner_est.named_steps.get("model", None)
                        if hasattr(inner_est, "named_steps") else inner_est
                    )
                    series = _extract_importance(inner_model, list(feats))
                    if series is None:
                        continue
                    kinds.add("importance" if hasattr(inner_model, "feature_importances_")
                              else "|coef|")
                    w = float(weight)
                    total_weight += w
                    for feat, val in series.items():
                        aggregated[feat] = aggregated.get(feat, 0.0) + w * float(val)
                if aggregated and total_weight > 0:
                    out = pd.Series(aggregated) / total_weight
                    kind = "importance" if kinds == {"importance"} else "|coef|"
                    return out, kind
            return None, ""

        importance_models = [n for n in bundle.estimators if _importance_for(n)[0] is not None]
        if not importance_models:
            empty_state(
                "No importance data",
                "None of the trained models expose feature importances or coefficients.",
            )
        else:
            picked = st.selectbox("Model", importance_models)
            series, kind = _importance_for(picked)
            if series is None or series.empty:
                empty_state("No importance data", "The selected model has no extractable importances.")
            else:
                title_suffix = "feature importance" if kind == "importance" else "coefficient magnitude"
                st.plotly_chart(
                    charts.feature_importance_bar(
                        series,
                        title=f"{picked} — {title_suffix}",
                    ),
                    width="stretch",
                )

    with tabs[4]:
        rows = [{"model": n, **bundle.best_params.get(n, {})} for n in bundle.estimators]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── Hooplytics Scout (chatbot) ──────────────────────────────────────────────
# 4-tuple: (icon, title, short_desc, full_prompt)
_CHAT_STARTERS: list[tuple[str, str, str, str]] = [
    (
        "\u25c6",
        "Top edges tonight",
        "Strongest projection-vs-line gaps, ranked by magnitude and confidence.",
        "What are the strongest projection-vs-line edges on tonight\u2019s slate? Show me the top 3 with edge magnitude and model confidence.",
    ),
    (
        "\u25c8",
        "Pick suggestion",
        "Structured MORE/LESS read with confidence rating and risk factors.",
        "Give me a MORE/LESS pick for the best edge tonight. Include a confidence rating and key risk factors.",
    ),
    (
        "\u25cb",
        "Player form",
        "5-game trend summary vs model projections for rostered players.",
        "Which rostered players are in their best recent form? Summarize 5-game trends vs model projections.",
    ),
    (
        "\u2248",
        "Model quality",
        "R\u00b2 scores per target with low-confidence flags.",
        "How reliable are the current models? Summarize R\u00b2 scores per target and flag any low-confidence areas.",
    ),
    (
        "\u2253",
        "Fade candidates",
        "Where the model disagrees with the market strongly enough to fade.",
        "Where does the model strongly disagree with the market in a way that suggests fading the posted line?",
    ),
    (
        "\u25cf",
        "Slate overview",
        "High-level analytics narrative for tonight\u2019s full slate.",
        "Give me a high-level narrative overview of tonight\u2019s slate from an analytics angle.",
    ),
]


def _chatbot_grounding(roster: dict, api_key: str) -> dict[str, Any]:
    """Assemble the grounding payload from current app state."""
    bundle = None
    edge_df = pd.DataFrame()
    projections: dict[str, pd.DataFrame] = {}
    try:
        bundle = _bundle_for_ui()
    except Exception:
        bundle = None
    try:
        edge_df = _build_edge_board(roster, api_key)
    except Exception:
        edge_df = pd.DataFrame()
    if bundle is not None and roster:
        store = _store()
        modeling_df = _modeling_frame(_roster_key())
        for player in list(roster.keys())[:8]:
            try:
                proj = project_next_game(
                    player,
                    bundle=bundle,
                    store=store,
                    modeling_df=modeling_df,
                )
                if isinstance(proj, pd.DataFrame) and not proj.empty:
                    projections[player] = proj
            except Exception:
                continue
    return build_grounding_payload(
        roster=roster,
        bundle=bundle,
        edge_df=edge_df if not edge_df.empty else None,
        projections=projections or None,
    )


def _scout_chart_figure(spec: dict) -> Any:
    """Build a Plotly figure from a validated Hooplytics Scout chart spec."""
    import plotly.graph_objects as go
    from hooplytics.web.styles import (
        COLOR_ACCENT,
        COLOR_AXIS,
        COLOR_GRID,
        COLOR_LESS,
        COLOR_MORE,
    )

    ctype = spec["type"]
    x = spec["x"]
    y = spec["y"]
    diverging = spec.get("diverging", False)

    if diverging:
        colors = [COLOR_MORE if v >= 0 else COLOR_LESS for v in y]
    else:
        colors = [COLOR_ACCENT] * len(y)

    fig = go.Figure()
    if ctype == "hbar":
        fig.add_trace(go.Bar(
            x=y, y=x, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.2f}" if diverging else f"{v:.2f}" for v in y],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_yaxes(autorange="reversed")
    elif ctype == "bar":
        fig.add_trace(go.Bar(
            x=x, y=y,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.2f}" if diverging else f"{v:.2f}" for v in y],
            textposition="outside",
            cliponaxis=False,
        ))
    elif ctype == "line":
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2.4),
            marker=dict(size=7, color=COLOR_ACCENT),
        ))
    else:  # scatter
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=10, color=colors if diverging else COLOR_ACCENT),
        ))

    if diverging:
        fig.add_hline(y=0, line=dict(color=COLOR_AXIS, width=1, dash="dot"))

    height = max(220, min(420, 38 * len(x) + 120)) if ctype == "hbar" else 320
    fig.update_layout(
        title=dict(text=spec.get("title", ""), x=0.0, xanchor="left", font=dict(size=14)),
        showlegend=False,
        height=height,
        margin=dict(t=44, b=36, l=12, r=12),
        xaxis=dict(title=spec.get("x_label", "") or None, gridcolor=COLOR_GRID),
        yaxis=dict(title=spec.get("y_label", "") or None, gridcolor=COLOR_GRID),
    )
    return fig


def _render_chat_message(role: str, body: str, evidence: list[str] | None = None) -> None:
    with st.chat_message(role):
        if role == "assistant":
            segments = parse_chart_blocks(body or "")
            rendered_any = False
            for i, seg in enumerate(segments):
                if seg["kind"] == "text":
                    text = str(seg.get("content", "")).strip()
                    if text:
                        st.markdown(text)
                        rendered_any = True
                else:
                    try:
                        fig = _scout_chart_figure(seg["spec"])
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config={"displayModeBar": False},
                            key=f"scout_chart_{role}_{id(body)}_{i}",
                        )
                        rendered_any = True
                    except Exception:
                        # Never let a malformed chart break the message render.
                        st.markdown(
                            f"_(chart could not be rendered: `{seg['spec'].get('title','')}`)_"
                        )
                        rendered_any = True
            if not rendered_any:
                st.markdown("_[No response content was returned by the model.]_")
        else:
            st.markdown(body or "")
        if evidence:
            st.markdown(
                '<div class="hl-chat-evidence">'
                + "".join(chip(c) for c in evidence)
                + "</div>",
                unsafe_allow_html=True,
            )


# ── Roster Report (PDF export) ──────────────────────────────────────────────
def _recent_form_for(player: str, games: pd.DataFrame, last_n: int = 10) -> dict[str, float]:
    """Compute a small dict of recent-form averages for the report player block."""
    if games is None or games.empty:
        return {}
    out: dict[str, float] = {}
    for col in ("pts", "reb", "ast", "pra", "fantasy_score", "min"):
        if col not in games.columns:
            continue
        s = pd.to_numeric(games[col], errors="coerce").dropna()
        if s.empty:
            continue
        out[col] = float(s.tail(last_n).mean())
    return out


# Mapping from Odds API market name to the box-score stat column used to score
# the line as an Over / Under outcome in the report's "Lines vs Outcomes" panel.
_HISTORY_MODEL_TO_STAT: dict[str, str] = {
    "points":   "pts",
    "rebounds": "reb",
    "assists":  "ast",
    "threepm":  "fg3m",
}


@st.cache_data(show_spinner=False, ttl=60 * 30, max_entries=4)
def _player_history_lines_vs_outcomes(roster_key: str, last_n: int = 10) -> dict[str, pd.DataFrame]:
    """Build per-player historical lines vs. actual outcomes for the PDF report.

    Joins cached pregame consensus lines (from the historical odds cache) onto
    each player's actual game-log results and labels each row Over / Under /
    Push. Returns ``{player: DataFrame}`` with the most recent ``last_n``
    resolved line/outcome rows per player. Players with no joinable history
    are omitted.
    """
    odds_df = load_cached_historical_odds()
    if odds_df.empty:
        return {}

    modeling_df = _modeling_frame(roster_key)
    if modeling_df.empty or "player" not in modeling_df.columns:
        return {}

    # Normalize join keys.
    odds = odds_df.copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce").dt.normalize()
    odds = odds[odds["model"].isin(_HISTORY_MODEL_TO_STAT)]

    games = modeling_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()

    # Canonicalize player names so 'Shai Gilgeous-Alexander' matches
    # 'Shai Gilgeous Alexander' between the two sources.
    import re as _re
    def _canon(s: str) -> str:
        return _re.sub(r"[^a-z]", "", str(s).lower())

    odds["_canon"] = odds["player"].map(_canon)
    games["_canon"] = games["player"].map(_canon)

    try:
        roster: dict = json.loads(roster_key)
    except Exception:
        roster = {}
    canon_to_roster = {_canon(p): p for p in roster.keys()}

    out: dict[str, pd.DataFrame] = {}
    for canon_name, roster_name in canon_to_roster.items():
        sub_odds = odds[odds["_canon"] == canon_name]
        sub_games = games[games["_canon"] == canon_name]
        if sub_odds.empty or sub_games.empty:
            continue

        rows: list[dict] = []
        for _, line_row in sub_odds.iterrows():
            stat_col = _HISTORY_MODEL_TO_STAT[line_row["model"]]
            if stat_col not in sub_games.columns:
                continue
            game = sub_games[sub_games["game_date"] == line_row["game_date"]]
            if game.empty:
                continue
            actual = pd.to_numeric(game[stat_col], errors="coerce").iloc[0]
            if pd.isna(actual):
                continue
            line = float(line_row["line"])
            margin = float(actual) - line
            if margin > 0:
                result = "Over"
            elif margin < 0:
                result = "Under"
            else:
                result = "Push"
            rows.append({
                "game_date": line_row["game_date"],
                "metric": str(line_row["model"]),
                "line": line,
                "actual": float(actual),
                "margin": margin,
                "result": result,
            })

        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("game_date", ascending=False).head(last_n)
        out[roster_name] = df.reset_index(drop=True)

    return out


def page_report(roster: dict, api_key: str) -> None:
    page_hero(
        "Roster Report",
        "Export a printable analytics report covering model quality, "
        "live edges, per-player projections, and AI-written rationale.",
    )

    if not roster:
        empty_state(
            "Roster is empty",
            "Add players in the sidebar to enable the report export.",
        )
        return

    # Resolve OpenAI connection (optional — report is also useful without prose).
    openai_api_key = (
        (st.session_state.get("session_openai_api_key") or "").strip()
        or _deployment_openai_api_key()
    )
    conn: OpenAIConnection | None = None
    if openai_api_key:
        conn = _resolve_openai_connection(openai_api_key)

    selected_model = st.session_state.get("openai_selected_model") or ""
    has_chat = bool(conn and selected_model)

    # ── Configuration card ─────────────────────────────────────────────────
    with st.container():
        cols = st.columns([2, 1, 1])
        include_ai = cols[0].toggle(
            "Include AI-written rationale (uses OpenAI)",
            value=has_chat,
            disabled=not has_chat,
            help=(
                "Generates an executive summary, slate outlook, and a paragraph "
                "of analyst-style rationale per player using your configured "
                "OpenAI key. One API call per export."
                if has_chat
                else "Add an OpenAI key in the sidebar to enable AI prose."
            ),
        )
        cols[1].markdown(
            (pill("AI READY", "live") if has_chat else pill("DATA-ONLY", "warn")),
            unsafe_allow_html=True,
        )
        cols[2].caption(
            f"Model · {selected_model}" if has_chat else "Add OpenAI key for prose"
        )

    # ── Preview / status ───────────────────────────────────────────────────
    bundle = _bundle_for_ui()
    edge_df = _build_edge_board(roster, api_key)

    overview_cols = st.columns(4)
    overview_cols[0].metric("Players", len(roster))
    overview_cols[1].metric(
        "Models trained",
        len(bundle.estimators) if bundle and bundle.estimators else 0,
    )
    overview_cols[2].metric(
        "Live edges",
        int(len(edge_df)) if isinstance(edge_df, pd.DataFrame) else 0,
    )
    median_r2 = (
        float(bundle.metrics["R²"].median())
        if bundle is not None and bundle.metrics is not None
        and "R²" in bundle.metrics.columns and not bundle.metrics.empty
        else float("nan")
    )
    overview_cols[3].metric(
        "Median R²",
        f"{median_r2:.2f}" if not np.isnan(median_r2) else "—",
    )

    divider()

    # ── Generate button ────────────────────────────────────────────────────
    generate = st.button(
        "Generate report",
        type="primary",
        width="stretch",
        key="report_generate_btn",
    )

    if not generate:
        st.caption(
            "The report is built on demand. Toggle AI rationale above to "
            "include analyst-style prose for the executive summary, slate "
            "outlook, and each player."
        )
        return

    # ── Build payload ──────────────────────────────────────────────────────
    store = _store()
    modeling_df = _modeling_frame(_roster_key())
    seasons = _active_training_seasons() or None

    projections: dict[str, pd.DataFrame] = {}
    recent_form: dict[str, dict[str, float]] = {}
    player_games: dict[str, pd.DataFrame] = {}
    progress = st.progress(0.0, text="Computing projections…")
    players = list(roster.keys())
    for i, player in enumerate(players, start=1):
        try:
            proj = project_next_game(
                player,
                bundle=bundle,
                store=store,
                modeling_df=modeling_df,
                seasons=seasons,
            )
            if isinstance(proj, pd.DataFrame) and not proj.empty:
                projections[player] = proj
        except Exception:
            pass
        try:
            games = _player_games(_roster_key(), player)
            recent_form[player] = _recent_form_for(player, games)
            if isinstance(games, pd.DataFrame) and not games.empty:
                player_games[player] = games
        except Exception:
            pass
        progress.progress(i / max(len(players), 1), text=f"Projection {i}/{len(players)}…")
    progress.empty()

    # ── AI prose (optional) ────────────────────────────────────────────────
    ai_sections: dict[str, Any] | None = None
    if include_ai and conn is not None and selected_model:
        with st.spinner("Generating AI rationale…"):
            try:
                grounding = build_grounding_payload(
                    roster=roster,
                    bundle=bundle,
                    edge_df=edge_df if isinstance(edge_df, pd.DataFrame) and not edge_df.empty else None,
                    projections=projections or None,
                )
                ai_sections = generate_report_sections(
                    connection=conn,
                    model=selected_model,
                    grounding_payload=grounding,
                )
            except Exception as exc:
                st.warning(f"AI rationale failed — exporting data-only report. ({exc})")
                ai_sections = None

    # ── Build PDF ──────────────────────────────────────────────────────────
    metrics_df = (
        bundle.metrics if bundle is not None and bundle.metrics is not None
        else None
    )
    try:
        pdf_bytes = build_pdf_report(
            roster={p: list(s) for p, s in roster.items()},
            bundle_metrics=metrics_df,
            edge_df=edge_df if isinstance(edge_df, pd.DataFrame) and not edge_df.empty else None,
            projections=projections or None,
            recent_form=recent_form or None,
            ai_sections=ai_sections,
            player_history=_player_history_lines_vs_outcomes(_roster_key()) or None,
            player_games=player_games or None,
        )
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
        return

    filename = f"hooplytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    st.success(f"Report ready · {len(pdf_bytes) / 1024:.1f} KB")
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        type="primary",
        width="stretch",
        key="report_download_btn",
    )


def page_chatbot(roster: dict, api_key: str) -> None:
    openai_api_key = (
        (st.session_state.get("session_openai_api_key") or "").strip()
        or _deployment_openai_api_key()
    )

    if not openai_api_key:
        # ── Beautiful landing state ──────────────────────────────────────────
        st.markdown(
            '<div class="hl-scout-landing">'
            '<div class="hl-scout-landing-mark">HS</div>'
            '<h2>Hooplytics Scout</h2>'
            '<p class="hl-scout-landing-sub">Your grounded NBA analytics assistant \u2014 '
            "anchored to live roster data, model projections, and the live edge board. "
            "Ask about pick confidence, player form, model quality, and more.</p>"
            '<div class="hl-scout-steps">'
            '<div class="hl-scout-step">'
            '<div class="hl-scout-step-num">1</div>'
            '<div class="hl-scout-step-text">Get an API key at '
            "<strong>platform.openai.com/api-keys</strong></div>"
            "</div>"
            '<div class="hl-scout-step">'
            '<div class="hl-scout-step-num">2</div>'
            '<div class="hl-scout-step-text">Paste it in the sidebar under '
            "<strong>Hooplytics Scout</strong> \u2192 click <strong>Connect</strong></div>"
            "</div>"
            '<div class="hl-scout-step">'
            '<div class="hl-scout-step-num">3</div>'
            '<div class="hl-scout-step-text">Return here and pick a starter prompt '
            "or type your own question below</div>"
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        divider()
        _wc1, _wc2, _wc3 = st.columns(3, gap="medium")
        with _wc1:
            insight_card("Edge analysis", "Ranked projection-vs-line gaps by strength and direction.", icon="\u25c6")
        with _wc2:
            insight_card("Pick suggestions", "Structured MORE/LESS with Confidence + Risk factors.", icon="\u25c8")
        with _wc3:
            insight_card("Model insight", "R\u00b2 quality checks and 5-game player form trends.", icon="\u25cb")
        return

    connection = _resolve_openai_connection(openai_api_key)
    if connection is None:
        err = st.session_state.get("openai_connect_error", "")
        st.markdown(
            '<div class="hl-scout-hero" style="border-color:rgba(255,107,107,0.28);'
            'background:linear-gradient(160deg,rgba(255,107,107,0.07),rgba(255,255,255,0));">'
            '<div class="hl-scout-hero-eyebrow" style="color:var(--hl-neg);">'
            "\u26a0 CONNECTION ERROR</div>"
            '<h2 style="font-size:1.3rem;color:var(--hl-neg);">Could not reach OpenAI</h2>'
            f"<p>{err or 'Click Connect in the sidebar to retry. Verify your API key has access.'}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    model = st.session_state.get("openai_selected_model", "") or (
        connection.default_model or ""
    )
    if not model:
        empty_state(
            "No chat model available",
            "Your key did not expose any chat-capable models. Try a different key.",
        )
        return

    payload = _chatbot_grounding(roster, api_key)
    chips = evidence_chips(payload)
    mode_label = "Strict" if st.session_state.get("chat_strict_grounded") else "Hybrid"

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hl-scout-hero">'
        '<div class="hl-scout-hero-eyebrow">'
        '<span class="hl-scout-pulse"></span>HOOPLYTICS SCOUT \u00b7 AI CONNECTED'
        "</div>"
        "<h2>What can I analyze for you?</h2>"
        "<p>Grounded on your roster, model metrics, projections, and live edge data. "
        "General reasoning is explicitly labeled as such.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Status bar ───────────────────────────────────────────────────────────
    model_display = (model[:35] + "\u2026") if len(model) > 36 else model
    chips_html = ""
    if chips:
        for _c in chips:
            chips_html += (
                '<span class="hl-scout-status-sep"></span>'
                f'<span class="hl-scout-status-val">{_c}</span>'
            )
    st.markdown(
        f'<div class="hl-scout-status">'
        f'<span class="hl-scout-status-lbl">Model</span>'
        f'<span class="hl-scout-status-val">{model_display}</span>'
        f'<span class="hl-scout-status-sep"></span>'
        f'<span class="hl-scout-status-lbl">Mode</span>'
        f'<span class="hl-scout-status-val">{mode_label}</span>'
        f"{chips_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Chat area ────────────────────────────────────────────────────────────
    history: list[dict[str, Any]] = st.session_state.chat_history
    if history:
        _, _ctrl = st.columns([6, 1])
        with _ctrl:
            if st.button("↺ Clear", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.session_state.chat_pending = ""
                st.rerun()
        for turn in history:
            _render_chat_message(
                turn.get("role", "user"),
                turn.get("content", ""),
                turn.get("evidence") or None,
            )
    else:
        st.markdown(
            '<p class="hl-scout-starters-label">Suggested prompts</p>',
            unsafe_allow_html=True,
        )
        _sc1, _sc2, _sc3 = st.columns(3, gap="medium")
        _sc_cols = [_sc1, _sc2, _sc3]
        for _si, (_icon, _title, _desc, _prompt) in enumerate(_CHAT_STARTERS):
            with _sc_cols[_si % 3]:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="hl-scout-sta-icon">{_icon}</div>'
                        f'<div class="hl-scout-sta-title">{_title}</div>'
                        f'<div class="hl-scout-sta-desc">{_desc}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Ask this \u2192", key=f"chat_starter_{_si}", width="stretch"):
                        st.session_state.chat_pending = _prompt
                        st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("How grounding works", expanded=False):
            st.markdown(
                "Local roster, model metrics, projections, and live edge rows are sent as "
                "authoritative context. Answers cite those values directly; anything beyond "
                "them is labeled **General context**. Pick suggestions include "
                "**What local data says**, **Confidence**, and **Risk factors** sections."
            )

    # ── Input ────────────────────────────────────────────────────────────────
    user_msg = st.chat_input("Ask Hooplytics Scout\u2026")
    effective_msg = user_msg or st.session_state.get("chat_pending", "")
    if effective_msg:
        st.session_state.chat_pending = ""
        history.append({"role": "user", "content": effective_msg})
        try:
            with st.spinner("Thinking\u2026"):
                reply = chat_complete(
                    connection=connection,
                    model=model,
                    user_message=effective_msg,
                    grounding_payload=payload,
                    history=history[:-1],
                    strict_grounded=bool(st.session_state.get("chat_strict_grounded")),
                )
        except Exception as exc:
            history.append(
                {
                    "role": "assistant",
                    "content": f"OpenAI error: {exc}",
                    "evidence": [],
                }
            )
        else:
            history.append(
                {"role": "assistant", "content": reply, "evidence": chips}
            )
        st.session_state.chat_history = history
        st.rerun()


# Page registry ───────────────────────────────────────────────────────────────
PAGES = {
    "Home":                page_home,
    "Player projection":   page_projection,
    "Analytics Dashboard": page_edge_board,
    "Compare players":     page_compare,
    "Player Line Lab":     page_scenario,
    "Model diagnostics":   page_diagnostics,
    "Hooplytics Scout":    page_chatbot,
    "Roster Report":       page_report,
}


def main() -> None:
    _init_state()
    page, api_key, _openai_api_key = _render_sidebar()
    PAGES[page](st.session_state.roster, api_key)


if __name__ == "__main__":
    main()
