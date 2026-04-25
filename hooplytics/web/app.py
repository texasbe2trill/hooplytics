"""Streamlit UI for Hooplytics — premium dashboard for the projection engine."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from hooplytics.constants import DEFAULT_ROSTER, MODEL_SPECS
from hooplytics.data import PlayerStore, nba_seasons
from hooplytics.models import ModelBundle, ensure_models
from hooplytics.odds import fetch_live_player_lines, load_api_key
from hooplytics.predict import (
    fantasy_decisions,
    predict_scenario,
    project_next_game,
)
from hooplytics.web import charts
from hooplytics.web.styles import (
    chip,
    disclaimer,
    divider,
    empty_state,
    inject_css,
    insight_card,
    kpi_grid,
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
@st.cache_resource(show_spinner=False)
def _store() -> PlayerStore:
    return PlayerStore()


@st.cache_data(show_spinner="Fetching season game logs…", ttl=60 * 60 * 6)
def _player_data(roster_key: str) -> pd.DataFrame:
    roster = json.loads(roster_key)
    return _store().load_player_data(roster)


@st.cache_resource(show_spinner="Training models…")
def _bundle(roster_key: str) -> ModelBundle:
    return ensure_models(_player_data(roster_key))


@st.cache_data(show_spinner=False, ttl=60 * 5)
def _live_lines(api_key: str, players_key: str, _bust: int = 0) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["player", "model", "line", "books", "matchup"])
    return fetch_live_player_lines(api_key, json.loads(players_key))


@st.cache_data(show_spinner=False)
def _modeling_frame(roster_key: str) -> pd.DataFrame:
    return _store().modeling_frame(_player_data(roster_key))


@st.cache_data(show_spinner=False)
def _player_games(roster_key: str, player: str) -> pd.DataFrame:
    df = _player_data(roster_key)
    return df[df["player"] == player].sort_values("game_date").reset_index(drop=True)


# State ───────────────────────────────────────────────────────────────────────
def _default_seasons() -> list[str]:
    today = datetime.now().date()
    start = today.year - (1 if today.month >= 10 else 2)
    return nba_seasons(start, today.year + 1)


def _init_state() -> None:
    if "roster" not in st.session_state:
        seasons = _default_seasons()
        st.session_state.roster = {p: list(seasons) for p in DEFAULT_ROSTER}
    if "live_bust" not in st.session_state:
        st.session_state.live_bust = 0


def _roster_key() -> str:
    roster = st.session_state.roster
    return json.dumps({p: list(s) for p, s in sorted(roster.items())}, sort_keys=True)


# Edge board builder ──────────────────────────────────────────────────────────
def _build_edge_board(roster: dict, api_key: str) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    players = list(roster)
    live = _live_lines(api_key, json.dumps(players), st.session_state.live_bust)
    if live.empty:
        return pd.DataFrame()

    bundle = _bundle(_roster_key())
    store = _store()
    modeling_df = _modeling_frame(_roster_key())

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
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["abs_edge"] = df["edge"].abs()
    df["side"] = np.where(df["edge"] > 0, "MORE", "LESS")
    return df.sort_values("abs_edge", ascending=False).reset_index(drop=True)


# Sidebar ─────────────────────────────────────────────────────────────────────
def _render_sidebar() -> tuple[str, str]:
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
            cols = st.columns([5, 1])
            cols[0].markdown(f"<div style='padding-top:0.4rem'>{player}</div>",
                             unsafe_allow_html=True)
            if cols[1].button("×", key=f"rm_{player}", help=f"Remove {player}"):
                roster.pop(player)
                st.rerun()

        new_player = st.text_input("Add player", placeholder="e.g. LeBron James",
                                   label_visibility="collapsed")
        seasons_csv = st.text_input("Seasons", value="2024-25,2025-26",
                                    label_visibility="collapsed",
                                    help="Comma-separated NBA seasons")
        if st.button("Add", width="stretch") and new_player.strip():
            seasons = [s.strip() for s in seasons_csv.split(",") if s.strip()]
            try:
                resolved = PlayerStore.resolve_player_name(new_player) or new_player
                roster[resolved] = seasons
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        divider()

        # API key
        st.markdown('<p class="hl-section">Live odds</p>', unsafe_allow_html=True)
        env_key = load_api_key() or ""
        api_key = st.text_input(
            "Odds API key", value=env_key, type="password",
            label_visibility="collapsed",
            placeholder="paste ODDS_API_KEY",
        )
        cols = st.columns(2)
        if cols[0].button("Refresh", width="stretch"):
            st.session_state.live_bust += 1
            st.rerun()
        cols[1].markdown(
            pill("LIVE", "live") if api_key else pill("OFFLINE", "warn"),
            unsafe_allow_html=True,
        )

    return page, api_key


# ── Pages ────────────────────────────────────────────────────────────────────


def page_home(roster: dict, api_key: str) -> None:
    bundle = _bundle(_roster_key())
    modeling_df = _modeling_frame(_roster_key())

    # Brand row
    last_refresh = datetime.now().strftime("%b %d · %H:%M")
    status = (
        f'<span class="hl-status-dot"></span>LIVE LINES · UPDATED {last_refresh}'
        if api_key else
        f'<span class="hl-status-dot" style="background:#f5b041;box-shadow:0 0 0 3px rgba(245,176,65,0.18)"></span>'
        f'OFFLINE · ADD ODDS API KEY FOR LIVE LINES'
    )
    st.markdown(
        f'<div class="hl-brand-row">'
        f'<div class="hl-brand"><span class="hl-brand-mark"></span>HOOPLYTICS</div>'
        f'<div class="hl-status">{status}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Tagline
    st.markdown(
        '<div class="hl-hero-wrap">'
        '<h1 class="hl-tagline">Today\'s slate, analyzed.</h1>'
        '<p class="hl-tagline-sub">A precision view of player form, projection gaps, '
        'model confidence, and matchup context.</p>'
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
    for player, seasons in roster.items():
        games = (modeling_df["player"] == player).sum()
        color = cmap[player]
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
    bundle = _bundle(_roster_key())
    modeling_df = _modeling_frame(_roster_key())
    store = _store()

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
            proj = project_next_game(
                st.session_state.last_proj_player,
                bundle=bundle, store=store,
                last_n=st.session_state.last_proj_n,
                modeling_df=modeling_df,
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
        f'<div class="hl-status" style="margin-top:0.9rem">{status}</div>'
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
    cards.append(_kpi_card("Last updated", last_refresh,
                           "refresh from the sidebar to pull new lines"))

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
        "books": "Books", "matchup": "Matchup",
    }
    display = display.rename(columns=rename)
    keep = [c for c in (
        "Player", "Metric", "Direction", "Signal",
        "Current line", "Model projection", "Gap", "|Gap|", "Gap %",
        "Books", "Matchup",
    ) if c in display.columns]
    st.dataframe(display[keep], width="stretch", hide_index=True)


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


def page_scenario(roster: dict, api_key: str) -> None:
    page_hero(
        "Scenario lab",
        "Hypothetical box-score inputs in, model projections out — see how every "
        "model responds to a change in usage, efficiency, or pace.",
    )
    bundle = _bundle(_roster_key())
    modeling_df = _modeling_frame(_roster_key())

    insight_card(
        "How this works",
        "Sliders below are grouped by analytical category. Use a player's recent "
        "median as a baseline, then adjust any inputs to explore counterfactuals. "
        "Each model produces an independent projection from the same scenario.",
        icon="i",
    )

    cols = st.columns([2, 1])
    base_player = cols[0].selectbox(
        "Preset baseline",
        ["(zeros)"] + list(roster),
        help="Fill the inputs with this player's recent (last-10) median.",
    )
    if cols[1].button("Reset all inputs", width="stretch"):
        for k in list(st.session_state):
            if k.startswith("scn_"):
                del st.session_state[k]
        st.rerun()

    if base_player != "(zeros)":
        rows = modeling_df[modeling_df["player"] == base_player].tail(10)
        defaults = rows.median(numeric_only=True).to_dict() if not rows.empty else {}
    else:
        defaults = {}

    feature_set: set[str] = set()
    for spec in MODEL_SPECS.values():
        feature_set.update(spec["features"])
    features = sorted(feature_set)
    grouped = _classify_scenario_features(features)

    scenario: dict[str, float] = {}
    for label, group_feats in grouped:
        with st.expander(label, expanded=(label == "Scoring")):
            grid_cols = st.columns(3)
            for i, feat in enumerate(group_feats):
                col = grid_cols[i % 3]
                default = float(defaults.get(feat, 0.0))
                scenario[feat] = col.number_input(
                    feat, value=round(default, 2), step=0.5, key=f"scn_{feat}",
                )

    df = predict_scenario(scenario, bundle)
    if df.empty:
        empty_state("No matching models",
                    "No models matched the supplied features — check the inputs above.")
        return

    if {"model", "prediction"}.issubset(df.columns):
        tiles = [
            mini_kpi(str(r["model"]), f"{float(r['prediction']):.1f}")
            for _, r in df.iterrows()
        ]
        st.markdown(
            f'<div class="hl-card">'
            f'<p class="hl-card-title">Scenario projections</p>'
            f'<div class="hl-kpi-grid">{"".join(tiles)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.scenario_output_bar(df), width="stretch")

    with st.expander("Full prediction table", expanded=False):
        st.dataframe(df, width="stretch", hide_index=True)


def page_diagnostics(roster: dict, api_key: str) -> None:
    page_hero(
        "Model diagnostics",
        "How well the trained models actually fit the held-out data — accuracy, "
        "ranking, residuals, and feature drivers.",
    )
    bundle = _bundle(_roster_key())
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

    # ── Predicted vs actual + residuals share the same panel construction ──
    panels: list[dict] = []
    try:
        from sklearn.model_selection import train_test_split
        cmap = player_color_map(list(roster))
        _, test = train_test_split(modeling_df, test_size=0.2, random_state=123)
        for name, est in bundle.estimators.items():
            spec = bundle.specs[name]
            X = test[spec["features"]]
            y = test[spec["target"]].to_numpy()
            yhat = est.predict(X)
            r2 = float(metrics.loc[metrics["model"] == name, "R²"].iloc[0])
            points = [
                {
                    "player": p, "date": str(d.date()) if pd.notna(d) else "",
                    "matchup": m or "", "actual": float(a), "pred": float(yh),
                    "color": cmap.get(p, "#888"),
                }
                for p, d, m, a, yh in zip(
                    test["player"],
                    test.get("game_date", pd.Series([pd.NaT] * len(test))),
                    test.get("MATCHUP", pd.Series([""] * len(test))),
                    y, yhat,
                )
            ]
            panels.append({"metric": name, "r2": r2, "points": points})
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
            "For tree-based models, this shows which input features the model relied "
            "on most when making predictions on the training data.",
            icon="i",
        )
        rf_models = [n for n, est in bundle.estimators.items()
                     if hasattr(est.named_steps.get("model", None), "feature_importances_")]
        if not rf_models:
            empty_state("No tree-based models",
                        "None of the trained models expose feature importances.")
        else:
            picked = st.selectbox("Model", rf_models)
            est = bundle.estimators[picked]
            feats = bundle.specs[picked]["features"]
            importances = pd.Series(est.named_steps["model"].feature_importances_, index=feats)
            st.plotly_chart(
                charts.feature_importance_bar(importances,
                                              title=f"Feature importance — {picked}"),
                width="stretch",
            )

    with tabs[4]:
        rows = [{"model": n, **bundle.best_params.get(n, {})} for n in bundle.estimators]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# Page registry ───────────────────────────────────────────────────────────────
PAGES = {
    "Home":                page_home,
    "Player projection":   page_projection,
    "Analytics Dashboard": page_edge_board,
    "Compare players":     page_compare,
    "Scenario lab":        page_scenario,
    "Model diagnostics":   page_diagnostics,
}


def main() -> None:
    _init_state()
    page, api_key = _render_sidebar()
    PAGES[page](st.session_state.roster, api_key)


if __name__ == "__main__":
    main()
