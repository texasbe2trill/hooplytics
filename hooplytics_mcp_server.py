"""
hooplytics_mcp_server.py
━━━━━━━━━━━━━━━━━━━━━━━━
Hooplytics MCP Server — exposes the full projection + analytics engine
to Claude Desktop (and any MCP-compatible client) as callable tools.

Install deps:
    pip install "mcp[cli]" hooplytics

Run (stdio mode for Claude Desktop):
    python hooplytics_mcp_server.py

Run (SSE mode for remote / web clients):
    python hooplytics_mcp_server.py --transport sse --port 8765
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Ensure we're in the repo root so relative paths (data/, .hooplytics_cache/) resolve correctly
# Claude Desktop may launch the server with cwd not set to the repo
os.chdir(Path(__file__).parent)

import pandas as pd

from mcp.server.fastmcp import FastMCP

from hooplytics import (
    PlayerStore,
    ensure_models,
    project_next_game,
    predict_scenario,
    custom_prop,
    fantasy_decisions,
    fetch_live_player_lines,
    load_api_key,
    DEFAULT_ROSTER,
    MODEL_SPECS,
)
from hooplytics.data import nba_seasons
from hooplytics.models import load_models
from hooplytics.ai_agent import (
    connect as ai_connect,
    chat_complete,
    generate_slate_brief,
)


_DEFAULT_ROSTER_PATH = Path.home() / ".hooplytics" / "roster.json"
_ROSTER_PATH = Path(os.environ.get("HOOPLYTICS_ROSTER_PATH", _DEFAULT_ROSTER_PATH))
_PREBUILT_BUNDLE = Path(
    os.environ.get("HOOPLYTICS_PREBUILT_BUNDLE", "bundles/race_fast.joblib")
)


_store: PlayerStore | None = None
_bundle = None


def _default_seasons() -> list[str]:
    """Current + previous NBA season, matching the CLI bootstrap logic."""
    from datetime import date

    today = date.today()
    if today.month >= 10:
        start_year = today.year - 1
        end_exclusive = today.year + 1
    else:
        start_year = today.year - 2
        end_exclusive = today.year
    return nba_seasons(start_year, end_exclusive)


def _default_roster() -> dict[str, list[str]]:
    seasons = _default_seasons()
    return {p: list(seasons) for p in DEFAULT_ROSTER}


def _bootstrap():
    """Lazy-init the PlayerStore and model bundle.

    Prefers the shipped prebuilt bundle to avoid a slow first-call train.
    Falls back to ``ensure_models`` over the tracked roster if the prebuilt
    file isn't present.
    """
    global _store, _bundle
    if _store is None:
        _store = PlayerStore()
    if _bundle is None:
        if _PREBUILT_BUNDLE.exists():
            _bundle = load_models(_PREBUILT_BUNDLE)
        else:
            roster = _load_roster()
            player_data = _store.load_player_data(roster)
            _bundle = ensure_models(player_data)
    return _store, _bundle


def _load_roster() -> dict[str, list[str]]:
    """Load the shared CLI/MCP roster as ``{name: [seasons]}``.

    Tolerates the legacy MCP list-of-names format and converts it on read.
    """
    if _ROSTER_PATH.exists():
        try:
            data = json.loads(_ROSTER_PATH.read_text())
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            return {k: list(v) for k, v in data.items()}
        if isinstance(data, list) and data:
            seasons = _default_seasons()
            return {name: list(seasons) for name in data}
    return _default_roster()


def _save_roster(roster: dict[str, list[str]]) -> None:
    _ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ROSTER_PATH.write_text(json.dumps(roster, indent=2))


def _projection_dict(proj_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert a project_next_game DataFrame into a model-keyed dict."""
    if proj_df is None or proj_df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in proj_df.iterrows():
        out[row["model"]] = {
            "projection": float(row["prediction"]),
            "target": row.get("target"),
            "uses": row.get("uses"),
        }
    return out


def _ai_connection_or_error(provider: str):
    """Return (connection, model, None) on success or (None, None, error_str)."""
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return None, None, (
            f"{api_key_env} not set. Add your API key to the environment or .env file."
        )
    try:
        conn = ai_connect(api_key, provider)
    except Exception as exc:
        return None, None, f"AI connection failed: {exc}"
    model = conn.default_model
    if not model:
        return None, None, f"No usable {provider} chat models available for this key."
    return conn, model, None


mcp = FastMCP(
    name="hooplytics",
    instructions=(
        "NBA player intelligence workbench. "
        "Project next-game stats, compare against live lines, analyse props, "
        "score hypothetical scenarios, and generate AI analytical prose — "
        "all grounded in Hooplytics' RACE ML models and The Odds API."
    ),
)


@mcp.tool()
def project_player(
    player_name: str,
    last_n: int = 5,
) -> dict[str, Any]:
    """
    Project a player's next-game stats across all 8 Hooplytics RACE models
    (points, rebounds, assists, PRA, 3PM, stl+blk, turnovers, fantasy score).

    Returns a dict keyed by stat name with each model's projection, target column,
    and feature inputs.

    Args:
        player_name: Full or partial NBA player name (fuzzy-matched).
        last_n: Number of recent games used for the rolling-window features (default 5).
    """
    store, bundle = _bootstrap()
    proj_df = project_next_game(player_name, bundle=bundle, store=store, last_n=last_n)
    return {
        "player": player_name,
        "last_n": last_n,
        "projections": _projection_dict(proj_df),
    }


@mcp.tool()
def analyze_prop(
    player_name: str,
    stat: str,
    line: float | None = None,
    last_n: int = 5,
    confidence_margin: float = 0.10,
) -> dict[str, Any]:
    """
    Compare a player's model projection against a sportsbook line for a single stat.
    Returns the projected value, MORE/LESS call, edge gap, historical hit-rate,
    and model confidence metadata.

    Args:
        player_name: Full or partial NBA player name.
        stat: One of: points, rebounds, assists, pra, threepm, stl_blk, turnovers, fantasy_score.
        line: The sportsbook line threshold. If omitted, auto-fetched from The Odds API.
        last_n: Recent-game window for rolling features.
        confidence_margin: Minimum edge (fraction of line) required to signal MORE/LESS.
    """
    if stat not in MODEL_SPECS:
        return {
            "error": f"Unknown stat '{stat}'. Valid options: {', '.join(MODEL_SPECS)}"
        }
    store, bundle = _bootstrap()
    odds_key = load_api_key() or None
    result = custom_prop(
        player_name,
        stat,
        line=line,
        bundle=bundle,
        store=store,
        odds_api_key=odds_key,
        last_n=last_n,
        confidence_margin=confidence_margin,
    )
    return result


@mcp.tool()
def player_decisions(
    player_name: str,
    last_n: int = 5,
    use_live_lines: bool = True,
) -> list[dict[str, Any]]:
    """
    Return an 8-stat decision table for a player — projection, live line (if available),
    signed edge, MORE/LESS call, and model-vs-market gap for every tracked stat.

    Args:
        player_name: Full or partial NBA player name.
        last_n: Recent-game window for rolling features.
        use_live_lines: If True and ODDS_API_KEY is set, fetch live lines before comparing.
    """
    store, bundle = _bootstrap()
    live_lines: dict[str, float] | None = None
    if use_live_lines:
        odds_key = load_api_key() or None
        if odds_key:
            ll = fetch_live_player_lines(odds_key, [player_name])
            if not ll.empty:
                live_lines = dict(zip(ll["model"], ll["line"]))

    df = fantasy_decisions(
        player_name, bundle=bundle, store=store, last_n=last_n, live_lines=live_lines
    )
    return df.to_dict(orient="records")


@mcp.tool()
def score_scenario(scenario_json: str) -> dict[str, Any]:
    """
    Score a hypothetical box-score row through the RACE models.
    Useful for "what-if" analysis — e.g. what happens to fantasy score if a player
    shoots 55% from the field in 36 minutes?

    Args:
        scenario_json: JSON string of feature overrides. Recognised keys include:
            fgm, fga, fg3m, ftm, min, fg_pct, ft_pct, oreb, dreb, ast, stl, blk, tov.
            Example: '{"fgm":9,"fga":16,"fg3m":3,"ftm":5,"min":36}'
    """
    try:
        row = json.loads(scenario_json)
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON: {exc}"}

    _, bundle = _bootstrap()
    df = predict_scenario(row, bundle=bundle)
    if df is None or df.empty:
        return {
            "scenario": row,
            "predictions": {},
            "message": "No model had all required features in the scenario.",
        }
    predictions = {
        r["model"]: {
            "prediction": float(r["prediction"]),
            "target": r.get("target"),
            "uses": r.get("uses"),
        }
        for r in df.to_dict(orient="records")
    }
    return {"scenario": row, "predictions": predictions}


@mcp.tool()
def live_line_board(
    players: list[str] | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Fetch the live sportsbook line board for a list of players (or the full tracked roster).
    Returns consensus lines, book counts, and stat labels — sorted by projection gap.

    Requires ODDS_API_KEY to be set in the environment (or .env file).

    Args:
        players: Player names to query. Defaults to the full tracked roster.
        refresh: Force a fresh fetch even if lines are cached for this session.
    """
    odds_key = load_api_key() or None
    if not odds_key:
        return [{"error": "ODDS_API_KEY not set. Add it to your .env file."}]

    target_players = players or list(_load_roster().keys())
    try:
        ll = fetch_live_player_lines(odds_key, target_players, force_refresh=refresh)
    except Exception as exc:
        return [{"error": f"Odds API request failed: {exc}"}]
    if ll.empty:
        return [{"message": "No live lines found for the requested players."}]

    store, bundle = _bootstrap()
    rows = []
    proj_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in ll.iterrows():
        player = row["player"]
        try:
            if player not in proj_cache:
                proj_df = project_next_game(player, bundle=bundle, store=store)
                proj_cache[player] = _projection_dict(proj_df)
            stat_key = row["model"]
            stat_proj = proj_cache[player].get(stat_key)
            projected = stat_proj["projection"] if stat_proj else None
            edge = round(projected - row["line"], 3) if projected is not None else None
            call = None if edge is None else ("MORE" if edge > 0 else "LESS")
        except Exception:
            projected, edge, call = None, None, None

        rows.append({
            "player": player,
            "stat": row["model"],
            "line": float(row["line"]) if row.get("line") is not None else None,
            "books": int(row["books"]) if row.get("books") is not None else None,
            "projection": projected,
            "edge": edge,
            "call": call,
        })

    rows.sort(key=lambda r: abs(r.get("edge") or 0), reverse=True)
    return rows


@mcp.tool()
def generate_scout_report(
    player_name: str,
    stat: str = "points",
    provider: str = "anthropic",
    mode: str = "hybrid",
    last_n: int = 5,
) -> str:
    """
    Generate NBA-style analytical prose for a player + stat using the Hooplytics
    AI layer. Grounds the response in projection data, live line context,
    recent form, and model diagnostics.

    Args:
        player_name: Full or partial NBA player name.
        stat: Primary stat to focus the report on (default: points).
        provider: LLM provider — 'anthropic' or 'openai'.
        mode: Grounding mode — 'hybrid' (AI adds context) or 'strict' (data only).
        last_n: Recent-game window for grounding payload.
    """
    store, bundle = _bootstrap()

    try:
        proj_df = project_next_game(player_name, bundle=bundle, store=store, last_n=last_n)
    except Exception as exc:
        return f"Error fetching projection for {player_name}: {exc}"

    projections = _projection_dict(proj_df)

    odds_key = load_api_key() or None
    live_line = None
    if odds_key:
        ll = fetch_live_player_lines(odds_key, [player_name])
        if not ll.empty:
            row = ll[ll["model"] == stat]
            if not row.empty:
                live_line = float(row.iloc[0]["line"])

    grounding = {
        "player": player_name,
        "stat_focus": stat,
        "projection": projections.get(stat, {}),
        "live_line": live_line,
        "all_projections": projections,
        "mode": mode,
    }

    conn, model, err = _ai_connection_or_error(provider)
    if err:
        return err

    user_message = (
        f"Write a concise NBA scouting report on {player_name} focused on "
        f"{stat}. Compare the Hooplytics projection to the live line (if "
        f"present), call out the bigger edges across the other tracked stats, "
        f"and end with a clear MORE/LESS lean for {stat}. Keep it editorial "
        f"and grounded in the LOCAL CONTEXT data."
    )
    try:
        return chat_complete(
            connection=conn,
            model=model,
            user_message=user_message,
            grounding_payload=grounding,
            strict_grounded=(mode == "strict"),
            max_output_tokens=1200,
        )
    except Exception as exc:
        return f"Scout report generation failed: {exc}"


@mcp.tool()
def player_analytics(
    player_name: str,
    seasons: list[str] | None = None,
    last_n: int = 10,
) -> dict[str, Any]:
    """
    Return rich analytical context for a player — recent-game log (last N games),
    season averages, rolling trend vs season baseline, and volatility metrics.

    Args:
        player_name: Full or partial NBA player name.
        seasons: List of seasons in 'YYYY-YY' format (default: current + previous season).
        last_n: Number of recent games to return in the game log.
    """
    store, _ = _bootstrap()
    target_seasons = seasons or _default_seasons()

    try:
        df = store.load_player_data({player_name: list(target_seasons)})
    except Exception as exc:
        return {"error": str(exc)}

    if df is None or df.empty:
        return {"error": f"No game log data found for '{player_name}'."}

    df = df.sort_values("game_date", ascending=False)

    core_cols = ["game_date", "MATCHUP", "min", "pts", "reb", "ast",
                 "fg3m", "stl", "blk", "tov", "fg_pct", "ft_pct"]
    available = [c for c in core_cols if c in df.columns]
    recent = df[available].head(last_n).copy()
    if "game_date" in recent.columns:
        recent["game_date"] = recent["game_date"].astype(str)

    season_avgs = {
        col: round(float(df[col].mean()), 2)
        for col in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
        if col in df.columns
    }
    recent_avgs = {
        col: round(float(df[col].head(last_n).mean()), 2)
        for col in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
        if col in df.columns
    }
    volatility = {
        col: round(float(df[col].head(last_n).std()), 2)
        for col in ["pts", "reb", "ast"]
        if col in df.columns
    }

    trend = {
        stat: {
            "recent_avg": recent_avgs.get(stat),
            "season_avg": season_avgs.get(stat),
            "delta": round(
                (recent_avgs.get(stat, 0) or 0) - (season_avgs.get(stat, 0) or 0), 2
            ),
            "trending": (
                "above" if (recent_avgs.get(stat, 0) or 0) > (season_avgs.get(stat, 0) or 0)
                else "below"
            ),
        }
        for stat in ["pts", "reb", "ast"]
        if stat in df.columns
    }

    return {
        "player": player_name,
        "games_in_sample": len(df),
        "seasons_loaded": list(target_seasons),
        "season_averages": season_avgs,
        "recent_averages": recent_avgs,
        "volatility_last_n": volatility,
        "trend_vs_baseline": trend,
        "recent_game_log": recent.to_dict(orient="records"),
    }


@mcp.tool()
def roster_manage(
    action: str,
    player_name: str | None = None,
) -> dict[str, Any]:
    """
    View or update the tracked roster of players used by Hooplytics tools.

    The roster is shared with the Hooplytics CLI at ``~/.hooplytics/roster.json``
    (override with the ``HOOPLYTICS_ROSTER_PATH`` environment variable). Each
    entry is ``{player_name: [seasons]}``; new players default to the current +
    previous NBA season.

    Args:
        action: One of 'list', 'add', 'remove', or 'reset'.
        player_name: Required for 'add' and 'remove' actions.
    """
    roster = _load_roster()

    if action == "list":
        return {"roster": roster, "count": len(roster)}

    if action == "reset":
        default = _default_roster()
        _save_roster(default)
        return {"message": "Roster reset to default.", "roster": default}

    if action in ("add", "remove") and not player_name:
        return {"error": f"'player_name' is required for action '{action}'."}

    if action == "add":
        if player_name in roster:
            return {"message": f"'{player_name}' is already on the roster.", "roster": roster}
        roster[player_name] = _default_seasons()
        _save_roster(roster)
        return {"message": f"Added '{player_name}'.", "roster": roster}

    if action == "remove":
        if player_name not in roster:
            return {"message": f"'{player_name}' not found in roster.", "roster": roster}
        del roster[player_name]
        _save_roster(roster)
        return {"message": f"Removed '{player_name}'.", "roster": roster}

    return {"error": f"Unknown action '{action}'. Use: list, add, remove, reset."}


@mcp.tool()
def slate_brief(
    provider: str = "anthropic",
    max_players: int = 8,
) -> str:
    """
    Generate a one-paragraph AI Slate Brief — tonight's loudest mispricings across
    the tracked roster, written in editorial NBA analytics style.

    Automatically pulls live lines, runs projections for each player, and feeds
    the edge board to the AI layer for a concise daily read.

    Args:
        provider: LLM provider — 'anthropic' or 'openai'.
        max_players: Maximum number of roster players to include (default 8).
    """
    odds_key = load_api_key() or None
    if not odds_key:
        return "ODDS_API_KEY not set — Slate Brief requires live line data."

    roster = list(_load_roster().keys())[:max_players]
    store, bundle = _bootstrap()

    edge_board: list[dict[str, Any]] = []
    for player in roster:
        try:
            ll = fetch_live_player_lines(odds_key, [player])
            if ll.empty:
                continue
            proj_df = project_next_game(player, bundle=bundle, store=store)
            projections = _projection_dict(proj_df)
            for _, row in ll.iterrows():
                stat = row["model"]
                stat_proj = projections.get(stat)
                if not stat_proj:
                    continue
                projected = stat_proj["projection"]
                edge = round(projected - float(row["line"]), 3)
                edge_board.append({
                    "player": player,
                    "stat": stat,
                    "projection": projected,
                    "line": float(row["line"]),
                    "edge": edge,
                    "call": "MORE" if edge > 0 else "LESS",
                })
        except Exception:
            continue

    if not edge_board:
        return "No edge board data available for the current slate."

    edge_board.sort(key=lambda r: abs(r["edge"]), reverse=True)

    conn, model, err = _ai_connection_or_error(provider)
    if err:
        raw = "\n".join(
            f"  {r['player']} {r['stat']}: proj {r['projection']} vs line {r['line']} "
            f"({r['call']}, edge {r['edge']:+.2f})"
            for r in edge_board[:5]
        )
        return f"{err}\n\nRaw edge board (top 5):\n{raw}"

    grounding = {
        "edge_board": edge_board[:10],
        "edges_summary": {
            "rows": len(edge_board),
            "side_counts": {
                side: sum(1 for r in edge_board if r["call"] == side)
                for side in ("MORE", "LESS")
            },
        },
    }
    try:
        return generate_slate_brief(
            connection=conn,
            model=model,
            grounding_payload=grounding,
        )
    except Exception as exc:
        return f"Slate brief generation failed: {exc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hooplytics MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (stdio for Claude Desktop, sse for remote clients).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE transport (default: 8765).",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
