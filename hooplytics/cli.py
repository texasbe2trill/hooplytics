"""Hooplytics CLI — Typer + Rich.

Commands:
    project   — project a player's next game from recent form
    prop      — single MORE/LESS prop call (auto-fetches line)
    decisions — full 8-stat decision table
    scenario  — predict from a hypothetical box-score JSON
    lines     — today's live lines + edge-sorted decisions
    roster    — manage your persisted roster
    train     — pre-warm the model cache
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import (
    DEFAULT_ROSTER,
    PlayerStore,
    custom_prop,
    ensure_models,
    fantasy_decisions,
    fetch_live_player_lines,
    ingest_historical_odds,
    load_api_key,
    nba_seasons,
    predict_scenario,
    project_next_game,
)
from .constants import MODEL_SPECS

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="🏀 Hooplytics — NBA player analytics and live line context.",
    no_args_is_help=True,
)
roster_app = typer.Typer(help="Manage your tracked roster.", no_args_is_help=True)
app.add_typer(roster_app, name="roster")

console = Console()
err_console = Console(stderr=True)

ROSTER_PATH = Path.home() / ".hooplytics" / "roster.json"


# ── Roster persistence ──────────────────────────────────────────────────────
def _load_roster() -> dict[str, list[str]]:
    if ROSTER_PATH.exists():
        try:
            data = json.loads(ROSTER_PATH.read_text())
            if isinstance(data, dict) and data:
                return {k: list(v) for k, v in data.items()}
        except Exception:  # noqa: BLE001
            pass
    from datetime import date

    today = date.today()
    start = today.year - (1 if today.month >= 10 else 2)
    # end is the exclusive start-year upper bound for nba_seasons:
    # in Oct+ we are in a new season (e.g. Oct 2026 → include 2026-27)
    # otherwise the current season's start year is today.year - 1
    end = today.year + 1 if today.month >= 10 else today.year
    seasons = nba_seasons(start, end)
    return {p: seasons for p in DEFAULT_ROSTER}


def _save_roster(roster: dict[str, list[str]]) -> None:
    ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROSTER_PATH.write_text(json.dumps(roster, indent=2))


# ── Shared bootstrap ────────────────────────────────────────────────────────
def _resolve_player(store: PlayerStore, query: str) -> str:
    """Fuzzy-resolve player name; exit on failure."""
    name = store.resolve_player_name(query)
    if name is None:
        err_console.print(f"[red]✗ No NBA player matches '{query}'.[/red]")
        raise typer.Exit(1)
    if name.lower() != query.lower():
        console.print(f"[dim]→ resolved '{query}' to '{name}'[/dim]")
    return name


def _bootstrap(verbose: bool = False) -> tuple[PlayerStore, "ModelBundle", dict[str, list[str]]]:  # noqa: F821
    store = PlayerStore()
    roster = _load_roster()
    if verbose:
        console.print(f"[dim]roster: {', '.join(roster)}[/dim]")
    player_data = store.load_player_data(roster, verbose=verbose)
    bundle = ensure_models(player_data, verbose=verbose)
    return store, bundle, roster


# ── Rendering helpers ───────────────────────────────────────────────────────
def _df_to_table(df, *, title: str | None = None) -> Table:
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            text = "" if val is None else str(val)
            style = ""
            if col in ("decision", "call"):
                style = "bold green" if "MORE" in text else "bold red"
            elif col == "edge":
                try:
                    f = float(val)
                    style = "green" if f > 0 else "red"
                except (TypeError, ValueError):
                    pass
            cells.append(Text(text, style=style))
        table.add_row(*cells)
    return table


# ── Commands ─────────────────────────────────────────────────────────────────
@app.command()
def project(
    player: str = typer.Argument(..., help="Player name (fuzzy match OK)."),
    last_n: int = typer.Option(10, "--last-n", "-n", help="Rolling window size for the feature row."),
    seasons: Optional[list[str]] = typer.Option(None, "--season", help="Seasons (e.g. 2024-25). Repeatable."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Project a player's next game across all 8 models."""
    store, bundle, roster = _bootstrap()
    name = _resolve_player(store, player)
    player_data = store.load_player_data(roster) if name in roster else None
    modeling_df = store.modeling_frame(player_data) if player_data is not None else None
    df = project_next_game(
        name, bundle=bundle, store=store,
        last_n=last_n, seasons=seasons, modeling_df=modeling_df,
    )
    if json_out:
        console.print_json(df.to_json(orient="records"))
        return
    console.print(_df_to_table(df, title=f"Projection — {name} (last {last_n})"))


@app.command()
def prop(
    player: str = typer.Argument(..., help="Player name."),
    stat: str = typer.Argument(..., help=f"Stat model. One of: {', '.join(MODEL_SPECS)}"),
    line: Optional[float] = typer.Option(None, "--line", "-l", help="Line threshold. Auto-fetched from The Odds API if ODDS_API_KEY is set."),
    last_n: int = typer.Option(5, "--last-n", "-n"),
    margin: float = typer.Option(0.10, "--margin", help="Edge margin required to signal above/below (live lines only)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Compare a player projection against a posted line for a single stat."""
    if stat not in MODEL_SPECS:
        err_console.print(f"[red]Unknown stat '{stat}'. Choices: {', '.join(MODEL_SPECS)}[/red]")
        raise typer.Exit(1)
    store, bundle, _ = _bootstrap()
    name = _resolve_player(store, player)
    api_key = load_api_key()
    try:
        result = custom_prop(
            name, stat, line=line, bundle=bundle, store=store,
            odds_api_key=api_key or None, last_n=last_n,
            confidence_margin=margin,
        )
    except (ValueError, KeyError) as exc:
        err_console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1) from exc

    if json_out:
        console.print_json(json.dumps(result))
        return
    body = "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in result.items())
    color = "green" if "MORE" in result["call"] else "red"
    console.print(Panel(body, title=f"{name} — {stat}", border_style=color))


@app.command()
def decisions(
    player: str = typer.Argument(..., help="Player name."),
    last_n: int = typer.Option(5, "--last-n", "-n"),
    live: bool = typer.Option(True, "--live/--no-live", help="Fetch live Odds API lines for comparison."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """8-stat projection summary with model-vs-line gap analysis."""
    store, bundle, roster = _bootstrap()
    name = _resolve_player(store, player)
    api_key = load_api_key() if live else ""
    live_lines: dict[str, float] | None = None
    if api_key:
        ll = fetch_live_player_lines(api_key, [name])
        if not ll.empty:
            live_lines = dict(zip(ll["model"], ll["line"]))
    df = fantasy_decisions(
        name, bundle=bundle, store=store, last_n=last_n, live_lines=live_lines,
    )
    if json_out:
        console.print_json(df.to_json(orient="records"))
        return
    title = f"Decisions — {name}" + (f"  ({len(live_lines)} live lines)" if live_lines else "")
    console.print(_df_to_table(df, title=title))


@app.command()
def scenario(
    spec: str = typer.Argument(..., help="JSON object of feature values, or path to .json file."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Predict from a hypothetical box-score row.

    Example:
      hooplytics scenario '{"fgm":8,"fga":15,"fg3m":4,"ftm":4,"min":32,"fg_pct":0.53,"ft_pct":1.0,"oreb":1,"dreb":4}'
    """
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    try:
        scenario_dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]✗ Invalid JSON: {exc}[/red]")
        raise typer.Exit(1) from exc
    _, bundle, _ = _bootstrap()
    df = predict_scenario(scenario_dict, bundle)
    if df.empty:
        console.print("[yellow]No model had all required features in your scenario.[/yellow]")
        return
    if json_out:
        console.print_json(df.to_json(orient="records"))
        return
    console.print(_df_to_table(df, title="Scenario predictions"))


@app.command()
def lines(
    refresh: bool = typer.Option(False, "--refresh", help="Bypass disk cache and re-fetch from The Odds API."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Live line board for the tracked roster, sorted by projection gap."""
    api_key = load_api_key()
    if not api_key:
        err_console.print("[red]✗ ODDS_API_KEY not set in env or .env[/red]")
        raise typer.Exit(1)
    store, bundle, roster = _bootstrap()
    ll = fetch_live_player_lines(api_key, list(roster), force_refresh=refresh)
    if ll.empty:
        console.print("[yellow]No live lines available right now.[/yellow]")
        return

    rows = []
    for _, r in ll.iterrows():
        try:
            res = custom_prop(
                r["player"], r["model"], line=float(r["line"]),
                bundle=bundle, store=store, last_n=5,
            )
        except Exception:  # noqa: BLE001
            continue
        res["matchup"] = r["matchup"]
        res["books"] = int(r["books"])
        rows.append(res)
    if not rows:
        console.print("[yellow]Lines fetched but no matching predictions.[/yellow]")
        return
    import pandas as pd
    df = pd.DataFrame(rows)[
        ["player", "matchup", "model", "posted line", "books",
         "model prediction", "5-game avg", "adj. threshold", "edge", "call"]
    ].sort_values("edge", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    if json_out:
        console.print_json(df.to_json(orient="records"))
        return
    console.print(_df_to_table(df, title="Live lines — sorted by |edge|"))


@app.command(name="ingest-odds")
def ingest_odds(
    start: str = typer.Option(..., "--start", help="Start date ISO, e.g. 2023-10-01"),
    end: str = typer.Option(..., "--end", help="End date ISO (exclusive), e.g. 2024-06-30"),
    force: bool = typer.Option(False, "--force", help="Re-fetch already-cached dates."),
) -> None:
    """Ingest historical player prop lines from The Odds API into the local cache.

    Example: hooplytics ingest-odds --start 2023-10-01 --end 2024-04-15

    NOTE: Requires a paid Odds API plan. Costs ~40 credits per game per date
    (4 markets × 1 region × 10x historical multiplier).
    """
    import datetime

    api_key = load_api_key()
    if not api_key:
        err_console.print("[red]ODDS_API_KEY not set in .env or environment.[/red]")
        raise typer.Exit(1)

    try:
        start_dt = datetime.date.fromisoformat(start)
        end_dt = datetime.date.fromisoformat(end)
    except ValueError as exc:
        err_console.print(f"[red]Invalid date: {exc}[/red]")
        raise typer.Exit(1)

    dates = [
        str(start_dt + datetime.timedelta(days=i))
        for i in range((end_dt - start_dt).days)
    ]

    console.print(f"[dim]Ingesting odds for {len(dates)} dates ({start} → {end})…[/dim]")
    from .constants import ODDS_PLAYER_PROPS_CUTOFF

    eligible = [d for d in dates if d >= ODDS_PLAYER_PROPS_CUTOFF]
    if len(eligible) < len(dates):
        console.print(
            f"[dim]Skipping {len(dates) - len(eligible)} date(s) before {ODDS_PLAYER_PROPS_CUTOFF}"
            " (player props not available).[/dim]"
        )

    try:
        result = ingest_historical_odds(api_key, eligible, force_refresh=force, verbose=True)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    n_rows = len(result)
    n_dates = result["game_date"].nunique() if not result.empty else 0
    console.print(
        f"[green]✓ Ingested {n_rows} player-prop lines across {n_dates} date(s).[/green]"
    )


@app.command()
def train(
    force: bool = typer.Option(False, "--force", help="Retrain even if a cached bundle exists."),
) -> None:
    """Pre-warm the model cache for the current roster."""
    store = PlayerStore()
    roster = _load_roster()
    console.print(f"[dim]Loading game logs for {len(roster)} player(s)…[/dim]")
    player_data = store.load_player_data(roster, verbose=True)
    bundle = ensure_models(player_data, force=force, verbose=True)
    body = (
        f"[green]✓ models ready[/green]  [dim]({len(bundle.estimators)} estimators)[/dim]\n"
        + (bundle.metrics.to_string(index=False) if bundle.metrics is not None else "")
    )
    if getattr(bundle, "uplift_report", None) is not None and not bundle.uplift_report.empty:
        body += "\n\n[bold]Context Feature Uplift[/bold]\n"
        body += bundle.uplift_report.to_string(index=False)
    console.print(Panel(body, title="Training complete"))


# ── Roster sub-commands ─────────────────────────────────────────────────────
@roster_app.command("list")
def roster_list() -> None:
    """Show the tracked roster."""
    r = _load_roster()
    table = Table(title="Tracked roster")
    table.add_column("Player", style="bold")
    table.add_column("Seasons")
    for name, seasons in r.items():
        table.add_row(name, ", ".join(seasons))
    console.print(table)


@roster_app.command("add")
def roster_add(
    player: str = typer.Argument(...),
    seasons: Optional[list[str]] = typer.Option(None, "--season"),
) -> None:
    """Add a player to the tracked roster."""
    store = PlayerStore()
    name = _resolve_player(store, player)
    r = _load_roster()
    if seasons is None:
        from datetime import date
        today = date.today()
        start = today.year - (1 if today.month >= 10 else 2)
        _end = today.year + 1 if today.month >= 10 else today.year
        seasons = next(iter(r.values()), nba_seasons(start, _end))
    r[name] = list(seasons)
    _save_roster(r)
    console.print(f"[green]✓ added {name}[/green] ({', '.join(seasons)})")


@roster_app.command("remove")
def roster_remove(player: str = typer.Argument(...)) -> None:
    """Remove a player from the tracked roster."""
    r = _load_roster()
    if player in r:
        r.pop(player)
        _save_roster(r)
        console.print(f"[green]✓ removed {player}[/green]")
        return
    # try fuzzy
    store = PlayerStore()
    name = store.resolve_player_name(player, active_only=False)
    if name and name in r:
        r.pop(name)
        _save_roster(r)
        console.print(f"[green]✓ removed {name}[/green]")
        return
    err_console.print(f"[red]✗ '{player}' not in roster[/red]")
    raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
