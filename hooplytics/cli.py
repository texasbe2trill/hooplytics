"""Hooplytics CLI — Typer + Rich.

Commands:
    project    — project a player's next game from recent form
    prop       — single MORE/LESS prop call (auto-fetches line)
    decisions  — full 8-stat decision table
    scenario   — predict from a hypothetical box-score JSON
    lines      — today's live lines + edge-sorted decisions
    role-shift — check for in-series player role transitions
    roster     — manage your persisted roster
    train      — pre-warm the model cache
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
    backtest_summary,
    custom_prop,
    ensure_models,
    fantasy_decisions,
    fetch_live_player_lines,
    ingest_historical_odds,
    load_api_key,
    nba_seasons,
    predict_scenario,
    project_next_game,
    retro_projection_table,
)
from .constants import MODEL_SPECS
from .role_shift_detector import RoleShiftDetector, Severity

_detector = RoleShiftDetector()

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


# ── Role-shift rendering helpers ────────────────────────────────────────────

def _render_shift_panel_cli(result) -> None:
    """Print a Rich panel above the projection table when a shift is detected."""
    from rich.panel import Panel

    sev = result.severity
    if sev == Severity.SUPPRESS:
        border = "red"
        title = "[bold red]🔴 ROLE SHIFT — SUPPRESS[/bold red]"
    else:
        border = "yellow"
        title = "[bold yellow]⚠ ROLE SHIFT — WARN[/bold yellow]"

    shift_names = ", ".join(s.value for s in result.shift_types)
    lines = [
        f"[bold]Shift type:[/bold]  {shift_names}",
        f"[bold]Severity:[/bold]    {sev.value}",
    ]
    if result.suppressed_stats:
        sup = ", ".join(result.suppressed_stats)
        lines.append(f"[bold]NO_CALL stats:[/bold] {sup}")
    if sev == Severity.WARN:
        lines.append(f"[bold]Confidence:[/bold]  -{result.confidence_penalty:.0%}")
    lines.append(f"[bold]Recommended last_n:[/bold] {result.recommended_last_n}")

    # Signal rows
    lines.append("")
    lines.append("[bold]Signals[/bold]")
    for sig in result.signals:
        action_str = (
            "[red]SUPPRESS[/red]" if sig.action == "SUPPRESS"
            else "[yellow]WARN[/yellow]" if sig.action == "WARN"
            else "[green]OK[/green]"
        )
        lines.append(
            f"  {sig.stat:<12} L3={sig.recent:>6.1f}  L30={sig.baseline:>5.1f}"
            f"  σ={sig.z_score:>+6.2f}  → {action_str}"
        )

    console.print(Panel("\n".join(lines), title=title, border_style=border))


def _df_to_table_with_suppression(df, suppressed: set[str], title: str | None = None) -> Table:
    """Like _df_to_table but annotates rows for suppressed stats with NO_CALL."""
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for _, row in df.iterrows():
        stat_name = str(row.get("model", ""))
        is_suppressed = stat_name in suppressed
        cells = []
        for col, val in row.items():
            text = "" if val is None else str(val)
            style = ""
            if is_suppressed and col in ("decision", "call", "model"):
                style = "bold red"
                if col in ("decision", "call"):
                    text = "NO_CALL"
            elif col in ("decision", "call"):
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
    import pandas as pd
    store, bundle, roster = _bootstrap()
    name = _resolve_player(store, player)
    player_data = store.load_player_data(roster) if name in roster else None
    modeling_df = store.modeling_frame(player_data) if player_data is not None else None
    df = project_next_game(
        name, bundle=bundle, store=store,
        last_n=last_n, seasons=seasons, modeling_df=modeling_df,
    )

    # Role-shift guard — extract features from the latest modeling row
    shift_result = None
    if modeling_df is not None and not modeling_df.empty:
        player_rows = modeling_df[modeling_df["player"] == name] if "player" in modeling_df.columns else modeling_df
        if not player_rows.empty:
            if "game_date" in player_rows.columns:
                player_rows = player_rows.sort_values("game_date")
            latest = player_rows.iloc[-1]
            _feature_keys = [
                "ast_l3", "ast_l30", "ast_std_l10",
                "pts_l3", "pts_l30", "pts_dev_s",
                "fga_l10", "fga_l30",
                "min_l3", "min_l30",
            ]
            features: dict[str, float] = {}
            for k in _feature_keys:
                val = latest.get(k) if hasattr(latest, "get") else (latest[k] if k in latest.index else None)
                if val is not None:
                    try:
                        features[k] = float(val)
                    except (TypeError, ValueError):
                        pass
            if features:
                shift_result = _detector.check(name, features)

    if json_out:
        out = json.loads(df.to_json(orient="records"))
        if shift_result:
            for row in out:
                stat = row.get("model", "")
                if shift_result.severity == Severity.SUPPRESS and stat in shift_result.suppressed_stats:
                    row["call"] = "NO_CALL"
            out_wrapper = {"projections": out, "role_shift": shift_result.to_dict()}
        else:
            out_wrapper = {"projections": out, "role_shift": None}
        console.print_json(json.dumps(out_wrapper))
        return

    # Print role-shift panel above projection table if shift detected
    if shift_result and shift_result.shift_detected:
        _render_shift_panel_cli(shift_result)

    # Build projection table, annotating suppressed stats with NO_CALL
    suppressed: set[str] = set()
    if shift_result and shift_result.severity == Severity.SUPPRESS:
        suppressed = set(shift_result.suppressed_stats)

    table = _df_to_table_with_suppression(df, suppressed=suppressed,
                                          title=f"Projection — {name} (last {last_n})")
    console.print(table)


@app.command(name="role-shift")
def role_shift(
    player: str = typer.Argument(..., help="Player name (fuzzy match OK)."),
    last_n: int = typer.Option(5, "--last-n", "-n", help="Rolling window size (unused in detection but sets context)."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON instead of a Rich table."),
) -> None:
    """Check for in-series role transitions that would break RACE directional calls.

    Exit codes:
        0 — No shift detected
        1 — WARN (|z| >= 1.5σ, confidence -20%)
        2 — SUPPRESS (|z| >= 2.0σ, props voided)
    """
    store, bundle, roster = _bootstrap()
    name = _resolve_player(store, player)

    player_data = store.load_player_data(roster) if name in roster else None
    modeling_df = store.modeling_frame(player_data) if player_data is not None else None

    features: dict[str, float] = {}
    if modeling_df is not None and not modeling_df.empty:
        player_rows = (
            modeling_df[modeling_df["player"] == name]
            if "player" in modeling_df.columns
            else modeling_df
        )
        if not player_rows.empty:
            if "game_date" in player_rows.columns:
                player_rows = player_rows.sort_values("game_date")
            latest = player_rows.iloc[-1]
            _feature_keys = [
                "ast_l3", "ast_l30", "ast_std_l10",
                "pts_l3", "pts_l30", "pts_dev_s",
                "fga_l10", "fga_l30",
                "min_l3", "min_l30",
            ]
            for k in _feature_keys:
                val = latest.get(k) if hasattr(latest, "get") else (latest[k] if k in latest.index else None)
                if val is not None:
                    try:
                        features[k] = float(val)
                    except (TypeError, ValueError):
                        pass

    result = _detector.check(name, features)

    if json_out:
        console.print_json(json.dumps(result.to_dict()))
        raise typer.Exit(0 if result.severity == Severity.NONE
                         else 1 if result.severity == Severity.WARN
                         else 2)

    if not result.shift_detected:
        console.print(
            Panel(
                f"[green]✓ No role shift detected for [bold]{name}[/bold].[/green]\n"
                "All rolling signals are within normal bounds — RACE calls are valid.",
                title="Role Shift Check",
                border_style="green",
            )
        )
        raise typer.Exit(0)

    _render_shift_panel_cli(result)

    # Signal table
    sig_table = Table(title="Signal details", show_lines=False, header_style="bold cyan")
    sig_table.add_column("Stat", style="bold")
    sig_table.add_column("L3", justify="right")
    sig_table.add_column("L30", justify="right")
    sig_table.add_column("σ", justify="right")
    sig_table.add_column("Action")

    for sig in result.signals:
        action_text = Text(sig.action)
        if sig.action == "SUPPRESS":
            action_text.stylize("bold red")
        elif sig.action == "WARN":
            action_text.stylize("bold yellow")
        else:
            action_text.stylize("green")
        sig_table.add_row(
            sig.stat,
            f"{sig.recent:.1f}",
            f"{sig.baseline:.1f}",
            f"{sig.z_score:+.2f}",
            action_text,
        )
    console.print(sig_table)

    exit_code = (
        2 if result.severity == Severity.SUPPRESS
        else 1 if result.severity == Severity.WARN
        else 0
    )
    raise typer.Exit(exit_code)


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


@app.command()
def compare(
    player: str = typer.Argument(..., help="Player name (fuzzy match OK)."),
    stat: str = typer.Argument("points", help=f"Stat model. One of: {', '.join(MODEL_SPECS)}"),
    games: int = typer.Option(15, "--games", "-g", help="Number of recent games to evaluate."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Compare model projections against actual game results (projection accuracy backtest).

    Reconstructs what the RACE model would have projected before each of the
    last N games, then shows how those projections compared to reality.

    Example:
      hooplytics compare "Shai Gilgeous-Alexander" points --games 20
    """
    if stat not in MODEL_SPECS:
        err_console.print(f"[red]Unknown stat '{stat}'. Choices: {', '.join(MODEL_SPECS)}[/red]")
        raise typer.Exit(1)

    store, bundle, _ = _bootstrap()
    name = _resolve_player(store, player)

    console.print(f"[dim]Evaluating last {games} game(s) for {name} / {stat}…[/dim]")

    try:
        table = retro_projection_table(name, stat, store=store, bundle=bundle, n_games=games)
    except Exception as exc:
        err_console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1) from exc

    if table.empty:
        console.print("[yellow]Insufficient game history to evaluate projections.[/yellow]")
        return

    summary = backtest_summary(table)

    if json_out:
        import json as _json
        console.print_json(_json.dumps({"games": table.to_dict(orient="records"), "summary": summary}))
        return

    # Game-by-game table
    import pandas as pd

    display = table.copy()
    t = Table(title=f"Projection vs Actual — {name} / {stat}", show_lines=False,
              header_style="bold cyan")
    for col in display.columns:
        t.add_column(str(col), overflow="fold")
    for _, row in display.iterrows():
        cells = []
        for col, val in row.items():
            text = "" if val is None else str(val)
            style = ""
            if col == "result":
                style = "green" if val == "Over" else ("red" if val == "Under" else "dim")
            elif col == "error":
                try:
                    style = "green" if float(val) > 0 else "red"
                except (TypeError, ValueError):
                    pass
            cells.append(Text(text, style=style))
        t.add_row(*cells)
    console.print(t)

    # Summary panel
    bias_sign = "+" if (summary["bias"] or 0) > 0 else ""
    dir_acc = summary.get("directional_accuracy")
    body = (
        f"[bold]Games evaluated:[/bold]  {summary['n_games']} [dim](sportsbook lines only)[/dim]\n"
        f"[bold]MAE:[/bold]             {summary['mae']}\n"
        f"[bold]RMSE:[/bold]            {summary['rmse']}\n"
        f"[bold]Bias:[/bold]            {bias_sign}{summary['bias']}  "
        f"[dim](+ = model under-predicted)[/dim]\n"
        f"[bold]Directional acc:[/bold] {f'{dir_acc:.1%}' if dir_acc is not None else '—'}  "
        f"[dim](call matched result vs sportsbook line)[/dim]\n"
        f"[bold]Median error:[/bold]    {bias_sign}{summary['median_error']}"
    )
    console.print(Panel(body, title="Accuracy summary", border_style="cyan"))


@app.command(name="ingest-odds")
def ingest_odds(
    start: str = typer.Option(..., "--start", help="Start date ISO, e.g. 2023-10-01"),
    end: str = typer.Option(..., "--end", help="End date ISO (exclusive), e.g. 2024-06-30"),
    force: bool = typer.Option(False, "--force", help="Re-fetch already-cached dates."),
    markets: Optional[str] = typer.Option(
        None,
        "--markets",
        help=(
            "Comma-separated subset of markets to fetch: points,rebounds,assists,threepm. "
            "Each market costs 10 credits per event, so 2 markets = half price. "
            "Re-running later with different markets fills in the missing ones."
        ),
    ),
) -> None:
    """Ingest historical player prop lines from The Odds API into the local cache.

    Example: hooplytics ingest-odds --start 2023-10-01 --end 2024-04-15

    NOTE: Requires a paid Odds API plan. Costs ~10 credits per market per event
    (default 4 markets × 1 region × 10x historical multiplier = 40 cr/event).
    Use ``--markets points,rebounds`` to halve the cost.
    """
    import datetime
    from .constants import ODDS_MARKETS, ODDS_PLAYER_PROPS_CUTOFF

    api_key = load_api_key()
    if not api_key:
        err_console.print("[red]ODDS_API_KEY not set in .env or environment.[/red]")
        raise typer.Exit(1)

    market_subset: list[str] | None = None
    if markets:
        valid = set(ODDS_MARKETS.values())
        market_subset = [m.strip() for m in markets.split(",") if m.strip()]
        bad = [m for m in market_subset if m not in valid]
        if bad:
            err_console.print(
                f"[red]Unknown market(s): {', '.join(bad)}. Valid: {', '.join(sorted(valid))}[/red]"
            )
            raise typer.Exit(1)
        cost = len(market_subset) * 10
        console.print(
            f"[dim]Market subset: {', '.join(market_subset)} (~{cost} credits/event)[/dim]"
        )

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

    eligible = [d for d in dates if d >= ODDS_PLAYER_PROPS_CUTOFF]
    if len(eligible) < len(dates):
        console.print(
            f"[dim]Skipping {len(dates) - len(eligible)} date(s) before {ODDS_PLAYER_PROPS_CUTOFF}"
            " (player props not available).[/dim]"
        )

    try:
        result = ingest_historical_odds(
            api_key, eligible, force_refresh=force, verbose=True, markets=market_subset,
        )
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
