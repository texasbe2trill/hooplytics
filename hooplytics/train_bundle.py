"""Interactive bundle trainer for shipping a prebuilt model artifact.

This command is optimized for producing a repo-shipped bundle
(e.g. bundles/race_fast.joblib) used by the Streamlit app to reduce cold-start
lag when users select players.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .constants import MODEL_SPECS, NBA_RENAME, TRAINING_ANCHOR_PLAYERS
from .data import PlayerStore, add_pregame_features, nba_seasons
from .features_context import build_context_features
from .features_market import build_market_features
from .features_role import build_role_features
from .fantasy import fantasy
from .models import ModelBundle, save_models, train_models

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Interactive trainer for shipping a prebuilt RACE bundle.",
)
console = Console()
err_console = Console(stderr=True)


class TrainingMode(str, Enum):
    fast = "fast"
    balanced = "balanced"
    exhaustive = "exhaustive"


class PlayerSource(str, Enum):
    roster = "roster"
    postseason_active = "postseason-active"
    postseason_plus_anchors = "postseason-plus-anchors"


def _current_season() -> str:
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _default_training_seasons() -> list[str]:
    today = date.today()
    if today.month >= 10:
        start, end = today.year - 1, today.year + 1
    else:
        start, end = today.year - 2, today.year
    return nba_seasons(start, end)


def _discover_postseason_players(season: str) -> list[str]:
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import players as nba_players

        logs = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="P",
            season_nullable=season,
            season_type_nullable="Playoffs",
            timeout=15,
        ).get_data_frames()[0]
        if logs.empty:
            return []

        name_col = "PLAYER_NAME" if "PLAYER_NAME" in logs.columns else "PLAYER"
        if name_col not in logs.columns:
            return []

        postseason_names = sorted({str(n).strip() for n in logs[name_col].dropna().tolist() if str(n).strip()})
        active_names = {
            p["full_name"]
            for p in nba_players.get_players()
            if bool(p.get("is_active"))
        }
        filtered = [n for n in postseason_names if n in active_names]
        return filtered if filtered else postseason_names
    except Exception:
        return []


def _resolve_players(source: PlayerSource, season: str) -> list[str]:
    if source == PlayerSource.roster:
        from .cli import _load_roster

        return sorted(_load_roster().keys())

    postseason = _discover_postseason_players(season)
    if not postseason:
        raise RuntimeError(
            "Could not discover postseason-active players from nba_api. "
            "Try --players-source roster or provide --player values manually."
        )

    if source == PlayerSource.postseason_plus_anchors:
        return sorted(set(postseason) | set(TRAINING_ANCHOR_PLAYERS))

    return postseason


def _load_and_engineer_player_data(
    store: PlayerStore,
    players: list[str],
    seasons: list[str],
) -> pd.DataFrame:
    raw_parts: list[pd.DataFrame] = []
    kept_players: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        fetch_task = progress.add_task("Fetching player logs", total=len(players))

        for player in players:
            try:
                df = store.fetch_player_seasons(player, seasons)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                raw_parts.append(df)
                kept_players.append(player)
            progress.advance(fetch_task)

        feature_task = progress.add_task("Building pregame-safe features", total=5)

        if not raw_parts:
            raise RuntimeError("No player logs were fetched. Training cannot continue.")

        raw = pd.concat(raw_parts, ignore_index=True)
        progress.advance(feature_task)

        data = raw.rename(columns=NBA_RENAME)
        data["game_date"] = pd.to_datetime(data["game_date"], format="%b %d, %Y", errors="coerce")
        data = data.sort_values(["player", "game_date"]).reset_index(drop=True)
        data["pra"] = data["pts"] + data["reb"] + data["ast"]
        data["stl_blk"] = data["stl"] + data["blk"]
        data["fantasy_score"] = fantasy(data)
        progress.advance(feature_task)

        data = add_pregame_features(data)
        progress.advance(feature_task)

        data = build_context_features(data)
        data = build_role_features(data)
        progress.advance(feature_task)

        data = build_market_features(data)
        progress.advance(feature_task)

    if not kept_players:
        raise RuntimeError("Fetched data was empty for all selected players.")

    console.print(
        f"[dim]Loaded {len(kept_players)} players with {len(data):,} total game rows.[/dim]"
    )
    return data


def _candidate_configs(mode: TrainingMode) -> list[dict[str, object]]:
    if mode == TrainingMode.fast:
        return [
            {
                "name": "fast",
                "fast_mode": True,
                "time_aware_validation": True,
                "random_state": 123,
            }
        ]
    if mode == TrainingMode.balanced:
        return [
            {
                "name": "balanced",
                "fast_mode": False,
                "time_aware_validation": True,
                "random_state": 123,
            }
        ]
    return [
        {
            "name": "fast",
            "fast_mode": True,
            "time_aware_validation": True,
            "random_state": 123,
        },
        {
            "name": "balanced",
            "fast_mode": False,
            "time_aware_validation": True,
            "random_state": 123,
        },
        {
            "name": "aggressive",
            "fast_mode": False,
            "time_aware_validation": False,
            "random_state": 123,
        },
    ]


def _score_bundle(bundle: ModelBundle) -> float:
    if bundle.metrics is None or bundle.metrics.empty or "R²" not in bundle.metrics.columns:
        return float("-inf")
    return float(pd.to_numeric(bundle.metrics["R²"], errors="coerce").fillna(-1e9).mean())


def _r2_gate(bundle: ModelBundle, min_r2: float, min_models: int) -> tuple[bool, int]:
    if bundle.metrics is None or bundle.metrics.empty or "R²" not in bundle.metrics.columns:
        return False, 0
    r2 = pd.to_numeric(bundle.metrics["R²"], errors="coerce")
    passing = int((r2 >= min_r2).sum())
    return passing >= min_models, passing


def _render_metrics(name: str, bundle: ModelBundle) -> None:
    if bundle.metrics is None or bundle.metrics.empty:
        console.print(f"[yellow]{name}: no metrics available.[/yellow]")
        return

    metrics = bundle.metrics.copy()
    for col in ("RMSE", "MAE", "R²"):
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce").round(3)

    table = Table(title=f"Validation Metrics - {name}")
    table.add_column("model", style="bold")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("R2", justify="right")

    for _, row in metrics.sort_values("R²", ascending=False).iterrows():
        table.add_row(
            str(row.get("model", "")),
            str(row.get("RMSE", "")),
            str(row.get("MAE", "")),
            str(row.get("R²", "")),
        )

    console.print(table)


@app.command()
def train_bundle(
    output: Path = typer.Option(
        Path("bundles/race_fast.joblib"),
        "--output",
        "-o",
        help="Output bundle path (repo-shipped by default).",
    ),
    mode: TrainingMode = typer.Option(
        TrainingMode.exhaustive,
        "--mode",
        help="Training strategy: fast, balanced, or exhaustive (choose best mean R2).",
    ),
    players_source: PlayerSource = typer.Option(
        PlayerSource.postseason_plus_anchors,
        "--players-source",
        help="Player pool strategy.",
    ),
    postseason_season: str = typer.Option(
        _current_season(),
        "--postseason-season",
        help="Season used for postseason player discovery (e.g. 2025-26).",
    ),
    season: list[str] = typer.Option(
        None,
        "--season",
        help="Training seasons to fetch (repeatable). Defaults to previous+current season windows.",
    ),
    player: list[str] = typer.Option(
        None,
        "--player",
        help="Optional manual players to add (repeatable).",
    ),
    min_r2: float = typer.Option(
        -0.05,
        "--min-r2",
        help="Minimum acceptable validation R2 for gate checks.",
    ),
    min_models_passing: int = typer.Option(
        5,
        "--min-models-passing",
        help="Minimum count of model targets that must satisfy --min-r2.",
    ),
    allow_failed_gate: bool = typer.Option(
        False,
        "--allow-failed-gate",
        help="Save bundle even if R2 gate fails.",
    ),
) -> None:
    """Train and export a prebuilt model bundle with interactive progress."""
    seasons = season if season else _default_training_seasons()

    try:
        base_players = _resolve_players(players_source, postseason_season)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    all_players = sorted(set(base_players) | set(player or []))
    if not all_players:
        err_console.print("[red]No players selected for training.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            "\n".join(
                [
                    f"[bold]Mode:[/bold] {mode.value}",
                    f"[bold]Players:[/bold] {len(all_players)}",
                    f"[bold]Seasons:[/bold] {', '.join(seasons)}",
                    f"[bold]Output:[/bold] {output}",
                    f"[bold]R2 gate:[/bold] min R2 >= {min_r2} on at least {min_models_passing}/{len(MODEL_SPECS)} models",
                ]
            ),
            title="Training Plan",
            border_style="cyan",
        )
    )

    store = PlayerStore()
    data = _load_and_engineer_player_data(store, all_players, seasons)

    configs = _candidate_configs(mode)
    candidates: list[tuple[str, ModelBundle]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        train_task = progress.add_task("Training candidate bundles", total=len(configs))
        for cfg in configs:
            name = str(cfg["name"])
            bundle = train_models(
                data,
                random_state=int(cfg["random_state"]), # type: ignore
                time_aware_validation=bool(cfg["time_aware_validation"]),
                fast_mode=bool(cfg["fast_mode"]),
                verbose=False,
            )
            candidates.append((name, bundle))
            progress.advance(train_task)

    ranked = sorted(candidates, key=lambda it: _score_bundle(it[1]), reverse=True)
    best_name, best_bundle = ranked[0]

    for name, bundle in ranked:
        mean_r2 = _score_bundle(bundle)
        console.print(f"[dim]{name} mean R2: {mean_r2:.3f}[/dim]")
        _render_metrics(name, bundle)

    passed, passing_count = _r2_gate(best_bundle, min_r2=min_r2, min_models=min_models_passing)
    if not passed and not allow_failed_gate:
        err_console.print(
            f"[red]R2 gate failed for best candidate '{best_name}': "
            f"{passing_count}/{len(MODEL_SPECS)} models met R2 >= {min_r2}. "
            "Use --allow-failed-gate to save anyway.[/red]"
        )
        raise typer.Exit(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_models(best_bundle, output)

    gate_text = "passed" if passed else "failed (saved due to --allow-failed-gate)"
    console.print(
        Panel(
            "\n".join(
                [
                    f"[green]Saved bundle:[/green] {output}",
                    f"[bold]Selected mode:[/bold] {best_name}",
                    f"[bold]Selected mean R2:[/bold] {_score_bundle(best_bundle):.3f}",
                    f"[bold]R2 gate:[/bold] {gate_text}",
                ]
            ),
            title="Done",
            border_style="green",
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
