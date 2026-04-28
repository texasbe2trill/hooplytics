"""CLI for fitting the historical-odds calibration artifact.

Usage::

    hooplytics-build-calibration                       # uses cached odds + auto seasons
    hooplytics-build-calibration --season 2024-25
    hooplytics-build-calibration --output bundles/cal.json --verbose

The artifact is consumed automatically by ``hooplytics.predict`` when present
at ``bundles/calibration_v1.json`` (or ``$HOOPLYTICS_CALIBRATION``).
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .calibration import DEFAULT_CALIBRATION_PATH, fit_from_cache, load_calibration

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Fit per-market + per-player calibration from cached historical odds.",
)
console = Console()


@app.command()
def build(
    output: Path = typer.Option(
        DEFAULT_CALIBRATION_PATH,
        "--output", "-o",
        help="Output JSON path (repo-shipped by default).",
    ),
    season: list[str] = typer.Option(
        None,
        "--season",
        help="NBA seasons to fetch player logs for (repeatable).",
    ),
    player: list[str] = typer.Option(
        None,
        "--player",
        help="Restrict to these players (repeatable). Defaults to every player in the cached odds history.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit and persist a calibration artifact from cached historical odds."""
    calib = fit_from_cache(
        seasons=season or None,
        players=player or None,
        output_path=output,
        verbose=verbose,
    )

    table = Table(title="Per-market calibration  (actual ≈ a + b·line)")
    table.add_column("model", style="bold")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("rmse", justify="right")
    table.add_column("n", justify="right")
    for model, stats in sorted(calib.per_market.items()):
        table.add_row(
            model,
            f"{stats['a']:.3f}",
            f"{stats['b']:.3f}",
            f"{stats['rmse']:.3f}",
            str(int(stats["n"])),
        )
    console.print(table)
    console.print(
        Panel(
            "\n".join([
                f"[green]Saved:[/green] {output}",
                f"[bold]Markets fitted:[/bold] {len(calib.per_market)}",
                f"[bold]Player residuals:[/bold] {len(calib.per_player)}",
            ]),
            title="Calibration",
            border_style="green",
        )
    )


@app.command()
def show(
    path: Path = typer.Option(DEFAULT_CALIBRATION_PATH, "--path"),
) -> None:
    """Print the contents of an existing calibration artifact."""
    calib = load_calibration(path)
    if calib is None:
        console.print(f"[yellow]No calibration found at {path}[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"Calibration {path}  (trained {calib.trained_at})")
    table.add_column("model", style="bold")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("rmse", justify="right")
    table.add_column("n", justify="right")
    table.add_column("blend_w", justify="right")
    for model, stats in sorted(calib.per_market.items()):
        table.add_row(
            model,
            f"{stats['a']:.3f}",
            f"{stats['b']:.3f}",
            f"{stats['rmse']:.3f}",
            str(int(stats["n"])),
            f"{calib.blend_weights.get(model, 0.5):.2f}",
        )
    console.print(table)
    console.print(f"[dim]Per-player residuals: {len(calib.per_player)}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
