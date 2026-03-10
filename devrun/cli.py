"""devrun CLI — Typer-based command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from devrun.runner import TaskRunner
from devrun.registry import list_tasks, list_executors
from devrun.utils.sync import sync_to_remote, fetch_from_remote

app = typer.Typer(
    name="devrun",
    help="Modular developer task orchestration system.",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


def _runner() -> TaskRunner:
    return TaskRunner()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context,
    target: str = typer.Argument(..., help="Config path, task, or task/variation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Prepare job without actually executing it"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Submit a task using a config file or variation. Trailing arguments are passed as OmegaConf overrides."""
    _setup_logging(verbose)
    runner = _runner()
    
    # Any trailing arguments will be passed as overrides
    overrides = ctx.args
    if overrides:
        console.print(f"[dim]Using overrides: {overrides}[/dim]")
        
    try:
        job_ids = runner.run(target, overrides=overrides, dry_run=dry_run)
        if dry_run:
            console.print("[yellow]Dry-run complete. No jobs were submitted.[/yellow]")
        else:
            for jid in job_ids:
                console.print(f"[green]✓[/green] Job submitted: [bold]{jid}[/bold]")
    except Exception as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("list")
def list_plugins() -> None:
    """List available task and executor plugins."""
    table = Table(title="Registered Plugins")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")

    for name in list_tasks():
        table.add_row("task", name)
    for name in list_executors():
        table.add_row("executor", name)
    console.print(table)


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to query"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check the status of a submitted job."""
    _setup_logging(verbose)
    info = _runner().status(job_id)
    if "error" in info:
        console.print(f"[red]{info['error']}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Job {job_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, val in info.items():
        table.add_row(key, str(val))
    console.print(table)


@app.command()
def logs(
    job_id: str = typer.Argument(..., help="Job ID to fetch logs for"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Retrieve logs for a submitted job."""
    _setup_logging(verbose)
    output = _runner().logs(job_id)
    console.print(output)


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent jobs to show"),
) -> None:
    """Show recent job history."""
    records = _runner().history(limit)
    if not records:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title="Job History")
    table.add_column("Job ID", style="bold")
    table.add_column("Task", style="cyan")
    table.add_column("Executor", style="green")
    table.add_column("Status")
    table.add_column("Created")

    for rec in records:
        status_style = {
            "completed": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "dim",
        }.get(rec.get("status", ""), "")
        table.add_row(
            rec["job_id"],
            rec["task_name"],
            rec["executor"],
            f"[{status_style}]{rec.get('status', 'unknown')}[/{status_style}]" if status_style else rec.get("status", "unknown"),
            rec.get("created_at", ""),
        )
    console.print(table)


@app.command()
def rerun(
    job_id: str = typer.Argument(..., help="Job ID to re-submit"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-submit a previous job with the same parameters."""
    _setup_logging(verbose)
    try:
        new_ids = _runner().rerun(job_id)
        for nid in new_ids:
            console.print(f"[green]✓[/green] Re-submitted as: [bold]{nid}[/bold]")
    except Exception as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def sync(
    source: str = typer.Argument(..., help="Source path (local or remote:path)"),
    destination: str = typer.Argument(..., help="Destination path (local or remote:path)"),
    delete: bool = typer.Option(False, "--delete", help="Delete extraneous files on destination"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be transferred"),
) -> None:
    """Sync files between local and remote hosts using rsync."""
    result = sync_to_remote(source, destination, delete=delete, dry_run=dry_run)
    if result.returncode == 0:
        console.print(f"[green]✓[/green] Sync complete: {source} → {destination}")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print(f"[red]✗ Sync failed:[/red] {result.stderr}")
        raise typer.Exit(code=1)


@app.command()
def fetch(
    source: str = typer.Argument(..., help="Remote source (host:path)"),
    destination: str = typer.Argument(..., help="Local destination path"),
    delete: bool = typer.Option(False, "--delete"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Fetch files from a remote host to local."""
    result = fetch_from_remote(source, destination, delete=delete, dry_run=dry_run)
    if result.returncode == 0:
        console.print(f"[green]✓[/green] Fetch complete: {source} → {destination}")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print(f"[red]✗ Fetch failed:[/red] {result.stderr}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
