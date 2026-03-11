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

def complete_target(incomplete: str):
    """Provide task name completions for the CLI."""
    for t in list_tasks():
        if t.startswith(incomplete):
            yield t


app = typer.Typer(
    name="devrun",
    help="Modular developer task orchestration system."
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


def _show_task_help(target: str, ctx: typer.Context) -> None:
    """Show help for a specific task based on its configuration schema."""
    runner = _runner()
    
    try:
        # _load_config handles merging project configs, user configs, and CLI overrides
        # We pass target which might be 'task' or 'task/variation'
        cfg = runner._load_config(target)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] No config found for target '{target}'.")
        console.print(f"Make sure it is a valid task and has a YAML config under configs/.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Failed to load configuration for '{target}':[/red] {e}")
        raise typer.Exit(code=1)
        
    try:
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        
        # We parse the target name for display
        task_name = target.split("/")[0]

        console.print(Panel(f"Help for target: [bold cyan]{target}[/bold cyan]", expand=False))
        console.print()

        # Gather main info
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Field", style="dim")
        info_table.add_column("Value")
        info_table.add_row("Task Class:", f"[cyan]{cfg.task}[/cyan]")
        info_table.add_row("Default Executor:", f"[green]{cfg.executor}[/green]")
        console.print(info_table)
        console.print()

        # Gather parameters
        params = cfg.params
        if params:
            param_table = Table(title="Task Parameters", show_edge=False, title_justify="left", header_style="bold cyan")
            param_table.add_column("Argument override")
            param_table.add_column("Resolved Value")
            
            for k, v in params.items():
                val_str = str(v)
                if val_str.startswith("<") and val_str.endswith(">"):
                    val_str = f"[yellow]{val_str}[/yellow]"
                
                param_table.add_row(f"params.[bold]{k}[/bold]", val_str)
            
            console.print(param_table)
        else:
            console.print("[dim]No default parameters defined.[/dim]")
            console.print()
        
        console.print("\n[dim]Usage Example:[/dim]")
        example_cmd = Text("devrun run ", style="bold")
        example_cmd.append(target, style="bold cyan")
        if params:
            first_param = next(iter(params.keys()))
            example_cmd.append(f" params.{first_param}=value", style="green")
        
        console.print(example_cmd)
        
    except Exception as e:
        console.print(f"[red]Failed to render documentation for {target}:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []}
)
def run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(
        None, 
        help="Config path, task, or task/variation",
        autocompletion=complete_target
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Prepare job without actually executing it"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    help: bool = typer.Option(False, "--help", "-h", help="Show this message and exit."),
) -> None:
    """Submit a task using a config file or variation. Trailing arguments are passed as OmegaConf overrides."""
    
    # Handle help manually so we can show task-specific help
    if help:
        if not target:
            # Show standard devrun run help
            console.print(ctx.get_help())
            raise typer.Exit()
        else:
            # Show task-specific help using its configuration template
            _show_task_help(target, ctx)
            raise typer.Exit()
            
    if not target:
        console.print("[red]Missing argument 'TARGET'.[/red]\n")
        console.print(ctx.get_help())
        raise typer.Exit(code=2)
        
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
