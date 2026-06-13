"""Typer subapp for ``devrun heartbeat`` — service control + foreground loop."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from devrun.db.jobs import JobStore, default_db_path

heartbeat_app = typer.Typer(
    name="heartbeat",
    help="Heartbeat scheduler control (foreground loop + service install/start/stop).",
    no_args_is_help=False,
)


def _tick_file_path() -> Path:
    return Path.home() / ".devrun" / "heartbeat.tick"


@heartbeat_app.callback(invoke_without_command=True)
def foreground(ctx: typer.Context) -> None:
    """Run the heartbeat loop in the foreground (refuses if the service is active)."""
    if ctx.invoked_subcommand is not None:
        return
    from devrun.heartbeat import run_loop
    from devrun.services import get_service

    svc = get_service()
    if svc.is_active():
        typer.echo(
            "Heartbeat daemon is already running; refusing to start foreground.",
            err=True,
        )
        raise typer.Exit(1)
    tick_file = _tick_file_path()
    tick_file.parent.mkdir(parents=True, exist_ok=True)
    run_loop(default_db_path(), interval=10.0, tick_file=tick_file)


@heartbeat_app.command()
def run() -> None:
    """Run foreground without service-active guard (for manual supervision)."""
    from devrun.heartbeat import run_loop

    tick_file = _tick_file_path()
    tick_file.parent.mkdir(parents=True, exist_ok=True)
    run_loop(default_db_path(), interval=10.0, tick_file=tick_file)


@heartbeat_app.command()
def install() -> None:
    """Install the heartbeat as a managed service (systemd --user / launchd) and start it."""
    from devrun.services import get_service

    svc = get_service()
    svc.install(python_path=sys.executable, db_path=str(default_db_path()))
    svc.start()
    typer.echo("Installed and started heartbeat service.")


@heartbeat_app.command()
def uninstall() -> None:
    """Uninstall the heartbeat service."""
    from devrun.services import get_service

    get_service().uninstall()
    typer.echo("Uninstalled heartbeat service.")


@heartbeat_app.command()
def start() -> None:
    """Start the installed heartbeat service."""
    from devrun.services import get_service

    get_service().start()
    typer.echo("Started.")


@heartbeat_app.command()
def stop() -> None:
    """Stop the installed heartbeat service."""
    from devrun.services import get_service

    get_service().stop()
    typer.echo("Stopped.")


@heartbeat_app.command()
def restart() -> None:
    """Restart the installed heartbeat service."""
    from devrun.services import get_service

    get_service().restart()
    typer.echo("Restarted.")


@heartbeat_app.command()
def status() -> None:
    """Print service status, job-status counts, and the last tick timestamp."""
    from devrun.services import get_service

    svc = get_service()
    active = svc.is_active()
    typer.echo(f"Service: {'active' if active else 'inactive'}")
    db = JobStore(default_db_path())
    try:
        counts = db.status_counts()
    finally:
        db.close()
    if counts:
        typer.echo("Job status counts:")
        for s, n in sorted(counts.items()):
            typer.echo(f"  {s}: {n}")
    else:
        typer.echo("Job status counts: (no jobs)")
    tick = _tick_file_path()
    if tick.exists():
        typer.echo(f"Last tick: {tick.read_text().strip()}")
    else:
        typer.echo("Last tick: never")
