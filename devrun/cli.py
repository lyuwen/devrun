"""devrun CLI — Typer-based command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import yaml

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
    help="Modular developer task orchestration system.",
    no_args_is_help=True,
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
    from_job: Optional[str] = typer.Option(
        None, "--from-job",
        help="Import params from an existing job (e.g. swe_bench_collect --from-job <agentic_job_id>)",
    ),
    after: list[str] = typer.Option(
        [], "--after",
        help="Wait for these job IDs to complete before submitting. Repeatable.",
    ),
    allow_failure_from: list[str] = typer.Option(
        [], "--allow-failure-from",
        help="Subset of --after jobs whose failure should not block this run. Repeatable.",
    ),
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

    # Validate --allow-failure-from requires --after
    if allow_failure_from and not after:
        console.print("[red]Error:[/red] --allow-failure-from requires at least one --after parent.")
        raise typer.Exit(code=1)

    runner = _runner()

    # Merge order: YAML base → from-job imported params → CLI trailing overrides.
    # Imported params are converted to dotlist entries and prepended so that
    # any matching CLI override wins.
    overrides: list[str] = []

    if from_job:
        # Resolve target → task name to drive the import hook.
        try:
            task_name = runner._load_config(target).task
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] No config found for target '{target}'.")
            raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[red]Failed to load configuration for '{target}':[/red] {exc}")
            raise typer.Exit(code=1)
        try:
            imported, source_task, resolved_job_id = runner.extract_task_params(
                from_job, task_name
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)
        console.print(
            f"[dim]From job {resolved_job_id} ({source_task}): {list(imported.keys())}[/dim]"
        )
        overrides.extend(f"params.{k}={v}" for k, v in imported.items())

    if ctx.args:
        console.print(f"[dim]Using overrides: {ctx.args}[/dim]")
        overrides.extend(ctx.args)

    try:
        job_ids = runner.run(
            target,
            overrides=overrides,
            dry_run=dry_run,
            after=after,
            allow_failure_from=set(allow_failure_from),
        )
        if dry_run:
            console.print("[yellow]Dry-run complete. No jobs were submitted.[/yellow]")
        else:
            for jid in job_ids:
                console.print(f"[green]✓[/green] Job queued: [bold]{jid}[/bold]")
            # Check if heartbeat service is running
            try:
                from devrun.services import get_service
                service = get_service()
                if not service.is_active():
                    console.print("[yellow]⚠ Warning:[/yellow] Heartbeat scheduler is not running. Jobs will remain in QUEUED state.")
                    console.print("  Start it with: [bold]devrun heartbeat start[/bold]")
            except Exception:
                # Silently skip warning if service check fails
                pass
    except ValueError as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("list")
def list_plugins() -> None:
    """List available task and executor plugins, including config variations."""
    from devrun.runner import get_config_dirs

    table = Table(title="Registered Plugins")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Variations", style="dim")

    # Collect variations per task/workflow by scanning config directories
    config_dirs = get_config_dirs()
    task_variations: dict[str, set[str]] = {}
    workflow_variations: dict[str, set[str]] = {}
    workflow_dirs: set[str] = set()  # dirs identified as workflows

    for d in config_dirs:
        if not d.is_dir():
            continue
        for child in d.iterdir():
            if child.is_dir():
                for yaml_file in child.glob("*.yaml"):
                    stem = yaml_file.stem
                    # Check if this YAML has a top-level 'workflow' key
                    try:
                        with open(yaml_file, "r") as fh:
                            data = yaml.safe_load(fh)
                        if isinstance(data, dict) and "workflow" in data:
                            workflow_variations.setdefault(child.name, set()).add(stem)
                            workflow_dirs.add(child.name)
                            continue
                    except Exception:
                        continue
                    task_variations.setdefault(child.name, set()).add(stem)

    # Remove any dirs classified as workflows from task_variations
    for wf_name in workflow_dirs:
        task_variations.pop(wf_name, None)

    for name in list_tasks():
        variations = sorted(task_variations.get(name, set()))
        variations_str = ", ".join(f"{name}/{v}" for v in variations if v != "default")
        table.add_row("task", name, variations_str)
    for name in list_executors():
        table.add_row("executor", name, "")
    for name in sorted(workflow_variations):
        variations = sorted(workflow_variations[name])
        variations_str = ", ".join(f"{name}/{v}" for v in variations if v != "default")
        table.add_row("workflow", name, variations_str)
    console.print(table)


_STATUS_STYLES: dict[str, str] = {
    "queued": "cyan",
    "submitting": "yellow",
    "submitted": "yellow",
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "canceling": "magenta",
    "cancelled": "magenta",
    "skipped": "dim",
    "timed_out": "red",
    "pending": "dim",
    "unknown": "dim",
}


def _style_status(status: str) -> str:
    """Return ``status`` wrapped in a Rich style tag for the table renderer.

    Normalises ``"timed_out"`` to a human-friendly ``"timed out"`` so the
    underscore doesn't trip the formatting expectations of the CLI tests.
    """
    label = status.replace("_", " ") if status == "timed_out" else status
    style = _STATUS_STYLES.get(status, "")
    return f"[{style}]{label}[/{style}]" if style else label


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to query"),
    with_deps: bool = typer.Option(
        False, "--with-deps", help="List parent job IDs and statuses for this job."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check the status of a submitted job (pure DB read)."""
    _setup_logging(verbose)
    runner = _runner()
    info = runner.status(job_id)
    if "error" in info:
        console.print(f"[red]{info['error']}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Job {job_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, val in info.items():
        if key == "status":
            table.add_row(key, _style_status(str(val)))
        else:
            table.add_row(key, str(val))
    console.print(table)

    if with_deps:
        deps = runner._db.list_dependencies(job_id)
        deps_table = Table(title=f"Parents of {job_id}")
        deps_table.add_column("Parent Job ID", style="bold")
        deps_table.add_column("Status")
        deps_table.add_column("allow_failure")
        if not deps:
            console.print("[dim]No parent dependencies.[/dim]")
        else:
            for dep in deps:
                parent_rec = runner._db.get(dep.parent_job_id)
                parent_status = parent_rec.status if parent_rec else "unknown"
                deps_table.add_row(
                    dep.parent_job_id,
                    _style_status(str(parent_status)),
                    "true" if dep.allow_failure else "false",
                )
            console.print(deps_table)


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
    all_records: bool = typer.Option(False, "--all", "-a", help="Show all records (no limit)"),
    no_pager: bool = typer.Option(False, "--no-pager", help="Disable pager for long output"),
) -> None:
    """Show recent job history."""
    effective_limit = None if all_records else limit
    records = _runner().history(effective_limit)
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
        table.add_row(
            rec["job_id"],
            rec["task_name"],
            rec["executor"],
            _style_status(str(rec.get("status", "unknown"))),
            rec.get("created_at", ""),
        )

    # Use pager if output is long and not disabled
    # Estimate: header(4) + rows + footer(1) lines
    table_lines = len(records) + 5
    if not no_pager and table_lines > console.height:
        with console.pager(styles=True):
            console.print(table)
    else:
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
def cancel(
    job_id: str = typer.Argument(..., help="Job ID to cancel"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cancel a running job."""
    _setup_logging(verbose)
    try:
        _runner().cancel(job_id)
        console.print(f"[green]✓[/green] Cancellation requested for: [bold]{job_id}[/bold]")
    except ValueError as exc:
        console.print(f"[yellow]⚠ Warning:[/yellow] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]✗ Error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix deprecated config parameters"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all checks, including passing"),
) -> None:
    """Check config health: validate schemas, detect deprecations, flag placeholders."""
    from devrun.doctor import run_doctor, Severity

    report = run_doctor(fix=fix, verbose=verbose)

    if report.diagnostics or verbose:
        table = Table(title="Doctor Report")
        table.add_column("Severity", style="bold")
        table.add_column("File")
        table.add_column("Rule", style="dim")
        table.add_column("Message")

        severity_styles = {
            Severity.ERROR: "red",
            Severity.WARNING: "yellow",
            Severity.INFO: "blue",
        }

        for d in report.diagnostics:
            style = severity_styles.get(d.severity, "")
            fixed = " [green](fixed)[/green]" if d.fix_applied else ""
            table.add_row(
                f"[{style}]{d.severity.value.upper()}[/{style}]",
                d.file_path,
                d.rule_id,
                d.message + fixed,
            )
        console.print(table)

    # Summary
    console.print(
        f"\n[bold]Summary:[/bold] {report.error_count} error(s), "
        f"{report.warning_count} warning(s), {report.info_count} info"
    )

    if report.has_errors:
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
# Workflow subcommands
# ---------------------------------------------------------------------------


workflow_app = typer.Typer(name="workflow", help="Manage multi-stage workflows.", no_args_is_help=True)
app.add_typer(workflow_app, name="workflow")

from devrun.cli_heartbeat import heartbeat_app  # noqa: E402

app.add_typer(heartbeat_app, name="heartbeat")


def _show_workflow_help(target: str) -> None:
    """Show help for a specific workflow based on its configuration."""
    from omegaconf import OmegaConf
    from devrun.runner import load_merged_config
    from rich.panel import Panel
    from rich.text import Text

    # Register cross-stage resolver so ${stages:...} tokens don't error
    if not OmegaConf.has_resolver("stages"):
        OmegaConf.register_new_resolver(
            "stages",
            lambda stage_name, param_key: f"<<STAGE_REF:{stage_name}:{param_key}>>",
        )

    try:
        raw = load_merged_config(target)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] No config found for workflow '{target}'.")
        console.print("Ensure the workflow config exists in one of the config search directories.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Failed to load configuration for '{target}':[/red] {e}")
        raise typer.Exit(code=1)

    workflow_name = raw.get("workflow", target)

    console.print(Panel(f"Workflow: [bold cyan]{workflow_name}[/bold cyan]  (config: {target})", expand=False))
    console.print()

    # Workflow-level params
    params = raw.get("params", {})
    if params:
        param_table = Table(title="Workflow Parameters", show_edge=False, title_justify="left", header_style="bold cyan")
        param_table.add_column("Override")
        param_table.add_column("Default Value")

        for k, v in params.items():
            val_str = str(v)
            if val_str.startswith("<") and val_str.endswith(">"):
                val_str = f"[yellow]{val_str}[/yellow]"
            param_table.add_row(f"params.[bold]{k}[/bold]", val_str)

        console.print(param_table)
        console.print()

    # Stages
    import re as _re
    _ref_pattern = _re.compile(r"<<STAGE_REF:([\w]+):([\w.]+)>>")

    stages = raw.get("stages", [])
    if stages:
        stage_table = Table(title="Stages", show_edge=False, title_justify="left", header_style="bold cyan")
        stage_table.add_column("Name")
        stage_table.add_column("Task", style="cyan")
        stage_table.add_column("Executor", style="green")
        stage_table.add_column("Depends On", style="dim")
        stage_table.add_column("Params", style="dim")

        for s in stages:
            deps = s.get("depends_on", None)
            if isinstance(deps, list):
                deps_str = ", ".join(deps)
            elif deps:
                deps_str = str(deps)
            else:
                deps_str = "—"
            # Summarise params — show cross-stage refs with readable format
            stage_params = s.get("params", {})
            param_parts: list[str] = []
            for pk, pv in list(stage_params.items())[:4]:
                pv_str = str(pv)
                # Format sentinel strings as readable cross-stage refs
                pv_str = _ref_pattern.sub(r"stages.\1.\2", pv_str)
                param_parts.append(f"{pk}={pv_str}")
            extra = len(stage_params) - 4
            if extra > 0:
                param_parts.append(f"+{extra} more")
            params_str = ", ".join(param_parts) if param_parts else "—"
            stage_table.add_row(
                s.get("name", "?"),
                s.get("task", "?"),
                s.get("executor", "?"),
                deps_str,
                params_str,
            )

        console.print(stage_table)
        console.print()

    # Note about auto-forwarded params
    if any(s.get("depends_on") for s in stages):
        console.print(
            "[dim]Note: Params not listed for a stage may be auto-forwarded "
            "from its dependencies at run time.[/dim]"
        )
        console.print()

    # Usage example
    console.print("[dim]Usage Example:[/dim]")
    example_cmd = Text("devrun workflow run ", style="bold")
    example_cmd.append(target, style="bold cyan")
    if params:
        first_param = next(iter(params.keys()))
        example_cmd.append(f" params.{first_param}=value", style="green")
    console.print(example_cmd)


@workflow_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []},
)
def workflow_run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(None, help="Workflow config path, name, or name/variation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without submitting"),
    start_after: Optional[str] = typer.Option(None, "--start-after", help="Skip this stage and its dependencies, start from the next"),
    from_job: Optional[str] = typer.Option(None, "--from-job", help="Extract workflow params from an existing job"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    help: bool = typer.Option(False, "--help", "-h", help="Show this message and exit."),
) -> None:
    """Run a multi-stage workflow from a YAML config.

    TARGET can be a file path, a workflow name (e.g. swe_bench_workflow),
    or name/variation. Configs are resolved through the same hierarchical
    search path as task configs. Trailing arguments are OmegaConf overrides.
    """
    if help:
        if not target:
            console.print(ctx.get_help())
            raise typer.Exit()
        else:
            _show_workflow_help(target)
            raise typer.Exit()

    if not target:
        console.print("[red]Missing argument 'TARGET'.[/red]\n")
        console.print(ctx.get_help())
        raise typer.Exit(code=2)

    _setup_logging(verbose)

    # Validate --start-after requires --from-job
    if start_after and not from_job:
        console.print(
            "[red]Error:[/red] --start-after requires --from-job to provide params for skipped stages."
        )
        raise typer.Exit(code=1)

    from omegaconf import OmegaConf
    from devrun.runner import find_configs
    import devrun.keystore  # noqa: F401  — registers ${key:…} resolver
    import devrun.presets  # noqa: F401  — registers ${preset:…} resolver
    from devrun.models import WorkflowConfig
    from devrun.workflow import WorkflowRunner

    # Register the cross-stage reference resolver.  At config-load time it
    # embeds a sentinel; WorkflowRunner resolves it at submit time.
    if not OmegaConf.has_resolver("stages"):
        OmegaConf.register_new_resolver(
            "stages",
            lambda stage_name, param_key: f"<<STAGE_REF:{stage_name}:{param_key}>>",
        )

    runner = WorkflowRunner()
    task_name: Optional[str] = None
    skipped_params: dict[str, dict] = {}

    # Merge order: YAML base (hierarchical) → from-job params → CLI overrides (highest priority)
    try:
        config_paths = find_configs(target)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Config not found for '{target}'.")
        console.print("Ensure the workflow config exists in one of the config search directories.")
        raise typer.Exit(code=1)

    try:
        raw_cfg = OmegaConf.load(config_paths[0])
        for extra_path in config_paths[1:]:
            raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.load(extra_path))

        if from_job:
            # Peek at the raw config's stage tasks so negative-index lookups
            # filter to history entries this workflow can actually consume.
            allowed_tasks: set[str] = set()
            raw_stages = OmegaConf.to_container(raw_cfg, resolve=False).get("stages", [])
            if isinstance(raw_stages, list):
                for st in raw_stages:
                    if isinstance(st, dict) and st.get("task"):
                        allowed_tasks.add(str(st["task"]))
            try:
                job_params, task_name, resolved_job_id = runner.extract_workflow_params(
                    from_job, allowed_source_tasks=allowed_tasks or None
                )
            except ValueError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(code=1)
            from_job = resolved_job_id  # propagate resolved ID downstream
            if job_params:
                console.print(f"[dim]From job {resolved_job_id} ({task_name}): {list(job_params.keys())}[/dim]")
                job_overrides = [f"{k}={v}" for k, v in job_params.items()]
                raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(job_overrides))

        if ctx.args:
            console.print(f"[dim]Using overrides: {ctx.args}[/dim]")
            for arg in ctx.args:
                key, _, value = arg.partition("=")
                if key and _ == "=":
                    # Parse value type (e.g. "30" → int, "true" → bool)
                    # so numeric/boolean overrides aren't stored as strings.
                    parsed = yaml.safe_load(value)
                    OmegaConf.update(raw_cfg, key, parsed)
                else:
                    console.print(f"[yellow]Warning:[/yellow] ignoring malformed override: {arg}")

        resolved = OmegaConf.to_container(raw_cfg, resolve=True)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error loading/resolving workflow config:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        cfg = WorkflowConfig(**resolved)
    except Exception as exc:
        console.print(f"[red]Error parsing workflow config:[/red] {exc}")
        raise typer.Exit(code=1)

    # Auto-detect stage to skip when --from-job is used without --start-after,
    # and populate skipped-stage resolved_params from the source job so that
    # downstream ${stages:X,...} references resolve correctly regardless of
    # whether --start-after was supplied explicitly or auto-detected.
    if from_job and task_name is not None:
        if not start_after:
            detected_stage = runner.detect_stage_for_task(task_name, cfg)
            if detected_stage:
                start_after = detected_stage
                console.print(
                    f"[dim]Auto-detected: skipping stage '{detected_stage}' "
                    f"based on job task type '{task_name}'[/dim]"
                )

        # Find the stage whose task matches the source job; if the user gave
        # an explicit --start-after for a different stage, we only seed
        # skipped_params when the source-task stage is actually in the skip
        # set (otherwise the params would belong to a stage that still runs).
        if start_after:
            source_stage = runner.detect_stage_for_task(task_name, cfg)
            if source_stage:
                record = runner._db.get(from_job)
                if record is not None:
                    skipped_params[source_stage] = record.params_dict

    try:
        result = runner.run(
            cfg,
            dry_run=dry_run,
            start_after=start_after,
            skipped_params=skipped_params or None,
            from_job=from_job,
        )
        if dry_run:
            console.print(result)
            console.print("[yellow]Dry-run complete. No jobs were submitted.[/yellow]")
        else:
            console.print(f"[green]Workflow enqueued:[/green] {result}")
            # Check if heartbeat service is running
            try:
                from devrun.services import get_service
                service = get_service()
                if not service.is_active():
                    console.print("[yellow]⚠ Warning:[/yellow] Heartbeat scheduler is not running. Jobs will remain in QUEUED state.")
                    console.print("  Start it with: [bold]devrun heartbeat start[/bold]")
            except Exception:
                # Silently skip warning if service check fails
                pass
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@workflow_app.command("status")
def workflow_status(
    workflow_id: str = typer.Argument(..., help="Workflow ID to query"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check status of a workflow."""
    _setup_logging(verbose)

    from devrun.workflow import WorkflowRunner

    runner = WorkflowRunner()
    record = runner.status(workflow_id)
    if not record:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Workflow {workflow_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Name", record["workflow_name"])
    table.add_row("Status", record["status"])
    table.add_row("Created", record["created_at"])
    table.add_row("Completed", record.get("completed_at") or "—")

    stages = json.loads(record.get("stages_state", "{}"))
    for name, state in stages.items():
        status_str = state.get("status", "unknown")
        job_id = state.get("job_id", "—")
        table.add_row(f"  Stage: {name}", f"{status_str} (job: {job_id})")

    console.print(table)


@workflow_app.command("list")
def workflow_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max workflows to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List recent workflows."""
    _setup_logging(verbose)

    from devrun.workflow import WorkflowRunner

    runner = WorkflowRunner()
    records = runner.list_workflows(limit=limit)

    if not records:
        console.print("[dim]No workflows found.[/dim]")
        return

    table = Table(title="Recent Workflows")
    table.add_column("ID", style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Created")

    for rec in records:
        status_val = rec.get("status", "unknown")
        style = {"completed": "green", "failed": "red", "running": "yellow", "timed_out": "red"}.get(status_val, "dim")
        table.add_row(
            rec["workflow_id"],
            rec["workflow_name"],
            f"[{style}]{status_val}[/{style}]",
            rec.get("created_at", ""),
        )
    console.print(table)


@workflow_app.command("logs")
def workflow_logs(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Specific stage name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show logs for a workflow or specific stage."""
    _setup_logging(verbose)

    from devrun.workflow import WorkflowRunner

    runner = WorkflowRunner()
    try:
        output = runner.logs(workflow_id, stage=stage)
        console.print(output)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@workflow_app.command("cancel")
def workflow_cancel(
    workflow_id: str = typer.Argument(..., help="Workflow ID to cancel"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cancel all active stages of a workflow."""
    _setup_logging(verbose)

    from devrun.workflow import WorkflowRunner

    runner = WorkflowRunner()
    try:
        runner.cancel(workflow_id)
        console.print(f"[green]Workflow {workflow_id} cancelled.[/green]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Keys subcommands
# ---------------------------------------------------------------------------


keys_app = typer.Typer(name="keys", help="Manage stored secrets.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")


@keys_app.command("set")
def keys_set(
    name: str = typer.Argument(..., help="Key name"),
    value: str = typer.Argument(..., help="Secret value to store"),
) -> None:
    """Store a secret value under a given name."""
    from devrun.keystore import KeyStore

    try:
        KeyStore().set(name, value)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]Key '{name}' stored.[/green]")


@keys_app.command("get")
def keys_get(
    name: str = typer.Argument(..., help="Key name"),
) -> None:
    """Print the plain-text value for a stored key."""
    from devrun.keystore import KeyStore

    try:
        value = KeyStore().get(name)
    except KeyError:
        console.print(f"[red]Key '{name}' not found.[/red]")
        raise typer.Exit(code=1)
    console.print(value)


@keys_app.command("list")
def keys_list() -> None:
    """List all stored key names."""
    from devrun.keystore import KeyStore

    names = KeyStore().list_keys()
    if not names:
        console.print("[dim]No keys stored.[/dim]")
        return

    table = Table(title="Stored Keys")
    table.add_column("Name", style="cyan")
    for n in names:
        table.add_row(n)
    console.print(table)


@keys_app.command("delete")
def keys_delete(
    name: str = typer.Argument(..., help="Key name"),
) -> None:
    """Delete a stored key."""
    from devrun.keystore import KeyStore

    try:
        KeyStore().delete(name)
    except KeyError:
        console.print(f"[red]Key '{name}' not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Key '{name}' deleted.[/green]")


# ---------------------------------------------------------------------------
# Presets subcommands
# ---------------------------------------------------------------------------


presets_app = typer.Typer(name="presets", help="Manage reusable config presets.", no_args_is_help=True)
app.add_typer(presets_app, name="presets")


@presets_app.command("set")
def presets_set(
    field: str = typer.Argument(..., help="Preset field (namespace)"),
    name: str = typer.Argument(..., help="Preset name"),
    value: Optional[str] = typer.Argument(None, help="Value (plain string)"),
    json_value: Optional[str] = typer.Option(None, "--json", help="Value as JSON string"),
    file_path: Optional[Path] = typer.Option(None, "--file", help="Read value from YAML file"),
) -> None:
    """Store a preset value under field/name."""
    from devrun.presets import PresetStore

    sources = sum(x is not None for x in (value, json_value, file_path))
    if sources != 1:
        console.print("[red]Error:[/red] Provide exactly one of: positional value, --json, or --file.")
        raise typer.Exit(code=1)

    if json_value is not None:
        try:
            parsed = json.loads(json_value)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Error:[/red] Invalid JSON: {exc}")
            raise typer.Exit(code=1)
    elif file_path is not None:
        try:
            with open(file_path) as fh:
                parsed = yaml.safe_load(fh)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] File not found: {file_path}")
            raise typer.Exit(code=1)
        except yaml.YAMLError as exc:
            console.print(f"[red]Error:[/red] Invalid YAML: {exc}")
            raise typer.Exit(code=1)
    else:
        parsed = value

    try:
        PresetStore().set(field, name, parsed)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]Preset '{field}.{name}' stored.[/green]")


@presets_app.command("get")
def presets_get(
    field: str = typer.Argument(..., help="Preset field"),
    name: str = typer.Argument(..., help="Preset name"),
) -> None:
    """Print the value of a preset formatted as YAML."""
    from devrun.presets import PresetStore

    try:
        val = PresetStore().get(field, name)
    except KeyError:
        console.print(f"[red]Preset '{field}.{name}' not found.[/red]")
        raise typer.Exit(code=1)
    console.print(yaml.dump(val, default_flow_style=False).rstrip())


@presets_app.command("list")
def presets_list(
    field: Optional[str] = typer.Argument(None, help="Optional field to filter by"),
) -> None:
    """List stored presets."""
    from devrun.presets import PresetStore

    presets = PresetStore().list_presets(field=field)
    if not presets:
        console.print("[dim]No presets stored.[/dim]")
        return

    table = Table(title="Stored Presets")
    table.add_column("Field", style="cyan")
    table.add_column("Name", style="green")
    for f, names in presets.items():
        for n in names:
            table.add_row(f, n)
    console.print(table)


@presets_app.command("delete")
def presets_delete(
    field: str = typer.Argument(..., help="Preset field"),
    name: str = typer.Argument(..., help="Preset name"),
) -> None:
    """Delete a stored preset."""
    from devrun.presets import PresetStore

    try:
        PresetStore().delete(field, name)
    except KeyError:
        console.print(f"[red]Preset '{field}.{name}' not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Preset '{field}.{name}' deleted.[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
