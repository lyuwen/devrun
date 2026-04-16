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
    """List available task and executor plugins, including config variations."""
    from devrun.runner import get_config_dirs

    table = Table(title="Registered Plugins")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Variations", style="dim")

    # Collect variations per task by scanning config directories
    config_dirs = get_config_dirs()
    task_variations: dict[str, set[str]] = {}
    for d in config_dirs:
        if not d.is_dir():
            continue
        for child in d.iterdir():
            if child.is_dir():
                for yaml_file in child.glob("*.yaml"):
                    stem = yaml_file.stem
                    task_variations.setdefault(child.name, set()).add(stem)

    for name in list_tasks():
        variations = sorted(task_variations.get(name, set()))
        variations_str = ", ".join(f"{name}/{v}" for v in variations if v != "default")
        table.add_row("task", name, variations_str)
    for name in list_executors():
        table.add_row("executor", name, "")
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


workflow_app = typer.Typer(name="workflow", help="Manage multi-stage workflows.")
app.add_typer(workflow_app, name="workflow")


@workflow_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def workflow_run(
    ctx: typer.Context,
    config_path: str = typer.Argument(..., help="Path to workflow YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without submitting"),
    start_after: Optional[str] = typer.Option(None, "--start-after", help="Skip this stage and its dependencies, start from the next"),
    from_job: Optional[str] = typer.Option(None, "--from-job", help="Extract workflow params from an existing job"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Run workflow in background, return immediately"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a multi-stage workflow from a YAML config. Trailing arguments are passed as OmegaConf overrides."""
    _setup_logging(verbose)

    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[red]Error:[/red] Config not found: {config_path}")
        raise typer.Exit(code=1)

    from omegaconf import OmegaConf
    import devrun.keystore  # noqa: F401  — registers ${key:…} resolver
    import devrun.presets  # noqa: F401  — registers ${preset:…} resolver
    from devrun.models import WorkflowConfig
    from devrun.workflow import WorkflowRunner

    runner = WorkflowRunner()
    task_name: Optional[str] = None

    # Merge order: YAML base → from-job params → CLI overrides (highest priority)
    try:
        raw_cfg = OmegaConf.load(config_file)

        if from_job:
            try:
                job_params, task_name = runner.extract_workflow_params(from_job)
            except ValueError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(code=1)
            if job_params:
                console.print(f"[dim]From job {from_job}: {list(job_params.keys())}[/dim]")
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

    # Auto-detect stage to skip when --from-job is used without --start-after
    if from_job and not start_after and task_name is not None:
        detected_stage = runner.detect_stage_for_task(task_name, cfg)
        if detected_stage:
            start_after = detected_stage
            console.print(
                f"[dim]Auto-detected: skipping stage '{detected_stage}' "
                f"based on job task type '{task_name}'[/dim]"
            )

    try:
        if detach:
            if dry_run:
                console.print("[red]Error:[/red] --detach and --dry-run cannot be used together.")
                raise typer.Exit(code=1)
            wf_id = runner.run_detached(cfg, start_after=start_after)
            console.print(
                f"[green]Workflow {wf_id} started in background.[/green]\n"
                f"Use [bold]devrun workflow status {wf_id}[/bold] to monitor."
            )
        else:
            result = runner.run(cfg, dry_run=dry_run, start_after=start_after)
            if dry_run:
                console.print(result)
                console.print("[yellow]Dry-run complete. No jobs were submitted.[/yellow]")
            else:
                console.print(f"[green]Workflow completed:[/green] {result}")
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


keys_app = typer.Typer(name="keys", help="Manage stored secrets.")
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


presets_app = typer.Typer(name="presets", help="Manage reusable config presets.")
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
