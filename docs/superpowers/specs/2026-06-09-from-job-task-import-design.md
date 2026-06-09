# Task Parameter Import via `--from-job` Flag

**Date:** 2026-06-09  
**Status:** Approved  
**Author:** AI Assistant

## Motivation

Users running multi-stage SWE-bench pipelines often execute `swe_bench_agentic` (inference) jobs first, then need to run `swe_bench_collect` with matching parameters. Manually copying parameters (output_dir, dataset, model_name, etc.) is error-prone and tedious.

The `swe_bench_workflow` already handles this via `workflow run --from-job <job_id>`, which:
- Extracts params from an existing job
- Auto-detects which stage to skip
- Merges params into the workflow config

This design extends the same interface to single-task runs via `devrun run --from-job <job_id>`, maintaining **interface consistency** between task and workflow commands.

## Use Case

```bash
# Run inference
devrun run swe_bench_agentic params.model_name=gpt-4 params.dataset=/data/swe-bench
# → creates job abc123

# Collect outputs (manual parameter copying - current)
devrun run swe_bench_collect \
  params.output_dir=logs/run1 \
  params.dataset=/data/swe-bench \
  params.model_name_or_path=gpt-4

# Collect outputs (with --from-job - proposed)
devrun run swe_bench_collect --from-job abc123
# Automatically imports: output_dir, dataset, split, model_name → model_name_or_path
```

## Design

### CLI Interface

Add `--from-job <job_id>` option to `devrun run`:

```python
@app.command("run")
def run(
    ctx: typer.Context,
    target: Optional[str] = ...,
    from_job: Optional[str] = typer.Option(None, "--from-job", help="Import params from existing job"),
    dry_run: bool = ...,
    verbose: bool = ...,
) -> None:
```

**Merge order** (same as workflow):
1. YAML config base (hierarchical: repo default < user < project)
2. `--from-job` extracted params
3. CLI trailing overrides (highest priority)

### Architecture

#### 1. TaskRunner.extract_task_params()

New method in `devrun/runner.py`:

```python
def extract_task_params(self, job_id: str, target_task: str) -> dict[str, str]:
    """Extract task-level params from an existing job record for a target task.
    
    Returns dotlist dict with keys like "params.output_dir" suitable for
    OmegaConf merging.
    
    Args:
        job_id: Source job ID
        target_task: Target task type (e.g., "swe_bench_collect")
        
    Returns:
        Dict mapping dotlist keys (params.X) to string values
        
    Raises:
        ValueError: If job not found or target task doesn't support import
    """
    record = self._db.get(job_id)
    if record is None:
        raise ValueError(
            f"Job '{job_id}' not found. Use `devrun history` to find job IDs."
        )
    
    source_task = record.task_name
    source_params = record.params_dict
    
    # Get target task class and delegate to import_from_job
    target_task_cls = get_task_class(target_task)
    imported = target_task_cls.import_from_job(source_task, source_params)
    
    # Convert to dotlist format
    dotlist = {f"params.{k}": str(v) for k, v in imported.items() if v}
    
    logger.info(
        "Extracted %d params from job %s (%s) for target task %s: %s",
        len(dotlist), job_id, source_task, target_task, list(dotlist.keys()),
    )
    return dotlist
```

#### 2. BaseTask.import_from_job() classmethod

New classmethod in `devrun/tasks/base.py`:

```python
class BaseTask:
    """Base interface for all task plugins."""
    
    @classmethod
    def import_from_job(cls, source_task: str, source_params: dict[str, Any]) -> dict[str, Any]:
        """Import parameters from a source job's params for this task type.
        
        Override in subclasses to define cross-task parameter mappings.
        Default implementation returns empty dict (no import support).
        
        Args:
            source_task: Source job's task_name (e.g., "swe_bench_agentic")
            source_params: Source job's params_dict
            
        Returns:
            Dict mapping this task's param names to values
            
        Raises:
            ValueError: If source_task is not supported for import
        """
        return {}
```

#### 3. SWEBenchCollectTask.import_from_job()

Implementation in `devrun/tasks/swe_bench_collect.py`:

```python
@classmethod
def import_from_job(cls, source_task: str, source_params: dict[str, Any]) -> dict[str, Any]:
    """Import parameters from swe_bench_agentic jobs.
    
    Mapping:
        output_dir: direct copy (or derive from logs_dir/run_name if not set)
        dataset: direct copy
        split: direct copy (default: test)
        model_name → model_name_or_path: rename
        working_dir: direct copy
    """
    if source_task != "swe_bench_agentic":
        raise ValueError(
            f"SWEBenchCollectTask can only import from 'swe_bench_agentic' jobs, "
            f"got '{source_task}'"
        )
    
    imported = {}
    
    # Direct mappings
    for key in ["dataset", "split", "working_dir"]:
        if key in source_params and source_params[key]:
            imported[key] = source_params[key]
    
    # output_dir: direct copy or derive
    if "output_dir" in source_params and source_params["output_dir"]:
        imported["output_dir"] = source_params["output_dir"]
    else:
        # Derive from logs_dir + run_name (matching swe_bench_agentic default)
        logs_dir = source_params.get("logs_dir", "logs")
        run_name = source_params.get("run_name", "run1")
        imported["output_dir"] = f"{logs_dir}/{run_name}"
    
    # Rename: model_name → model_name_or_path
    if "model_name" in source_params and source_params["model_name"]:
        imported["model_name_or_path"] = source_params["model_name"]
    
    # Default split if not present
    if "split" not in imported:
        imported["split"] = "test"
    
    return imported
```

### CLI Integration

Update `devrun/cli.py` run command:

```python
def run(
    ctx: typer.Context,
    target: Optional[str] = ...,
    from_job: Optional[str] = typer.Option(None, "--from-job", help="Import params from existing job"),
    dry_run: bool = ...,
    verbose: bool = ...,
) -> None:
    """Submit a task using a config file or variation. Trailing arguments are OmegaConf overrides."""
    
    _setup_logging(verbose)
    
    from omegaconf import OmegaConf
    import devrun.keystore
    import devrun.presets
    
    runner = _runner()
    
    # Load base config via hierarchical merge
    config_paths = runner._find_configs(target)
    raw_cfg = OmegaConf.load(config_paths[0])
    for extra_path in config_paths[1:]:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.load(extra_path))
    
    # Extract task type for import
    task_type = OmegaConf.to_container(raw_cfg, resolve=True).get("task")
    if not task_type:
        console.print("[red]Error:[/red] Config missing 'task' field.")
        raise typer.Exit(code=1)
    
    # Apply --from-job params
    if from_job:
        try:
            job_params = runner.extract_task_params(from_job, task_type)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)
        if job_params:
            console.print(f"[dim]From job {from_job}: {list(job_params.keys())}[/dim]")
            job_overrides = [f"{k}={v}" for k, v in job_params.items()]
            raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(job_overrides))
    
    # Apply CLI trailing overrides
    if ctx.args:
        console.print(f"[dim]Using overrides: {ctx.args}[/dim]")
        for arg in ctx.args:
            key, _, value = arg.partition("=")
            if key and _ == "=":
                parsed = yaml.safe_load(value)
                OmegaConf.update(raw_cfg, key, parsed)
            else:
                console.print(f"[yellow]Warning:[/yellow] ignoring malformed override: {arg}")
    
    # Convert to plain dict for runner
    overrides = []
    resolved = OmegaConf.to_container(raw_cfg, resolve=True)
    
    # Convert resolved config to override list format
    # The existing runner.run() can handle this via the overrides parameter
    overrides_list = []
    for key, value in resolved.get("params", {}).items():
        overrides_list.append(f"params.{key}={value}")
    
    job_ids = runner.run(target, overrides=overrides_list, dry_run=dry_run)
    
    for job_id in job_ids:
        console.print(f"[green]✓[/green] Job submitted: {job_id}")
```

**Implementation Note:** The existing `TaskRunner.run()` already handles the full merge via `_load_config()`, which calls `load_merged_config()`. We need to **bypass** `_load_config()` when `--from-job` is used, since we've already done the merge in the CLI layer.

**Solution:** Add a private `TaskRunner._run_from_merged_config()` method that skips `_load_config()` and goes straight to sweep expansion + submission:

```python
def _run_from_merged_config(self, cfg_dict: dict, dry_run: bool = False) -> list[str]:
    """Internal: run with pre-merged config dict, bypassing _load_config()."""
    cfg = TaskConfig(**cfg_dict)
    param_combos = self._expand_sweep(cfg)
    # ... rest of run() logic
```

The CLI `run()` command will call `_run_from_merged_config()` when `--from-job` is used, otherwise delegates to the existing `run()` method.

## Testing Plan

### Unit Tests

**File:** `tests/test_runner.py`

1. **Test `TaskRunner.extract_task_params()` method:**
   - Valid job exists, target task supports import → returns dotlist dict
   - Job not found → raises ValueError
   - Target task doesn't support source task → raises ValueError

2. **Test `BaseTask.import_from_job()` default:**
   - Returns empty dict

3. **Test `SWEBenchCollectTask.import_from_job()`:**
   - Source task = `swe_bench_agentic`, all params present → maps correctly
   - Source task = `swe_bench_agentic`, missing `output_dir` → derives from `logs_dir/run_name`
   - Source task = `swe_bench_agentic`, missing `split` → defaults to "test"
   - Source task = other → raises ValueError

**File:** `tests/test_cli.py`

4. **Test CLI integration:**
   - `devrun run swe_bench_collect --from-job abc123` → calls `extract_task_params` and merges
   - Merge order: YAML < from-job < CLI overrides
   - Invalid job ID → shows error message

### Integration Test

**File:** `tests/test_e2e.py`

5. **End-to-end test:**
   - Run `swe_bench_agentic` (dry-run), capture job record
   - Run `swe_bench_collect --from-job <job_id>` (dry-run)
   - Verify collected params match source job params with correct mapping

## Future Extensibility

The `import_from_job()` classmethod pattern allows easy extension:

1. **Other task pairs:**
   - `SWEBenchEvalTask.import_from_job("swe_bench_collect", ...)` could extract `predictions_path`, `dataset`
   - Custom task classes can define their own mappings

2. **Generic fallback:**
   - Could add a `BaseTask._generic_import()` helper that maps overlapping param names 1:1
   - Individual tasks can override for custom logic (renaming, derivation)

3. **Validation hooks:**
   - `import_from_job()` could call a validation method to check compatibility
   - E.g., warn if source job's dataset doesn't match target's expected format

## Edge Cases

1. **Source job has no output_dir (agentic job used default):**
   - Derive from `logs_dir/run_name` (matches agentic default)

2. **User overrides imported param on CLI:**
   - CLI override takes precedence (merge order)

3. **Target task doesn't support source task:**
   - Raise clear ValueError with supported source tasks

4. **Sensitive params (api_key, token):**
   - Skip in generic fallback (if implemented later)
   - Explicit mappings don't copy sensitive keys

## Summary

This design adds `--from-job` to `devrun run`, mirroring the existing workflow interface:

- **CLI:** `devrun run <target> --from-job <job_id> [overrides...]`
- **Mechanism:** Task classes define `import_from_job()` classmethods for cross-task param mapping
- **Merge order:** YAML base → from-job → CLI overrides
- **Initial scope:** `swe_bench_agentic` → `swe_bench_collect`
- **Extensible:** Other tasks can add support via classmethod override

This maintains interface consistency between single-task and workflow commands while staying focused on the immediate use case.
