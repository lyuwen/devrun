# Devrun Project Overview

**Devrun** is a modular developer task orchestration system for running parameterised tasks across heterogeneous compute backends.
It is built heavily with Python, Typer (CLI), Pydantic (data validation), PyYAML (declarative configuration), and SQLite (state).

## Architecture & Data Flow

```text
PRODUCER SIDE:
CLI (typer)
  ↓
TaskRunner / WorkflowRunner  ← enqueue jobs to QUEUED, return immediately
  ↓
JobStore (SQLite)

CONSUMER SIDE (async):
HeartbeatScheduler           ← tick loop drives all async work
  ↓
JobStore (read QUEUED/SUBMITTED/RUNNING/CANCELING)
  ↓
ExecutorRouter               ← resolves executor name → instance
  ↓
Executor Plugins             ← LocalExecutor | SSHExecutor | SlurmExecutor | HTTPExecutor
  ↓
Compute Backends
```

All interactions are driven entirely by YAML configuration files.

## Critical Design Principle: Producer-Consumer Separation

**IMPORTANT**: The job queuing (producer) and execution/submission (consumer) are completely decoupled:

- **Producer side** (`devrun run`, `devrun workflow run`): Runs in the user's local environment with their local file paths, environment variables, and working directory. Writes jobs to SQLite as QUEUED with `params_template` (unresolved YAML) and `parameters` (dict snapshot).

- **Consumer side** (`devrun heartbeat`): Runs as a background service, potentially in a different environment, with different paths and env vars. Reads QUEUED jobs, resolves `${jobs:...}` references via `JobRefContext`, expands templates, and submits to executors.

**Key implications for development:**

1. **Variable tracking**: All parameter references (`${jobs:...}`, `${stages:...}`) must be rewritten at enqueue time to use job IDs, not stage names. The consumer has no access to the workflow config or stage names—only the `params_template` stored in the job row.

2. **Environment isolation**: Do NOT assume the heartbeat sees the same environment as the producer. Paths, env vars, and working directories can differ. All necessary context must be serialized into the job row.

3. **Template resolution timing**: `params_template` stored at enqueue is the YAML string with `${jobs:...}` placeholders. Resolution happens at promotion time in the heartbeat, not at enqueue time.

4. **Idempotency**: The same `params_template` must resolve to the same command/env/resources regardless of when/where the heartbeat runs, given the same parent job outputs.

5. **Breaking changes**: Any change to how `params_template` is generated, how references are rewritten, or how resolution works at promotion time is a **breaking change** that can cause in-flight workflows to fail. Test thoroughly with workflows that have cross-stage dependencies.

## High-Level Modules

* **`devrun/models.py`:** Central Pydantic models. Core entity is `TaskSpec` which holds the final executed `command`, `env`, and `resources` and is returned by `Task.prepare()`.
* **`devrun/registry.py`:** Custom decorator-based registry (`@register_task`, `@register_executor`) for automated discovery of plugins.
* **`devrun/runner.py`:** Orchestration logic. Loads YAML configs, expands sweeps via Cartesian product, invokes task preparation, writes to DB, invokes executor plugin `submit()`, and updates DB on error. Exposes two public module-level functions shared by both `TaskRunner` and the workflow CLI: `find_configs(target, config_dirs=None)` resolves a target name or path to config file paths across the 3-layer search directories (repo defaults, user overrides, project overrides); `load_merged_config(target, overrides=None, config_dirs=None)` loads and deep-merges those configs via OmegaConf, applies CLI overrides, and returns a resolved dict. `TaskRunner._find_configs` and `_load_config` delegate to these functions internally. `TaskRunner.extract_task_params(job_id, target_task)` translates a source job's stored params into the target task's schema by delegating to that task's `BaseTask.import_from_job` classmethod — this powers `devrun run <task> --from-job <id>` and mirrors `WorkflowRunner.extract_workflow_params` at the task level.
* **`devrun/cli.py`:** User entry point with Typer commands: `run`, `list`, `status`, `logs`, `history`, `rerun`, `sync`, `fetch`, plus the `workflow` subcommand group (`workflow run`, `workflow status`, `workflow list`, `workflow logs`, `workflow cancel`). All four Typer apps (main, workflow, keys, presets) set `no_args_is_help=True` so invoking any group without arguments displays its help text. Supports command-line autocompletion for tasks and context-aware task help via `devrun run <task> --help`. `devrun list` discovers and displays both task configs and workflow configs (those with a `workflow` top-level key), showing each entry's type accordingly. `devrun run` accepts `--after <job_id>` to add a dependency edge and `--allow-failure-from <subset>` to mark edges as allow_failure. `devrun workflow run` accepts `--from-job <job_id>` and `--start-after <stage>` (requires --from-job) to import params from a previous job; the source job's params are translated to the target's schema (via `BaseTask.import_from_job` for tasks, `_PARAM_MAPPING` for workflows) and merged before CLI overrides. The job id may also be a negative integer (`-1`, `-2`, …) referring to the N-th most recent compatible job in history — for `devrun run` "compatible" means the source task's params translate via the target's `import_from_job`; for `workflow run` it means the source task matches at least one stage of the workflow. `workflow run` accepts a target as a name, name/variation, or literal file path, using the same hierarchical config resolution as task configs via `load_merged_config()` (3-layer merge: repo defaults < user overrides < project overrides). It also accepts `--start-after <stage>` (requires `--from-job`) to skip completed stages, `--dry-run` for plan preview, and trailing OmegaConf overrides (e.g. `params.model_name=X`). Merge order for both `run` and `workflow run`: YAML base → from-job params → CLI overrides.
* **`devrun/db/jobs.py`:** Persistent SQLite job store with both `jobs` and `workflows` tables.
* **`devrun/workflow.py`:** Multi-stage workflow producer. Builds stage plan, validates `<REQUIRED:…>` placeholders, calls `JobStore.enqueue_workflow()` atomically, and returns immediately. No polling loop. Supports `--start-after <stage>` (requires `--from-job`) to skip upstream stages, `--from-job` to import params and rewrite `${stages:...}` → `${jobs:...}` references, `extract_workflow_params()` to pull params from existing jobs, and dry-run mode for plan preview. Workflow configs go through the same hierarchical search and OmegaConf merge pipeline as task configs via `load_merged_config()`.
* **`devrun/heartbeat.py`:** Global scheduler and async consumer. Loops `tick()` at a configurable interval. Each tick runs six phases in order: stale-lease reclaim, workflow-deadline expiry, cascade-skip of dependents, promotion of ready `QUEUED` jobs (claim CAS → resolve params with `JobRefContext` → executor submit → `finalize_submit`), poll of active jobs (SUBMITTED/RUNNING/CANCELING), and workflow status aggregation (aggregate stage job states into workflow status: all completed/skipped → completed, any failed/cancelled/timed_out → failed). Producers (`devrun run`, `devrun workflow run`) enqueue to QUEUED; heartbeat drives all async work.
* **`devrun/services/`:** Cross-platform service management. `get_service()` dispatches on `sys.platform` to `SystemdUserService` (Linux, `systemctl --user`) or `LaunchdService` (macOS, `launchctl` + LaunchAgent plist). Both implement `install/uninstall/start/stop/restart/is_active`.
* **`devrun/cli_heartbeat.py`:** Typer subapp wired as `devrun heartbeat`. Subcommands: foreground (default), `install`, `uninstall`, `start`, `stop`, `restart`, `status`.
* **`devrun/utils/templates.py`:** Jinja2 template rendering utility with `shell_quote` filter (`shlex.quote`) and `StrictUndefined` mode. Templates live in `devrun/templates/`.
* **`devrun/utils/swebench.py`:** Shared SWE-bench utilities including `derive_ds_dir()` for consistent DS_DIR path naming across tasks.
* **`configs/`:** Example YAML configurations for execution (`executors.yaml`), deployment (`deploy_ray.yaml`), and logic (`eval_math.yaml`, `eval_sweep.yaml`, `inference.yaml`).

## Available Plugin Systems

### Executors
All subclasses of `BaseExecutor` (`devrun/executors/base.py`) must implement `submit()`, `status()`, and `logs()`.
Registered types: `local`, `ssh`, `slurm`, `http`.

* **`LocalExecutor`:** Executes locally using `subprocess`, saving stout logs in `.devrun/logs/`.
* **`SSHExecutor`:** Remote execution using `nohup bash` with a heredoc to safely pass arbitrary commands. Each job is tracked by a composite `pid:token` job ID where `token` is a UUID-derived string used as the remote log file name (`/tmp/devrun_ssh_{token}.log`). This ensures `logs()`, `status()`, and `cancel()` all reliably find the right file and process even after the SSH session ends.
* **`SlurmExecutor`:** Generates `sbatch` scripts natively with unique UUID-suffixed filenames to prevent sweep collisions. If the `host` property is specified, it automatically uploads the script via `scp` and submits it via SSH, allowing local machines to natively dispatch to remote head nodes. After submission, the absolute log path is stored in `task_spec.metadata["log_path"]`, persisted to the DB, and used by `logs()` for reliable retrieval regardless of CWD. Status tracking uses `sacct --json` as the primary source (covers both active and finished jobs), with `squeue --json` as a fallback for very fresh jobs. Array jobs are aggregated: task states are counted and an overall status is derived (`running` if any active, `completed` if all done, `failed` if any failed). The optional `progress()` method returns per-task state counts for array job progress tracking by merging sacct (authoritative for terminal states) with squeue (catches pending/active tasks sacct may miss), using `merge_array_counts()`. The sacct result from the preceding `status()` call is cached in `_status_cache` to avoid a duplicate SSH round-trip.
* **`HTTPExecutor`:** JSON POST payload to a REST API.

### File Synchronization (`devrun/utils/sync.py`)
Provides native `devrun sync` and `devrun fetch` CLI wrappers over `rsync` to pull/push artifacts natively between the host and remote target aliases (e.g. `devrun sync ./data swedev2:/data`).

### Tasks
All subclasses of `BaseTask` (`devrun/tasks/base.py`) must implement `prepare(params: dict) -> TaskSpec`.
Registered types: `eval`, `inference`, `deploy_ray`, `swe_bench_eval`, `swe_bench_agentic`, `swe_bench_collect`.

* **`swe_bench_agentic`:** Generates Slurm array job scripts for OpenHands SWE-bench inference via Jinja2 template (`swe_bench_agentic.sh.j2`). Includes retry loop, completion checking, failed-run archiving, and `set_e=False` metadata for the executor. Uses `derive_ds_dir()` for consistent DS_DIR naming.
* **`swe_bench_collect`:** Scans inference output directories and produces `predictions.jsonl` via `jq`. Uses `json.dumps()` (not `shlex.quote()`) for values inside jq expressions. Bridges inference outputs to evaluation input. Implements `import_from_job("swe_bench_agentic", source_params)` so `devrun run swe_bench_collect --from-job <agentic_job_id>` forwards `output_dir` (or `{logs_dir}/{run_name}`), `dataset`, `split`, `working_dir`, and renames `model_name` → `model_name_or_path`.
* **`swe_bench_eval`:** Runs `swebench.harness.run_evaluation` with configurable resources.

## Database Schema (`devrun/db/jobs.py`)

Table: `jobs`
Columns: `job_id`, `task_name`, `executor`, `parameters`, `params_template`, `remote_job_id`, `status`, `created_at`, `completed_at`, `log_path`, `skip_reason`, `claimed_by`, `claimed_at`, `claim_expires_at`

Table: `workflows`
Columns: `workflow_id`, `workflow_name`, `stages_state` (JSON), `status`, `created_at`, `completed_at`, `deadline_at`

Table: `job_dependencies`
Columns: `child_job_id`, `parent_job_id`, `allow_failure`

Table: `workflow_jobs`
Columns: `workflow_id`, `stage_name`, `ordinal`, `job_id`, `source_job_id`

## Heartbeat Scheduler

The global heartbeat scheduler (`devrun heartbeat`) drains the `QUEUED` job queue through the dependency-aware promotion pipeline. It runs as a background service (systemd `--user` on Linux, launchd LaunchAgent on macOS) and executes a five-phase tick every 10 seconds:

1. **Reclaim stale leases** — `claim_expires_at < now` SUBMITTING rows with no `remote_job_id` are returned to QUEUED with a `skip_reason` annotation.
2. **Expire workflow deadlines** — workflows past `deadline_at` get every non-terminal job transitioned (QUEUED→SKIPPED, SUBMITTED/RUNNING→CANCELING) and the workflow row → TIMED_OUT.
3. **Cascade-skip dependents** — children whose blocking parent (`allow_failure=0`) reached a terminal failure state (failed/skipped/cancelled/timed_out) are flipped to SKIPPED with `skip_reason='parent <id> <status>'`.
4. **Promote ready-queued** — for each QUEUED job whose every parent edge is satisfied: claim CAS to SUBMITTING, install a `JobRefContext`, resolve `${jobs:<id>,<path>}` references via OmegaConf, scan for residual `<REQUIRED:…>`, prepare the task, submit via the executor router, and atomically `finalize_submit` to SUBMITTED with the resolved parameters + remote_job_id + log_path persisted.
5. **Poll active jobs** — SUBMITTED/RUNNING/CANCELING rows have `executor.status()` polled and remapped through `map_status()`; CANCELING entries call `executor.cancel(remote_id)` then unconditionally transition to CANCELLED.

CLI surface:
- `devrun heartbeat` — foreground loop (refuses when the managed service is already active).
- `devrun heartbeat run` — foreground loop alias that skips the service-active guard (user-supervised).
- `devrun heartbeat install` — write the service unit/plist and enable it (resolves `python_path=sys.executable`, `db_path=default_db_path()`).
- `devrun heartbeat start|stop|restart` — delegate to the platform service backend.
- `devrun heartbeat status` — print `is_active()`, job-status counts (via `JobStore.status_counts()`), and the last-tick timestamp from `~/.devrun/heartbeat.tick`.
- `devrun heartbeat uninstall` — stop, disable, remove the unit/plist, daemon-reload.

Service files: `~/.config/systemd/user/devrun-heartbeat.service` (Linux) or `~/Library/LaunchAgents/com.devrun.heartbeat.plist` (macOS). The systemd unit ExecStart is `{{ python_path }} -m devrun.heartbeat --db {{ db_path }}` with `Restart=on-failure` and `KillSignal=SIGTERM` so `_shutdown_event` unwinds cleanly after the current tick. PR2 lands the scheduler and CLI; producers (`devrun run`, `devrun workflow run`) still use legacy in-process polling — that flips in PR3.

## Development Specifics

* **Dependency Management:** Configured via `pyproject.toml` using standard `setuptools.build_meta` build backend.
* **Environment:** Installed globally in local `.venv`.
* **Formatting/Linting:** Not explicitly enforced, but code follows typing-heavy, strict schema practices. Supports Python 3.10+.
* **Issue to note:** Handled a regression related to Pydantic v2 incompatibilities in `models.py` (migrated away from `__root__` logic and inner `class Config` usage to `model_config = {}` dictionary and `.model_dump(mode="json")` for parsing `run_history` datetimes).
* **JobStore typed API surface (PR1):** The `JobStore` now exposes a dependency-aware API surface for the new dependency model: `enqueue`, `enqueue_workflow`, `insert_dependency`, `claim_for_submit` (CAS), `finalize_submit`, `fail_promotion`, `reclaim_expired_leases`, `cascade_skip_dependents`, `fetch_ready_queued`, `fetch_active_jobs`, `fetch_expired_workflows`, `expire_workflow`, `request_cancel`, `list_dependencies`, `get_parent_parameters`, `get_workflow_stages`. New `JobStatus` values: `QUEUED`, `SUBMITTING`, `CANCELING`, `SKIPPED`, `TIMED_OUT`. New module `devrun/jobref.py` registers a promotion-scoped `${jobs:<id>,<path>}` OmegaConf resolver. New helper `devrun.runner.load_merged_omegaconf` returns the merged config without resolving interpolations. PR1 lands schema + APIs only; no producer/consumer code uses them yet — that's PR2 (heartbeat) and PR3 (producer flip).
* **Shell safety:** All executor plugins use `shlex.quote()` when interpolating user-supplied values (env vars, paths, command args) into shell strings. SSH commands use heredocs rather than `bash -c '...'` wrapping to handle single quotes in commands. Jinja2 templates use the `shell_quote` filter for shell contexts and `json.dumps()` for jq expression values.
* **Datetime:** All timestamps use `datetime.now(timezone.utc)` (timezone-aware). `datetime.utcnow()` is deprecated in Python 3.12+ and must not be re-introduced.
* **Log path propagation pattern:** When an executor knows the log path at submit time, it writes it to `task_spec.metadata["log_path"]`. The runner reads this after `submit_with_retry()` and passes it to `db.update_status(..., log_path=...)`. The runner then retrieves `record.log_path` and passes it as `executor.logs(remote_id, log_path=record.log_path)`. All `logs()` implementations accept the optional `log_path` kwarg.
* **set_e passthrough:** Tasks that need `set -x` without `set -e` (e.g., retry loops) set `metadata["set_e"] = False`. `SlurmExecutor.submit()` reads this and passes it to `generate_sbatch_script(set_e=...)`. The default is `True` (preserving `set -ex` for all existing tasks).
* **Workflow OmegaConf overrides:** `workflow run` resolves `${params.X}` interpolations via OmegaConf. Merge order: YAML base → `--from-job` extracted params → CLI trailing overrides. From-job params use `OmegaConf.merge()` (dict-only keys), while CLI overrides use `OmegaConf.update()` per-key to correctly handle list-indexed paths like `stages.0.params.X`. The `_PARAM_MAPPING` dict in `WorkflowRunner.extract_workflow_params()` maps task-level param keys to workflow-level dotlist keys.
* **Inter-stage parameter passing:** Downstream workflow stages can reference params from completed upstream stages via two mechanisms:
* **Workflow placeholder validation:** Required params use `<REQUIRED: description>` markers. `WorkflowRunner._validate_no_placeholders()` matches `^<REQUIRED(?::\s*.*?)?>$` and raises `ValueError` listing all unfilled fields before any submission.

## Testing

The project uses **pytest** for testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_cli.py

# Run with verbose output
python -m pytest tests/ -v
```

### Test Structure

| File | Description |
|------|-------------|
| `conftest.py` | Shared pytest fixtures (temp directories, `tmp_path`-backed SQLite job store, executors.yaml, CLI runner) |
| `test_models.py` | Unit tests for Pydantic models (JobStatus, TaskSpec, TaskConfig, ExecutorEntry, JobRecord) |
| `test_registry.py` | Tests for plugin registry decorators (@register_task, @register_executor) |
| `test_db.py` | Tests for SQLite job store operations (insert, update_status, get, list_all) |
| `test_router.py` | Tests for executor configuration loading and resolution |
| `test_runner.py` | Tests for TaskRunner orchestration (config loading, sweep expansion, job submission) |
| `test_tasks.py` | Tests for task plugins (EvalTask and BaseTask interface) |
| `test_executors.py` | Tests for executor plugins (LocalExecutor, BaseExecutor) |
| `test_cli.py` | Tests for all CLI commands (run, list, status, logs, history, rerun, cancel, sync, fetch, workflow subcommands) |
| `test_utils.py` | Tests for utility functions (sync, SSH, Slurm including set_e and --output/--error dedup) |
| `test_e2e.py` | End-to-end integration tests for complete workflows |
| `test_ssh_executor.py` | Unit tests for SSHExecutor: composite job ID, log token stability, shell quoting |
| `test_swe_bench_eval.py` | Unit tests for SWEBenchEvalTask: placeholder validation, command generation, shlex safety |
| `test_swe_bench_agentic.py` | Unit tests for SWEBenchAgenticTask: Jinja2 template rendering, retry loop, ds_dir derivation, set_e metadata |
| `test_swe_bench_collect.py` | Unit tests for SWEBenchCollectTask: jq command generation, DS_DIR consistency, json escaping |
| `test_templates.py` | Unit tests for Jinja2 template utility and swe_bench_agentic.sh.j2 template |
| `test_swebench_utils.py` | Unit tests for derive_ds_dir shared utility |
| `test_slurm_status.py` | Unit tests for Slurm job status JSON parsing (sacct/squeue), array status aggregation, and SlurmExecutor status/progress methods |
| `test_workflow_models.py` | Unit tests for WorkflowStage and WorkflowConfig Pydantic models |
| `test_workflow.py` | Unit tests for WorkflowRunner: dependency ordering, failure handling, timeout, cancel, logs |
| `test_workflow_simulation.py` | Simulation tests verifying full execution plan consistency across stages |
| `test_heartbeat_loop.py` | `tick()` no-op against empty DB, poll-active phase status transitions (SUBMITTED→RUNNING, RUNNING→COMPLETED/FAILED), skip of QUEUED/terminal jobs, executor-status exception is swallowed |
| `test_heartbeat_cascade.py` | Cascade-skip across ticks: single-hop, allow_failure respected, multi-hop A→B→C across two ticks, idempotency |
| `test_heartbeat_promotion.py` | Promotion phase: QUEUED→SUBMITTED happy path, resolved-params persisted, blocked-by-dependency skip; failure paths (unfilled `<REQUIRED:…>`, unauthorized `${jobs:…}` ref, executor.submit_with_retry raises, claim columns cleared on failure) |
| `test_heartbeat_claim.py` | Lease reclaim: expired SUBMITTING + NULL remote_job_id → QUEUED with `skip_reason` annotation, rows with remote_job_id are untouched, live lease is untouched |
| `test_heartbeat_timeout.py` | Workflow deadline expiry across a tick: QUEUED→SKIPPED, SUBMITTED/RUNNING→CANCELING, workflow→TIMED_OUT |
| `test_heartbeat_cancel.py` | CANCELING → CANCELLED after executor.cancel(); executor.cancel() raising still finalizes CANCELLED; cancel branch does not call status(); plain RUNNING jobs are untouched |
| `test_heartbeat_service.py` | systemd + launchd backends with mocked subprocess: platform dispatch, install path + daemon-reload + enable, start/stop/restart, uninstall flow, is_active returncode mapping |
| `test_cli_heartbeat.py` | `devrun heartbeat` Typer subapp: `--help`, `status` against isolated DB, foreground refusal when service is active, install kwargs forwarding, start/stop/restart/uninstall delegation |
| `test_data/sample_configs/` | Sample YAML config files for testing |

### Test Coverage

- **889 tests passing**, **16 skipped** (infrastructure-dependent: require real SSH/Slurm connectivity)
- Unit tests for all major components (models, registry, database, router, runner, tasks, executors, workflow engine)
- Integration tests between modules
- End-to-end workflow tests
- Workflow simulation tests (full plan verification without remote execution)
- CLI command tests with proper mocking
- Test isolation using temp directories and `tmp_path`-backed SQLite files

### Test Guidelines

- Use `@pytest.fixture` for reusable test setup
- Use `tempfile` and `Path` for isolated file operations
- Use `unittest.mock.patch` for mocking external dependencies
- Use `tmp_path`-backed real SQLite files for database isolation (do **not** use in-memory `:memory:` databases — `JobStore` caches `_db_path` so swapping connections after construction silently breaks things)
- Skip tests that require remote machine access with `@pytest.mark.skip(reason="Requires remote machine access")`
- When registering test executor/task names in tests, always use a unique `uuid`-based name (e.g. `f"test_{uuid.uuid4().hex[:6]}"`) to avoid polluting the global plugin registry for the rest of the test session

## User specific preferences
User strictly requested production quality python code, no pseudo-code, properly verified, strongly structured logs, robustness (retry policies) and standard python libraries when possible without framework overhead (only `typer`, `pydantic`, `pyyaml`, `requests`, `rich`, `jinja2`).
The Git initialisation requires excluding personal data carefully. `.gitignore` successfully covers `.env`, `.devrun/`, macOS system files, and `.venv/`.
