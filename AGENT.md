# Devrun Project Overview

**Devrun** is a modular developer task orchestration system for running parameterised tasks across heterogeneous compute backends.
It is built heavily with Python, Typer (CLI), Pydantic (data validation), PyYAML (declarative configuration), and SQLite (state).

## Architecture & Data Flow

```text
CLI (typer)
  ↓
TaskRunner          ← orchestration engine (runner.py)
  ↓
ExecutorRouter      ← resolves executor name → instance (router.py)
  ↓
Executor Plugins    ← LocalExecutor | SSHExecutor | SlurmExecutor | HTTPExecutor
  ↓
Compute Backends
```

All interactions are driven entirely by YAML configuration files.

## High-Level Modules

* **`devrun/models.py`:** Central Pydantic models. Core entity is `TaskSpec` which holds the final executed `command`, `env`, and `resources` and is returned by `Task.prepare()`.
* **`devrun/registry.py`:** Custom decorator-based registry (`@register_task`, `@register_executor`) for automated discovery of plugins.
* **`devrun/runner.py`:** Orchestration logic. Loads YAML configs, expands sweeps via Cartesian product, invokes task preparation, writes to DB, invokes executor plugin `submit()`, and updates DB on error.
* **`devrun/cli.py`:** User entry point with 8 Typer commands: `run`, `list`, `status`, `logs`, `history`, `rerun`, `sync`, `fetch`. Supports command-line autocompletion for tasks (`devrun --show-completion bash`) and context-aware task help via `devrun run <task> --help` (parses YAML configurations).
* **`devrun/db/jobs.py`:** Persistent SQLite job store.
* **`configs/`:** Example YAML configurations for execution (`executors.yaml`), deployment (`deploy_ray.yaml`), and logic (`eval_math.yaml`, `eval_sweep.yaml`, `inference.yaml`).

## Available Plugin Systems

### Executors
All subclasses of `BaseExecutor` (`devrun/executors/base.py`) must implement `submit()`, `status()`, and `logs()`.
Registered types: `local`, `ssh`, `slurm`, `http`.

* **`LocalExecutor`:** Executes locally using `subprocess`, saving stout logs in `.devrun/logs/`.
* **`SSHExecutor`:** Remote execution using `nohup bash` with a heredoc to safely pass arbitrary commands. Each job is tracked by a composite `pid:token` job ID where `token` is a UUID-derived string used as the remote log file name (`/tmp/devrun_ssh_{token}.log`). This ensures `logs()`, `status()`, and `cancel()` all reliably find the right file and process even after the SSH session ends.
* **`SlurmExecutor`:** Generates `sbatch` scripts natively with unique UUID-suffixed filenames to prevent sweep collisions. If the `host` property is specified, it automatically uploads the script via `scp` and submits it via SSH, allowing local machines to natively dispatch to remote head nodes. After submission, the absolute log path is stored in `task_spec.metadata["log_path"]`, persisted to the DB, and used by `logs()` for reliable retrieval regardless of CWD.
* **`HTTPExecutor`:** JSON POST payload to a REST API.

### File Synchronization (`devrun/utils/sync.py`)
Provides native `devrun sync` and `devrun fetch` CLI wrappers over `rsync` to pull/push artifacts natively between the host and remote target aliases (e.g. `devrun sync ./data swedev2:/data`).

### Tasks
All subclasses of `BaseTask` (`devrun/tasks/base.py`) must implement `prepare(params: dict) -> TaskSpec`.
Registered types: `eval`, `inference`, `deploy_ray`.

## Database Schema (`devrun/db/jobs.py`)

Table: `jobs`
Columns: `job_id`, `task_name`, `executor`, `parameters`, `remote_job_id`, `status`, `created_at`, `completed_at`, `log_path`

## Development Specifics

* **Dependency Management:** Configured via `pyproject.toml` using standard `setuptools.build_meta` build backend.
* **Environment:** Installed globally in local `.venv`.
* **Formatting/Linting:** Not explicitly enforced, but code follows typing-heavy, strict schema practices. Supports Python 3.10+.
* **Issue to note:** Handled a regression related to Pydantic v2 incompatibilities in `models.py` (migrated away from `__root__` logic and inner `class Config` usage to `model_config = {}` dictionary and `.model_dump(mode="json")` for parsing `run_history` datetimes).
* **Shell safety:** All executor plugins use `shlex.quote()` when interpolating user-supplied values (env vars, paths, command args) into shell strings. SSH commands use heredocs rather than `bash -c '...'` wrapping to handle single quotes in commands.
* **Datetime:** All timestamps use `datetime.now(timezone.utc)` (timezone-aware). `datetime.utcnow()` is deprecated in Python 3.12+ and must not be re-introduced.
* **Log path propagation pattern:** When an executor knows the log path at submit time, it writes it to `task_spec.metadata["log_path"]`. The runner reads this after `submit_with_retry()` and passes it to `db.update_status(..., log_path=...)`. The runner then retrieves `record.log_path` and passes it as `executor.logs(remote_id, log_path=record.log_path)`. All `logs()` implementations accept the optional `log_path` kwarg.

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
| `test_cli.py` | Tests for all CLI commands (run, list, status, logs, history, rerun, cancel, sync, fetch) |
| `test_utils.py` | Tests for utility functions (sync, SSH, Slurm) |
| `test_e2e.py` | End-to-end integration tests for complete workflows |
| `test_ssh_executor.py` | Unit tests for SSHExecutor: composite job ID, log token stability, shell quoting |
| `test_swe_bench_eval.py` | Unit tests for SWEBenchEvalTask: placeholder validation, command generation, shlex safety |
| `test_swe_bench_agentic.py` | Unit tests for SWEBenchAgenticTask: array jobs, mkdir, backslash fix, oversubscribe opt-in |
| `test_data/sample_configs/` | Sample YAML config files for testing |

### Test Coverage

- **249 tests passing**, **10 skipped** (infrastructure-dependent: require real SSH/Slurm connectivity)
- Unit tests for all major components (models, registry, database, router, runner, tasks, executors)
- Integration tests between modules
- End-to-end workflow tests
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
User strictly requested production quality python code, no pseudo-code, properly verified, strongly structured logs, robustness (retry policies) and standard python libraries when possible without framework overhead (only `typer`, `pydantic`, `pyyaml`, `requests`, `rich`).
The Git initialisation requires excluding personal data carefully. `.gitignore` successfully covers `.env`, `.devrun/`, macOS system files, and `.venv/`.
