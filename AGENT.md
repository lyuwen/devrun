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
* **`devrun/cli.py`:** User entry point with 8 Typer commands: `run`, `list`, `status`, `logs`, `history`, `rerun`, `sync`, `fetch`.
* **`devrun/db/jobs.py`:** Persistent SQLite job store.
* **`configs/`:** Example YAML configurations for execution (`executors.yaml`), deployment (`deploy_ray.yaml`), and logic (`eval_math.yaml`, `eval_sweep.yaml`, `inference.yaml`).

## Available Plugin Systems

### Executors
All subclasses of `BaseExecutor` (`devrun/executors/base.py`) must implement `submit()`, `status()`, and `logs()`.
Registered types: `local`, `ssh`, `slurm`, `http`.

* **`LocalExecutor`:** Executes locally using `subprocess`, saving stout logs in `.devrun/logs/`.
* **`SSHExecutor`:** Remote SSH via `nohup bash` background processes.
* **`SlurmExecutor`:** Generates `sbatch` scripts, uploads via `scp`, submits via SSH, queries status with `squeue`/`sacct`.
* **`HTTPExecutor`:** JSON POST payload to a REST API.

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

## User specific preferences
User strictly requested production quality python code, no pseudo-code, properly verified, strongly structured logs, robustness (retry policies) and standard python libraries when possible without framework overhead (only `typer`, `pydantic`, `pyyaml`, `requests`, `rich`).
The Git initialisation requires excluding personal data carefully. `.gitignore` successfully covers `.env`, `.devrun/`, macOS system files, and `.venv/`.
