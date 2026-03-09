# devrun

**Modular developer task orchestration system** for running parameterised tasks across heterogeneous compute backends (local, SSH, SLURM, HTTP/API).

## Architecture

```
CLI (typer)
  ↓
TaskRunner          ← orchestration engine
  ↓
ExecutorRouter      ← resolves executor name → instance
  ↓
Executor Plugins    ← LocalExecutor | SSHExecutor | SlurmExecutor | HTTPExecutor
  ↓
Compute Backends
```

### Key modules

| Module | Responsibility |
|---|---|
| `devrun/models.py` | Pydantic data models (`TaskSpec`, `TaskConfig`, `JobRecord`, …) |
| `devrun/registry.py` | Decorator-based plugin registry for executors & tasks |
| `devrun/router.py` | Loads `executors.yaml`, instantiates the right executor |
| `devrun/runner.py` | Central orchestration: config loading, sweep expansion, DB tracking |
| `devrun/cli.py` | Typer CLI exposing all commands |
| `devrun/executors/` | Executor plugins: `local`, `ssh`, `slurm`, `http` |
| `devrun/tasks/` | Task plugins: `eval`, `inference`, `deploy_ray` |
| `devrun/db/jobs.py` | SQLite job store |
| `devrun/utils/` | SSH, SLURM, and rsync helpers |

## Installation

```bash
# From the project root
pip install -e .
```

## Quick start

```bash
# List available plugins
devrun list

# Run a task
devrun run configs/eval_math.yaml

# Run a parameter sweep (launches 4 jobs)
devrun run configs/eval_sweep.yaml

# Check status
devrun status <job_id>

# View logs
devrun logs <job_id>

# Job history
devrun history

# Re-run a previous job
devrun rerun <job_id>

# Sync files to/from a cluster
devrun sync ./data cluster-a:/data
devrun fetch cluster-a:/results ./results
```

## Task configuration

Tasks are defined in YAML:

```yaml
# configs/eval_math.yaml
task: eval
executor: local

params:
  model: deepseek-r1
  dataset: math500
  batch_size: 16
```

### Parameter sweeps

```yaml
sweep:
  model:
    - deepseek-r1
    - qwen-72b
  batch_size:
    - 8
    - 16
```

This automatically expands to **4 jobs** (cartesian product).

## Executor configuration

Executors are defined in `configs/executors.yaml`:

```yaml
local:
  type: local

slurm_cluster_a:
  type: slurm
  host: cluster-a.company.com
  user: dev
  partition: gpu

inference_cluster:
  type: http
  endpoint: https://inference.cluster/api
```

## Extending devrun

### Add a new task

1. Create `devrun/tasks/my_task.py`.
2. Subclass `BaseTask`, implement `prepare(params) → TaskSpec`.
3. Decorate with `@register_task("my_task")`.
4. Import in `devrun/tasks/__init__.py`.

### Add a new executor

1. Create `devrun/executors/my_executor.py`.
2. Subclass `BaseExecutor`, implement `submit`, `status`, `logs`.
3. Decorate with `@register_executor("my_type")`.
4. Import in `devrun/executors/__init__.py`.

## License

MIT
