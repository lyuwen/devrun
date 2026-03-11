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

# File Synchronization
The `sync` command provides a simplified wrapper around `rsync` for pushing files to your remote clusters:
```bash
# Sync files to a remote host (e.g. your SSH/Slurm node)
devrun sync ./data swedev2:/remote/data

# Fetch results back
devrun fetch swedev2:/remote/results ./results
```

## Executor configuration

Executors define compute targets in `~/.devrun/configs/executors.yaml` or `configs/executors.yaml`:

```yaml
local:
  type: local

# Remote SSH execution
remote_ssh:
  type: ssh
  host: swedev2

# Remote Slurm submission (via SSH)
swedev1_slurm:
  type: slurm
  host: swedev1           # If host is provided, sbatch/squeue is executed over SSH!
  partition: compute

swedev2_slurm:
  type: slurm
  host: swedev2
  partition: agent
```

### Switching executors on the fly
You can dynamically override the target executor from the command line without modifying your configured task:
```bash
devrun run eval_math executor=swedev1_slurm
devrun run eval_math executor=swedev2_slurm
```

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
