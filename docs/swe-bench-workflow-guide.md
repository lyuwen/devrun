# SWE-bench Workflow Guide

Run the full SWE-bench evaluation pipeline -- inference, result collection, and patch evaluation -- as a single orchestrated workflow.

## Overview

The SWE-bench workflow chains three stages:

```
inference (swe_bench_agentic)
    ↓  Slurm array job: runs an LLM agent against benchmark instances
collect (swe_bench_collect)
    ↓  SSH command: aggregates per-instance results into predictions.jsonl
evaluate (swe_bench_eval)
       Slurm job: validates produced patches with swebench.harness
```

Each stage depends on the previous one. The workflow engine handles dependency resolution, status polling, and failure propagation automatically via a heartbeat loop.

## Quick Start

### 1. Configure your executor

Ensure `executors.yaml` (or your project/user config layer) defines both a `slurm` and `ssh` executor pointing to your compute cluster:

```yaml
# executors.yaml
slurm:
  type: slurm
  host: cluster.example.com
  user: myuser
  python_env:
    type: conda
    name: openhands

ssh:
  type: ssh
  host: cluster.example.com
  user: myuser
```

### 2. Run the full pipeline

```bash
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  params.model_name=openai/gpt-4 \
  params.dataset=/data/SWE-bench_Verified \
  params.working_dir=/remote/project/root
```

The three required parameters (`model_name`, `dataset`, `working_dir`) must be set -- either in a config overlay or as CLI overrides. All other parameters have sensible defaults.

The workflow blocks in the foreground, printing stage transitions as they happen. Use `--detach` (see [Background Execution](#background-execution)) to return immediately.

### 3. Monitor progress

```bash
# Check stage-by-stage status
devrun workflow status <workflow_id>

# View logs for a specific stage
devrun workflow logs <workflow_id> --stage inference

# List recent workflows
devrun workflow list
```

## Running from Existing Inference

If you already have a completed `swe_bench_agentic` job (inference is done and outputs exist on the remote host), you can launch only the collect and evaluate stages.

### Using `--from-job`

The simplest approach: pass the existing job's ID. The workflow engine extracts all relevant parameters (model name, dataset, output directory, etc.) from the job record and auto-detects which stage to skip.

```bash
# Find the job ID of your completed inference run
devrun history

# Launch collect + eval using that job's parameters
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  --from-job abc123
```

What happens:
1. Parameters are extracted from job `abc123` (model_name, dataset, split, output_dir, working_dir, run_name).
2. The job's task type (`swe_bench_agentic`) is matched to the `inference` stage.
3. `--start-after inference` is automatically applied -- the inference stage is skipped, and the workflow begins at `collect`.

You can still apply CLI overrides on top of the extracted parameters. CLI overrides take highest priority:

```bash
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  --from-job abc123 \
  params.working_dir=/different/remote/path
```

### Using `--start-after` manually

If you prefer explicit control, combine `--start-after` with manual parameter overrides:

```bash
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  --start-after inference \
  params.model_name=openai/gpt-4 \
  params.dataset=/data/SWE-bench_Verified \
  params.output_dir=logs/run1 \
  params.working_dir=/remote/project/root
```

`--start-after <stage>` skips the named stage **and all its transitive dependencies**. For the SWE-bench workflow, `--start-after inference` skips only the inference stage (since it has no dependencies of its own), leaving collect and evaluate to run.

### Finding existing job IDs

Use `devrun history` to list recent jobs:

```bash
# Show the 20 most recent jobs (default)
devrun history

# Show all jobs
devrun history --all

# Limit to a specific count
devrun history -n 50
```

The output table shows Job ID, Task, Executor, Status, and Created timestamp. Look for `swe_bench_agentic` jobs with status `completed`.

## Background Execution

For long-running workflows, use `--detach` (or `-d`) to start the workflow in the background and return immediately:

```bash
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  --detach \
  params.model_name=openai/gpt-4 \
  params.dataset=/data/SWE-bench_Verified \
  params.working_dir=/remote/project/root
```

Output:
```
Workflow abc12345 started in background.
Use devrun workflow status abc12345 to monitor.
```

The background process logs to `~/.devrun/logs/workflow_<workflow_id>.log`.

Detached mode validates all parameters and creates the workflow DB record **before** forking, so configuration errors appear immediately in your terminal rather than silently failing in the background.

`--detach` and `--dry-run` cannot be combined.

### Monitoring a detached workflow

```bash
# Stage-by-stage status with timing
devrun workflow status <workflow_id>

# Tail logs for a specific stage
devrun workflow logs <workflow_id> --stage inference

# Cancel all active stages
devrun workflow cancel <workflow_id>
```

## Parameter Overrides

Trailing arguments on `devrun workflow run` are OmegaConf dotlist overrides. They are merged in this order (last wins):

1. **YAML config file** -- base configuration
2. **`--from-job` params** -- extracted from an existing job record
3. **CLI overrides** -- trailing arguments

### Syntax

```bash
# Simple value
params.model_name=openai/gpt-4

# Path with spaces (quote the whole arg)
"params.dataset=/data/my dataset/SWE-bench_Verified"

# Nested stage params
stages.0.params.max_iterations=200

# Multiple overrides
devrun workflow run config.yaml \
  params.model_name=mymodel \
  params.split=dev \
  params.run_name=experiment-2
```

### Common overrides

| Override | Purpose |
|----------|---------|
| `params.model_name=X` | Model identifier |
| `params.dataset=/path` | Dataset location on remote host |
| `params.working_dir=/path` | Remote project root |
| `params.split=dev` | Dataset split (default: `test`) |
| `params.run_name=X` | Run identifier (default: `run1`) |
| `params.output_dir=logs/X` | Output directory (default: `logs/run1`) |

## Dry-Run Verification

Preview the full execution plan without submitting any jobs:

```bash
devrun workflow run devrun/configs/swe_bench_workflow/default.yaml \
  --dry-run \
  params.model_name=openai/gpt-4 \
  params.dataset=/data/SWE-bench_Verified \
  params.working_dir=/remote/project/root
```

The dry-run output shows each stage with:
- Task type and executor
- Dependencies
- Working directory
- Key parameters (up to 5)
- First 500 characters of the rendered command

When combined with `--start-after`, skipped stages are clearly tagged:

```
Workflow: swe_bench
Timeout: 172800s (48h)

Stage 1: inference [SKIPPED -- start-after]
  Task: swe_bench_agentic
  Executor: slurm
  Depends on: (none)

Stage 2: collect [WILL RUN]
  Task: swe_bench_collect
  Executor: ssh
  Depends on: inference
  Working dir: /remote/project/root
  Params: output_dir=logs/run1, dataset=/data/SWE-bench_Verified, ...
  Command preview (first 500 chars):
    ...

Stage 3: evaluate [WILL RUN]
  Task: swe_bench_eval
  Executor: slurm
  Depends on: collect
  ...

Summary: 1 stage(s) skipped, 2 stage(s) will run
```

Always dry-run before launching a real workflow to verify parameter resolution and command generation.

## Configuration Reference

### Workflow-level parameters

Defined under `params:` in the workflow config. All stages reference these via `${params.X}` interpolation.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_name` | Yes | -- | Model identifier (e.g., `openai/gpt-4`) |
| `dataset` | Yes | -- | Absolute path to SWE-bench dataset on remote host |
| `split` | No | `test` | Dataset split |
| `run_name` | No | `run1` | Run identifier, used in output paths |
| `output_dir` | No | `logs/run1` | Output directory (relative to `working_dir`) |
| `working_dir` | Yes | -- | Remote project root directory |

Required parameters use `<REQUIRED: description>` placeholders in the default config. The workflow engine validates that all placeholders are filled before submission and reports which ones are missing.

### Stage: inference (`swe_bench_agentic`)

Submits a Slurm array job that runs the OpenHands agent against each benchmark instance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `100` | Max agent iterations per instance |
| `max_attempts` | `5` | Retry count per array element |
| `array` | `000-499` | Slurm array range |
| `concurrency_limit` | `10` | Max concurrent array elements |
| `cpus_per_task` | `4` | CPUs per array element |
| `mem` | `32G` | Memory per array element |
| `walltime` | `24:00:00` | Slurm time limit |
| `job_name` | `swe-inference` | Slurm job name |

### Stage: collect (`swe_bench_collect`)

Runs via SSH. Scans inference output directories and produces `predictions.jsonl` using `jq`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predictions_path` | `${params.output_dir}/predictions.jsonl` | Output file path |
| `model_name_or_path` | `${params.model_name}` | Model identifier for prediction records |

Instances with missing `git_patch` values are excluded with a warning. The command prints a summary of collected vs. skipped instances.

### Stage: evaluate (`swe_bench_eval`)

Submits a Slurm job running `swebench.harness.run_evaluation`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpus_per_task` | `32` | CPUs for evaluation |
| `mem` | `64G` | Memory for evaluation |
| `max_workers` | `32` | Parallel evaluation workers |
| `walltime` | `24:00:00` | Slurm time limit |

### Workflow engine settings

Set at the top level of the workflow config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `172800` (48h) | Max wall-clock time in seconds for the entire workflow |
| `heartbeat_interval` | `30.0` | Poll interval in seconds between status checks |

## Troubleshooting

### "Workflow config has unfilled required parameters"

You forgot to set one or more required parameters. The error message lists exactly which ones:

```
Workflow config has unfilled required parameters:
  params.model_name: <REQUIRED: model identifier (e.g. openai/gpt-4)>
  params.dataset: <REQUIRED: absolute path to SWE-bench dataset>

Set them via CLI overrides:
  devrun workflow run config.yaml params.model_name=mymodel params.dataset=/path/to/data
```

Fix: add the missing overrides to your command.

### "Job 'xyz' not found"

The `--from-job` flag references a job ID that doesn't exist in the database.

```bash
# List all jobs to find the correct ID
devrun history --all
```

### "Stage 'X' depends_on 'Y' which does not exist"

A `depends_on` reference in the workflow config points to a stage name that doesn't exist. Check for typos in your config file. Available stage names are listed in the error message.

### Stage stuck in "submitted" state

The workflow engine polls executor status at the heartbeat interval (default 30s). If a stage remains in `submitted` for longer than expected:

- Check the Slurm queue: `squeue -u $USER`
- Verify the executor configuration (host, user, SSH key)
- Check workflow logs: `devrun workflow logs <workflow_id> --stage <stage>`

### Inference succeeded but collect finds no outputs

Verify that the `output_dir` and `working_dir` parameters match between the inference job and the workflow config. The collect stage looks for files at:

```
{working_dir}/{output_dir}/*/{DS_DIR}/*/*/output.jsonl
```

where `DS_DIR` is derived from the dataset path (e.g., `/data/SWE-bench_Verified` with split `test` becomes `__data__SWE-bench_Verified-test`).

### OmegaConf interpolation errors

If you see errors about unresolved interpolations (`${params.X}`), ensure:
- The referenced key exists in the top-level `params:` section
- The key name matches exactly (case-sensitive)
- You haven't introduced a typo in `${params.…}` syntax

### Workflow timed out

The default timeout is 48 hours. Override it for longer runs:

```bash
devrun workflow run config.yaml timeout=259200 ...  # 72 hours
```
