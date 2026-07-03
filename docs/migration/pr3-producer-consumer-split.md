# PR3 Migration Guide: Producer/Consumer Split

**Date:** 2026-06-14  
**Target Branch:** `feat/dependency`  
**Status:** Breaking Release

## Overview

PR3 fundamentally changes how devrun executes work. Previously, `devrun run` and `devrun workflow run` would submit jobs to executors synchronously and wait for results (or detach to background). Now, all commands are **enqueue-only producers** that write jobs to a queue and return immediately. A separate **heartbeat service** acts as the consumer, continuously polling the queue, resolving dependencies, and submitting jobs to executors.

This producer/consumer architecture enables:

- **Job-level dependencies**: Any `devrun run` can wait on another job's completion
- **Unified orchestration**: Single-job runs and multi-stage workflows share the same scheduler
- **Improved reliability**: Crashed CLI processes don't orphan running jobs
- **Better observability**: All state lives in the database, not in-memory

**This is a breaking release.** User-visible behavior changes significantly.

---

## What Changed

### Architecture Shift

**Before (PR2):**
```text
devrun run <task>
  ↓
TaskRunner.run()
  ↓ (synchronous)
Executor.submit()  →  Remote job starts
  ↓ (poll until done)
TaskRunner.status() → Executor.status()
  ↓
Result printed to terminal
```

**After (PR3):**
```text
devrun run <task>
  ↓
TaskRunner.run()
  ↓ (enqueue only)
JobStore.enqueue() → jobs table (QUEUED)
  ↓
Returns immediately
                            ┌─────────────────────────┐
                            │ devrun heartbeat (loop) │
                            └───────────┬─────────────┘
                                        ↓
                            Read QUEUED jobs from DB
                                        ↓
                            Resolve dependencies
                                        ↓
                            Executor.submit() → Remote job
                                        ↓
                            Poll & update DB
```

The CLI is now a thin write-only client. The heartbeat owns all executor communication.

### New Job Lifecycle

PR3 introduces several new job statuses to reflect the producer/consumer split:

| Status | Meaning | Terminal? |
|--------|---------|-----------|
| `QUEUED` | Enqueued by producer, waiting for heartbeat to promote | No |
| `SUBMITTING` | Heartbeat is actively calling `executor.submit()` | No |
| `SUBMITTED` | Successfully submitted to executor, not yet running | No |
| `RUNNING` | Executor reports job is executing | No |
| `COMPLETED` | Finished successfully | Yes |
| `FAILED` | Execution failed or submission error | Yes |
| `CANCELING` | Cancel requested, heartbeat polling for confirmation | No |
| `CANCELLED` | Successfully cancelled | Yes |
| `SKIPPED` | Blocked by failed dependency or unfilled `<REQUIRED:...>` placeholder | Yes |
| `TIMED_OUT` | Workflow deadline exceeded | Yes |

**Legacy status removed:** `PENDING` (pre-PR3 synchronous submission state) is no longer written. Existing `PENDING` rows in your database are ignored by the heartbeat.

---

## Breaking Changes

### 1. `devrun run` No Longer Blocks

**Before:**
```bash
devrun run eval params.model=gpt-4
# Blocks until job completes, prints logs to terminal
```

**After:**
```bash
devrun run eval params.model=gpt-4
# Returns immediately with job ID
# Job ID: abc123
# Status: QUEUED
# WARNING: Heartbeat service not running. Jobs will remain queued until 'devrun heartbeat start'.
```

**Migration:**

If your scripts rely on `devrun run` blocking until completion, you must now poll status explicitly:

```bash
job_id=$(devrun run eval params.model=gpt-4 | grep "Job ID:" | awk '{print $3}')

# Wait for completion
while true; do
  status=$(devrun status "$job_id" --format json | jq -r '.status')
  if [[ "$status" == "completed" ]]; then
    echo "Job succeeded"
    break
  elif [[ "$status" =~ ^(failed|cancelled|skipped)$ ]]; then
    echo "Job terminal: $status"
    exit 1
  fi
  sleep 5
done
```

Or use the new `--after` flag to chain jobs as dependencies (see below).

### 2. `workflow run --detach` Flag Removed

**Before:**
```bash
devrun workflow run swe_bench          # Blocks in foreground
devrun workflow run swe_bench --detach # Forks to background
```

**After:**
```bash
devrun workflow run swe_bench
# Always returns immediately after enqueue
# Workflow ID: wf_abc123
# Enqueued 3 stages (5 jobs total)
```

All workflow runs are now async by default. The `--detach` / `-d` flag has been removed.

**Migration:**

Remove `--detach` from all `workflow run` invocations. The behavior is now the default.

### 3. `devrun status` and `devrun history` No Longer Ping Executors

**Before:**

`devrun status <job_id>` would call `executor.status()` to fetch live job state from the remote system (Slurm, SSH, etc.) and update the database.

**After:**

`devrun status` is a pure database read. It shows the last state written by the heartbeat. Status updates happen only during heartbeat ticks.

**Implication:**

Status may lag by up to one tick interval (default: 10 seconds). If you need real-time status, check the heartbeat's own logs or reduce the tick interval (see Advanced Configuration in the [Heartbeat Service Guide](../guides/heartbeat-service.md)).

### 4. `devrun cancel` No Longer Kills Remote Jobs Directly

**Before:**

`devrun cancel <job_id>` would immediately call `executor.cancel()` to terminate the remote job.

**After:**

`devrun cancel <job_id>` writes a cancellation request to the database (transitions job to `CANCELING`). The heartbeat picks up the request on the next tick and calls `executor.cancel()`. Once the executor confirms cancellation, the job transitions to `CANCELLED`.

**Migration:**

Cancellation is now asynchronous. Poll `devrun status <job_id>` until status is `CANCELLED` to confirm termination.

### 5. Heartbeat Service Required for Jobs to Progress

**Critical:** Without a running heartbeat, jobs remain in `QUEUED` state forever.

After upgrading to PR3, you must install and start the heartbeat service:

```bash
devrun heartbeat install
devrun heartbeat start
```

Verify it's running:

```bash
devrun heartbeat status
# Service: active
# Job status counts:
#   QUEUED: 0
#   RUNNING: 0
```

See the [Heartbeat Service Installation Guide](../guides/heartbeat-service.md) for details.

---

## New Features

### Job Dependencies: `--after` and `--allow-failure-from`

Single-job runs can now wait for other jobs to complete before starting.

**Example:**

```bash
# Run inference
job1=$(devrun run swe_bench_agentic params.model=gpt-4 | grep "Job ID:" | awk '{print $3}')

# Collect predictions only after inference completes
job2=$(devrun run swe_bench_collect --after "$job1" | grep "Job ID:" | awk '{print $3}')

# Evaluate predictions
devrun run swe_bench_eval --after "$job2"
```

The heartbeat automatically sequences these jobs: `job2` won't submit until `job1` is `COMPLETED`, and `job3` waits for `job2`.

**Failure propagation:**

By default, if a parent job fails, all dependent children are automatically marked `SKIPPED`. To allow a child to run even if the parent fails, use `--allow-failure-from`:

```bash
devrun run cleanup --after main_job --allow-failure-from main_job
```

This is useful for cleanup tasks that should run regardless of success.

### Cross-Job Parameter References: `${jobs:<id>,<path>}`

Jobs can reference parameters from completed dependencies using interpolation syntax:

```bash
job1=$(devrun run inference params.output_dir=/results/run1 | grep "Job ID:" | awk '{print $3}')

# Reference job1's output_dir in job2
devrun run collect --after "$job1" 'params.input_dir=${jobs:'"$job1"',output_dir}'
```

The heartbeat resolves `${jobs:...}` references at promotion time (when moving `QUEUED` → `SUBMITTING`).

**Requirements:**
- The referenced job must be a dependency (via `--after`)
- The referenced parameter must exist in the parent's resolved parameters
- Resolution happens at submit time, so the parent must be `COMPLETED`

### Workflow Partial Reruns: `--start-after` and `--from-job`

Rerun a workflow starting from a specific stage, reusing outputs from a previous run.

**Example:**

You ran a workflow and inference succeeded, but evaluation failed. Instead of re-running inference, skip to evaluation:

```bash
# Original run
devrun workflow run swe_bench
# Workflow ID: wf_001
# inference stage: job_abc (COMPLETED)
# collect stage:   job_def (COMPLETED)
# eval stage:      job_ghi (FAILED)

# Rerun starting after collect, reusing job_def's outputs
devrun workflow run swe_bench --start-after collect --from-job job_def
# Workflow ID: wf_002
# inference stage: SKIPPED (source: job_abc from wf_001)
# collect stage:   SKIPPED (source: job_def from wf_001)
# eval stage:      job_xyz (QUEUED)
```

The `--from-job` argument provides the source job whose parameters are imported for skipped stages. The `--start-after` argument specifies which stages to skip.

**Cross-stage references (`${stages:...}`) are automatically rewritten** to point to the source jobs from the original workflow, ensuring downstream stages get correct inputs.

### Dependency-Free Job Copies: `devrun rerun`

`devrun rerun <job_id>` creates a new job with the same parameters as an existing job, but **without copying its dependencies**.

**Use case:** You want to retry a failed job in isolation, without waiting for its original parents.

```bash
# Original job had --after dep1,dep2
devrun run task --after dep1 --after dep2 params.x=1
# Job ID: job_abc (FAILED)

# Rerun without waiting for dep1/dep2
devrun rerun job_abc
# Job ID: job_xyz (QUEUED, no dependencies)
```

---

## Migration Steps

### 1. Install the Heartbeat Service

**Before upgrading**, ensure the heartbeat service is installed and running on any host where you submit devrun jobs:

```bash
# Install service (creates systemd unit or launchd plist)
devrun heartbeat install

# Start the service
devrun heartbeat start

# Verify it's active
devrun heartbeat status
```

The service will auto-start on boot/login going forward.

**Platform support:**
- **Linux:** systemd user service (`~/.config/systemd/user/devrun-heartbeat.service`)
- **macOS:** launchd agent (`~/Library/LaunchAgents/com.devrun.heartbeat.plist`)

See the [Heartbeat Service Installation Guide](../guides/heartbeat-service.md) for troubleshooting.

### 2. Update Scripts That Assumed Blocking Behavior

**Find usage:**

```bash
grep -r "devrun run" scripts/
grep -r "devrun workflow run" scripts/
```

**Pattern 1: Scripts that capture exit codes**

**Before:**
```bash
if devrun run eval params.model=gpt-4; then
  echo "Success"
else
  echo "Failed"
  exit 1
fi
```

**After (poll for completion):**
```bash
job_id=$(devrun run eval params.model=gpt-4 | grep "Job ID:" | awk '{print $3}')

while true; do
  status=$(devrun status "$job_id" --format json | jq -r '.status')
  case "$status" in
    completed)
      echo "Success"
      break
      ;;
    failed|cancelled|skipped)
      echo "Failed: $status"
      exit 1
      ;;
  esac
  sleep 5
done
```

**Pattern 2: Sequential pipeline scripts**

**Before:**
```bash
devrun run step1
devrun run step2
devrun run step3
```

**After (use `--after` for dependency chaining):**
```bash
job1=$(devrun run step1 | grep "Job ID:" | awk '{print $3}')
job2=$(devrun run step2 --after "$job1" | grep "Job ID:" | awk '{print $3}')
job3=$(devrun run step3 --after "$job2" | grep "Job ID:" | awk '{print $3}')

# Optional: wait for final job
while [[ $(devrun status "$job3" --format json | jq -r '.status') =~ ^(queued|submitting|submitted|running)$ ]]; do
  sleep 5
done
```

### 3. Remove `--detach` from Workflow Invocations

**Find usage:**

```bash
grep -r "\-\-detach\|\-d" scripts/ | grep "workflow run"
```

**Before:**
```bash
devrun workflow run swe_bench --detach
```

**After:**
```bash
devrun workflow run swe_bench
# Returns immediately (detach is now the default)
```

### 4. Handle Legacy `PENDING` Rows in Database

If you have jobs in `PENDING` state from pre-PR3 runs, they will be ignored by the heartbeat. To clean them up:

```bash
# List legacy pending jobs
devrun history --status pending

# Manually mark them as failed
sqlite3 ~/.devrun/jobs.db "UPDATE jobs SET status='failed', skip_reason='Legacy PENDING job ignored by heartbeat' WHERE status='pending';"
```

Alternatively, leave them — they won't interfere with new jobs.

---

## Troubleshooting

### Jobs Stuck in `QUEUED` State

**Symptom:**

```bash
devrun status job_abc
# Status: QUEUED
# (never progresses)
```

**Causes:**

1. **Heartbeat not running**

   Check heartbeat status:
   ```bash
   devrun heartbeat status
   ```

   If `inactive`, start it:
   ```bash
   devrun heartbeat start
   ```

2. **Dependencies not satisfied**

   If the job has `--after` dependencies, check parent statuses:
   ```bash
   devrun status job_abc --with-deps
   # Job: job_abc (QUEUED)
   # Dependencies:
   #   job_xyz (RUNNING) ← blocking
   ```

   The job will remain `QUEUED` until `job_xyz` completes.

3. **Unfilled `<REQUIRED:...>` placeholders**

   The heartbeat validates parameters at promotion time. Jobs with unresolved `<REQUIRED:...>` placeholders are marked `SKIPPED`.

   Check for skip reason:
   ```bash
   devrun status job_abc
   # Status: SKIPPED
   # Skip reason: Required parameter 'params.model_name' not provided
   ```

   Fix by providing the missing parameter and rerun.

### Heartbeat Service Won't Start

**Check logs:**

**Linux:**
```bash
journalctl --user -u devrun-heartbeat.service -n 50
```

**macOS:**
```bash
log show --predicate 'processImagePath contains "python"' --info --last 10m | grep heartbeat
```

**Common issues:**

1. **Missing `executors.yaml`**

   The heartbeat requires `~/.devrun/executors.yaml` to resolve executor configurations.

   Create a minimal config:
   ```bash
   mkdir -p ~/.devrun
   cat > ~/.devrun/executors.yaml <<EOF
   local:
     type: local
   EOF
   ```

2. **Python environment mismatch**

   The service uses the Python interpreter active at `install` time. If you've changed virtual environments, reinstall:
   ```bash
   devrun heartbeat uninstall
   devrun heartbeat install
   devrun heartbeat start
   ```

See the [Heartbeat Service Installation Guide](../guides/heartbeat-service.md) for more troubleshooting steps.

### How to Check Job Dependencies

Use `--with-deps` to view a job's dependency tree:

```bash
devrun status job_abc --with-deps
# Job: job_abc (QUEUED)
# Dependencies:
#   job_xyz (COMPLETED)
#   job_def (RUNNING) ← blocking
```

This shows which parent jobs must complete before `job_abc` can proceed.

### Workflow Stage Logs Not Found

**Symptom:**

```bash
devrun workflow logs wf_001 inference
# <queued — no logs yet>
```

**Cause:**

If a stage is in `QUEUED` or `SUBMITTING` state, logs don't exist yet. The heartbeat hasn't submitted the job to the executor.

Wait for the stage to reach `SUBMITTED` or `RUNNING`, then retry:

```bash
devrun workflow status wf_001
# Stage: inference (RUNNING)

devrun workflow logs wf_001 inference
# [log output]
```

For `SKIPPED` stages with `source_job_id`, logs are automatically fetched from the source job:

```bash
devrun workflow logs wf_002 inference
# (fetching logs from source job job_abc)
# [log output from job_abc]
```

---

## Summary of CLI Changes

| Command | Before PR3 | After PR3 |
|---------|-----------|-----------|
| `devrun run <task>` | Blocks until done | Returns immediately (enqueues) |
| `devrun run <task> --after <id>` | ❌ Not supported | ✅ Wait for job `<id>` |
| `devrun workflow run <wf>` | Blocks (or `--detach` to background) | Always async (enqueues) |
| `devrun workflow run <wf> --detach` | Forks to background | ❌ Flag removed |
| `devrun workflow run <wf> --start-after <stage>` | ❌ Not supported | ✅ Skip stages up to `<stage>` |
| `devrun status <id>` | Pings executor for live state | Pure DB read (shows last heartbeat update) |
| `devrun cancel <id>` | Kills job immediately | Enqueues cancel request (heartbeat processes it) |
| `devrun rerun <id>` | ❌ Not supported | ✅ Copy job without dependencies |
| `devrun heartbeat ...` | ❌ Not supported | ✅ Manage heartbeat service |

---

## Further Reading

- [Heartbeat Service Installation Guide](../guides/heartbeat-service.md)
- [Heartbeat CLI Reference](../cli/heartbeat.md)
- [Job Dependency Mechanism Design Spec](../superpowers/specs/2026-06-12-job-dependency-mechanism-design.md)
- [PR3 Implementation Plan](../superpowers/plans/2026-06-12-job-dependency-pr3-producer-flip.md)

---

**Questions or issues?** Check the [troubleshooting section](#troubleshooting) above or review the heartbeat service logs.
