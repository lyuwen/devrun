# Job-Level Dependency Mechanism & Global Heartbeat Scheduler

**Date:** 2026-06-12
**Status:** Approved
**Branch:** `feat/dependency`

## Motivation

Today, multi-stage execution is owned by `WorkflowRunner`, which drives a heartbeat loop in-process (foreground or detached via `run_detached()`) and submits each stage synchronously inside that loop. This creates several problems:

1. **No real dependency model for single-job runs.** `devrun run` jobs cannot wait on another job. Pipelining is only possible through workflow YAML.
2. **`--from-job` + `--start-after` interpolation gap.** When `--from-job` is combined with `--start-after`, `skipped_params` is not populated from the source job's resolved params, so downstream `${stages:inference,...}` references can silently resolve to defaults instead of the real predecessor outputs.
3. **No "wait for source job to finish" semantics.** If `--from-job` references a still-running job, the collect stage fires immediately and either fails (no predictions) or collects partial outputs.
4. **Two parallel orchestration mechanisms.** Workflow stage dependencies live inside `WorkflowRunner`; single jobs have none. Cancellation, status refresh, and submission logic are duplicated across `TaskRunner` and `WorkflowRunner`.

This design replaces the in-process workflow loop with a **global heartbeat scheduler** and adds **job-level dependencies** that work for both single-job runs and workflows. Workflows become thin producers that enqueue jobs and edges into a shared queue; the heartbeat is the single consumer.

## Use Cases

```bash
# Single-job dependency
devrun run swe_bench_agentic params.model_name=gpt-4    # → job abc123
devrun run swe_bench_collect --after abc123             # → job def456, waits for abc123
devrun run swe_bench_eval    --after def456             # → job ghi789, waits for def456

# allow-failure
devrun run cleanup --after main_job --allow-failure-from main_job

# Workflow (now just a batch enqueue)
devrun workflow run swe_bench_workflow                  # → enqueues N jobs + edges, returns
devrun heartbeat status                                 # → shows queued/running/failed counts
```

## Architecture

```text
Producers (synchronous, write-only)
  devrun run <task> [--after <id> ...]   → INSERT job(QUEUED) + edges
  devrun workflow run <wf>               → INSERT N jobs(QUEUED) + edges (one tx)
  ↓
  jobs table  +  job_dependencies table  ←  ${jobs:<id>,<path>} resolver
  ↑
Consumer (single long-running process; foreground OR systemd/launchd)
  devrun heartbeat
    loop every N seconds:
      cascade-skip dependents of failed/skipped/cancelled parents
      promote ready QUEUED → SUBMITTED (resolve params, prepare, submit)
      poll SUBMITTED/RUNNING/CANCELING → terminal (or CANCELLED)
```

The producer side returns immediately. The heartbeat is the only component that talks to executors.

## Data Model

### New table: `job_dependencies`

```sql
CREATE TABLE IF NOT EXISTS job_dependencies (
    child_job_id  TEXT NOT NULL,
    parent_job_id TEXT NOT NULL,
    allow_failure INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (child_job_id, parent_job_id),
    FOREIGN KEY (child_job_id)  REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobdeps_child  ON job_dependencies(child_job_id);
CREATE INDEX IF NOT EXISTS idx_jobdeps_parent ON job_dependencies(parent_job_id);
```

Pure edge list. The scheduler retrieves all parents for a child in one indexed query.

### `jobs` table additions

```sql
ALTER TABLE jobs ADD COLUMN params_template TEXT;   -- unresolved JSON for late resolution
ALTER TABLE jobs ADD COLUMN skip_reason     TEXT;   -- human-readable on SKIPPED/FAILED-at-enqueue/promotion
```

- `params_template` stores the merged-but-unresolved config (OmegaConf-serialized JSON). Cross-job references like `${jobs:abc123,output_dir}` stay as strings here and are resolved only when the scheduler promotes QUEUED → SUBMITTED.
- `parameters` (existing column) holds the **final resolved** params, written at promotion time. For jobs with no cross-job references, `params_template` and `parameters` agree after promotion.
- `skip_reason` is populated when a job transitions to `SKIPPED` (parent failed and `allow_failure=0`) or to `FAILED` at enqueue/promotion time (unfilled placeholder, unresolved reference, executor.submit exception).

### `JobStatus` extension

```python
class JobStatus(str, Enum):
    PENDING    = "pending"      # legacy — kept for migration; new code uses QUEUED
    QUEUED     = "queued"       # NEW
    SUBMITTED  = "submitted"
    RUNNING    = "running"
    CANCELING  = "canceling"    # NEW — cancel requested, executor.cancel() pending
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    SKIPPED    = "skipped"      # NEW — blocking parent failed
    UNKNOWN    = "unknown"
```

### Cross-job interpolation resolver

New OmegaConf resolver registered in `devrun/jobref.py`:

```
${jobs:<job_id>,<dotted.path>}
```

Reads from `jobs.parameters` JSON of a completed predecessor. Resolves **only** against parents listed in `job_dependencies` for the calling job — prevents arbitrary cross-job reads. Imported wherever `${key:...}` / `${preset:...}` are imported.

### `workflows` table addition

```sql
ALTER TABLE workflows ADD COLUMN job_ids TEXT;   -- JSON array of job IDs in this workflow
```

`stages_state` is kept for backward compatibility with existing rows but new rows leave it `'{}'`. Workflow status/logs queries reconstruct stage info by joining `job_ids` → `jobs` + `job_dependencies`. Removal of `stages_state` is a follow-up.

### Edge semantics

A child is **promotable** iff every edge is satisfied:

| Parent status | `allow_failure=0` | `allow_failure=1` |
|---|---|---|
| COMPLETED  | satisfied | satisfied |
| FAILED     | child → SKIPPED  | satisfied |
| SKIPPED    | child → SKIPPED  | satisfied |
| CANCELLED  | child → SKIPPED  | satisfied |
| QUEUED / SUBMITTED / RUNNING / CANCELING | wait | wait |

## Heartbeat Scheduler

`devrun/heartbeat.py` — one module, one loop. Three independent phases per tick.

### Loop

```python
def run_loop(db_path: Path, interval: float = 10.0, tick_file: Path = ...) -> None:
    db = JobStore(db_path)
    while not _shutdown_requested():
        try:
            tick(db)
        except Exception:
            logger.exception("heartbeat tick failed; continuing")
        tick_file.write_text(datetime.now(timezone.utc).isoformat())
        time.sleep(interval)


def tick(db: JobStore) -> None:
    _cascade_skip_failed_dependents(db)   # propagate SKIPPED first
    _promote_ready_queued(db)             # QUEUED → SUBMITTED
    _poll_active_jobs(db)                 # SUBMITTED/RUNNING/CANCELING → terminal
```

### Phase 1 — cascade skip

```sql
SELECT DISTINCT j.job_id
FROM   jobs j
JOIN   job_dependencies d ON d.child_job_id = j.job_id
JOIN   jobs p             ON p.job_id      = d.parent_job_id
WHERE  j.status = 'queued'
  AND  d.allow_failure = 0
  AND  p.status IN ('failed','skipped','cancelled');
```

Each match → `UPDATE jobs SET status='skipped', skip_reason='parent <id> <status>', completed_at=now`. Multi-hop cascade resolves across consecutive ticks (fixpoint-stable). Simple SQL, no recursion.

### Phase 2 — promote ready queued

```sql
SELECT j.job_id
FROM   jobs j
WHERE  j.status = 'queued'
  AND  NOT EXISTS (
    SELECT 1 FROM job_dependencies d
    JOIN   jobs p ON p.job_id = d.parent_job_id
    WHERE  d.child_job_id = j.job_id
      AND  NOT (
            p.status = 'completed'
         OR (d.allow_failure = 1 AND p.status IN ('failed','skipped','cancelled'))
      )
  )
ORDER BY j.created_at;
```

For each ready job:
1. Load `params_template`; fetch parent records.
2. Build OmegaConf with `${jobs:...}` resolver pointed at the parents map.
3. Resolve. If `<REQUIRED:...>` survives or any `${jobs:...}` fails → `FAILED` with `skip_reason`.
4. `task = get_task_class(task_name)(); spec = task.prepare(resolved_params)`.
5. `executor.submit_with_retry(spec)` → `SUBMITTED` with `remote_job_id` and `log_path`. Persist resolved `parameters`.

All exceptions in steps 3–5 mark the job `FAILED` with the exception text in `skip_reason`. Phase 1 of the next tick cascades the failure.

### Phase 3 — poll active

```sql
SELECT job_id, executor, remote_job_id FROM jobs
WHERE status IN ('submitted','running','canceling');
```

For each row:
- `SUBMITTED`/`RUNNING`: `executor.status(remote_id)` → mapped via a module-level `_map_status()` (moved from `TaskRunner._map_status` so it can be imported without a TaskRunner instance). On terminal status, set `completed_at`.
- `CANCELING`: `executor.cancel(remote_id)`, then transition to `CANCELLED` with `completed_at`. If `cancel()` raises, log + retry next tick.

Batching seam: group by executor name and call `executor.status_many(remote_ids)` if available; default `BaseExecutor.status_many` falls back to a per-job loop. Real batched implementations for Slurm/SSH are out of scope here but the seam is added.

### Cancellation flow

`devrun cancel <job_id>` is producer-side only — writes a state transition, never talks to executors:

- `QUEUED` → `CANCELLED` (immediate).
- `SUBMITTED` / `RUNNING` → `CANCELING` (heartbeat completes the cancellation).
- terminal states → error "already <status>".

`CANCELING` is itself the user-visible signal. `devrun history` renders it in yellow; `cancelled` in dim red. If the heartbeat is down, the job sits in `CANCELING` until it comes back. Producer never falls back to direct executor cancel — that would re-introduce dual ownership.

### Process model & shutdown

- Foreground (`devrun heartbeat`) and the systemd/launchd unit both call `run_loop()`.
- SIGTERM/SIGINT set a module-level shutdown event; the loop exits cleanly after the current tick.
- systemd: `Type=simple`, `KillSignal=SIGTERM`, `TimeoutStopSec=60`, `Restart=on-failure`.
- launchd: equivalent `KeepAlive` + `ProcessType=Background`.
- Every exception inside `tick()` is caught and logged. Only signals exit the loop.

### Configuration

Optional `~/.devrun/config.yaml`:

```yaml
heartbeat:
  interval: 10.0
  poll_concurrency: 1        # placeholder for future batched executor support
  log_path: ~/.devrun/heartbeat.log
```

Defaults apply when missing. No CLI flags — keep producer commands clean.

## Heartbeat CLI

```
devrun heartbeat                  # foreground; refuses if daemon is active
devrun heartbeat start            # systemctl --user start / launchctl load
devrun heartbeat stop             # systemctl --user stop  / launchctl unload
devrun heartbeat restart          # restart via service manager
devrun heartbeat install          # write unit file + reload + enable
devrun heartbeat uninstall        # disable + remove unit file + reload
devrun heartbeat status           # service state + DB summary + last-tick timestamp
```

### Platform dispatch

`devrun/services/heartbeat_service.py` exposes:

```python
class HeartbeatService(Protocol):
    def install(self) -> None: ...
    def uninstall(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_active(self) -> bool: ...
```

Implementations:
- `devrun/services/linux.py` — drives `systemctl --user`; renders unit from `devrun/templates/devrun-heartbeat.service.j2`. Unit path: `~/.config/systemd/user/devrun-heartbeat.service`. `ExecStart=<sys.executable> -m devrun.heartbeat`.
- `devrun/services/darwin.py` — drives `launchctl`; renders plist from `devrun/templates/devrun-heartbeat.plist.j2`. Path: `~/Library/LaunchAgents/com.devrun.heartbeat.plist`. `Label=com.devrun.heartbeat`.

Selection happens in `devrun.heartbeat.get_service()` via `sys.platform`. The CLI never sees the platform.

### Foreground vs daemon coexistence

Foreground mode calls `service.is_active()` first. If true, refuses to start with a clear message. Source of truth is the service manager — no separate PID lockfile.

### `status` output

```
Service:    active (since 2026-06-12 14:03:21 UTC)
Last tick:  2026-06-12 14:08:54 UTC (5s ago)
Queued:     12
Running:    3
Canceling:  0
Failed:     1
Completed:  47
```

## Producer-Side CLI Changes

### `devrun run` — new options

```
--after <job_id>                    repeatable
--allow-failure-from <job_id>       repeatable; subset of --after
```

Flow:
1. Existing merge pipeline runs (`load_merged_config` → `TaskConfig` → sweep expansion).
2. For each combo, instead of calling `executor.submit_with_retry()`:
   - Insert job row with `status=QUEUED`, `params_template` = unresolved merged dict (JSON).
   - For each `--after`, insert a `job_dependencies` edge; `allow_failure=1` if listed in `--allow-failure-from`, else `0`.
3. Print `Job queued: <job_id> (waiting on N deps)`.

`--after` IDs validated against local DB; unknown IDs → typer.Exit with clear error. `--allow-failure-from` must be a subset of `--after`; otherwise error.

### `devrun workflow run` — pure producer

1. Merge config (unchanged).
2. Topologically validate stages (unchanged, Pydantic).
3. In a single SQLite transaction:
   - Insert one job row per stage (`status=QUEUED`, `params_template` = stage's params with cross-stage references rewritten as `${jobs:<id>,...}`).
   - Insert `job_dependencies` rows mapping the stage DAG to the job DAG.
   - Insert the workflow row with `job_ids = [...]`.
4. Return `workflow_id` and the list of queued job IDs.

**Dropped:**
- `--detach/-d` flag (everything is detached).
- `run_detached()` and the `python -m devrun.workflow --state-file` entry point.
- `WorkflowRunner.run()`'s `while True` loop and submission/polling helpers.
- `WorkflowRunner.cancel()`'s direct executor cancellation.

**Kept:**
- `--start-after <stage>` — skips ancestor stages by **not enqueuing them**. Dependent stages' `${jobs:...}` references must resolve from either `--from-job` source or the merged config; otherwise enqueue refuses with a clear error. This eliminates the `skipped_params` shortcut that silently leaks defaults — the bug is fixed by removing the path.
- `--from-job <job_id>` — extracts params into the merge (unchanged).
- `--dry-run` — prints the plan, writes nothing.

### `devrun cancel <job_id>`

State machine drives behavior:
- `QUEUED` → `CANCELLED`
- `SUBMITTED` / `RUNNING` → `CANCELING`
- terminal → error "already <status>"

No executor calls. Heartbeat completes the transition.

### `devrun status` / `devrun history`

Pure DB reads — no executor calls.
- `history` renders `QUEUED` (cyan), `CANCELING` (yellow), `SKIPPED` (dim).
- `status --with-deps` joins `job_dependencies` and prints the dep graph for that job.

### `devrun workflow status <wf_id>`

Reconstructs the stage view from `workflows.job_ids` → `jobs` rows. Output unchanged from the user's POV.

### Heartbeat-aware warning

After a successful enqueue, producer commands check `HeartbeatService.is_active()`. If inactive:

```
⚠ Heartbeat is not running — queued jobs will not progress.
  Start it with: devrun heartbeat start    (systemd/launchd)
              or: devrun heartbeat          (foreground)
```

Warning only. Never auto-spawn. Never a hard error.

## Removed Surface

| Component | Why |
|---|---|
| `WorkflowRunner._submit_stage`, `_poll_job_status` | Moved into heartbeat scheduler |
| `WorkflowRunner.run()` heartbeat loop | Replaced by global heartbeat |
| `WorkflowRunner.cancel()` direct executor cancel | Replaced by state transition + heartbeat |
| `run_detached()` and `--detach` flag | All runs are detached now |
| `python -m devrun.workflow --state-file` entry | Workflow is no longer a runnable module |
| `TaskRunner.status()` live-refresh-from-executor branch | Heartbeat owns this |
| `TaskRunner.cancel()` direct `executor.cancel()` call | State transition only |

## Migration

Schema migration on `JobStore.__init__`:
- All `CREATE TABLE IF NOT EXISTS` (existing pattern).
- `ALTER TABLE jobs ADD COLUMN params_template TEXT` etc., wrapped in `try/except sqlite3.OperationalError` for idempotent re-runs.
- New `job_dependencies` table created if absent.
- A `schema_version` table tracks current version; forward migrations run in order on startup.

Data migration:
- Existing `PENDING` rows are conceptually pre-QUEUED jobs without deps — left untouched. The heartbeat ignores any status not in {QUEUED, SUBMITTED, RUNNING, CANCELING}.
- No backfill of `params_template` for historical rows.

Mixed-version coexistence (old producers + new heartbeat against the same DB) is unsupported and documented.

## Testing Plan

### New test files

| File | Coverage |
|---|---|
| `tests/test_job_dependencies.py` | edge table CRUD, FK cascade on job delete, idempotent inserts |
| `tests/test_jobref_resolver.py` | `${jobs:id,path}` resolver: hit, miss, unauthorized (no edge), nested paths |
| `tests/test_heartbeat_loop.py` | `tick()` phases in isolation; populated DB → expected transitions |
| `tests/test_heartbeat_cascade.py` | parent FAILED + `allow_failure=0` → child SKIPPED; `allow_failure=1` → child promotes; multi-hop cascade across ticks |
| `tests/test_heartbeat_promotion.py` | unresolved `${jobs:...}` → FAILED with `skip_reason`; `<REQUIRED:...>` survives → FAILED; happy path → SUBMITTED + `parameters` persisted |
| `tests/test_heartbeat_cancel.py` | `CANCELING` → `CANCELLED` after `executor.cancel()`; `QUEUED` → `CANCELLED` direct; dependent cascade picks up |
| `tests/test_heartbeat_service.py` | platform dispatch (Linux→systemd, Darwin→launchd) with both backends mocked; install/uninstall/start/stop/is_active |
| `tests/test_cli_after.py` | `devrun run --after <id>` writes edges; `--allow-failure-from` subset; unknown ID → exit 1 |
| `tests/test_workflow_as_producer.py` | `workflow run` writes N jobs + edges in one transaction; `--start-after` skips ancestors; no heartbeat invoked |

### Refactored test files

- `tests/test_workflow.py` — drop `while True` loop tests; replace with "enqueues correctly and returns" tests.
- `tests/test_workflow_simulation.py` — rebuild around feeding a sequence of `tick()` calls against a mocked executor.
- `tests/test_runner.py` — drop "refresh from executor in status()" tests.
- `tests/test_cli.py` — update `cancel`, `status`, `history` expectations.

### Test fixtures

- `mock_executor` — records `submit/status/cancel` calls; can be told to return staged statuses across ticks.
- `heartbeat_driver` — calls `tick(db)` N times against a populated store; asserts state transitions deterministically without sleeping.

## Rollout

Three PRs on `feat/dependency`:

1. **PR1 — schema + edges + resolver.** Adds `job_dependencies` table, `params_template`/`skip_reason` columns, new `JobStatus` values, `${jobs:...}` resolver. No behavior change yet. Tests for schema/resolver only.

2. **PR2 — heartbeat + service.** Adds `devrun/heartbeat.py`, `devrun/services/`, the `devrun heartbeat` CLI subcommand, and the systemd/launchd templates. Heartbeat runs against the new schema; no producer enqueues into it yet (test by manually inserting QUEUED rows).

3. **PR3 — producer flip + cleanup.** `devrun run` and `devrun workflow run` switch to enqueue-only. `run_detached`, `--detach`, in-runner polling all removed. CLI changes for `cancel`/`status`/`history`. AGENT.md updated. This is the breaking PR.

Each PR ends green with `pytest tests/` before the next starts.

## AGENT.md Updates (after PR3)

- Architecture diagram: producers → DB → heartbeat → executors.
- New module entries: `devrun/heartbeat.py`, `devrun/services/`, `devrun/jobref.py`.
- Workflow module description: pure producer.
- Schema section: `jobs` new columns, `job_dependencies` table, new statuses.
- Remove references to `run_detached()` and `--detach`.

## Open Items (deferred)

- Batched executor status polling (Slurm `sacct --jobs=a,b,c`, SSH multi-status). Seam is added; implementations later.
- Removal of legacy `stages_state` column. Kept for backward compat; remove in a follow-up after one release cycle.
- Heartbeat metrics/observability beyond `last-tick` timestamp.
