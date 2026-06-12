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

## Unresolved Config Pipeline

`load_merged_config()` today returns a fully-resolved `dict` via `OmegaConf.to_container(..., resolve=True)`. Producers in the new model need **both** views, so the merge pipeline is split:

```python
def load_merged_omegaconf(target, overrides=None, config_dirs=None) -> DictConfig:
    """Return the merged DictConfig without resolving interpolations."""

def load_merged_config(target, overrides=None, config_dirs=None) -> dict:
    """Existing behavior — fully resolved dict. Implemented via load_merged_omegaconf + to_container(resolve=True)."""
```

Producer flow:
1. `cfg = load_merged_omegaconf(target, overrides)` — unresolved DictConfig with `${jobs:...}` / `${stages:...}` / `<REQUIRED:...>` strings intact.
2. **Validate enqueue-safe fields** by attempting `OmegaConf.to_container(cfg, resolve=True)` inside a try/except. If only cross-job references fail, those are deferred; any other failure (missing `${key:...}`, malformed YAML) is fatal at enqueue time.
3. `params_template = OmegaConf.to_yaml(cfg, resolve=False)` — stored verbatim in the DB column.
4. The eagerly-resolved view (where possible) is used to print dry-run plans and to populate the existing `parameters` column with the best-effort resolved form for display in `devrun history`. The authoritative resolution still happens at promotion.

Tests must assert that `${jobs:<id>,...}` substrings remain literally present inside `params_template` after enqueue.

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
    SUBMITTING = "submitting"   # NEW — claimed by one heartbeat tick; mid-submit
    SUBMITTED  = "submitted"
    RUNNING    = "running"
    CANCELING  = "canceling"    # NEW — cancel requested, executor.cancel() pending
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    SKIPPED    = "skipped"      # NEW — blocking parent failed (no remote_job_id)
    UNKNOWN    = "unknown"
```

### Heartbeat claim columns

```sql
ALTER TABLE jobs ADD COLUMN claimed_by         TEXT;     -- heartbeat instance ID (hostname:pid)
ALTER TABLE jobs ADD COLUMN claimed_at         TEXT;     -- ISO timestamp
ALTER TABLE jobs ADD COLUMN claim_expires_at   TEXT;     -- ISO timestamp; lease horizon
```

Used to make promotion atomic and crash-safe — see "Atomic Claim & Recovery" below.

### Cross-job interpolation resolver

New OmegaConf resolver registered in `devrun/jobref.py`:

```
${jobs:<job_id>,<dotted.path>}
```

Reads from `jobs.parameters` JSON of a completed predecessor.

**Scoping — promotion-local, not global.** A globally registered resolver has no calling-job context, so the resolver itself is a thin shim that reads from a `contextvars.ContextVar[JobRefContext]`. The heartbeat installs that context before resolving any single job's template, with:

```python
@dataclass(frozen=True)
class JobRefContext:
    allowed_parents: dict[str, dict]   # job_id → parameters dict
    calling_job_id: str
```

The resolver looks up `<job_id>` against `allowed_parents` only. If the requested ID is not in the set (i.e. no edge declared), it raises a clear error. The context is cleared in a `finally` block after each promotion attempt. There is no path by which an arbitrary code site can read another job's params via the resolver.

Tests must cover: unauthorized read (no edge), missing parent, nested-path access, two back-to-back promotions of different jobs where the allowed-parent maps differ.

### `workflows` table — new schema

The plain `job_ids` array proposed in the first draft cannot represent skipped-but-satisfied source stages, stage→job mapping, or parallel stages with similar shapes. Replaced with a normalized join table:

```sql
CREATE TABLE IF NOT EXISTS workflow_jobs (
    workflow_id    TEXT NOT NULL,
    stage_name     TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,                -- topo order within the workflow
    job_id         TEXT,                            -- NULL when stage is satisfied by source_job_id
    source_job_id  TEXT,                            -- non-NULL when stage was skipped via --from-job
    PRIMARY KEY (workflow_id, stage_name),
    FOREIGN KEY (workflow_id)   REFERENCES workflows(workflow_id)  ON DELETE CASCADE,
    FOREIGN KEY (job_id)        REFERENCES jobs(job_id)            ON DELETE SET NULL,
    FOREIGN KEY (source_job_id) REFERENCES jobs(job_id)            ON DELETE SET NULL,
    CHECK (job_id IS NOT NULL OR source_job_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_wfjobs_workflow ON workflow_jobs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wfjobs_job      ON workflow_jobs(job_id);
```

Each row binds one stage of one workflow to either:
- a freshly-enqueued `job_id` (new work), or
- a `source_job_id` (the stage was skipped via `--start-after` and is satisfied by a pre-existing job — typically from `--from-job`).

`workflow status` / `workflow logs` reconstruct the stage view by joining `workflow_jobs` → `jobs` ordered by `ordinal`. The legacy `stages_state` column on `workflows` remains for one release for historical rows but is no longer written; removal is queued for a follow-up after one release cycle.

This preserves the richer state the current `WorkflowRunner` keeps (`resolved_params` lives on the underlying jobs row's `parameters`; stage name and ordering live in `workflow_jobs`; skipped-source-job mapping is first-class).

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

Two-step **atomic claim** + submit:

**Step 2a — claim.** For each candidate row from the readiness query, run a compare-and-set:

```sql
UPDATE jobs
SET    status            = 'submitting',
       claimed_by        = :instance_id,
       claimed_at        = :now,
       claim_expires_at  = :now_plus_lease
WHERE  job_id = :job_id
  AND  status = 'queued';
```

`instance_id = f"{socket.gethostname()}:{os.getpid()}"`. `lease = 2 * heartbeat.interval` (default 20 s). If the `UPDATE` affected zero rows, another process won the claim — skip. Only the winner proceeds to submit.

**Step 2b — readiness query** (run before 2a; same shape as before):

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

For each ready job (after winning the claim):
1. Load `params_template`; fetch parent records (both `job_dependencies` parents and any `workflow_jobs.source_job_id` if this job belongs to a workflow).
2. Install `JobRefContext(allowed_parents=…, calling_job_id=…)` and build the OmegaConf load with the `${jobs:...}` resolver active.
3. Resolve. If `<REQUIRED:...>` survives or any `${jobs:...}` fails → `FAILED` with `skip_reason`. Release the claim columns to NULL on failure.
4. `task = get_task_class(task_name)(); spec = task.prepare(resolved_params)`.
5. `executor.submit_with_retry(spec)` → on success, atomic transition `SUBMITTING` → `SUBMITTED` setting `remote_job_id`, `log_path`, persisted `parameters`, and clearing claim columns.

All exceptions in steps 3–5 mark the job `FAILED` with the exception text in `skip_reason` and clear claim columns. Phase 1 of the next tick cascades the failure.

### Atomic Claim & Recovery

A heartbeat may crash between step 5 (executor.submit returned a `remote_job_id`) and the DB write that persists it. To recover, every tick runs a **claim sweep** before phase 1:

```sql
-- Reclaim leases that have expired (heartbeat died mid-submit)
UPDATE jobs
SET    status            = 'queued',
       claimed_by        = NULL,
       claimed_at        = NULL,
       claim_expires_at  = NULL,
       skip_reason       = COALESCE(skip_reason || ' ; ', '') || 'reclaimed after stale lease'
WHERE  status = 'submitting'
  AND  claim_expires_at < :now
  AND  remote_job_id IS NULL;
```

Cases:
- **Crashed before `executor.submit` returned** — no remote job exists. Lease expires; job is reclaimed to `QUEUED` and tried again. Idempotency relies on `submit_with_retry` already being safe on retry (it generates unique sbatch filenames, UUID tokens, etc., per existing convention).
- **Crashed after `executor.submit` returned but before DB write** — remote job exists, `remote_job_id` was never persisted. This is the orphan case. We accept that the orphan remote job leaks; the local row gets reclaimed and submitted again. A `skip_reason` annotation records that this happened so users can investigate.

This is a deliberate trade-off: the alternative (write `remote_job_id=PENDING` before submit, then patch) requires a two-phase commit across SQLite + executor, which we don't have. Documented as a known limitation. Detection helper: `devrun heartbeat status --orphans` lists jobs with `skip_reason LIKE '%reclaimed%'`.

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
3. **Compute the skip set.** If `--start-after <stage>` is supplied, the producer marks `<stage>` and all its ancestors as skipped. For each skipped stage, identify its **source job**:
   - If `--from-job <id>` is supplied and the source job's `task_name` matches the skipped stage's task type, that job becomes the source for that stage.
   - For other skipped stages, walk back through `--from-job` history (the existing negative-index lookup logic continues to apply) or refuse to enqueue if a downstream stage references the skipped stage and no source job is available.
4. **Rewrite cross-stage references.** Any `${stages:<stage>,<path>}` in remaining stages' `params_template` is rewritten at enqueue time:
   - If `<stage>` is a freshly-enqueued stage → `${jobs:<new_job_id>,<path>}` (we know the new ID because we insert in topo order).
   - If `<stage>` is in the skip set with a source job → `${jobs:<source_job_id>,<path>}`.
   - Otherwise → enqueue refuses with a clear error naming the unresolvable reference.
5. In a single SQLite transaction:
   - Insert one job row per non-skipped stage (`status=QUEUED`, rewritten `params_template`).
   - Insert `job_dependencies` rows. Edges to skipped stages point at the **source job**, not the absent stage — so cascade-skip and the resolver both see real, persisted parents.
   - Insert the workflow row.
   - Insert `workflow_jobs` rows for every stage (skipped or not), recording `job_id` xor `source_job_id` and `ordinal`.
6. Return `workflow_id`, the list of queued job IDs, and a summary of which stages were satisfied by source jobs.

This is the explicit successor to today's `skipped_params` seeding: instead of carrying resolved param dicts in workflow state, we point dep edges and `${jobs:...}` references directly at the satisfying source job. The job's `parameters` column is the single source of truth; the resolver reads it.

**Dropped:**
- `--detach/-d` flag (everything is detached).
- `run_detached()` and the `python -m devrun.workflow --state-file` entry point.
- `WorkflowRunner.run()`'s `while True` loop and submission/polling helpers.
- `WorkflowRunner.cancel()`'s direct executor cancellation.

**Kept:**
- `--start-after <stage>` — see step 3 above. Cross-stage references to skipped stages are rewritten to `${jobs:<source_job_id>,...}`. The previous `skipped_params` shortcut is gone; the dep graph itself now encodes the satisfying source.
- `--from-job <job_id>` — extracts params into the merge (unchanged) AND, when combined with `--start-after`, serves as the explicit source for the matched skipped stage. The negative-index lookup (`-1`, `-2`, …) continues to work; "compatible" history filtering is unchanged.
- `--dry-run` — prints the plan, writes nothing.

### Workflow Timeout

`WorkflowConfig.timeout` remains a first-class feature, enforced by the heartbeat (not the producer).

```sql
ALTER TABLE workflows ADD COLUMN deadline_at TEXT;   -- ISO timestamp; NULL = no deadline
```

Set at enqueue time to `now + config.timeout`. The heartbeat runs a pre-phase each tick:

```sql
SELECT workflow_id FROM workflows
WHERE  status NOT IN ('completed','failed','cancelled','timed_out')
  AND  deadline_at < :now;
```

For each expired workflow, every non-terminal job in `workflow_jobs` for that workflow is transitioned: `QUEUED` → `SKIPPED` (`skip_reason='workflow deadline'`), `SUBMITTED`/`RUNNING` → `CANCELING`. The workflow row goes to `timed_out`. Cascade-skip handles transitive dependents on the next tick.

### Producer-side Open Behaviors

- **`devrun run` with no `--after`** still enqueues as `QUEUED`. There is no synchronous mode. This is a deliberate breaking change — surfaced via the heartbeat warning at every producer command and documented in PR3's release notes.
- **Legacy `PENDING` rows** are conceptually closed. The heartbeat ignores them in phases 1–3 (explicit status filters). Existing in-flight `PENDING` jobs from before the upgrade may still be live remotely; documented as a migration caveat.
- **`devrun rerun <job_id>`** copies the original job's `task_name`, `executor`, and resolved `parameters` into a **dependency-free new** `QUEUED` row. It does not copy `--after` edges. Users wanting a re-attempt that preserves the dep DAG should use `workflow run --from-job` instead.
- **`workflow logs` for skipped source stages** (where `workflow_jobs.source_job_id` is set and `job_id` is NULL) shows logs from the source job's `log_path` — same retrieval path, just dereferences `source_job_id`. For `QUEUED` jobs with no `remote_job_id` yet, output is `"<queued — no logs yet>"`.

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
| `tests/test_jobstore_api.py` | `enqueue`, `enqueue_workflow` single-tx atomicity, `claim_for_submit` CAS (two concurrent winners → exactly one), `reclaim_expired_leases`, `cascade_skip_dependents`, `request_cancel` state machine |
| `tests/test_jobref_resolver.py` | `${jobs:id,path}` resolver: hit, miss, unauthorized (no edge), nested paths, `JobRefContext` scoping across back-to-back promotions with different parent maps |
| `tests/test_unresolved_config.py` | `load_merged_omegaconf` preserves `${jobs:...}` and `${stages:...}`; `params_template` round-trip retains literal strings |
| `tests/test_heartbeat_loop.py` | `tick()` phases in isolation; populated DB → expected transitions |
| `tests/test_heartbeat_cascade.py` | parent FAILED + `allow_failure=0` → child SKIPPED; `allow_failure=1` → child promotes; multi-hop cascade across ticks |
| `tests/test_heartbeat_promotion.py` | unresolved `${jobs:...}` → FAILED with `skip_reason`; `<REQUIRED:...>` survives → FAILED; happy path → SUBMITTED + `parameters` persisted; SUBMITTING → SUBMITTED atomic transition |
| `tests/test_heartbeat_claim.py` | claim CAS under simulated concurrency; expired lease reclaim with NULL `remote_job_id`; orphan annotation on `skip_reason` |
| `tests/test_heartbeat_cancel.py` | `CANCELING` → `CANCELLED` after `executor.cancel()`; `QUEUED` → `CANCELLED` direct; dependent cascade picks up |
| `tests/test_heartbeat_timeout.py` | workflow `deadline_at` expiry → non-terminal jobs CANCELING/SKIPPED; workflow → timed_out |
| `tests/test_heartbeat_service.py` | platform dispatch (Linux→systemd, Darwin→launchd) with both backends mocked; install/uninstall/start/stop/is_active |
| `tests/test_cli_after.py` | `devrun run --after <id>` writes edges; `--allow-failure-from` subset; unknown ID → exit 1 |
| `tests/test_workflow_as_producer.py` | `workflow run` writes N jobs + edges in one transaction; `--start-after` skips ancestors; `--start-after` + `--from-job` rewrites `${stages:source,...}` → `${jobs:<source_job_id>,...}` and edges point at the source job; downstream stage references resolve from the real source-job `parameters` at promotion (regression test for the open bug); no heartbeat invoked |

### Refactored test files

- `tests/test_workflow.py` — drop `while True` loop tests; replace with "enqueues correctly and returns" tests.
- `tests/test_workflow_simulation.py` — rebuild around feeding a sequence of `tick()` calls against a mocked executor.
- `tests/test_runner.py` — drop "refresh from executor in status()" tests.
- `tests/test_cli.py` — update `cancel`, `status`, `history` expectations.

### Test fixtures

- `mock_executor` — records `submit/status/cancel` calls; can be told to return staged statuses across ticks.
- `heartbeat_driver` — calls `tick(db)` N times against a populated store; asserts state transitions deterministically without sleeping.

## JobStore API Surface (new)

The heartbeat and producers must not issue ad-hoc SQL. PR1 lands the following typed methods on `JobStore`; the heartbeat/producer code calls only these. All multi-row writes use a single transaction (`with self._conn:` block).

```python
class JobStore:
    # --- enqueue ---
    def enqueue(
        self,
        *,
        task_name: str,
        executor: str,
        params_template: str,          # OmegaConf yaml, unresolved
        parameters: dict | None,       # best-effort resolved view for display
        initial_status: JobStatus = JobStatus.QUEUED,
    ) -> str: ...

    def insert_dependency(
        self, *, child_job_id: str, parent_job_id: str, allow_failure: bool
    ) -> None: ...

    def enqueue_workflow(
        self,
        *,
        workflow_name: str,
        deadline_at: datetime | None,
        stage_rows: list[WorkflowStageRow],   # see below
        edges: list[tuple[str, str, bool]],   # (child_job_id, parent_job_id, allow_failure)
    ) -> str:
        """Single-transaction insert of workflow + stage jobs + workflow_jobs + edges."""

    # --- heartbeat: claim & promote ---
    def fetch_ready_queued(self, limit: int = 100) -> list[JobRecord]: ...
    def claim_for_submit(
        self, *, job_id: str, instance_id: str, lease_seconds: float
    ) -> bool:
        """Atomic CAS QUEUED → SUBMITTING. Returns True if this caller won."""

    def finalize_submit(
        self,
        *,
        job_id: str,
        remote_job_id: str,
        log_path: str | None,
        resolved_parameters: dict,
    ) -> None:
        """SUBMITTING → SUBMITTED, clear claim columns."""

    def fail_promotion(self, *, job_id: str, skip_reason: str) -> None:
        """SUBMITTING or QUEUED → FAILED, clear claim columns."""

    def reclaim_expired_leases(self, *, now: datetime) -> list[str]:
        """Reset SUBMITTING rows with expired lease + NULL remote_job_id back to QUEUED."""

    # --- heartbeat: cascade & poll ---
    def cascade_skip_dependents(self) -> list[str]: ...
    def fetch_active_jobs(self) -> list[JobRecord]: ...   # status IN submitted/running/canceling

    # --- heartbeat: workflow deadline ---
    def fetch_expired_workflows(self, *, now: datetime) -> list[str]: ...
    def expire_workflow(self, workflow_id: str) -> None: ...

    # --- producer-side cancel ---
    def request_cancel(self, job_id: str) -> JobStatus:
        """QUEUED → CANCELLED direct; SUBMITTED/RUNNING → CANCELING; else error."""

    # --- queries ---
    def list_dependencies(self, child_job_id: str) -> list[Dependency]: ...
    def get_parent_parameters(self, child_job_id: str) -> dict[str, dict]: ...
    def get_workflow_stages(self, workflow_id: str) -> list[WorkflowStageRow]: ...
```

All heartbeat-side mutations are confined to these methods. Tests live in `tests/test_jobstore_api.py` and exercise the CAS semantics (two concurrent `claim_for_submit` calls — only one wins).

## Rollout

Three PRs on `feat/dependency`:

1. **PR1 — schema + edges + resolver + JobStore API.** Adds `job_dependencies` and `workflow_jobs` tables, all new `jobs`/`workflows` columns (`params_template`, `skip_reason`, claim columns, `deadline_at`), new `JobStatus` values, `${jobs:...}` resolver with `JobRefContext` scoping, and the full `JobStore` API surface above. No behavior change yet for `run`/`workflow run`. Tests for schema, resolver scoping, CAS claiming, and `JobStore` APIs.

2. **PR2 — heartbeat + service.** Adds `devrun/heartbeat.py` (loop + four phases including workflow deadline pre-phase + claim sweep), `devrun/services/` with the Linux/Darwin backends, the `devrun heartbeat` CLI subcommand, and the systemd/launchd templates. Heartbeat runs against the new schema; producer code still uses the old path (test by manually inserting QUEUED rows via `JobStore.enqueue`).

3. **PR3 — producer flip + cleanup.** `devrun run` and `devrun workflow run` switch to enqueue-only via `JobStore.enqueue` / `enqueue_workflow`. `--from-job` + `--start-after` interaction reimplemented around `workflow_jobs.source_job_id` (no `skipped_params` shortcut). `run_detached`, `--detach`, in-runner polling all removed. CLI changes for `cancel`/`status`/`history`/`workflow logs`. AGENT.md updated. This is the breaking PR.

Each PR ends green with `pytest tests/` before the next starts.

## AGENT.md Updates (after PR3)

- Architecture diagram: producers → DB → heartbeat → executors.
- New module entries: `devrun/heartbeat.py`, `devrun/services/`, `devrun/jobref.py`.
- Workflow module description: pure producer.
- Schema section: `jobs` new columns, `job_dependencies` table, new statuses.
- Remove references to `run_detached()` and `--detach`.

## Review Round 1 — Addressed

This spec was reviewed against the recent `--from-job`/`--start-after` PR. The following findings from `2026-06-12-job-dependency-mechanism-design-review.md` are folded into the sections above:

1. **Unresolved config preservation** — new "Unresolved Config Pipeline" section with `load_merged_omegaconf(resolve=False)` and explicit `params_template` round-trip test.
2. **`--from-job` + `--start-after` regression risk** — workflow producer rewrites `${stages:...}` to `${jobs:<source_job_id>,...}` and points dep edges at the source job. The previous `skipped_params` shortcut is replaced by first-class graph state in `workflow_jobs`.
3. **Atomic claim & recovery** — new `SUBMITTING` status + claim columns + CAS in phase 2, lease sweep per tick, documented orphan trade-off.
4. **`workflow_jobs` join table** — replaces the proposed `workflows.job_ids` JSON array; stores `stage_name`, `ordinal`, `job_id` xor `source_job_id`.
5. **Resolver scoping** — `JobRefContext` ContextVar installed per promotion; resolver reads from allowed-parents map only.
6. **PR1 store APIs** — full `JobStore` API surface section; heartbeat never issues ad-hoc SQL.
7. **Workflow timeout** — kept as a first-class feature; `workflows.deadline_at` + heartbeat pre-phase.

Open behaviors (`devrun run` always async, `PENDING` legacy, `rerun` semantics, `workflow logs` for skipped sources) are documented under "Producer-side Open Behaviors".

## Open Items (deferred)

- Batched executor status polling (Slurm `sacct --jobs=a,b,c`, SSH multi-status). Seam is added; implementations later.
- Removal of legacy `stages_state` column. Kept for backward compat; remove in a follow-up after one release cycle.
- Heartbeat metrics/observability beyond `last-tick` timestamp.
