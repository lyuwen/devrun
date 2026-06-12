# PR3 — Producer Flip + Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-06-12-job-dependency-mechanism-design.md` (sections "Producer-side CLI changes", "Workflow as producer", "Producer-side Open Behaviors", and "Rollout").

**Goal:** Flip `devrun run` and `devrun workflow run` to enqueue-only producers that write `QUEUED` rows plus `job_dependencies` / `workflow_jobs` edges via the PR1 JobStore API. Remove `WorkflowRunner.run_detached`, `--detach`, the in-runner polling loop, `TaskRunner.status()` executor refresh, and direct `TaskRunner.cancel()` executor calls. The heartbeat (PR2) takes over everything async.

**This is the breaking PR.** It changes user-visible behavior: every `devrun run` is now async, `devrun status`/`history` no longer ping executors, `cancel` no longer kills remote jobs directly, and `workflow run` returns immediately without `-d`.

**Architecture:** Producers do (1) hierarchical config merge via `load_merged_omegaconf` (PR1), (2) `<REQUIRED:...>` validation on enqueue-safe fields, (3) `from-job` + `start-after` rewriting of `${stages:...}` → `${jobs:<source_job_id>,...}` and source-job edge pointers, (4) atomic `JobStore.enqueue` / `enqueue_workflow`. Then they emit a heartbeat-liveness warning and exit.

**Tech Stack:** OmegaConf (unresolved), JobStore API (PR1), Typer (CLI), no new dependencies.

**Exit criteria:** Full suite passes, including new producer tests (`test_cli_after.py`, `test_workflow_as_producer.py`) and the rewritten `test_workflow.py` / `test_workflow_simulation.py`. Manual smoke: `devrun heartbeat &` then `devrun workflow run swe_full --start-after collect --from-job <agentic_id>` enqueues the workflow, the heartbeat drives stages to completion, and `devrun workflow status` reflects them.

---

## File Structure

| File | Change |
|---|---|
| `devrun/runner.py` | `TaskRunner.run()` → calls `JobStore.enqueue`; `status()` becomes pure DB; `cancel()` calls `JobStore.request_cancel`. |
| `devrun/workflow.py` | `WorkflowRunner.run()` → producer-only: build stage rows + edges + reference rewrites, call `JobStore.enqueue_workflow`. Drop `run_detached`, `_poll_job_status`, `_submit_stage`, `cancel()`, the while-True loop. |
| `devrun/cli.py` | `run`: add `--after` / `--allow-failure-from`. `workflow run`: drop `--detach`. `cancel`/`status`/`history`: render new statuses. `workflow logs`: handle `source_job_id` dereferencing. |
| `devrun/db/jobs.py` | Minor: `status_counts` already in PR2. |
| `tests/test_cli_after.py` | New — `devrun run --after`. |
| `tests/test_workflow_as_producer.py` | New — `workflow run` writes job + edges + workflow_jobs atomically; `--start-after` + `--from-job` rewrites `${stages:...}` to `${jobs:<source>,...}`. |
| `tests/test_workflow.py` | Rewrite — drop while-True loop tests; assert post-enqueue DB state. |
| `tests/test_workflow_simulation.py` | Rewrite — feed sequential `tick()` calls; assert end state. |
| `tests/test_runner.py` | Drop "refresh from executor in status()" tests; assert pure DB. |
| `tests/test_cli.py` | Update `cancel`/`status`/`history` expectations. |
| `AGENT.md` | Producer/consumer split; remove `run_detached`/`--detach` mentions; new status table; warning behavior. |

---

## Task 1: Add `--after` and `--allow-failure-from` to `devrun run`

**Files:**
- Modify: `devrun/cli.py` (the `run` command)
- Modify: `devrun/runner.py` (`TaskRunner.run`)
- Create: `tests/test_cli_after.py`

The producer path moves into `TaskRunner.run()` directly: it computes params (existing merge pipeline), calls `JobStore.enqueue()`, then `JobStore.insert_dependency()` for each `--after` entry.

- [ ] **Step 1: Write failing test**

```python
def test_run_after_writes_edges(jobstore, cli_runner):
    parent = jobstore.enqueue(task_name="t", executor="local",
                              params_template="", parameters={})
    result = cli_runner.invoke(app, ["run", "eval",
                                     "--after", parent,
                                     "params.model=x"])
    assert result.exit_code == 0
    new_jobs = [j for j in jobstore.list_all() if j.job_id != parent]
    assert len(new_jobs) == 1
    deps = jobstore.list_dependencies(new_jobs[0].job_id)
    assert deps[0].parent_job_id == parent
    assert deps[0].allow_failure is False
```

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Add CLI flags + plumb to runner**

```python
# devrun/cli.py — inside run()
after: list[str] = typer.Option(
    None, "--after",
    help="Wait for these job IDs to complete before submitting. Repeatable."),
allow_failure_from: list[str] = typer.Option(
    None, "--allow-failure-from",
    help="Subset of --after jobs whose failure should not block this run."),
...
runner.run(
    target=target,
    overrides=overrides,
    from_job=from_job,
    after=after or [],
    allow_failure_from=set(allow_failure_from or []),
)
```

- [ ] **Step 4: Change `TaskRunner.run()` to enqueue**

(Keep the existing merge pipeline. Replace the `executor.submit_with_retry(...)` call and the `db.insert(...)` immediate-submission with `db.enqueue(...)` + `db.insert_dependency(...)`.)

```python
def run(self, target, overrides=None, from_job=None,
        after=None, allow_failure_from=None):
    after = list(after or [])
    allow_failure_from = set(allow_failure_from or [])

    # Validate --after IDs against local DB upfront
    for jid in after:
        if self.db.get(jid) is None:
            raise typer.BadParameter(f"unknown job id: {jid}")

    cfg_unresolved = load_merged_omegaconf(target, overrides=overrides)
    # … existing from-job merge …
    cfg_unresolved = self._apply_from_job(cfg_unresolved, from_job)

    # Build params_template + best-effort resolved parameters per sweep combo
    sweep_iter = self._expand_sweep(cfg_unresolved)
    for params_template_yaml, resolved_params in sweep_iter:
        job_id = self.db.enqueue(
            task_name=target,
            executor=resolved_params.get("executor", "local"),
            params_template=params_template_yaml,
            parameters=resolved_params,
            initial_status=JobStatus.QUEUED,
        )
        for parent_id in after:
            self.db.insert_dependency(
                child_job_id=job_id,
                parent_job_id=parent_id,
                allow_failure=parent_id in allow_failure_from,
            )
        typer.echo(f"Queued {job_id} (after={after or 'none'})")

    self._warn_if_no_heartbeat()
```

- [ ] **Step 5: `_warn_if_no_heartbeat()` helper**

```python
def _warn_if_no_heartbeat(self) -> None:
    from devrun.services import get_service
    if not get_service().is_active():
        typer.echo(
            "⚠ Heartbeat is not running — queued jobs will not progress.\n"
            "  Start it with: devrun heartbeat start    (systemd/launchd)\n"
            "              or: devrun heartbeat          (foreground)",
            err=True,
        )
```

- [ ] **Step 6: Run test** — Expected: PASS.

- [ ] **Step 7: Add `--allow-failure-from` test**

```python
def test_run_after_with_allow_failure(jobstore, cli_runner):
    p = jobstore.enqueue(...)
    cli_runner.invoke(app, ["run", "eval", "--after", p,
                            "--allow-failure-from", p])
    new_jobs = [...]
    deps = jobstore.list_dependencies(new_jobs[0].job_id)
    assert deps[0].allow_failure is True
```

- [ ] **Step 8: Add unknown-ID test** — Expected exit code 1.

- [ ] **Step 9: Commit**

```bash
git add devrun/cli.py devrun/runner.py tests/test_cli_after.py
git commit -m "feat(cli): devrun run becomes async producer with --after edges"
```

## Task 2: Drop `TaskRunner.status()` executor refresh and `cancel()` direct call

**Files:**
- Modify: `devrun/runner.py`
- Modify: `tests/test_runner.py`
- Modify: `devrun/cli.py` — the `cancel` command calls `self.db.request_cancel(job_id)` instead of `runner.cancel(...)`.

- [ ] **Step 1: Update failing tests in `test_runner.py`** that assume executor.status is called during `status()`. Replace them with pure-DB assertions.

- [ ] **Step 2: Simplify `TaskRunner.status()`**

```python
def status(self, job_id: str) -> JobRecord:
    rec = self.db.get(job_id)
    if rec is None:
        raise ValueError(f"unknown job: {job_id}")
    return rec
```

- [ ] **Step 3: Delete `TaskRunner.cancel()`** in favor of CLI calling `JobStore.request_cancel`:

```python
# devrun/cli.py — cancel command
@app.command()
def cancel(job_id: str):
    db = JobStore(default_db_path())
    new_status = db.request_cancel(job_id)
    typer.echo(f"{job_id} → {new_status.value}")
```

- [ ] **Step 4: Run tests** — `pytest tests/test_runner.py tests/test_cli.py -v`. Expected: PASS after the deletions/updates.

- [ ] **Step 5: Commit**

```bash
git add devrun/runner.py devrun/cli.py tests/test_runner.py tests/test_cli.py
git commit -m "refactor(runner): pure-DB status; cancel routes through request_cancel"
```

## Task 3: Render new statuses in `history` and `status`

**Files:**
- Modify: `devrun/cli.py` — Rich rendering for the `history` and `status` tables.

- [ ] **Step 1: Add styles for new statuses**

```python
_STATUS_STYLE = {
    "queued":     "cyan",
    "submitting": "blue",
    "submitted":  "blue",
    "running":    "green",
    "canceling":  "yellow",
    "completed":  "bold green",
    "failed":     "bold red",
    "cancelled":  "dim red",
    "skipped":    "dim",
    "timed_out":  "magenta",
}
```

- [ ] **Step 2: Add `--with-deps` flag to `status`** that joins `job_dependencies` and prints parent IDs + statuses.

```python
@app.command()
def status(job_id: str,
           with_deps: bool = typer.Option(False, "--with-deps")):
    db = JobStore(default_db_path())
    rec = db.get(job_id)
    if rec is None:
        raise typer.Exit(1)
    _render_job(rec)
    if with_deps:
        for d in db.list_dependencies(job_id):
            p = db.get(d.parent_job_id)
            typer.echo(f"  parent {d.parent_job_id}: {p.status.value} "
                       f"(allow_failure={d.allow_failure})")
```

- [ ] **Step 3: Test the rendering** — extend `tests/test_cli.py` to assert each new status appears with the correct style label, and that `--with-deps` lists parents.

- [ ] **Step 4: Commit**

```bash
git add devrun/cli.py tests/test_cli.py
git commit -m "feat(cli): render queued/canceling/skipped/timed_out; add status --with-deps"
```

## Task 4: Workflow producer — basic enqueue (no skip/from-job yet)

**Files:**
- Modify: `devrun/workflow.py` — replace `WorkflowRunner.run()` entirely.
- Create: `tests/test_workflow_as_producer.py`

- [ ] **Step 1: Write failing test for happy path**

```python
def test_workflow_run_writes_jobs_and_edges_atomically(jobstore, cli_runner):
    result = cli_runner.invoke(app, ["workflow", "run", "swe_full"])
    assert result.exit_code == 0
    # Two stages: inference → collect
    wf_id = result.stdout.strip().split()[-1]
    stages = jobstore.get_workflow_stages(wf_id)
    assert [s.stage_name for s in stages] == ["inference", "collect"]
    inf_job = stages[0].job_id
    col_job = stages[1].job_id
    assert jobstore.get(inf_job).status == JobStatus.QUEUED
    deps = jobstore.list_dependencies(col_job)
    assert deps[0].parent_job_id == inf_job
```

- [ ] **Step 2: Rewrite `WorkflowRunner.run()` as producer**

```python
def run(self, *, from_job: str | None = None, start_after: str | None = None,
        overrides: list[str] | None = None, dry_run: bool = False) -> str:
    cfg = self._merge_unresolved(overrides=overrides, from_job=from_job)
    self._validate_topology(cfg)

    skip_set, source_jobs = self._compute_skip_set(cfg, start_after, from_job)
    stage_jobs = {}  # stage_name → new job_id (or source_job_id if skipped)
    stage_rows = []
    edges = []

    for ordinal, stage in enumerate(self._topo_order(cfg)):
        if stage.name in skip_set:
            src = source_jobs.get(stage.name)
            if src is None:
                raise typer.BadParameter(
                    f"stage '{stage.name}' is in --start-after's ancestor set "
                    f"but no source job is available; pass --from-job")
            stage_jobs[stage.name] = src
            stage_rows.append(WorkflowStageRow(
                stage_name=stage.name, ordinal=ordinal,
                job_id=None, source_job_id=src,
                task_name=None, executor=None,
                params_template=None, parameters=None,
            ))
            continue

        # rewrite ${stages:X,Y} → ${jobs:<id>,Y} where X is already resolved
        rewritten = self._rewrite_stage_refs(stage.params_template,
                                             stage_jobs)
        new_id = uuid.uuid4().hex[:12]
        stage_jobs[stage.name] = new_id
        stage_rows.append(WorkflowStageRow(
            stage_name=stage.name, ordinal=ordinal,
            job_id=new_id, source_job_id=None,
            task_name=stage.task, executor=stage.executor,
            params_template=rewritten,
            parameters=self._best_effort_resolve(rewritten),
        ))
        for dep in stage.depends_on:
            edges.append((new_id, stage_jobs[dep],
                          stage.allow_failure_from.get(dep, False)))

    if dry_run:
        _print_plan(stage_rows, edges); return ""

    deadline = (datetime.now(timezone.utc) + cfg.workflow.timeout
                if cfg.workflow.timeout else None)
    wf_id = self.db.enqueue_workflow(
        workflow_name=cfg.workflow.name,
        deadline_at=deadline,
        stage_rows=stage_rows,
        edges=edges,
    )
    typer.echo(f"Queued workflow {wf_id}")
    self._warn_if_no_heartbeat()
    return wf_id
```

(`_rewrite_stage_refs` does a textual substitution on the OmegaConf-serialized YAML: every `${stages:<stage>,<path>}` becomes `${jobs:<resolved_id>,<path>}` where the resolved id comes from `stage_jobs`. Reuse the regex pattern already used in PR1's resolver tests.)

- [ ] **Step 3: Run test** — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add devrun/workflow.py tests/test_workflow_as_producer.py
git commit -m "feat(workflow): producer-only run() — enqueue stages + edges atomically"
```

## Task 5: Workflow producer — `--start-after` + `--from-job` rewriting

**Files:**
- Modify: `devrun/workflow.py` — `_compute_skip_set`, `_rewrite_stage_refs` interaction with skipped stages.
- Modify: `tests/test_workflow_as_producer.py`

This is the regression fix for the open `skipped_params` bug. The skipped stage's downstream references must resolve from the **real source job's `parameters`** in the DB, not from a transient `skipped_params` dict.

- [ ] **Step 1: Write failing test**

```python
def test_start_after_with_from_job_rewrites_to_source_jobref(jobstore, cli_runner):
    # Pre-seed a swe_bench_agentic job in DB with known output_dir
    agentic_id = jobstore.enqueue(
        task_name="swe_bench_agentic", executor="slurm",
        params_template="",
        parameters={"output_dir": "/runs/abc", "model_name": "x"},
    )
    jobstore.update_status(agentic_id, JobStatus.COMPLETED)

    result = cli_runner.invoke(app, [
        "workflow", "run", "swe_full",
        "--start-after", "collect",
        "--from-job", agentic_id,
    ])
    assert result.exit_code == 0
    wf_id = result.stdout.strip().split()[-1]
    stages = jobstore.get_workflow_stages(wf_id)

    # inference stage was skipped — points at agentic_id
    inf = next(s for s in stages if s.stage_name == "inference")
    assert inf.job_id is None
    assert inf.source_job_id == agentic_id

    # collect stage's params_template references ${jobs:<agentic_id>,output_dir}
    col = next(s for s in stages if s.stage_name == "collect")
    col_rec = jobstore.get(col.job_id)
    assert f"${{jobs:{agentic_id}," in col_rec.params_template

    # dep edge from collect points at agentic_id (the source job)
    deps = jobstore.list_dependencies(col.job_id)
    assert any(d.parent_job_id == agentic_id for d in deps)
```

- [ ] **Step 2: Implement `_compute_skip_set` per spec section "workflow run" step 3**

```python
def _compute_skip_set(self, cfg, start_after, from_job):
    if start_after is None:
        return set(), {}
    skip_set = self._ancestors_inclusive(cfg, start_after)
    source_jobs = {}
    if from_job:
        src_rec = self.db.get(from_job)
        for stage_name in skip_set:
            stage = cfg.stages_by_name[stage_name]
            if src_rec.task_name == stage.task:
                source_jobs[stage_name] = from_job
                break  # one source job satisfies one stage
    # Future: walk back history to satisfy other skipped stages
    return skip_set, source_jobs
```

(For the first cut, exactly one skipped stage gets the `--from-job` source; the rest fail enqueue if anyone references them. Matches the spec's "refuse if downstream references and no source is available" rule.)

- [ ] **Step 3: Adjust `_rewrite_stage_refs`** — when a stage is in `skip_set`, the entry in `stage_jobs[stage_name]` is the `source_job_id`, so the existing rewrite logic Just Works.

- [ ] **Step 4: Adjust edge generation** — `for dep in stage.depends_on:` produces an edge pointing at whatever's in `stage_jobs[dep]`, which is the source job for skipped stages.

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add devrun/workflow.py tests/test_workflow_as_producer.py
git commit -m "fix(workflow): rewrite ${stages:X,...} to ${jobs:<source_job_id>,...} when X is skipped"
```

## Task 6: Drop `run_detached`, `--detach`, in-runner polling

**Files:**
- Modify: `devrun/workflow.py` — delete `run_detached`, `_submit_stage`, `_poll_job_status`, `cancel`, `_resolve_stage_params`, the while-True loop.
- Modify: `devrun/cli.py` — remove `-d/--detach` flag and its handler from `workflow run`.
- Modify: `tests/test_workflow.py` — drop tests for the deleted code.
- Modify: `tests/test_workflow_simulation.py` — rewrite around `tick()` calls.

- [ ] **Step 1: Identify all callers** — `grep -rn "run_detached\|--detach\| -d \|_submit_stage\|_poll_job_status\|_resolve_stage_params\|WorkflowRunner.cancel" devrun/ tests/`.

- [ ] **Step 2: Delete the code in `devrun/workflow.py`** — the producer-only `run()` from Task 4–5 stays; everything else listed above goes.

- [ ] **Step 3: Drop `--detach` flag from `devrun/cli.py`** — remove the option and any branches that depend on it.

- [ ] **Step 4: Remove the `python -m devrun.workflow --state-file …` entry point** — the file's `__main__` block.

- [ ] **Step 5: Rewrite `test_workflow_simulation.py`** — feed sequential `tick()` calls from PR2 against a mocked executor:

```python
def test_full_workflow_simulation(jobstore, mock_executor_router):
    runner = WorkflowRunner(db=jobstore)
    wf_id = runner.run(target_config="swe_full")
    # Seed parent statuses across ticks
    mock_executor_router.executors["slurm"].status_queue = [
        "running", "running", "completed",
    ]
    for _ in range(10):
        tick(jobstore, mock_executor_router)
        if all(jobstore.get(s.job_id).status in TERMINAL
               for s in jobstore.get_workflow_stages(wf_id)
               if s.job_id):
            break
    final = [jobstore.get(s.job_id).status for s in jobstore.get_workflow_stages(wf_id) if s.job_id]
    assert all(s == JobStatus.COMPLETED for s in final)
```

- [ ] **Step 6: Run full suite** — `pytest tests/`. Expected: PASS (some tests may need their imports/fixtures updated; do that inline).

- [ ] **Step 7: Commit**

```bash
git add devrun/workflow.py devrun/cli.py tests/test_workflow.py tests/test_workflow_simulation.py
git commit -m "refactor(workflow): drop run_detached, --detach, in-runner polling — heartbeat owns lifecycle"
```

## Task 7: `workflow logs` handles skipped source stages

**Files:**
- Modify: `devrun/cli.py` — the `workflow logs` command.
- Modify: `tests/test_cli.py`

When a stage is satisfied by a `source_job_id`, `workflow logs <wf_id> <stage_name>` should fetch logs from the source job's `log_path`. When the stage's `job_id` is QUEUED with no `remote_job_id` yet, print `"<queued — no logs yet>"`.

- [ ] **Step 1: Write failing test** — workflow with one skipped (source) stage; `workflow logs <wf_id> <skipped_stage>` returns the source job's stored logs.

- [ ] **Step 2: Implement**

```python
@workflow_app.command("logs")
def workflow_logs(workflow_id: str, stage_name: str):
    db = JobStore(default_db_path())
    stages = db.get_workflow_stages(workflow_id)
    stage = next((s for s in stages if s.stage_name == stage_name), None)
    if stage is None:
        raise typer.Exit(1)
    target_job_id = stage.job_id or stage.source_job_id
    rec = db.get(target_job_id)
    if rec.remote_job_id is None:
        typer.echo("<queued — no logs yet>")
        return
    executor = ExecutorRouter().get(rec.executor)
    typer.echo(executor.logs(rec.remote_job_id, log_path=rec.log_path))
```

- [ ] **Step 3: Run test** — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add devrun/cli.py tests/test_cli.py
git commit -m "feat(cli): workflow logs dereferences source_job_id for skipped stages"
```

## Task 8: `devrun rerun` — dependency-free copy

**Files:**
- Modify: `devrun/cli.py` (or wherever `rerun` lives today)
- Modify: `tests/test_cli.py`

Per spec "Producer-side Open Behaviors": `rerun <job_id>` copies `task_name`, `executor`, and resolved `parameters`. It does NOT copy `--after` edges. New row is `QUEUED`.

- [ ] **Step 1: Write failing test** — rerun a job that had `--after` edges; assert the new job has no edges.

- [ ] **Step 2: Update rerun implementation**

```python
@app.command()
def rerun(job_id: str):
    db = JobStore(default_db_path())
    src = db.get(job_id)
    if src is None:
        raise typer.Exit(1)
    new_id = db.enqueue(
        task_name=src.task_name,
        executor=src.executor,
        params_template=src.params_template or "",
        parameters=src.parameters,
    )
    typer.echo(f"Queued {new_id} (rerun of {job_id}; no dependencies copied)")
```

- [ ] **Step 3: Run tests** — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add devrun/cli.py tests/test_cli.py
git commit -m "feat(cli): rerun creates dependency-free queued copy"
```

## Task 9: End-to-end smoke test (producer + heartbeat + executor)

**Files:**
- Create: `tests/test_e2e_dependency.py`

- [ ] **Step 1: Write integration test** — enqueue a 2-stage workflow against `LocalExecutor`, drive `tick()` in a loop until both stages complete or 10 ticks pass.

```python
def test_e2e_workflow_through_heartbeat(jobstore, tmp_path):
    """A real local 2-stage workflow drained by the heartbeat tick loop."""
    runner = WorkflowRunner(db=jobstore)
    wf_id = runner.run(target_config="tests/test_data/sample_configs/two_stage_local.yaml")
    router = ExecutorRouter()
    for _ in range(20):
        tick(jobstore, router)
        stages = jobstore.get_workflow_stages(wf_id)
        statuses = [jobstore.get(s.job_id).status for s in stages if s.job_id]
        if all(s in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED)
               for s in statuses):
            break
        time.sleep(0.2)
    final = [jobstore.get(s.job_id).status for s in jobstore.get_workflow_stages(wf_id) if s.job_id]
    assert final == [JobStatus.COMPLETED, JobStatus.COMPLETED]
```

(Create a minimal `two_stage_local.yaml` sample config under `tests/test_data/sample_configs/` if one doesn't already exist — two `eval` stages, second depends on first, both running `echo hello` via `LocalExecutor`.)

- [ ] **Step 2: Run test** — Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_dependency.py tests/test_data/sample_configs/two_stage_local.yaml
git commit -m "test(e2e): workflow flows through heartbeat to local executor completion"
```

## Task 10: AGENT.md rewrite + full suite

**Files:**
- Modify: `AGENT.md`

This is the biggest doc change — many sentences across "High-Level Modules", "Database Schema", "Development Specifics" are now stale.

- [ ] **Step 1: Update the architecture diagram**

```
CLI (typer)
  ↓                                ┌──────────────────────────┐
TaskRunner / WorkflowRunner   →    │  jobs + job_dependencies │  ← writes only
(enqueue-only producers)            │  + workflow_jobs (SQLite)│
                                   └──────────┬───────────────┘
                                              │ reads
                                              ↓
                                   devrun heartbeat (scheduler)
                                              │
                                              ↓
                              ExecutorRouter → Executor Plugins → Backends
```

- [ ] **Step 2: Rewrite `devrun/runner.py` and `devrun/workflow.py` bullets** to reflect producer-only role.

- [ ] **Step 3: Add bullets for `devrun/heartbeat.py`, `devrun/services/`, `devrun/jobref.py`** (heartbeat one was added in PR2 — confirm it's still accurate).

- [ ] **Step 4: Update Database Schema section** with all PR1 + PR2 + PR3 columns/tables.

- [ ] **Step 5: Remove references** to `run_detached()`, `--detach`, `WorkflowRunner._submit_stage`, `_poll_job_status`, `_resolve_stage_params`, `skipped_params`.

- [ ] **Step 6: Add new "Job Dependency & Heartbeat Lifecycle" subsection** under Development Specifics summarizing the new model in 5–8 sentences. Reference the spec for full detail.

- [ ] **Step 7: Update test count** at the end (run `pytest tests/ --co -q | tail -3` and patch the number).

- [ ] **Step 8: Full suite** — `pytest tests/`. Expected: all green.

- [ ] **Step 9: Manual smoke** — described in the PR description, not a unit test:
  1. `devrun heartbeat install && devrun heartbeat start`
  2. `devrun workflow run <some real workflow>`
  3. `devrun workflow status <wf_id>` — eventually shows COMPLETED.
  4. `devrun heartbeat stop && devrun heartbeat uninstall`.

- [ ] **Step 10: Commit**

```bash
git add AGENT.md
git commit -m "docs(agent): rewrite for enqueue-only producers + heartbeat consumer (PR3)"
```

## Task 11: Open PR with detailed migration notes

- [ ] **Step 1: Push branch** — `git push -u origin feat/dependency`.

- [ ] **Step 2: Open PR** — title: "feat: job-level dependencies + heartbeat scheduler". Body includes:
  - Link to spec.
  - Three-PR rationale.
  - Breaking changes list: `--detach` gone; `devrun run` always async; `devrun cancel` no longer kills remote directly; `devrun status` / `history` are pure DB reads; new `devrun heartbeat …` subcommand required for anything to progress.
  - Migration notes: existing `PENDING` rows are ignored by the heartbeat; legacy in-flight remote jobs may need manual cleanup.
  - Required follow-up: install `devrun heartbeat start` on any production host before merging.

---

## Self-review checklist (PR3)

- [ ] `devrun run` always enqueues; never calls `executor.submit` directly.
- [ ] `--after` validated against local DB; unknown ID → exit 1.
- [ ] `--allow-failure-from` subset rejects entries not in `--after`.
- [ ] `devrun workflow run` writes the workflow row + all stage jobs + `workflow_jobs` rows + `job_dependencies` edges in a single transaction.
- [ ] `--start-after` + `--from-job` rewrites `${stages:X,...}` → `${jobs:<source_job_id>,...}` and points the dep edge at the source job. (Regression test exists.)
- [ ] No `skipped_params` shortcut anywhere in `devrun/workflow.py`.
- [ ] `run_detached`, `--detach`, `_submit_stage`, `_poll_job_status`, `_resolve_stage_params`, `WorkflowRunner.cancel` all deleted.
- [ ] `TaskRunner.status()` is pure DB read; no executor calls.
- [ ] `devrun cancel` routes through `JobStore.request_cancel`.
- [ ] `devrun status`, `history` render queued/canceling/skipped/timed_out distinctively.
- [ ] `status --with-deps` lists parents + their statuses.
- [ ] `workflow logs` dereferences `source_job_id` for skipped stages; shows `<queued — no logs yet>` when appropriate.
- [ ] `rerun` is dependency-free.
- [ ] All producer commands warn (don't error) when heartbeat is inactive.
- [ ] E2E test drives a workflow to completion through heartbeat ticks.
- [ ] AGENT.md fully reflects new architecture.
- [ ] Full test suite green.
