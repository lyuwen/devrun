"""Producer-flip tests for WorkflowRunner.run() (PR3 Task 4)."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.models import JobStatus, WorkflowConfig, WorkflowStage
from devrun.workflow import WorkflowRunner


@contextmanager
def _hard_timeout(seconds: float):
    """SIGALRM-based hard timeout so legacy heartbeat loops can't hang the suite.

    The legacy WorkflowRunner.run() enters a heartbeat poll loop that blocks
    until all stages terminate. These tests assert the producer-flip behavior
    (return promptly after DB writes); the timeout makes the failure mode
    visible as a TimeoutError rather than a hung test.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"test timed out after {seconds}s — producer-flip likely incomplete")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _two_stage_config() -> WorkflowConfig:
    return WorkflowConfig(
        workflow="producer_test",
        stages=[
            WorkflowStage(
                name="inference",
                task="eval",
                executor="local",
                params={"model": "gpt-4"},
            ),
            WorkflowStage(
                name="collect",
                task="eval",
                executor="local",
                depends_on="inference",
                params={"model": "gpt-4"},
            ),
        ],
        params={},
    )


def test_workflow_run_writes_jobs_and_edges_atomically(tmp_path: Path):
    """`WorkflowRunner.run()` must enqueue stages + edges via JobStore.enqueue_workflow.

    The new contract:
    - Returns the workflow_id (string).
    - Two stages: inference → collect.
    - Both new jobs rows are QUEUED (no executor.submit_with_retry was called).
    - A single dep edge: collect → inference, allow_failure=False.
    - get_workflow_stages returns the stages in ordinal order.
    """
    runner = WorkflowRunner(db_path=tmp_path / "jobs.db")

    with _hard_timeout(8):
        wf_id = runner.run(_two_stage_config())

    assert isinstance(wf_id, str) and len(wf_id) > 0

    db = JobStore(tmp_path / "jobs.db")
    stages = db.get_workflow_stages(wf_id)
    assert [s.stage_name for s in stages] == ["inference", "collect"]
    assert [s.ordinal for s in stages] == [0, 1]

    inf_job = stages[0].job_id
    col_job = stages[1].job_id
    assert inf_job is not None
    assert col_job is not None

    inf_rec = db.get(inf_job)
    col_rec = db.get(col_job)
    assert inf_rec is not None and JobStatus(inf_rec.status) == JobStatus.QUEUED
    assert col_rec is not None and JobStatus(col_rec.status) == JobStatus.QUEUED

    deps = db.list_dependencies(col_job)
    assert len(deps) == 1
    assert deps[0].parent_job_id == inf_job
    assert bool(deps[0].allow_failure) is False


def test_workflow_run_is_atomic(tmp_path: Path):
    """All stages + edges land in a single transaction — no half-written workflows."""
    runner = WorkflowRunner(db_path=tmp_path / "jobs.db")

    with _hard_timeout(8):
        wf_id = runner.run(_two_stage_config())

    db = JobStore(tmp_path / "jobs.db")
    # Workflow row exists in QUEUED state.
    wf_row = db._conn.execute(
        "SELECT workflow_name, status FROM workflows WHERE workflow_id = ?",
        (wf_id,),
    ).fetchone()
    assert wf_row is not None
    assert wf_row["workflow_name"] == "producer_test"
    # Every stage row points at a real jobs row.
    stages = db.get_workflow_stages(wf_id)
    assert len(stages) == 2, f"expected 2 workflow_jobs rows, got {len(stages)}"
    for stage in stages:
        assert stage.job_id is not None
        assert db.get(stage.job_id) is not None


def test_workflow_run_does_not_block_on_heartbeat(tmp_path: Path):
    """The producer must return immediately — no in-runner polling loop.

    A 1-stage workflow whose stage never completes used to hang for the
    full timeout in legacy WorkflowRunner.run().  After the producer flip,
    run() returns the workflow_id and exits.
    """
    import time

    runner = WorkflowRunner(db_path=tmp_path / "jobs.db")
    cfg = WorkflowConfig(
        workflow="nonblock",
        stages=[
            WorkflowStage(
                name="solo",
                task="eval",
                executor="local",
                params={"model": "gpt-4"},
            )
        ],
        params={},
        # If the legacy polling loop is still wired, this short timeout
        # would make the test fail with a timeout error or hang.
        timeout=600.0,
    )

    started = time.monotonic()
    with _hard_timeout(8):
        wf_id = runner.run(cfg)
    elapsed = time.monotonic() - started

    # Generous threshold: enqueue + DB writes should be well under one second.
    # The legacy heartbeat loop sleeps `heartbeat_interval` (default 30s)
    # between polls and would never complete in this window with a stub task.
    assert elapsed < 5.0, f"run() did not return promptly (elapsed={elapsed:.2f}s)"
    assert isinstance(wf_id, str) and len(wf_id) > 0


def test_workflow_run_persists_params_template(tmp_path: Path):
    """Each stage's params_template is stored on its jobs row.

    The heartbeat will later read params_template + ${jobs:...} refs to
    promote each stage, so the producer must persist the unresolved template.
    """
    runner = WorkflowRunner(db_path=tmp_path / "jobs.db")
    with _hard_timeout(8):
        wf_id = runner.run(_two_stage_config())

    db = JobStore(tmp_path / "jobs.db")
    stages = db.get_workflow_stages(wf_id)
    assert len(stages) == 2, f"expected 2 workflow_jobs rows, got {len(stages)}"
    for stage in stages:
        row = db._conn.execute(
            "SELECT params_template FROM jobs WHERE job_id = ?",
            (stage.job_id,),
        ).fetchone()
        assert row is not None
        assert row["params_template"] is not None
        assert row["params_template"] != ""


# ============================================================================
# Task 5 — `--start-after` + `--from-job` rewriting
# ============================================================================


def _three_stage_config() -> WorkflowConfig:
    """inference → collect → evaluate; downstream stages use ${stages:X,key} refs."""
    return WorkflowConfig(
        workflow="rewrite_test",
        stages=[
            WorkflowStage(
                name="inference",
                task="eval",
                executor="local",
                params={"output_dir": "/runs/abc", "model": "gpt-4"},
            ),
            WorkflowStage(
                name="collect",
                task="eval",
                executor="local",
                depends_on="inference",
                params={
                    "predictions_path": "${stages:inference,output_dir}/predictions.jsonl",
                    "model": "${stages:inference,model}",
                },
            ),
            WorkflowStage(
                name="evaluate",
                task="eval",
                executor="local",
                depends_on="collect",
                params={
                    "predictions_path": "${stages:collect,predictions_path}",
                },
            ),
        ],
        params={},
    )


def _run_with_from_job(runner, cfg, *, start_after, from_job_id):
    """Tolerant invocation: accept either the plan signature (`from_job=`)
    or the existing one (`skipped_params=`), so the test is robust to whichever
    Implementer ships.
    """
    import inspect
    sig = inspect.signature(runner.run)
    if "from_job" in sig.parameters:
        return runner.run(cfg, start_after=start_after, from_job=from_job_id)
    # Fall back to skipped_params shape — Implementer must also populate
    # source_job_id from the job id internally; tests assert that downstream.
    src_rec = runner._db.get(from_job_id)
    return runner.run(
        cfg,
        start_after=start_after,
        skipped_params={"inference": src_rec.params_dict},
    )


def test_start_after_with_from_job_sets_source_job_id(tmp_path: Path):
    """Skipped stage's workflow_jobs row points at the source job via source_job_id."""
    db_path = tmp_path / "jobs.db"
    db = JobStore(db_path)

    # Pre-seed an upstream COMPLETED eval job to act as `--from-job` source.
    agentic_id = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="output_dir: /runs/abc\nmodel: gpt-4\n",
        parameters={"output_dir": "/runs/abc", "model": "gpt-4"},
    )
    db.update_status(agentic_id, JobStatus.COMPLETED)
    db.close()

    runner = WorkflowRunner(db_path=db_path)
    with _hard_timeout(8):
        wf_id = _run_with_from_job(
            runner, _three_stage_config(),
            start_after="collect", from_job_id=agentic_id,
        )

    db = JobStore(db_path)
    stages = db.get_workflow_stages(wf_id)
    inf = next(s for s in stages if s.stage_name == "inference")
    assert inf.job_id is None, "skipped stage must not get a new job_id"
    assert inf.source_job_id == agentic_id, (
        f"skipped stage's source_job_id must point at the --from-job id; "
        f"got {inf.source_job_id!r}, expected {agentic_id!r}"
    )


def test_start_after_with_from_job_rewrites_stage_refs_to_jobs(tmp_path: Path):
    """Downstream stage's params_template rewrites ${stages:X,…} → ${jobs:<src_id>,…}."""
    db_path = tmp_path / "jobs.db"
    db = JobStore(db_path)

    agentic_id = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="output_dir: /runs/abc\nmodel: gpt-4\n",
        parameters={"output_dir": "/runs/abc", "model": "gpt-4"},
    )
    db.update_status(agentic_id, JobStatus.COMPLETED)
    db.close()

    runner = WorkflowRunner(db_path=db_path)
    with _hard_timeout(8):
        wf_id = _run_with_from_job(
            runner, _three_stage_config(),
            start_after="collect", from_job_id=agentic_id,
        )

    db = JobStore(db_path)
    stages = db.get_workflow_stages(wf_id)
    collect = next(s for s in stages if s.stage_name == "collect")
    assert collect.job_id is not None

    row = db._conn.execute(
        "SELECT params_template FROM jobs WHERE job_id = ?", (collect.job_id,),
    ).fetchone()
    assert row is not None
    template = row["params_template"] or ""
    # ${stages:inference,...} must have been replaced by ${jobs:<agentic_id>,...}.
    assert f"${{jobs:{agentic_id}," in template, (
        f"expected ${{jobs:{agentic_id},...}} substitution in params_template; "
        f"got: {template!r}"
    )
    # The original ${stages:inference,...} ref must be gone.
    assert "${stages:inference," not in template


def test_start_after_with_from_job_edges_point_at_source_job(tmp_path: Path):
    """Downstream dep edges point at the source_job_id, not a None placeholder."""
    db_path = tmp_path / "jobs.db"
    db = JobStore(db_path)

    agentic_id = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="output_dir: /runs/abc\nmodel: gpt-4\n",
        parameters={"output_dir": "/runs/abc", "model": "gpt-4"},
    )
    db.update_status(agentic_id, JobStatus.COMPLETED)
    db.close()

    runner = WorkflowRunner(db_path=db_path)
    with _hard_timeout(8):
        wf_id = _run_with_from_job(
            runner, _three_stage_config(),
            start_after="collect", from_job_id=agentic_id,
        )

    db = JobStore(db_path)
    stages = db.get_workflow_stages(wf_id)
    collect = next(s for s in stages if s.stage_name == "collect")
    deps = db.list_dependencies(collect.job_id)
    assert any(d.parent_job_id == agentic_id for d in deps), (
        f"collect's dep edge must point at the source job {agentic_id}; "
        f"got parents {[d.parent_job_id for d in deps]}"
    )


def test_start_after_alone_without_from_job_raises_error(tmp_path: Path):
    """`--start-after` without `--from-job` raises ValueError (approach A validation).

    Per PR3 decision: --start-after requires --from-job to populate source_job_id
    for skipped stages. This prevents CHECK constraint violations and ensures
    ${stages:...} references can be resolved during promotion.
    """
    import pytest
    db_path = tmp_path / "jobs.db"
    runner = WorkflowRunner(db_path=db_path)

    with pytest.raises(ValueError, match="--start-after requires --from-job"):
        runner.run(_three_stage_config(), start_after="collect")
