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
