"""Heartbeat workflow-deadline expiry tests (PR2 Task 8)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from devrun.db.jobs import JobStore, WorkflowStageRow
from devrun.heartbeat import tick
from devrun.models import JobStatus


def _stage(**overrides):
    base = dict(
        stage_name="stage",
        ordinal=0,
        job_id="j-x",
        source_job_id=None,
        task_name="t",
        executor="local",
        params_template="x: 1",
        parameters={"x": 1},
    )
    base.update(overrides)
    return WorkflowStageRow(**base)


def test_tick_expires_workflow_with_past_deadline(tmp_path: Path):
    """Past deadline → QUEUED→SKIPPED, RUNNING→CANCELING, workflow→TIMED_OUT."""
    db = JobStore(tmp_path / "jobs.db")
    past = datetime.now(timezone.utc) - timedelta(minutes=5)

    wf_id = db.enqueue_workflow(
        workflow_name="dline",
        deadline_at=past,
        stage_rows=[
            _stage(stage_name="a", ordinal=0, job_id="j-a"),
            _stage(stage_name="b", ordinal=1, job_id="j-b"),
        ],
        edges=[],
    )
    db.update_status("j-b", JobStatus.RUNNING)

    tick(db, executor_router=None)

    rec_a = db.get("j-a")
    rec_b = db.get("j-b")
    assert rec_a is not None and JobStatus(rec_a.status) == JobStatus.SKIPPED
    assert rec_b is not None and JobStatus(rec_b.status) == JobStatus.CANCELING

    wf_row = db._conn.execute(
        "SELECT status FROM workflows WHERE workflow_id = ?", (wf_id,)
    ).fetchone()
    assert wf_row["status"] == JobStatus.TIMED_OUT.value


def test_tick_does_not_expire_future_deadline(tmp_path: Path):
    """A workflow whose deadline is in the future is left alone."""
    db = JobStore(tmp_path / "jobs.db")
    future = datetime.now(timezone.utc) + timedelta(hours=1)

    wf_id = db.enqueue_workflow(
        workflow_name="future",
        deadline_at=future,
        stage_rows=[_stage(job_id="j-f1")],
        edges=[],
    )

    tick(db, executor_router=None)

    rec = db.get("j-f1")
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.QUEUED

    wf_row = db._conn.execute(
        "SELECT status FROM workflows WHERE workflow_id = ?", (wf_id,)
    ).fetchone()
    assert wf_row["status"] != JobStatus.TIMED_OUT.value


def test_tick_does_not_expire_null_deadline(tmp_path: Path):
    """A workflow with no deadline_at is never expired."""
    db = JobStore(tmp_path / "jobs.db")
    wf_id = db.enqueue_workflow(
        workflow_name="no-deadline",
        deadline_at=None,
        stage_rows=[_stage(job_id="j-n1")],
        edges=[],
    )

    tick(db, executor_router=None)

    rec = db.get("j-n1")
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.QUEUED

    wf_row = db._conn.execute(
        "SELECT status FROM workflows WHERE workflow_id = ?", (wf_id,)
    ).fetchone()
    assert wf_row["status"] != JobStatus.TIMED_OUT.value
