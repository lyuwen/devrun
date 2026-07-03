"""Heartbeat cascade-skip phase integration tests (PR2 Task 2)."""

from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick
from devrun.models import JobStatus


def _enqueue(db: JobStore, **overrides):
    kwargs = dict(task_name="t", executor="local", params_template="", parameters={})
    kwargs.update(overrides)
    return db.enqueue(**kwargs)


def test_cascade_single_hop(tmp_path: Path):
    """Parent FAILED + blocking edge → child SKIPPED after one tick."""
    db = JobStore(tmp_path / "jobs.db")
    parent = _enqueue(db)
    child = _enqueue(db)
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)
    db.update_status(parent, JobStatus.FAILED)

    tick(db, executor_router=None)

    rec = db.get(child)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SKIPPED


def test_cascade_respects_allow_failure(tmp_path: Path):
    """allow_failure=1 child is not skipped after one tick."""
    db = JobStore(tmp_path / "jobs.db")
    parent = _enqueue(db)
    child = _enqueue(db)
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=True)
    db.update_status(parent, JobStatus.FAILED)

    tick(db, executor_router=None)

    rec = db.get(child)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.QUEUED


def test_cascade_multihop(tmp_path: Path):
    """A FAILED → B SKIPPED → C SKIPPED after two ticks."""
    db = JobStore(tmp_path / "jobs.db")
    a = _enqueue(db)
    b = _enqueue(db)
    c = _enqueue(db)
    db.insert_dependency(child_job_id=b, parent_job_id=a, allow_failure=False)
    db.insert_dependency(child_job_id=c, parent_job_id=b, allow_failure=False)
    db.update_status(a, JobStatus.FAILED)

    for _ in range(2):
        tick(db, executor_router=None)

    rec_b = db.get(b)
    rec_c = db.get(c)
    assert rec_b is not None and JobStatus(rec_b.status) == JobStatus.SKIPPED
    assert rec_c is not None and JobStatus(rec_c.status) == JobStatus.SKIPPED


def test_cascade_idempotent(tmp_path: Path):
    """Repeated ticks after the cascade completes don't disturb terminal rows."""
    db = JobStore(tmp_path / "jobs.db")
    parent = _enqueue(db)
    child = _enqueue(db)
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)
    db.update_status(parent, JobStatus.FAILED)

    for _ in range(3):
        tick(db, executor_router=None)

    rec = db.get(child)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SKIPPED
