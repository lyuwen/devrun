"""Heartbeat lease-reclaim / orphan-protection tests (PR2 Task 7)."""

import time
from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick
from devrun.models import JobStatus


def _enqueue(db: JobStore):
    return db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )


def test_expired_lease_reclaim_to_queued(tmp_path: Path):
    """SUBMITTING row with expired lease + NULL remote_job_id → QUEUED, annotated."""
    db = JobStore(tmp_path / "jobs.db")
    jid = _enqueue(db)
    assert db.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=0.001) is True

    # Wait long enough that the lease expires.
    time.sleep(0.05)

    tick(db, executor_router=None)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.QUEUED
    assert "reclaimed" in (rec.skip_reason or "").lower()


def test_non_null_remote_job_id_is_not_reclaimed(tmp_path: Path):
    """SUBMITTING + expired lease but remote_job_id set → NOT reclaimed (orphan-safe)."""
    db = JobStore(tmp_path / "jobs.db")
    jid = _enqueue(db)
    assert db.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=0.001) is True

    # Simulate a row that crashed after executor.submit returned but before
    # finalize_submit ran — remote_job_id is set, but the row is still SUBMITTING.
    db._conn.execute(
        "UPDATE jobs SET remote_job_id = ? WHERE job_id = ?",
        ("remote-orphan", jid),
    )
    db._conn.commit()

    time.sleep(0.05)
    tick(db, executor_router=None)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SUBMITTING
    assert rec.remote_job_id == "remote-orphan"


def test_live_lease_is_not_reclaimed(tmp_path: Path):
    """A SUBMITTING row whose lease is still in the future is untouched."""
    db = JobStore(tmp_path / "jobs.db")
    jid = _enqueue(db)
    assert db.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=600) is True

    tick(db, executor_router=None)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SUBMITTING
