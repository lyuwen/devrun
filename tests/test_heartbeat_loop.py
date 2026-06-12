"""Heartbeat module loop tests (PR2 Task 1)."""

from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick


def test_tick_empty_db_is_noop(tmp_path: Path):
    """A tick against a fresh empty DB does nothing and does not raise."""
    db = JobStore(tmp_path / "jobs.db")
    tick(db, executor_router=None)


def test_tick_returns_none(tmp_path: Path):
    """tick() returns None (side-effect only API)."""
    db = JobStore(tmp_path / "jobs.db")
    assert tick(db, executor_router=None) is None


# ============================================================================
# Task 5 — Poll-active phase + status mapping
# ============================================================================


class _PollExecutor:
    """Duck-typed executor with a status() that returns canned raw strings."""

    def __init__(self, status_map: dict[str, str]):
        self._status_map = status_map
        self.calls: list[str] = []

    def status(self, remote_job_id: str) -> str:
        self.calls.append(remote_job_id)
        return self._status_map.get(remote_job_id, "running")


class _PollRouter:
    def __init__(self, executor: _PollExecutor):
        self._executor = executor

    def get(self, name: str) -> _PollExecutor:
        return self._executor


def test_poll_active_transitions_running_to_completed(tmp_path: Path):
    """A RUNNING row whose executor reports 'completed' is updated to COMPLETED."""
    from devrun.models import JobStatus

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-1")

    router = _PollRouter(_PollExecutor({"r-1": "completed"}))
    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.COMPLETED


def test_poll_active_transitions_submitted_to_running(tmp_path: Path):
    """A SUBMITTED row whose executor reports 'running' is bumped to RUNNING."""
    from devrun.models import JobStatus

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.SUBMITTED, remote_job_id="r-2")

    router = _PollRouter(_PollExecutor({"r-2": "running"}))
    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.RUNNING


def test_poll_active_transitions_running_to_failed(tmp_path: Path):
    """A RUNNING row whose executor reports 'failed' is updated to FAILED."""
    from devrun.models import JobStatus

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-3")

    router = _PollRouter(_PollExecutor({"r-3": "failed"}))
    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.FAILED


def test_poll_active_skips_queued_and_terminal_jobs(tmp_path: Path):
    """Only SUBMITTED/RUNNING/CANCELING rows are polled — QUEUED and COMPLETED are skipped."""
    from devrun.models import JobStatus

    db = JobStore(tmp_path / "jobs.db")
    # Parent stays QUEUED forever (never satisfies its own deps) so the child
    # is never "ready" — keeps promotion phase out of the picture.
    parent = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(parent, JobStatus.RUNNING, remote_job_id="r-parent")
    queued = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.insert_dependency(child_job_id=queued, parent_job_id=parent, allow_failure=False)
    done = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(done, JobStatus.COMPLETED, remote_job_id="r-done")

    executor = _PollExecutor({"r-parent": "running"})
    tick(db, executor_router=_PollRouter(executor))

    # Only the active parent was polled; queued + completed were skipped.
    assert "r-done" not in executor.calls
    assert JobStatus(db.get(queued).status) == JobStatus.QUEUED
    assert JobStatus(db.get(done).status) == JobStatus.COMPLETED


def test_poll_active_no_op_when_status_unchanged(tmp_path: Path):
    """If executor returns the same status, no DB write is needed (idempotent)."""
    from devrun.models import JobStatus

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-x")

    router = _PollRouter(_PollExecutor({"r-x": "running"}))
    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.RUNNING


def test_poll_active_executor_status_exception_does_not_crash(tmp_path: Path):
    """A failing executor.status() is logged but does not halt the tick."""
    from devrun.models import JobStatus

    class _BoomExecutor:
        def status(self, remote_job_id: str) -> str:
            raise RuntimeError("network down")

    class _BoomRouter:
        def get(self, name: str):
            return _BoomExecutor()

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-broken")

    # Must not raise.
    tick(db, executor_router=_BoomRouter())

    # Row left untouched.
    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.RUNNING
