"""Heartbeat cancel-phase tests (PR2 Task 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick
from devrun.models import JobStatus


class _CancelExecutor:
    """Duck-typed executor that records cancel() calls and tolerates status()."""

    def __init__(self):
        self.cancel_calls: list[str] = []
        self.status_calls: list[str] = []

    def cancel(self, remote_job_id: str) -> None:
        self.cancel_calls.append(remote_job_id)

    def status(self, remote_job_id: str) -> str:
        # Not called when cancel branch fires, but provide a safe default.
        self.status_calls.append(remote_job_id)
        return "running"


class _Router:
    def __init__(self, executor: Any):
        self._executor = executor

    def get(self, name: str) -> Any:
        return self._executor


def test_canceling_transitions_to_cancelled(tmp_path: Path):
    """A CANCELING row → executor.cancel() called → CANCELLED."""
    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-1")
    db.request_cancel(jid)
    assert JobStatus(db.get(jid).status) == JobStatus.CANCELING

    executor = _CancelExecutor()
    tick(db, executor_router=_Router(executor))

    assert executor.cancel_calls == ["r-1"]
    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.CANCELLED


def test_canceling_executor_cancel_failure_retries(tmp_path: Path):
    """executor.cancel() raising must NOT prematurely finalize CANCELLED.

    The row stays CANCELING so the next tick can retry the executor call.
    Without this, a transient network blip would silently mark the remote job
    as cancelled while it's still running.
    """
    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-2")
    db.request_cancel(jid)

    class _BoomExecutor:
        def cancel(self, remote_job_id: str) -> None:
            raise RuntimeError("network down")

        def status(self, remote_job_id: str) -> str:
            return "running"

    tick(db, executor_router=_Router(_BoomExecutor()))

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.CANCELING


def test_canceling_does_not_call_status(tmp_path: Path):
    """A CANCELING row goes through cancel(), not status() — these are exclusive branches."""
    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-3")
    db.request_cancel(jid)

    executor = _CancelExecutor()
    tick(db, executor_router=_Router(executor))

    # Only the cancel branch fires; status() should not have been called for r-3.
    assert "r-3" not in executor.status_calls


def test_running_job_is_not_cancelled(tmp_path: Path):
    """A plain RUNNING job (no cancel request) is not transitioned by the cancel branch."""
    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.RUNNING, remote_job_id="r-4")

    executor = _CancelExecutor()
    tick(db, executor_router=_Router(executor))

    assert executor.cancel_calls == []
    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.RUNNING
