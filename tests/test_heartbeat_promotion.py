"""Heartbeat promotion-phase tests (PR2 Task 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick
from devrun.models import JobStatus


class _FakeExecutor:
    """Minimal duck-typed executor for tick() tests.

    submit_with_retry() returns a string remote_job_id (matching the
    current BaseExecutor contract used by TaskRunner).
    """

    def __init__(self, *, remote_job_id: str = "remote-x", log_path: str | None = None):
        self.name = "local"
        self._next_remote_id = remote_job_id
        self._next_log_path = log_path
        self.submitted: list[Any] = []

    def submit_with_retry(self, spec, retries: int = 3, retry_delay: float = 5.0):
        self.submitted.append(spec)
        # Mirror SlurmExecutor convention: log_path is written into
        # spec.metadata so the runner can later pull it for finalize_submit.
        if self._next_log_path is not None:
            spec.metadata["log_path"] = self._next_log_path
        return self._next_remote_id


class _FakeRouter:
    """Router with the single `.get(name)` method heartbeat uses."""

    def __init__(self, executor: _FakeExecutor):
        self._executor = executor

    def get(self, name: str) -> _FakeExecutor:
        return self._executor


def test_promotion_happy_path(tmp_path: Path):
    """QUEUED + ready → SUBMITTED with remote_job_id and log_path persisted."""
    db = JobStore(tmp_path / "jobs.db")
    executor = _FakeExecutor(remote_job_id="remote-123", log_path="/tmp/log")
    router = _FakeRouter(executor)

    jid = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\nseed: 1\n",
        parameters={"model": "gpt-4", "seed": 1},
    )

    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SUBMITTED
    assert rec.remote_job_id == "remote-123"
    assert rec.log_path == "/tmp/log"
    # Executor saw exactly one submission
    assert len(executor.submitted) == 1


def test_promotion_persists_resolved_parameters(tmp_path: Path):
    """finalize_submit must write the resolved params (not the unresolved template)."""
    db = JobStore(tmp_path / "jobs.db")
    executor = _FakeExecutor(remote_job_id="r-1")
    router = _FakeRouter(executor)

    jid = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\nbatch_size: 8\n",
        parameters={"model": "gpt-4", "batch_size": 8},
    )

    tick(db, executor_router=router)

    rec = db.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) == JobStatus.SUBMITTED
    assert rec.params_dict.get("model") == "gpt-4"
    assert rec.params_dict.get("batch_size") == 8


def test_promotion_skips_when_dependencies_unsatisfied(tmp_path: Path):
    """A child whose parent is still RUNNING must not be promoted."""
    db = JobStore(tmp_path / "jobs.db")
    executor = _FakeExecutor()
    router = _FakeRouter(executor)

    parent = db.enqueue(
        task_name="eval", executor="local",
        params_template="", parameters={},
    )
    child = db.enqueue(
        task_name="eval", executor="local",
        params_template="model: gpt-4\n", parameters={"model": "gpt-4"},
    )
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)
    db.update_status(parent, JobStatus.RUNNING, remote_job_id="r-parent")

    # Stub poll-phase router lookup: parent is RUNNING and tick() will poll it;
    # we don't care about the parent's status update here, only that the child
    # is NOT promoted. Give the executor a status() method that returns the
    # raw string the poll phase will tolerate (defaults to "running").
    executor.status = MagicMock(return_value="running")  # type: ignore[attr-defined]

    tick(db, executor_router=router)

    rec_child = db.get(child)
    assert rec_child is not None
    assert JobStatus(rec_child.status) == JobStatus.QUEUED
