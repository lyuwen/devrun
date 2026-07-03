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


# ============================================================================
# Review-gate fixes (BLOCK 1 + BLOCK 2)
# ============================================================================


def test_poll_pending_stays_in_active_set(tmp_path: Path):
    """Executor reporting 'pending' must map to SUBMITTED so the row keeps polling.

    Slurm reports 'pending' for queued-on-cluster jobs that haven't started yet.
    If heartbeat wrote PENDING, ``fetch_active_jobs()`` would drop the row and
    it would strand forever. The fix maps 'pending' → SUBMITTED so the next
    tick re-polls until the cluster scheduler moves it.
    """
    from devrun.models import JobStatus

    transitions: list[str] = []

    class _SequencingExecutor:
        """status() returns 'pending', 'pending', 'running', 'completed'."""

        def __init__(self):
            self._seq = iter(["pending", "pending", "running", "completed"])

        def status(self, remote_job_id: str) -> str:
            return next(self._seq)

    class _Router:
        def __init__(self, ex): self._ex = ex
        def get(self, name): return self._ex

    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    db.update_status(jid, JobStatus.SUBMITTED, remote_job_id="r-pending")
    router = _Router(_SequencingExecutor())

    for _ in range(4):
        tick(db, executor_router=router)
        transitions.append(JobStatus(db.get(jid).status).value)

    # Each tick must observe the row in the active set; pending must NOT have
    # stranded it.
    assert transitions == ["submitted", "submitted", "running", "completed"]


def test_run_loop_promotes_queued_row_one_tick(tmp_path: Path, monkeypatch):
    """`run_loop` (not just `tick`) promotes a QUEUED row using a real ExecutorRouter."""
    import threading

    from devrun.heartbeat import _shutdown_event, run_loop
    from devrun.models import JobStatus
    from devrun.registry import _EXECUTOR_REGISTRY  # type: ignore[attr-defined]
    from devrun.executors.base import BaseExecutor
    from devrun.router import ExecutorEntry
    import devrun.router as router_module

    class _SmokeExecutor(BaseExecutor):
        def submit(self, task_spec):
            return "remote-loop-1"

        def status(self, remote_id, **kwargs):
            return "running"

        def logs(self, remote_id, **kwargs):
            return ""

        def cancel(self, remote_id):
            pass

    # Register the smoke executor under a unique name so we don't pollute the
    # global registry for other tests.
    import uuid
    type_name = f"smoke_{uuid.uuid4().hex[:6]}"
    _EXECUTOR_REGISTRY[type_name] = _SmokeExecutor

    def _fake_loader(_path):
        return {"local": ExecutorEntry(type=type_name)}

    monkeypatch.setattr(router_module, "load_executor_configs", _fake_loader)

    # Pre-populate the DB.
    db = JobStore(tmp_path / "jobs.db")
    jid = db.enqueue(
        task_name="eval",
        executor="local",
        params_template="benchmark_name: gsm8k\nmodel_name: gpt-4\nsample_size: 1\n",
        parameters={},
    )
    db.close()

    # Need the eval task registered.
    import devrun.tasks  # noqa: F401

    # Reset shutdown flag so we have a clean run, then schedule a shutdown
    # after the first tick lands.
    _shutdown_event.clear()
    timer = threading.Timer(0.5, _shutdown_event.set)
    timer.daemon = True
    timer.start()
    try:
        run_loop(tmp_path / "jobs.db", interval=0.05, tick_file=tmp_path / "tick")
    finally:
        timer.cancel()
        _shutdown_event.clear()  # leave the module state clean for other tests

    db2 = JobStore(tmp_path / "jobs.db")
    rec = db2.get(jid)
    assert rec is not None
    assert JobStatus(rec.status) in (JobStatus.SUBMITTED, JobStatus.RUNNING), (
        f"expected SUBMITTED or RUNNING after run_loop tick, got {rec.status}"
    )
    assert rec.remote_job_id == "remote-loop-1"
    db2.close()
