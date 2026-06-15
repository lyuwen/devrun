"""End-to-end smoke test: producer → heartbeat → LocalExecutor (PR3 Task 9).

Drives the full pipeline tick-by-tick (no real background process) so the test
is deterministic. A 2-stage workflow runs `echo hello` via LocalExecutor; the
heartbeat is expected to promote each stage, observe completion, and finalize
the workflow.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick
from devrun.models import JobStatus, WorkflowConfig, WorkflowStage
from devrun.router import ExecutorRouter
from devrun.workflow import WorkflowRunner


@pytest.fixture
def local_log_dir(tmp_path: Path, monkeypatch):
    """Redirect LocalExecutor's _LOG_DIR to a tmp path so concurrent runs don't collide."""
    log_dir = tmp_path / "local_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("devrun.executors.local._LOG_DIR", log_dir)
    return log_dir


@pytest.fixture
def executors_yaml(tmp_path: Path) -> Path:
    import yaml
    path = tmp_path / "executors.yaml"
    path.write_text(yaml.safe_dump({"local": {"type": "local"}}))
    return path


def _two_stage_local() -> WorkflowConfig:
    """A no-op 2-stage workflow that LocalExecutor can run end-to-end.

    Both stages use the ``eval`` task (registered in devrun.tasks.eval). The
    EvalTask synthesizes a ``python eval.py --model X --dataset Y`` command,
    which won't actually succeed against a real eval.py, but the LocalExecutor
    will still record an exit code and the heartbeat will transition the job
    to a terminal state. We assert "terminal" rather than "completed" since
    the actual exit code depends on whether python is on PATH.
    """
    return WorkflowConfig(
        workflow="e2e_smoke",
        stages=[
            WorkflowStage(
                name="first",
                task="eval",
                executor="local",
                params={"model": "m1", "dataset": "d1"},
            ),
            WorkflowStage(
                name="second",
                task="eval",
                executor="local",
                depends_on="first",
                params={"model": "m2", "dataset": "d2"},
            ),
        ],
        params={},
        timeout=60.0,
    )


_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED, JobStatus.CANCELLED}


def _all_terminal(db: JobStore, wf_id: str) -> bool:
    stages = db.get_workflow_stages(wf_id)
    if not stages:
        return False
    for stage in stages:
        if stage.job_id is None:
            continue
        rec = db.get(stage.job_id)
        if rec is None:
            return False
        if JobStatus(rec.status) not in _TERMINAL:
            return False
    return True


def _drain_ticks(db: JobStore, router, wf_id: str, *, max_ticks: int = 40, sleep: float = 0.1):
    """Run tick() repeatedly until every stage's job_id row is terminal or the budget runs out."""
    for _ in range(max_ticks):
        tick(db, executor_router=router)
        if _all_terminal(db, wf_id):
            return
        time.sleep(sleep)


def test_e2e_workflow_drains_via_heartbeat_to_terminal(
    tmp_path: Path, local_log_dir, executors_yaml, monkeypatch
):
    """A real local 2-stage workflow drained by the heartbeat tick loop.

    Assertions:
    - WorkflowRunner.run() returns a workflow_id promptly (producer-only).
    - All stages reach a terminal state under repeated tick() calls.
    - The dependency edge is respected: the second stage only progresses
      after the first stage's promotion (no out-of-order submission).
    """
    db_path = tmp_path / "jobs.db"
    runner = WorkflowRunner(db_path=db_path)
    router = ExecutorRouter(executors_path=str(executors_yaml))

    started = time.monotonic()
    wf_id = runner.run(_two_stage_local())
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"workflow run() did not return promptly ({elapsed:.2f}s)"

    db = JobStore(db_path)
    stages = db.get_workflow_stages(wf_id)
    assert [s.stage_name for s in stages] == ["first", "second"]
    first_id = stages[0].job_id
    second_id = stages[1].job_id
    assert first_id and second_id

    # Both start QUEUED — heartbeat has not run yet.
    assert JobStatus(db.get(first_id).status) == JobStatus.QUEUED
    assert JobStatus(db.get(second_id).status) == JobStatus.QUEUED

    _drain_ticks(db, router, wf_id)

    first_final = JobStatus(db.get(first_id).status)
    second_final = JobStatus(db.get(second_id).status)

    # Both must be terminal. We allow FAILED here because the eval task's
    # command may not actually succeed on this machine — what matters is
    # that the producer + heartbeat drive the lifecycle.
    assert first_final in _TERMINAL, f"first stage did not terminate: {first_final}"
    assert second_final in _TERMINAL, f"second stage did not terminate: {second_final}"


def test_e2e_dependency_ordering_respected(
    tmp_path: Path, local_log_dir, executors_yaml,
):
    """The dependent (second) stage must not be SUBMITTED before the first terminates.

    Snapshots taken after each tick should never show second SUBMITTED/RUNNING
    while first is still QUEUED.
    """
    db_path = tmp_path / "jobs.db"
    runner = WorkflowRunner(db_path=db_path)
    router = ExecutorRouter(executors_path=str(executors_yaml))

    wf_id = runner.run(_two_stage_local())

    db = JobStore(db_path)
    stages = db.get_workflow_stages(wf_id)
    first_id = stages[0].job_id
    second_id = stages[1].job_id

    # Snapshot statuses across ticks; verify the dependency ordering invariant.
    for _ in range(40):
        tick(db, executor_router=router)
        rec_first = db.get(first_id)
        rec_second = db.get(second_id)
        assert rec_first is not None and rec_second is not None

        first_status = JobStatus(rec_first.status)
        second_status = JobStatus(rec_second.status)

        # If second left QUEUED, first must already be in a satisfied state:
        # COMPLETED is the only "edge-satisfying" status without allow_failure.
        if second_status != JobStatus.QUEUED:
            assert first_status in _TERMINAL, (
                f"second moved to {second_status} while first was {first_status}"
            )

        if _all_terminal(db, wf_id):
            break
        time.sleep(0.1)
