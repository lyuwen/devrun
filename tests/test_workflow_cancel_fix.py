"""Tests for workflow cancel with new producer model."""

import pytest
from devrun.db.jobs import JobStore, JobStatus, WorkflowStageRow
from devrun.workflow import WorkflowRunner


def test_workflow_cancel_with_queued_jobs(tmp_path):
    """Workflow cancel should cancel all queued stage jobs."""
    db = JobStore(tmp_path / "test.db")
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    # Enqueue a workflow with 2 stages
    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id="job1",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
        WorkflowStageRow(
            stage_name="stage2",
            ordinal=1,
            job_id="job2",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
    ]
    wf_id = db.enqueue_workflow(
        workflow_name="test_workflow",
        deadline_at=None,
        stage_rows=stage_rows,
        edges=[("job2", "job1", False)],
    )

    # Both jobs start as queued
    job1 = db.get("job1")
    job2 = db.get("job2")
    assert job1.status == JobStatus.QUEUED
    assert job2.status == JobStatus.QUEUED

    # Cancel the workflow
    runner.cancel(wf_id)

    # Both jobs should be cancelled
    job1 = db.get("job1")
    job2 = db.get("job2")
    assert job1.status == JobStatus.CANCELLED
    assert job2.status == JobStatus.CANCELLED

    # Workflow should be marked cancelled
    wf = db.get_workflow(wf_id)
    assert wf["status"] == "cancelled"
    assert wf["completed_at"] is not None


def test_workflow_cancel_with_running_jobs(tmp_path):
    """Workflow cancel should request cancel for running jobs."""
    db = JobStore(tmp_path / "test.db")
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id="job1",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
    ]
    wf_id = db.enqueue_workflow(
        workflow_name="test_workflow",
        deadline_at=None,
        stage_rows=stage_rows,
        edges=[],
    )

    # Mark job as running
    db.update_status("job1", JobStatus.RUNNING, remote_job_id="remote123")

    # Cancel the workflow
    runner.cancel(wf_id)

    # Job should be in CANCELING state (heartbeat will finish it)
    job1 = db.get("job1")
    assert job1.status == JobStatus.CANCELING

    # Workflow should still be marked cancelled
    wf = db.get_workflow(wf_id)
    assert wf["status"] == "cancelled"


def test_workflow_cancel_skips_completed_jobs(tmp_path):
    """Workflow cancel should skip already completed jobs."""
    db = JobStore(tmp_path / "test.db")
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id="job1",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
        WorkflowStageRow(
            stage_name="stage2",
            ordinal=1,
            job_id="job2",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
    ]
    wf_id = db.enqueue_workflow(
        workflow_name="test_workflow",
        deadline_at=None,
        stage_rows=stage_rows,
        edges=[],
    )

    # Mark one job as completed, one as queued
    db.update_status("job1", JobStatus.COMPLETED)

    # Cancel the workflow
    runner.cancel(wf_id)

    # Completed job should remain completed
    job1 = db.get("job1")
    assert job1.status == JobStatus.COMPLETED

    # Queued job should be cancelled
    job2 = db.get("job2")
    assert job2.status == JobStatus.CANCELLED


def test_workflow_cancel_with_skipped_stages(tmp_path):
    """Workflow cancel should handle skipped stages (job_id=None)."""
    db = JobStore(tmp_path / "test.db")
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id="job1",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
        WorkflowStageRow(
            stage_name="stage2",
            ordinal=1,
            job_id=None,  # Skipped stage
            source_job_id="source123",
            task_name=None,
            executor=None,
            params_template=None,
            parameters=None,
        ),
    ]
    wf_id = db.enqueue_workflow(
        workflow_name="test_workflow",
        deadline_at=None,
        stage_rows=stage_rows,
        edges=[],
    )

    # Cancel the workflow - should not crash on None job_id
    runner.cancel(wf_id)

    # The non-skipped job should be cancelled
    job1 = db.get("job1")
    assert job1.status == JobStatus.CANCELLED

    # Workflow should be cancelled
    wf = db.get_workflow(wf_id)
    assert wf["status"] == "cancelled"


def test_workflow_cancel_nonexistent_workflow(tmp_path):
    """Workflow cancel should raise ValueError for nonexistent workflow."""
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    with pytest.raises(ValueError, match="Workflow .* not found"):
        runner.cancel("nonexistent_id")
