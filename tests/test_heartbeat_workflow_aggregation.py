"""Tests for workflow status aggregation in heartbeat tick."""

import pytest
from devrun.db.jobs import JobStore, JobStatus, WorkflowStageRow
from devrun.heartbeat import tick


def test_workflow_aggregation_all_completed(tmp_path):
    """Workflow transitions to completed when all stage jobs are completed."""
    db = JobStore(tmp_path / "test.db")

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
        edges=[],
    )

    # Mark both jobs as completed
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.COMPLETED)

    # Run tick with no executor (aggregation only)
    tick(db, executor_router=None)

    # Verify workflow is now completed
    wf = db.get_workflow(wf_id)
    assert wf is not None
    assert wf["status"] == JobStatus.COMPLETED.value
    assert wf["completed_at"] is not None


def test_workflow_aggregation_with_skipped_jobs(tmp_path):
    """Workflow transitions to completed when all jobs are completed or skipped."""
    db = JobStore(tmp_path / "test.db")

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

    # Mark one completed, one skipped
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.SKIPPED)

    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.COMPLETED.value


def test_workflow_aggregation_any_failed(tmp_path):
    """Workflow transitions to failed when any stage job fails."""
    db = JobStore(tmp_path / "test.db")

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

    # Mark one completed, one failed
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.FAILED)

    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.FAILED.value
    assert wf["completed_at"] is not None


def test_workflow_aggregation_any_cancelled(tmp_path):
    """Workflow transitions to failed when any stage job is cancelled."""
    db = JobStore(tmp_path / "test.db")

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

    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.CANCELLED)

    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.FAILED.value


def test_workflow_aggregation_remains_queued_when_running(tmp_path):
    """Workflow stays queued when jobs are still running."""
    db = JobStore(tmp_path / "test.db")

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

    # One completed, one still running
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.RUNNING)

    tick(db, executor_router=None)

    # Workflow should remain queued
    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.QUEUED.value
    assert wf["completed_at"] is None


def test_workflow_aggregation_skips_terminal_workflows(tmp_path):
    """Already terminal workflows are not re-aggregated."""
    db = JobStore(tmp_path / "test.db")

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

    # Mark job and workflow as completed
    db.update_status("job1", JobStatus.COMPLETED)
    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.COMPLETED.value
    first_completed_at = wf["completed_at"]

    # Run tick again - should not update completed_at
    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["completed_at"] == first_completed_at


def test_workflow_aggregation_multiple_workflows(tmp_path):
    """Aggregation handles multiple workflows in one tick."""
    db = JobStore(tmp_path / "test.db")

    # Create two workflows
    stage_rows_1 = [
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
    wf_id_1 = db.enqueue_workflow(
        workflow_name="workflow1",
        deadline_at=None,
        stage_rows=stage_rows_1,
        edges=[],
    )

    stage_rows_2 = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id="job2",
            source_job_id=None,
            task_name="eval",
            executor="local",
            params_template="{}",
            parameters={},
        ),
    ]
    wf_id_2 = db.enqueue_workflow(
        workflow_name="workflow2",
        deadline_at=None,
        stage_rows=stage_rows_2,
        edges=[],
    )

    # Complete both jobs
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.FAILED)

    tick(db, executor_router=None)

    # Both workflows should be aggregated
    wf1 = db.get_workflow(wf_id_1)
    wf2 = db.get_workflow(wf_id_2)
    assert wf1["status"] == JobStatus.COMPLETED.value
    assert wf2["status"] == JobStatus.FAILED.value


def test_workflow_aggregation_with_no_jobs(tmp_path):
    """Workflow with no jobs (all stages skipped) stays queued."""
    db = JobStore(tmp_path / "test.db")

    # Workflow with only skipped stages (job_id=None)
    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id=None,
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

    tick(db, executor_router=None)

    # Workflow should remain queued (no jobs to aggregate)
    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.QUEUED.value
    assert wf["completed_at"] is None


def test_workflow_aggregation_timed_out_job(tmp_path):
    """Workflow transitions to failed when a job is timed out."""
    db = JobStore(tmp_path / "test.db")

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

    db.update_status("job1", JobStatus.TIMED_OUT)

    tick(db, executor_router=None)

    wf = db.get_workflow(wf_id)
    assert wf["status"] == JobStatus.FAILED.value
