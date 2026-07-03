"""Tests for workflow status CLI display with producer model."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from devrun.cli import app
from devrun.db.jobs import JobStore, JobStatus, WorkflowStageRow
from devrun.workflow import WorkflowRunner


def test_workflow_status_displays_stages_from_workflow_jobs(tmp_path):
    """workflow status should display stage information from workflow_jobs table."""
    runner = CliRunner()
    db_path = tmp_path / "test.db"

    # Create a workflow with stages
    db = JobStore(db_path)
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

    # Mark jobs with different statuses
    db.update_status("job1", JobStatus.COMPLETED)
    db.update_status("job2", JobStatus.RUNNING)

    # Mock WorkflowRunner to use our test database
    with patch("devrun.workflow.WorkflowRunner", wraps=WorkflowRunner) as mock_runner_class:
        mock_runner_class.return_value = WorkflowRunner(db_path=db_path)
        result = runner.invoke(app, ["workflow", "status", wf_id])

    # Check output contains stage information
    assert result.exit_code == 0
    assert "test_workflow" in result.stdout
    assert "Stage: stage1" in result.stdout
    assert "Stage: stage2" in result.stdout
    assert "job1" in result.stdout
    assert "job2" in result.stdout
    assert "completed" in result.stdout.lower()
    assert "running" in result.stdout.lower()


def test_workflow_status_displays_skipped_stages(tmp_path):
    """workflow status should display skipped stages with source_job_id."""
    runner = CliRunner()
    db_path = tmp_path / "test.db"

    db = JobStore(db_path)
    stage_rows = [
        WorkflowStageRow(
            stage_name="stage1",
            ordinal=0,
            job_id=None,  # Skipped stage
            source_job_id="source123",
            task_name=None,
            executor=None,
            params_template=None,
            parameters=None,
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

    db.update_status("job2", JobStatus.QUEUED)

    with patch("devrun.workflow.WorkflowRunner", wraps=WorkflowRunner) as mock_runner_class:
        mock_runner_class.return_value = WorkflowRunner(db_path=db_path)
        result = runner.invoke(app, ["workflow", "status", wf_id])

    assert result.exit_code == 0
    assert "Stage: stage1" in result.stdout
    assert "skipped" in result.stdout.lower()
    assert "source123" in result.stdout
    assert "Stage: stage2" in result.stdout
    assert "job2" in result.stdout


def test_workflow_status_empty_workflow(tmp_path):
    """workflow status should handle workflows with no stages."""
    runner = CliRunner()
    db_path = tmp_path / "test.db"

    db = JobStore(db_path)
    wf_id = db.enqueue_workflow(
        workflow_name="empty_workflow",
        deadline_at=None,
        stage_rows=[],
        edges=[],
    )

    with patch("devrun.workflow.WorkflowRunner", wraps=WorkflowRunner) as mock_runner_class:
        mock_runner_class.return_value = WorkflowRunner(db_path=db_path)
        result = runner.invoke(app, ["workflow", "status", wf_id])

    assert result.exit_code == 0
    assert "empty_workflow" in result.stdout
    # Should still show workflow-level info even with no stages
    assert "Status" in result.stdout or "status" in result.stdout.lower()


def test_workflow_status_not_found(tmp_path):
    """workflow status should handle nonexistent workflow ID."""
    runner = CliRunner()
    db_path = tmp_path / "test.db"

    with patch("devrun.workflow.WorkflowRunner", wraps=WorkflowRunner) as mock_runner_class:
        mock_runner_class.return_value = WorkflowRunner(db_path=db_path)
        result = runner.invoke(app, ["workflow", "status", "nonexistent"])

    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()
