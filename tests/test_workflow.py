"""Unit tests for WorkflowRunner."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from devrun.models import WorkflowConfig, WorkflowStage
from devrun.workflow import WorkflowRunner


@pytest.fixture
def simple_config():
    return WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(name="step1", task="eval", executor="local", params={"model": "x"}),
            WorkflowStage(name="step2", task="eval", executor="local", depends_on="step1"),
        ],
        heartbeat_interval=0.001,
    )


@pytest.fixture
def workflow_runner(tmp_path):
    return WorkflowRunner(db_path=tmp_path / "test.db")


class TestWorkflowRunner:
    def test_dry_run_does_not_submit(self, workflow_runner, simple_config):
        """Dry-run should not create any DB records or submit jobs."""
        result = workflow_runner.run(simple_config, dry_run=True)
        assert isinstance(result, str)
        assert "Stage 1: step1" in result
        assert "Stage 2: step2" in result

    def test_dependency_ordering(self, workflow_runner, simple_config):
        """step2 depends on step1, so step1 must run first."""
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                workflow_runner.run(simple_config)
                calls = [c.args[0] for c in mock_submit.call_args_list]
                assert calls[0] == "step1"
                assert calls[1] == "step2"

    def test_failure_stops_workflow(self, workflow_runner, simple_config):
        """If step1 fails, step2 should never be submitted."""
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="failed"):
                wf_id = workflow_runner.run(simple_config)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "failed"
                assert mock_submit.call_count == 1

    def test_all_stages_complete(self, workflow_runner, simple_config):
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                wf_id = workflow_runner.run(simple_config)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "completed"

    def test_timeout_cancels_workflow(self, workflow_runner):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="slow", task="eval", executor="local")],
            timeout=0.001,
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="running"):
                wf_id = workflow_runner.run(cfg)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "timed_out"

    def test_submit_failure_marks_stage_failed(self, workflow_runner):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="bad", task="eval", executor="local")],
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage", side_effect=RuntimeError("boom")):
            wf_id = workflow_runner.run(cfg)
            record = workflow_runner._db.get_workflow(wf_id)
            assert record["status"] == "failed"
            stages = json.loads(record["stages_state"])
            assert stages["bad"]["status"] == "failed"

    def test_skipped_stage_when_dep_fails(self, workflow_runner, simple_config):
        """step2 should be skipped when step1 fails."""
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="failed"):
                wf_id = workflow_runner.run(simple_config)
                record = workflow_runner._db.get_workflow(wf_id)
                stages = json.loads(record["stages_state"])
                assert stages["step1"]["status"] == "failed"
                # step2 was never submitted
                assert mock_submit.call_count == 1

    def test_cancel_workflow(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "running", "job_id": "j1"},
            "s2": {"status": "pending", "job_id": None},
        })
        workflow_runner._db.update_workflow(wf_id, status="running")
        workflow_runner.cancel(wf_id)
        record = workflow_runner._db.get_workflow(wf_id)
        assert record["status"] == "cancelled"
        stages = json.loads(record["stages_state"])
        assert stages["s1"]["status"] == "cancelled"
        assert stages["s2"]["status"] == "pending"  # was not running

    def test_cancel_nonexistent_raises(self, workflow_runner):
        with pytest.raises(ValueError, match="not found"):
            workflow_runner.cancel("nonexistent")

    def test_logs_all_stages(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "completed", "job_id": "j1"},
            "s2": {"status": "running", "job_id": "j2"},
        })
        result = workflow_runner.logs(wf_id)
        assert "s1" in result
        assert "s2" in result
        assert "completed" in result
        assert "running" in result

    def test_logs_specific_stage(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "completed", "job_id": "j1"},
        })
        result = workflow_runner.logs(wf_id, stage="s1")
        assert "j1" in result

    def test_logs_nonexistent_workflow(self, workflow_runner):
        with pytest.raises(ValueError, match="not found"):
            workflow_runner.logs("nonexistent")

    def test_status_returns_record(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {})
        record = workflow_runner.status(wf_id)
        assert record is not None
        assert record["workflow_name"] == "test"

    def test_status_nonexistent_returns_none(self, workflow_runner):
        assert workflow_runner.status("nonexistent") is None

    def test_list_workflows(self, workflow_runner):
        workflow_runner._db.insert_workflow("wf1", {})
        workflow_runner._db.insert_workflow("wf2", {})
        result = workflow_runner.list_workflows()
        assert len(result) == 2


class TestWorkflowRunnerParallelStages:
    """Test workflows with independent (parallel-eligible) stages."""

    def test_independent_stages_both_submitted(self, workflow_runner):
        """Two stages with no dependencies should both be submitted."""
        cfg = WorkflowConfig(
            workflow="parallel_test",
            stages=[
                WorkflowStage(name="a", task="eval", executor="local"),
                WorkflowStage(name="b", task="eval", executor="local"),
            ],
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                wf_id = workflow_runner.run(cfg)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "completed"
                assert mock_submit.call_count == 2

    def test_diamond_dependency(self, workflow_runner):
        """A depends on nothing, B & C depend on A, D depends on B & C."""
        cfg = WorkflowConfig(
            workflow="diamond",
            stages=[
                WorkflowStage(name="A", task="eval", executor="local"),
                WorkflowStage(name="B", task="eval", executor="local", depends_on="A"),
                WorkflowStage(name="C", task="eval", executor="local", depends_on="A"),
                WorkflowStage(name="D", task="eval", executor="local", depends_on=["B", "C"]),
            ],
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                wf_id = workflow_runner.run(cfg)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "completed"
                assert mock_submit.call_count == 4
                # A must be submitted before B, C; D must be last
                names = [c.args[0] for c in mock_submit.call_args_list]
                assert names.index("A") < names.index("B")
                assert names.index("A") < names.index("C")
                assert names.index("B") < names.index("D")
                assert names.index("C") < names.index("D")
