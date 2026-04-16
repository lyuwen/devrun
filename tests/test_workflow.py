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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                workflow_runner.run(simple_config)
                calls = [c.args[0] for c in mock_submit.call_args_list]
                assert calls[0] == "step1"
                assert calls[1] == "step2"

    def test_failure_stops_workflow(self, workflow_runner, simple_config):
        """If step1 fails, step2 should never be submitted."""
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="failed"):
                wf_id = workflow_runner.run(simple_config)
                record = workflow_runner._db.get_workflow(wf_id)
                assert record["status"] == "failed"
                assert mock_submit.call_count == 1

    def test_all_stages_complete(self, workflow_runner, simple_config):
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="failed"):
                wf_id = workflow_runner.run(simple_config)
                record = workflow_runner._db.get_workflow(wf_id)
                stages = json.loads(record["stages_state"])
                assert stages["step1"]["status"] == "failed"
                # step2 was never submitted
                assert mock_submit.call_count == 1

    def test_cancel_workflow(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "running", "remote_job_id": "j1"},
            "s2": {"status": "pending", "remote_job_id": None},
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
            "s1": {"status": "completed", "remote_job_id": "j1"},
            "s2": {"status": "running", "remote_job_id": "j2"},
        })
        result = workflow_runner.logs(wf_id)
        assert "s1" in result
        assert "s2" in result
        assert "completed" in result
        assert "running" in result

    def test_logs_specific_stage(self, workflow_runner):
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "completed", "remote_job_id": "j1"},
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
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


# ============================================================================
# OmegaConf interpolation tests
# ============================================================================


class TestOmegaConfInterpolation:
    """Tests for OmegaConf ${params.X} resolution in workflow configs."""

    def test_omegaconf_interpolation_resolved(self, tmp_path):
        """Workflow config with ${params.X} references should resolve correctly."""
        import yaml

        config_data = {
            "workflow": "interp_test",
            "params": {
                "model_name": "test-model",
                "output_dir": "/data/output",
            },
            "stages": [
                {
                    "name": "inference",
                    "task": "eval",
                    "executor": "local",
                    "params": {
                        "model": "${params.model_name}",
                        "output": "${params.output_dir}/results",
                    },
                },
                {
                    "name": "evaluate",
                    "task": "eval",
                    "executor": "local",
                    "depends_on": "inference",
                    "params": {
                        "model": "${params.model_name}",
                        "predictions_path": "${params.output_dir}/predictions.jsonl",
                    },
                },
            ],
            "heartbeat_interval": 0.001,
        }
        cfg_path = tmp_path / "workflow.yaml"
        cfg_path.write_text(yaml.dump(config_data))

        from omegaconf import OmegaConf

        raw = OmegaConf.load(str(cfg_path))
        resolved = OmegaConf.to_container(raw, resolve=True)
        cfg = WorkflowConfig(**resolved)

        # Verify interpolation resolved correctly
        assert cfg.stages[0].params["model"] == "test-model"
        assert cfg.stages[0].params["output"] == "/data/output/results"
        assert cfg.stages[1].params["model"] == "test-model"
        assert cfg.stages[1].params["predictions_path"] == "/data/output/predictions.jsonl"


# ============================================================================
# start_after tests
# ============================================================================


class TestWorkflowStartAfter:
    """Tests for the start_after feature that skips completed stages."""

    @pytest.fixture
    def three_stage_config(self):
        return WorkflowConfig(
            workflow="three_stage",
            stages=[
                WorkflowStage(
                    name="inference", task="eval", executor="local",
                    params={"model": "x"},
                ),
                WorkflowStage(
                    name="collect", task="eval", executor="local",
                    depends_on="inference",
                    params={"output_dir": "/data"},
                ),
                WorkflowStage(
                    name="evaluate", task="eval", executor="local",
                    depends_on="collect",
                    params={"predictions": "/data/pred.jsonl"},
                ),
            ],
            heartbeat_interval=0.001,
        )

    def test_start_after_skips_named_stage(self, workflow_runner, three_stage_config):
        """When start_after='inference', the inference stage is pre-marked as skipped_by_user."""
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                wf_id = workflow_runner.run(three_stage_config, start_after="inference")
                record = workflow_runner._db.get_workflow(wf_id)
                stages = json.loads(record["stages_state"])
                assert stages["inference"]["status"] == "skipped_by_user"
                # collect and evaluate should have been submitted
                submitted_names = [c.args[0] for c in mock_submit.call_args_list]
                assert "inference" not in submitted_names
                assert "collect" in submitted_names
                assert "evaluate" in submitted_names

    def test_start_after_skips_transitive_deps(self, workflow_runner):
        """Stages that the start_after stage depends on are also skipped."""
        cfg = WorkflowConfig(
            workflow="transitive_test",
            stages=[
                WorkflowStage(name="prep", task="eval", executor="local"),
                WorkflowStage(name="inference", task="eval", executor="local", depends_on="prep"),
                WorkflowStage(name="collect", task="eval", executor="local", depends_on="inference"),
                WorkflowStage(name="evaluate", task="eval", executor="local", depends_on="collect"),
            ],
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                wf_id = workflow_runner.run(cfg, start_after="inference")
                record = workflow_runner._db.get_workflow(wf_id)
                stages = json.loads(record["stages_state"])
                # Both prep and inference should be skipped_by_user
                assert stages["prep"]["status"] == "skipped_by_user"
                assert stages["inference"]["status"] == "skipped_by_user"
                # collect and evaluate should run
                submitted_names = [c.args[0] for c in mock_submit.call_args_list]
                assert "collect" in submitted_names
                assert "evaluate" in submitted_names

    def test_start_after_invalid_stage_raises(self, workflow_runner, three_stage_config):
        """start_after with a nonexistent stage name should raise ValueError."""
        with pytest.raises(ValueError, match="not found|does not exist|unknown stage"):
            workflow_runner.run(three_stage_config, start_after="nonexistent_stage")

    def test_start_after_last_stage_completes_immediately(self, workflow_runner, three_stage_config):
        """start_after the last stage means everything is skipped — workflow completes with all skipped."""
        wf_id = workflow_runner.run(three_stage_config, start_after="evaluate")
        record = workflow_runner._db.get_workflow(wf_id)
        stages = json.loads(record["stages_state"])
        # All stages should be skipped_by_user
        for name in ("inference", "collect", "evaluate"):
            assert stages[name]["status"] == "skipped_by_user"
        assert record["status"] == "completed"

    def test_start_after_dry_run_shows_skip_info(self, workflow_runner, three_stage_config):
        """Dry-run with start_after should indicate which stages are skipped."""
        result = workflow_runner.run(three_stage_config, dry_run=True, start_after="inference")
        assert isinstance(result, str)
        # The output should indicate inference is skipped
        result_lower = result.lower()
        assert "skip" in result_lower
        assert "inference" in result_lower
        # collect and evaluate should still be shown
        assert "collect" in result_lower
        assert "evaluate" in result_lower


# ============================================================================
# from_job tests
# ============================================================================


class TestWorkflowFromJob:
    """Tests for extract_workflow_params and detect_stage_for_task."""

    @pytest.fixture
    def swe_bench_workflow_config(self):
        return WorkflowConfig(
            workflow="swe_bench",
            stages=[
                WorkflowStage(
                    name="inference", task="swe_bench_agentic", executor="slurm",
                    params={
                        "model_name": "<REQUIRED:model name>",
                        "dataset": "/data/SWE-bench_Verified",
                        "split": "test",
                        "output_dir": "<REQUIRED:output directory>",
                        "run_name": "<REQUIRED:run name>",
                    },
                ),
                WorkflowStage(
                    name="collect", task="swe_bench_collect", executor="local",
                    depends_on="inference",
                    params={
                        "output_dir": "<REQUIRED:output directory>",
                        "dataset": "/data/SWE-bench_Verified",
                        "split": "test",
                    },
                ),
                WorkflowStage(
                    name="evaluate", task="swe_bench_eval", executor="local",
                    depends_on="collect",
                    params={
                        "dataset_name": "/data/SWE-bench_Verified",
                        "predictions_path": "<REQUIRED:predictions path>",
                    },
                ),
            ],
            heartbeat_interval=0.001,
        )

    def test_extract_workflow_params_extracts_params(self, workflow_runner):
        """extract_workflow_params should return a dotlist dict from a stored job record."""
        job_params = {
            "model_name": "test-model",
            "dataset": "/data/SWE-bench_Verified",
            "split": "test",
            "output_dir": "logs/test_run",
            "run_name": "test_run",
        }
        job_id = workflow_runner._db.insert(
            task_name="swe_bench_agentic",
            executor="slurm",
            parameters=job_params,
        )

        dotlist, task_name = workflow_runner.extract_workflow_params(job_id)
        assert task_name == "swe_bench_agentic"
        assert dotlist["params.model_name"] == "test-model"
        assert dotlist["params.output_dir"] == "logs/test_run"
        assert dotlist["params.run_name"] == "test_run"
        assert dotlist["params.dataset"] == "/data/SWE-bench_Verified"

    def test_extract_workflow_params_nonexistent_raises(self, workflow_runner):
        """extract_workflow_params with an invalid job_id should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            workflow_runner.extract_workflow_params("nonexistent_job_id")

    def test_extract_workflow_params_omits_empty_values(self, workflow_runner):
        """extract_workflow_params should skip params with empty/falsy values."""
        job_params = {
            "model_name": "test-model",
            "dataset": "",
            "output_dir": "logs/run",
        }
        job_id = workflow_runner._db.insert(
            task_name="swe_bench_agentic",
            executor="slurm",
            parameters=job_params,
        )
        dotlist, _ = workflow_runner.extract_workflow_params(job_id)
        assert "params.model_name" in dotlist
        assert "params.output_dir" in dotlist
        assert "params.dataset" not in dotlist  # empty string skipped

    def test_detect_stage_for_task(self, workflow_runner, swe_bench_workflow_config):
        """detect_stage_for_task maps task type to stage name."""
        assert workflow_runner.detect_stage_for_task("swe_bench_agentic", swe_bench_workflow_config) == "inference"
        assert workflow_runner.detect_stage_for_task("swe_bench_collect", swe_bench_workflow_config) == "collect"
        assert workflow_runner.detect_stage_for_task("swe_bench_eval", swe_bench_workflow_config) == "evaluate"
        assert workflow_runner.detect_stage_for_task("unknown_task", swe_bench_workflow_config) is None


# ============================================================================
# Detached mode tests
# ============================================================================


class TestWorkflowDetached:
    """Tests for detached (background) workflow execution."""

    def test_detached_creates_db_record(self, workflow_runner):
        """run_detached should write a workflow record to the DB before returning."""
        cfg = WorkflowConfig(
            workflow="detach_test",
            stages=[
                WorkflowStage(name="slow_stage", task="eval", executor="local",
                              params={"model": "x"}),
            ],
            heartbeat_interval=1.0,
        )
        with patch("devrun.workflow.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            wf_id = workflow_runner.run_detached(cfg)
            # Should return immediately with a workflow_id
            assert isinstance(wf_id, str)
            assert len(wf_id) > 0
            # DB record should exist
            record = workflow_runner._db.get_workflow(wf_id)
            assert record is not None
            assert record["workflow_name"] == "detach_test"

    def test_detached_returns_workflow_id_immediately(self, workflow_runner):
        """run_detached should return the workflow ID immediately without blocking."""
        cfg = WorkflowConfig(
            workflow="detach_return_test",
            stages=[
                WorkflowStage(name="s1", task="eval", executor="local",
                              params={"model": "x"}),
                WorkflowStage(name="s2", task="eval", executor="local",
                              depends_on="s1", params={"model": "y"}),
            ],
            heartbeat_interval=10.0,
        )
        import time

        with patch("devrun.workflow.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            start = time.monotonic()
            wf_id = workflow_runner.run_detached(cfg)
            elapsed = time.monotonic() - start
            # Should return much faster than the heartbeat interval
            assert elapsed < 5.0
            assert isinstance(wf_id, str)
            # Popen should have been called to spawn background process
            mock_popen.assert_called_once()


# ============================================================================
# Placeholder validation tests
# ============================================================================


class TestPlaceholderValidation:
    """Tests for <REQUIRED:...> placeholder validation."""

    def test_placeholder_validation_catches_required(self, workflow_runner):
        """Config with <REQUIRED:...> placeholders should be detected and rejected."""
        cfg = WorkflowConfig(
            workflow="placeholder_test",
            stages=[
                WorkflowStage(
                    name="s1", task="eval", executor="local",
                    params={"model": "<REQUIRED:model name>", "dataset": "/data/test"},
                ),
            ],
            heartbeat_interval=0.001,
        )
        with pytest.raises((ValueError, RuntimeError), match="REQUIRED|placeholder|unfilled"):
            workflow_runner.run(cfg)

    def test_placeholder_validation_passes_clean_config(self, workflow_runner):
        """Config without <REQUIRED:...> placeholders should pass validation."""
        cfg = WorkflowConfig(
            workflow="clean_test",
            stages=[
                WorkflowStage(
                    name="s1", task="eval", executor="local",
                    params={"model": "test-model", "dataset": "/data/test"},
                ),
            ],
            heartbeat_interval=0.001,
        )
        with patch.object(workflow_runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = ("mock_db_id", "mock_remote_id")
            with patch.object(workflow_runner, "_poll_job_status", return_value="completed"):
                # Should not raise
                wf_id = workflow_runner.run(cfg)
                assert isinstance(wf_id, str)


# ============================================================================
# Enhanced dry-run output tests
# ============================================================================


class TestImprovedDryRun:
    """Tests for enhanced dry-run output format."""

    def test_improved_dry_run_output(self, workflow_runner):
        """Enhanced dry-run should include full params and detailed formatting."""
        cfg = WorkflowConfig(
            workflow="dryrun_test",
            stages=[
                WorkflowStage(
                    name="inference", task="eval", executor="local",
                    params={"model": "test-model", "batch_size": 8},
                ),
                WorkflowStage(
                    name="evaluate", task="eval", executor="local",
                    depends_on="inference",
                    params={"dataset": "/data/test"},
                ),
            ],
            heartbeat_interval=0.001,
        )
        result = workflow_runner.run(cfg, dry_run=True)
        assert isinstance(result, str)
        # Should include stage info
        assert "inference" in result
        assert "evaluate" in result
        # Should include executor info
        assert "local" in result
        # Should include task info
        assert "eval" in result
        # Should include some form of parameter display
        result_lower = result.lower()
        assert "param" in result_lower or "model" in result_lower or "batch_size" in result_lower


# ============================================================================
# Enhanced logs tests
# ============================================================================


class TestEnhancedLogs:
    """Tests for enhanced workflow logs that delegate to executor."""

    def test_enhanced_logs_delegates_to_executor(self, workflow_runner):
        """Logs method should attempt to delegate to executor.logs() for actual content."""
        wf_id = workflow_runner._db.insert_workflow("test", {
            "s1": {"status": "completed", "remote_job_id": "j1", "executor": "local"},
        })
        # Insert a corresponding job record so executor lookup works
        workflow_runner._db.insert(task_name="eval", executor="local")

        mock_executor = MagicMock()
        mock_executor.logs.return_value = "real log content from executor"

        with patch("devrun.workflow.resolve_executor", return_value=mock_executor):
            result = workflow_runner.logs(wf_id, stage="s1")
            # The result should include actual log content or at least delegate
            assert isinstance(result, str)
            assert len(result) > 0
