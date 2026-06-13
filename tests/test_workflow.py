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


_LEGACY_HEARTBEAT_LOOP_SKIP = pytest.mark.skip(
    reason="PR3: WorkflowRunner.run() is now a pure producer (atomic enqueue + return). "
    "Tests that exercise the legacy in-runner heartbeat loop, stage-by-stage submit, "
    "and stages_state mutation are being removed in PR3 Task #37 "
    "(drop run_detached, --detach, in-runner polling)."
)



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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id", {})
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id", {})
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

        dotlist, task_name, _ = workflow_runner.extract_workflow_params(job_id)
        assert task_name == "swe_bench_agentic"
        assert dotlist["params.model_name"] == "test-model"
        assert dotlist["params.output_dir"] == "logs/test_run"
        assert dotlist["params.run_name"] == "test_run"
        assert dotlist["params.dataset"] == "/data/SWE-bench_Verified"

    def test_extract_workflow_params_nonexistent_raises(self, workflow_runner):
        """extract_workflow_params with an invalid job_id should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            workflow_runner.extract_workflow_params("nonexistent_job_id")

    def test_extract_workflow_params_negative_index(self, workflow_runner):
        """`-1` resolves to the latest job; `-2` to the one before."""
        first = workflow_runner._db.insert(
            task_name="swe_bench_agentic",
            executor="slurm",
            parameters={"model_name": "m-old", "dataset": "/d"},
        )
        second = workflow_runner._db.insert(
            task_name="swe_bench_agentic",
            executor="slurm",
            parameters={"model_name": "m-new", "dataset": "/d"},
        )
        _, _, rid = workflow_runner.extract_workflow_params("-1")
        assert rid == second
        _, _, rid2 = workflow_runner.extract_workflow_params("-2")
        assert rid2 == first

    def test_extract_workflow_params_negative_index_filtered(self, workflow_runner):
        """`allowed_source_tasks` should skip jobs whose task is not in the set."""
        workflow_runner._db.insert("eval", "local", {"foo": "bar"})  # newest, but excluded
        agentic = workflow_runner._db.insert(
            "swe_bench_agentic", "slurm", {"model_name": "m"}
        )
        _, task_name, rid = workflow_runner.extract_workflow_params(
            "-1", allowed_source_tasks={"swe_bench_agentic"}
        )
        assert rid == agentic
        assert task_name == "swe_bench_agentic"

    def test_extract_workflow_params_negative_index_too_deep(self, workflow_runner):
        workflow_runner._db.insert("swe_bench_agentic", "slurm", {"model_name": "m"})
        with pytest.raises(ValueError, match="only 1 job"):
            workflow_runner.extract_workflow_params("-5")

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
        dotlist, _, _ = workflow_runner.extract_workflow_params(job_id)
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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id", {})
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


# ============================================================================
# Cross-stage parameter resolution tests
# ============================================================================


class TestResolveStageParams:
    """Tests for WorkflowRunner._resolve_stage_params()."""

    def test_cross_ref_simple(self, workflow_runner):
        """A sentinel string is replaced with the dep's actual value."""
        stage = WorkflowStage(
            name="collect",
            task="eval",
            executor="local",
            params={"output_dir": "<<STAGE_REF:inference:output_dir>>"},
            depends_on="inference",
        )
        stages_state = {
            "inference": {
                "status": "completed",
                "resolved_params": {"output_dir": "/data/logs/run1", "model": "gpt-4"},
            },
        }
        resolved, auto_keys = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["output_dir"] == "/data/logs/run1"

    def test_cross_ref_concatenation(self, workflow_runner):
        """Sentinel embedded in a larger string is replaced correctly."""
        stage = WorkflowStage(
            name="collect",
            task="eval",
            executor="local",
            params={
                "predictions": "<<STAGE_REF:inference:output_dir>>/predictions.jsonl",
            },
            depends_on="inference",
        )
        stages_state = {
            "inference": {
                "status": "completed",
                "resolved_params": {"output_dir": "logs/run1"},
            },
        }
        resolved, _ = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["predictions"] == "logs/run1/predictions.jsonl"

    def test_cross_ref_preserves_non_string_types(self, workflow_runner):
        """When a sentinel spans the full value and the ref is non-string, preserve the type."""
        stage = WorkflowStage(
            name="step2",
            task="eval",
            executor="local",
            params={"count": "<<STAGE_REF:step1:count>>"},
            depends_on="step1",
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"count": 42},
            },
        }
        resolved, _ = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["count"] == 42
        assert isinstance(resolved["count"], int)

    def test_auto_forward_missing_param(self, workflow_runner):
        """Params missing from the stage are auto-forwarded from the dep."""
        stage = WorkflowStage(
            name="collect",
            task="eval",
            executor="local",
            params={"model_name_or_path": "gpt-4"},
            depends_on="inference",
        )
        stages_state = {
            "inference": {
                "status": "completed",
                "resolved_params": {
                    "output_dir": "/logs/run1",
                    "dataset": "/data/swebench",
                    "split": "test",
                    "working_dir": "/home/user",
                },
            },
        }
        resolved, auto_keys = workflow_runner._resolve_stage_params(stage, stages_state)
        # Existing explicit param preserved
        assert resolved["model_name_or_path"] == "gpt-4"
        # Auto-forwarded params
        assert resolved["output_dir"] == "/logs/run1"
        assert resolved["dataset"] == "/data/swebench"
        assert resolved["split"] == "test"
        assert resolved["working_dir"] == "/home/user"
        # Track which keys were auto-forwarded
        assert auto_keys == {"output_dir", "dataset", "split", "working_dir"}

    def test_explicit_param_not_overwritten(self, workflow_runner):
        """An explicitly set param is never overwritten by auto-forward."""
        stage = WorkflowStage(
            name="step2",
            task="eval",
            executor="local",
            params={"model": "my-model"},
            depends_on="step1",
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"model": "dep-model", "extra": "val"},
            },
        }
        resolved, auto_keys = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["model"] == "my-model"  # NOT dep-model
        assert resolved["extra"] == "val"
        assert "model" not in auto_keys
        assert "extra" in auto_keys

    def test_missing_ref_raises(self, workflow_runner):
        """Referencing a param that doesn't exist in the dep raises ValueError."""
        stage = WorkflowStage(
            name="collect",
            task="eval",
            executor="local",
            params={"x": "<<STAGE_REF:inference:nonexistent>>"},
            depends_on="inference",
        )
        stages_state = {
            "inference": {
                "status": "completed",
                "resolved_params": {"output_dir": "/logs"},
            },
        }
        with pytest.raises(ValueError, match="nonexistent"):
            workflow_runner._resolve_stage_params(stage, stages_state)

    def test_missing_stage_raises(self, workflow_runner):
        """Referencing a stage that doesn't exist raises ValueError."""
        stage = WorkflowStage(
            name="collect",
            task="eval",
            executor="local",
            params={"x": "<<STAGE_REF:unknown_stage:param>>"},
            depends_on="inference",
        )
        stages_state = {
            "inference": {
                "status": "completed",
                "resolved_params": {"param": "val"},
            },
        }
        with pytest.raises(ValueError, match="unknown_stage"):
            workflow_runner._resolve_stage_params(stage, stages_state)

    def test_multi_dep_auto_forward_first_wins(self, workflow_runner):
        """With multiple deps, first dep's value wins for auto-forward."""
        stage = WorkflowStage(
            name="step3",
            task="eval",
            executor="local",
            params={},
            depends_on=["step1", "step2"],
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"shared": "from_step1", "only_1": "val1"},
            },
            "step2": {
                "status": "completed",
                "resolved_params": {"shared": "from_step2", "only_2": "val2"},
            },
        }
        resolved, auto_keys = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["shared"] == "from_step1"  # first dep wins
        assert resolved["only_1"] == "val1"
        assert resolved["only_2"] == "val2"
        assert auto_keys == {"shared", "only_1", "only_2"}

    def test_no_deps_no_auto_forward(self, workflow_runner):
        """A stage with no dependencies gets no auto-forwarded params."""
        stage = WorkflowStage(
            name="step1",
            task="eval",
            executor="local",
            params={"model": "gpt-4"},
        )
        resolved, auto_keys = workflow_runner._resolve_stage_params(stage, {})
        assert resolved == {"model": "gpt-4"}
        assert auto_keys == set()

    def test_nested_dict_sentinel_resolved(self, workflow_runner):
        """Sentinels inside nested dicts are resolved."""
        stage = WorkflowStage(
            name="step2",
            task="eval",
            executor="local",
            params={"config": {"path": "<<STAGE_REF:step1:output>>"}},
            depends_on="step1",
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"output": "/data/out"},
            },
        }
        resolved, _ = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["config"]["path"] == "/data/out"

    def test_list_sentinel_resolved(self, workflow_runner):
        """Sentinels inside lists are resolved."""
        stage = WorkflowStage(
            name="step2",
            task="eval",
            executor="local",
            params={"paths": ["<<STAGE_REF:step1:output>>", "fixed"]},
            depends_on="step1",
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"output": "/data/out"},
            },
        }
        resolved, _ = workflow_runner._resolve_stage_params(stage, stages_state)
        assert resolved["paths"] == ["/data/out", "fixed"]

    def test_complex_ref_in_concatenation_raises(self, workflow_runner):
        """Interpolating a dict/list into a string concatenation raises ValueError."""
        stage = WorkflowStage(
            name="step2",
            task="eval",
            executor="local",
            params={"path": "prefix/<<STAGE_REF:step1:config>>/suffix"},
            depends_on="step1",
        )
        stages_state = {
            "step1": {
                "status": "completed",
                "resolved_params": {"config": {"key": "val"}},
            },
        }
        with pytest.raises(ValueError, match="complex value"):
            workflow_runner._resolve_stage_params(stage, stages_state)



