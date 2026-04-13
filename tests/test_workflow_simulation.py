"""Simulation tests for workflow execution plans.

These tests mock all remote execution and verify the full execution plan
including parameter propagation, DS_DIR consistency, and working directory
correctness across stages. No real remote connections are made.
"""
from __future__ import annotations

import pytest

from devrun.models import WorkflowConfig, WorkflowStage
from devrun.utils.swebench import derive_ds_dir


@pytest.fixture
def swe_bench_config():
    return WorkflowConfig(
        workflow="swe_bench",
        params={
            "model_name": "test-model",
            "dataset": "/mnt/data/SWE-bench_Verified",
            "split": "test",
            "run_name": "sim_run",
            "output_dir": "logs/sim_run",
            "working_dir": "/remote/project",
        },
        stages=[
            WorkflowStage(
                name="inference",
                task="swe_bench_agentic",
                executor="slurm",
                params={
                    "model_name": "test-model",
                    "dataset": "/mnt/data/SWE-bench_Verified",
                    "split": "test",
                    "run_name": "sim_run",
                    "output_dir": "logs/sim_run",
                    "llm_config": "/fake/config.json",
                    "max_iterations": 100,
                    "max_attempts": 5,
                    "array": "000-004",
                    "working_dir": "/remote/project",
                    "base_url": "http://localhost:8000",
                    "api_key": "sk-test",
                    "temperature": "0.7",
                    "top_p": "0.95",
                    "env_commands": ["source /opt/conda/bin/activate"],
                    "env": {},
                },
            ),
            WorkflowStage(
                name="collect",
                task="swe_bench_collect",
                executor="local",
                depends_on="inference",
                params={
                    "output_dir": "logs/sim_run",
                    "dataset": "/mnt/data/SWE-bench_Verified",
                    "split": "test",
                    "model_name_or_path": "test-model",
                    "predictions_path": "logs/sim_run/predictions.jsonl",
                    "working_dir": "/remote/project",
                },
            ),
            WorkflowStage(
                name="evaluate",
                task="swe_bench_eval",
                executor="local",
                depends_on="collect",
                params={
                    "dataset_name": "/mnt/data/SWE-bench_Verified",
                    "predictions_path": "logs/sim_run/predictions.jsonl",
                    "working_dir": "/remote/project",
                    "run_id": "sim_test",
                },
            ),
        ],
        heartbeat_interval=0.001,
    )


class TestWorkflowSimulation:
    def test_dry_run_produces_execution_plan(self, swe_bench_config, tmp_path):
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "test.db")
        result = runner.run(swe_bench_config, dry_run=True)
        assert isinstance(result, str)
        assert "Stage 1: inference" in result
        assert "Stage 2: collect" in result
        assert "Stage 3: evaluate" in result

    def test_working_dir_consistent_across_stages(self, swe_bench_config):
        """All stages should resolve to the same working_dir."""
        for stage in swe_bench_config.stages:
            wd = stage.params.get("working_dir")
            assert wd == "/remote/project", f"Stage {stage.name} has working_dir={wd}"

    def test_ds_dir_consistent_between_inference_and_collect(self, swe_bench_config):
        """The DS_DIR used by inference must match what collect scans."""
        inf_params = swe_bench_config.stages[0].params
        col_params = swe_bench_config.stages[1].params
        inf_ds_dir = derive_ds_dir(inf_params["dataset"], inf_params["split"])
        col_ds_dir = derive_ds_dir(col_params["dataset"], col_params["split"])
        assert inf_ds_dir == col_ds_dir

    def test_predictions_path_matches_between_collect_and_eval(self, swe_bench_config):
        col_pred = swe_bench_config.stages[1].params["predictions_path"]
        eval_pred = swe_bench_config.stages[2].params["predictions_path"]
        assert col_pred == eval_pred

    def test_inference_stage_generates_valid_script(self, swe_bench_config):
        """The inference stage should produce a valid TaskSpec with retry loop."""
        from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask

        task = SWEBenchAgenticTask()
        spec = task.prepare(swe_bench_config.stages[0].params)
        assert "for attempt in" in spec.command
        assert "__mnt__data__SWE-bench_Verified-test" in spec.command
        assert spec.metadata.get("set_e") is False

    def test_collect_stage_generates_valid_script(self, swe_bench_config):
        from devrun.tasks.swe_bench_collect import SWEBenchCollectTask

        task = SWEBenchCollectTask()
        spec = task.prepare(swe_bench_config.stages[1].params)
        assert "jq" in spec.command
        assert "predictions.jsonl" in spec.command
        assert "__mnt__data__SWE-bench_Verified-test" in spec.command

    def test_eval_stage_generates_valid_command(self, swe_bench_config):
        from devrun.tasks.swe_bench_eval import SWEBenchEvalTask

        task = SWEBenchEvalTask()
        spec = task.prepare(swe_bench_config.stages[2].params)
        assert "run_evaluation" in spec.command
        assert "predictions_path" in spec.command

    def test_inference_working_dir_in_command(self, swe_bench_config):
        """The inference TaskSpec should set working_dir (executor handles cd)."""
        from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask

        task = SWEBenchAgenticTask()
        spec = task.prepare(swe_bench_config.stages[0].params)
        assert spec.working_dir == "/remote/project"

    def test_collect_working_dir_in_command(self, swe_bench_config):
        """The collect command should cd to working_dir."""
        from devrun.tasks.swe_bench_collect import SWEBenchCollectTask

        task = SWEBenchCollectTask()
        spec = task.prepare(swe_bench_config.stages[1].params)
        assert "/remote/project" in spec.command

    def test_eval_working_dir_propagated(self, swe_bench_config):
        """The eval TaskSpec should have working_dir set."""
        from devrun.tasks.swe_bench_eval import SWEBenchEvalTask

        task = SWEBenchEvalTask()
        spec = task.prepare(swe_bench_config.stages[2].params)
        assert spec.working_dir == "/remote/project"

    def test_dependency_chain_is_linear(self, swe_bench_config):
        """inference → collect → evaluate — each depends on the previous."""
        stages = {s.name: s for s in swe_bench_config.stages}
        assert stages["inference"].depends_on is None
        assert stages["collect"].depends_on == "inference"
        assert stages["evaluate"].depends_on == "collect"

    def test_all_stages_use_same_dataset(self, swe_bench_config):
        """Dataset path must be consistent across all stages that use it."""
        inf_dataset = swe_bench_config.stages[0].params["dataset"]
        col_dataset = swe_bench_config.stages[1].params["dataset"]
        eval_dataset = swe_bench_config.stages[2].params["dataset_name"]
        assert inf_dataset == col_dataset == eval_dataset
