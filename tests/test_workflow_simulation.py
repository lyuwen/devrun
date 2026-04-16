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
        assert "python3" in spec.command
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


class TestWorkflowSimulationStartAfter:
    """Simulation tests for start_after and from_job workflows."""

    @pytest.fixture
    def swe_bench_three_stage_config(self):
        """A three-stage SWE-bench workflow matching the production pattern."""
        return WorkflowConfig(
            workflow="swe_bench",
            params={
                "model_name": "test-model",
                "dataset": "/mnt/data/SWE-bench_Verified",
                "split": "test",
                "output_dir": "logs/sim_run",
                "run_name": "sim_run",
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

    def test_start_after_inference_runs_collect_eval(
        self, swe_bench_three_stage_config, tmp_path
    ):
        """Core use case: skip inference, run collect and evaluate.

        When start_after='inference', only collect and evaluate should execute.
        This simulates the primary scenario where inference was run via an
        existing swe_bench_agentic task and we want to continue with
        collect + eval.
        """
        from unittest.mock import patch
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "sim_test.db")

        with patch.object(runner, "_submit_stage") as mock_submit:
            mock_submit.return_value = "mock_job_id"
            with patch.object(runner, "_poll_job_status", return_value="completed"):
                wf_id = runner.run(
                    swe_bench_three_stage_config, start_after="inference"
                )
                record = runner._db.get_workflow(wf_id)
                import json
                stages = json.loads(record["stages_state"])

                # Inference skipped, collect + evaluate completed
                assert stages["inference"]["status"] == "skipped"
                assert stages["collect"]["status"] == "completed"
                assert stages["evaluate"]["status"] == "completed"

                # Only collect and evaluate were submitted (in order)
                submitted = [c.args[0] for c in mock_submit.call_args_list]
                assert submitted == ["collect", "evaluate"]

    def test_from_job_populates_downstream_stages(self, tmp_path):
        """Params from a source job should flow correctly to collect + eval stages.

        When using from_job with a swe_bench_agentic job, the extracted params
        (model_name, dataset, split, output_dir) should propagate to the
        collect and evaluate stages, maintaining DS_DIR consistency.
        """
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "from_job_sim.db")

        # Insert a source job record simulating a completed swe_bench_agentic run
        job_params = {
            "model_name": "gpt-4o",
            "dataset": "/mnt/data/SWE-bench_Verified",
            "split": "test",
            "output_dir": "logs/gpt4o_run",
            "run_name": "gpt4o_run",
            "working_dir": "/remote/project",
            "llm_config": "/fake/config.json",
            "max_iterations": 100,
        }
        job_id = runner._db.insert(
            task_name="swe_bench_agentic",
            executor="slurm",
            parameters=job_params,
        )

        # Template config with placeholders
        template_config = WorkflowConfig(
            workflow="swe_bench",
            stages=[
                WorkflowStage(
                    name="inference",
                    task="swe_bench_agentic",
                    executor="slurm",
                    params={
                        "model_name": "<REQUIRED:model name>",
                        "dataset": "/mnt/data/SWE-bench_Verified",
                        "split": "test",
                        "output_dir": "<REQUIRED:output directory>",
                        "run_name": "<REQUIRED:run name>",
                    },
                ),
                WorkflowStage(
                    name="collect",
                    task="swe_bench_collect",
                    executor="local",
                    depends_on="inference",
                    params={
                        "output_dir": "<REQUIRED:output directory>",
                        "dataset": "/mnt/data/SWE-bench_Verified",
                        "split": "test",
                    },
                ),
                WorkflowStage(
                    name="evaluate",
                    task="swe_bench_eval",
                    executor="local",
                    depends_on="collect",
                    params={
                        "dataset_name": "/mnt/data/SWE-bench_Verified",
                        "predictions_path": "<REQUIRED:predictions path>",
                    },
                ),
            ],
            heartbeat_interval=0.001,
        )

        # Extract config from the source job
        populated = runner.from_job(job_id, template_config)

        # Verify params were populated correctly
        inf_params = populated.stages[0].params
        assert inf_params["model_name"] == "gpt-4o"
        assert inf_params["output_dir"] == "logs/gpt4o_run"
        assert inf_params["run_name"] == "gpt4o_run"

        # Verify downstream stages got consistent params
        col_params = populated.stages[1].params
        assert col_params["output_dir"] == "logs/gpt4o_run"

        # DS_DIR should be consistent between inference and collect
        inf_ds_dir = derive_ds_dir(
            inf_params["dataset"], inf_params["split"]
        )
        col_ds_dir = derive_ds_dir(
            col_params["dataset"], col_params["split"]
        )
        assert inf_ds_dir == col_ds_dir
