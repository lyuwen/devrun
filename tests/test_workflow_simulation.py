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
            mock_submit.return_value = ("mock_db_id", "mock_remote_id", {})
            with patch.object(runner, "_poll_job_status", return_value="completed"):
                wf_id = runner.run(
                    swe_bench_three_stage_config, start_after="inference"
                )
                record = runner._db.get_workflow(wf_id)
                import json
                stages = json.loads(record["stages_state"])

                # Inference skipped, collect + evaluate completed
                assert stages["inference"]["status"] == "skipped_by_user"
                assert stages["collect"]["status"] == "completed"
                assert stages["evaluate"]["status"] == "completed"

                # Only collect and evaluate were submitted (in order)
                submitted = [c.args[0] for c in mock_submit.call_args_list]
                assert submitted == ["collect", "evaluate"]

    def test_from_job_populates_downstream_stages(self, tmp_path):
        """Params extracted from a job should propagate correctly via OmegaConf merge.

        This simulates the full CLI flow: extract_workflow_params returns a
        dotlist dict, which is merged into the raw OmegaConf config before
        resolution.  The merged config should have REQUIRED placeholders
        replaced with actual values from the source job, maintaining DS_DIR
        consistency across stages.
        """
        import yaml as _yaml
        from omegaconf import OmegaConf
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

        # Template YAML config with OmegaConf interpolation and REQUIRED placeholders
        template_yaml = {
            "workflow": "swe_bench",
            "params": {
                "model_name": "<REQUIRED:model name>",
                "dataset": "/mnt/data/SWE-bench_Verified",
                "split": "test",
                "output_dir": "<REQUIRED:output directory>",
                "run_name": "<REQUIRED:run name>",
            },
            "stages": [
                {
                    "name": "inference",
                    "task": "swe_bench_agentic",
                    "executor": "slurm",
                    "params": {
                        "model_name": "${params.model_name}",
                        "dataset": "${params.dataset}",
                        "split": "${params.split}",
                        "output_dir": "${params.output_dir}",
                        "run_name": "${params.run_name}",
                    },
                },
                {
                    "name": "collect",
                    "task": "swe_bench_collect",
                    "executor": "local",
                    "depends_on": "inference",
                    "params": {
                        "output_dir": "${params.output_dir}",
                        "dataset": "${params.dataset}",
                        "split": "${params.split}",
                    },
                },
                {
                    "name": "evaluate",
                    "task": "swe_bench_eval",
                    "executor": "local",
                    "depends_on": "collect",
                    "params": {
                        "dataset_name": "${params.dataset}",
                    },
                },
            ],
            "heartbeat_interval": 0.001,
        }
        cfg_path = tmp_path / "workflow.yaml"
        cfg_path.write_text(_yaml.dump(template_yaml))

        # Simulate CLI flow: extract params → merge → resolve
        dotlist, task_name, _ = runner.extract_workflow_params(job_id)
        assert task_name == "swe_bench_agentic"
        assert dotlist["params.model_name"] == "gpt-4o"

        raw_cfg = OmegaConf.load(str(cfg_path))
        job_overrides = [f"{k}={v}" for k, v in dotlist.items()]
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(job_overrides))
        resolved = OmegaConf.to_container(raw_cfg, resolve=True)
        cfg = WorkflowConfig(**resolved)

        # Auto-detect stage
        detected = runner.detect_stage_for_task(task_name, cfg)
        assert detected == "inference"

        # Verify params were populated correctly in all stages
        inf_params = cfg.stages[0].params
        assert inf_params["model_name"] == "gpt-4o"
        assert inf_params["output_dir"] == "logs/gpt4o_run"
        assert inf_params["run_name"] == "gpt4o_run"

        col_params = cfg.stages[1].params
        assert col_params["output_dir"] == "logs/gpt4o_run"
        assert col_params["dataset"] == "/mnt/data/SWE-bench_Verified"

        # DS_DIR should be consistent between inference and collect
        inf_ds_dir = derive_ds_dir(inf_params["dataset"], inf_params["split"])
        col_ds_dir = derive_ds_dir(col_params["dataset"], col_params["split"])
        assert inf_ds_dir == col_ds_dir


class TestCrossStageRefSimulation:
    """Simulation tests for the cross-stage reference mechanism.

    These verify the full param flow through a 3-stage workflow where
    downstream stages use ${stages:...} refs and auto-forwarded params.
    """

    def test_three_stage_cross_stage_resolution(self, tmp_path):
        """Full 3-stage simulation: inference → collect → evaluate with cross-stage refs."""
        from unittest.mock import patch, MagicMock
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "cross_stage_sim.db")

        cfg = WorkflowConfig(
            workflow="swe_bench_cross",
            stages=[
                WorkflowStage(
                    name="inference",
                    task="swe_bench_agentic",
                    executor="slurm",
                    params={
                        "model_name": "gpt-4",
                        "dataset": "/data/SWE-bench_Verified",
                        "split": "test",
                        "run_name": "run1",
                        "output_dir": "logs/run1",
                        "working_dir": "/remote/project",
                        "max_iterations": 100,
                        "llm_config": "/fake/config.json",
                        "max_attempts": 5,
                        "array": "000-499",
                        "concurrency_limit": 10,
                    },
                ),
                WorkflowStage(
                    name="collect",
                    task="swe_bench_collect",
                    executor="ssh",
                    depends_on="inference",
                    params={
                        # Explicit cross-stage refs
                        "model_name_or_path": "<<STAGE_REF:inference:model_name>>",
                        "predictions_path": "<<STAGE_REF:inference:output_dir>>/predictions.jsonl",
                        # output_dir, dataset, split, working_dir: auto-forwarded
                    },
                ),
                WorkflowStage(
                    name="evaluate",
                    task="swe_bench_eval",
                    executor="slurm",
                    depends_on="collect",
                    params={
                        "dataset_name": "<<STAGE_REF:inference:dataset>>",
                        "predictions_path": "<<STAGE_REF:collect:predictions_path>>",
                        # working_dir: auto-forwarded
                        "mem": "64G",
                        "cpus_per_task": 32,
                        "max_workers": 32,
                    },
                ),
            ],
            heartbeat_interval=0.001,
        )

        submitted_stages: list[tuple[str, dict]] = []

        def fake_submit(stage_name, stage, stages_state):
            resolved, _ = runner._resolve_stage_params(stage, stages_state)
            submitted_stages.append((stage_name, resolved))
            return ("db_id", f"remote_{stage_name}", resolved)

        with patch.object(runner, "_submit_stage", side_effect=fake_submit):
            with patch.object(runner, "_poll_job_status", return_value="completed"):
                wf_id = runner.run(cfg)

        # All 3 stages submitted in order
        assert [s[0] for s in submitted_stages] == ["inference", "collect", "evaluate"]

        # --- inference stage ---
        inf_params = submitted_stages[0][1]
        assert inf_params["model_name"] == "gpt-4"
        assert inf_params["output_dir"] == "logs/run1"
        assert inf_params["dataset"] == "/data/SWE-bench_Verified"

        # --- collect stage ---
        col_params = submitted_stages[1][1]
        # Cross-stage refs resolved
        assert col_params["model_name_or_path"] == "gpt-4"
        assert col_params["predictions_path"] == "logs/run1/predictions.jsonl"
        # Auto-forwarded from inference
        assert col_params["output_dir"] == "logs/run1"
        assert col_params["dataset"] == "/data/SWE-bench_Verified"
        assert col_params["split"] == "test"
        assert col_params["working_dir"] == "/remote/project"

        # --- evaluate stage ---
        eval_params = submitted_stages[2][1]
        # Cross-stage refs resolved
        assert eval_params["dataset_name"] == "/data/SWE-bench_Verified"
        assert eval_params["predictions_path"] == "logs/run1/predictions.jsonl"
        # Auto-forwarded through collect (which got it from inference)
        assert eval_params["working_dir"] == "/remote/project"
        # Explicit params preserved
        assert eval_params["mem"] == "64G"
        assert eval_params["cpus_per_task"] == 32

        # Workflow completed
        record = runner._db.get_workflow(wf_id)
        assert record["status"] == "completed"

    def test_cross_stage_with_skipped_inference(self, tmp_path):
        """When inference is skipped via --start-after, collect still resolves refs."""
        from unittest.mock import patch
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "skip_sim.db")

        cfg = WorkflowConfig(
            workflow="swe_bench_skip",
            stages=[
                WorkflowStage(
                    name="inference",
                    task="swe_bench_agentic",
                    executor="slurm",
                    params={"model_name": "gpt-4", "output_dir": "logs/run1"},
                ),
                WorkflowStage(
                    name="collect",
                    task="swe_bench_collect",
                    executor="ssh",
                    depends_on="inference",
                    params={
                        "model_name_or_path": "<<STAGE_REF:inference:model_name>>",
                        "predictions_path": "<<STAGE_REF:inference:output_dir>>/pred.jsonl",
                    },
                ),
            ],
            heartbeat_interval=0.001,
        )

        skipped_params = {
            "inference": {
                "model_name": "gpt-4-turbo",
                "output_dir": "/existing/logs/turbo_run",
                "dataset": "/data/swe",
                "split": "test",
                "working_dir": "/remote/project",
            },
        }

        submitted_stages: list[tuple[str, dict]] = []

        def fake_submit(stage_name, stage, stages_state):
            resolved, _ = runner._resolve_stage_params(stage, stages_state)
            submitted_stages.append((stage_name, resolved))
            return ("db_id", f"remote_{stage_name}", resolved)

        with patch.object(runner, "_submit_stage", side_effect=fake_submit):
            with patch.object(runner, "_poll_job_status", return_value="completed"):
                runner.run(
                    cfg,
                    start_after="inference",
                    skipped_params=skipped_params,
                )

        # Only collect submitted (inference skipped)
        assert len(submitted_stages) == 1
        assert submitted_stages[0][0] == "collect"

        col_params = submitted_stages[0][1]
        # Cross-stage refs resolved from skipped stage's params
        assert col_params["model_name_or_path"] == "gpt-4-turbo"
        assert col_params["predictions_path"] == "/existing/logs/turbo_run/pred.jsonl"
        # Auto-forwarded from skipped stage
        assert col_params["dataset"] == "/data/swe"
        assert col_params["working_dir"] == "/remote/project"

    def test_dry_run_with_cross_stage_refs(self, tmp_path):
        """Dry-run should simulate cross-stage resolution and show resolved values."""
        from devrun.workflow import WorkflowRunner

        runner = WorkflowRunner(db_path=tmp_path / "dry_run_sim.db")

        cfg = WorkflowConfig(
            workflow="test_dry",
            stages=[
                WorkflowStage(
                    name="step1",
                    task="eval",
                    executor="local",
                    params={"output_dir": "/logs/run1", "model": "gpt-4"},
                ),
                WorkflowStage(
                    name="step2",
                    task="eval",
                    executor="local",
                    depends_on="step1",
                    params={
                        "path": "<<STAGE_REF:step1:output_dir>>/data.json",
                    },
                ),
            ],
            heartbeat_interval=0.001,
        )

        result = runner.run(cfg, dry_run=True)

        # Resolved value should appear in the output
        assert "/logs/run1/data.json" in result
        # Sentinel should not appear in the output
        assert "<<STAGE_REF" not in result
        # Auto-forwarded params should be annotated
        assert "[auto]" in result
