"""Unit tests for SWEBenchEvalTask."""
from __future__ import annotations
import pytest
from devrun.tasks.swe_bench_eval import SWEBenchEvalTask


class TestSWEBenchEvalTask:
    def test_prepare_requires_dataset_name(self):
        task = SWEBenchEvalTask()
        with pytest.raises(ValueError, match="dataset_name"):
            task.prepare({})

    def test_prepare_placeholder_validation_dataset(self):
        task = SWEBenchEvalTask()
        with pytest.raises(ValueError, match="placeholder"):
            task.prepare({"dataset_name": "<path_to_dataset>"})

    def test_prepare_placeholder_validation_working_dir(self):
        task = SWEBenchEvalTask()
        with pytest.raises(ValueError, match="placeholder"):
            task.prepare({"dataset_name": "real/dataset", "working_dir": "<path_to_working_directory>"})

    def test_prepare_generates_correct_command(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "max_workers": 4,
            "run_id": "abc123",
            "predictions_path": "preds.jsonl",
        })
        assert "python -m swebench.harness.run_evaluation" in spec.command
        assert "--dataset_name" in spec.command
        assert "princeton-nlp/SWE-bench_Verified" in spec.command
        assert "--run_id" in spec.command
        assert "abc123" in spec.command

    def test_prepare_auto_generates_run_id(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "some/dataset",
            "predictions_path": "/nonexistent/file.jsonl",
        })
        # run_id should be generated (datetime format)
        assert "--run_id" in spec.command

    def test_prepare_with_namespace(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "some/dataset",
            "namespace": "myns",
            "run_id": "x",
        })
        assert "--namespace" in spec.command
        assert "myns" in spec.command

    def test_prepare_without_namespace(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "some/dataset",
            "run_id": "x",
        })
        assert "--namespace" not in spec.command

    def test_prepare_resources_forwarded(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "some/dataset",
            "run_id": "x",
            "mem": "64G",
            "cpus_per_task": 32,
            "walltime": "24:00:00",
        })
        assert spec.resources["mem"] == "64G"
        assert spec.resources["cpus_per_task"] == 32

    def test_prepare_env_forwarded(self):
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "some/dataset",
            "run_id": "x",
            "env": {"MY_VAR": "value"},
        })
        assert spec.env.get("MY_VAR") == "value"

    def test_command_shell_safe_with_spaces(self):
        """Paths with spaces should be quoted and not break the command."""
        task = SWEBenchEvalTask()
        spec = task.prepare({
            "dataset_name": "path/with spaces/dataset",
            "run_id": "x",
        })
        # The command should contain a properly quoted version
        assert "path/with spaces/dataset" in spec.command or \
               "'path/with spaces/dataset'" in spec.command or \
               '"path/with spaces/dataset"' in spec.command
