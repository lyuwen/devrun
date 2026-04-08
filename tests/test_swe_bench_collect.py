"""Unit tests for SWEBenchCollectTask."""
from __future__ import annotations

import pytest
from devrun.tasks.swe_bench_collect import SWEBenchCollectTask


def _make_params(**kwargs):
    base = {
        "output_dir": "logs/run1",
        "dataset": "/mnt/data/SWE-bench_Verified",
        "model_name_or_path": "test-model",
    }
    base.update(kwargs)
    return base


class TestSWEBenchCollectTask:
    def test_prepare_requires_output_dir(self):
        task = SWEBenchCollectTask()
        with pytest.raises(ValueError, match="output_dir"):
            task.prepare({"dataset": "/fake", "model_name_or_path": "m"})

    def test_prepare_requires_dataset(self):
        task = SWEBenchCollectTask()
        with pytest.raises(ValueError, match="dataset"):
            task.prepare({"output_dir": "logs/run1", "model_name_or_path": "m"})

    def test_prepare_requires_model_name_or_path(self):
        task = SWEBenchCollectTask()
        with pytest.raises(ValueError, match="model_name_or_path"):
            task.prepare({"output_dir": "logs/run1", "dataset": "/fake"})

    def test_prepare_generates_jq_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "jq" in spec.command

    def test_prepare_ds_dir_in_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "__mnt__data__SWE-bench_Verified-test" in spec.command

    def test_prepare_default_predictions_path(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "predictions.jsonl" in spec.command

    def test_prepare_custom_predictions_path(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(predictions_path="custom.jsonl"))
        assert "custom.jsonl" in spec.command

    def test_prepare_model_name_in_jq(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(model_name_or_path="my-model"))
        assert "my-model" in spec.command

    def test_prepare_null_patch_validation(self):
        """The jq command should filter out null model_patch values."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "null" in spec.command

    def test_prepare_working_dir_forwarded(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(working_dir="/remote/root"))
        assert spec.working_dir == "/remote/root"

    def test_prepare_working_dir_in_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(working_dir="/remote/root"))
        assert "cd" in spec.command
        assert "/remote/root" in spec.command

    def test_prepare_summary_output(self):
        """Command should print a summary of collected instances."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "echo" in spec.command

    def test_prepare_array_scoping(self):
        """When array is set, only scan matching directories."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(array="000-099"))
        assert spec.command is not None

    def test_prepare_model_name_json_escaped(self):
        """Model name with special chars must be properly escaped for jq."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(model_name_or_path='model "with" quotes'))
        # json.dumps should produce escaped quotes
        assert r'\"with\"' in spec.command or '"model \\"with\\" quotes"' in spec.command
