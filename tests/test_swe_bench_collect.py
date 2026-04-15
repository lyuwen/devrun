"""Unit tests for SWEBenchCollectTask."""
from __future__ import annotations

import json

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
    # ---- Validation ---------------------------------------------------

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

    # ---- Command structure --------------------------------------------

    def test_prepare_generates_python_collector(self):
        """Command must invoke python3 on a temp script, not bash+jq."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "python3" in spec.command
        assert "mktemp" in spec.command

    def test_prepare_heredoc_structure(self):
        """Command must use a quoted heredoc to embed the Python script."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "__DEVRUN_COLLECT_EOF__" in spec.command
        # Quoted delimiter prevents bash expansion
        assert "'__DEVRUN_COLLECT_EOF__'" in spec.command

    def test_prepare_tempfile_cleanup(self):
        """Command must remove the temp script after execution."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "rm -f" in spec.command

    def test_prepare_exit_code_preserved(self):
        """Command must preserve the python3 exit code."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "_RC=$?" in spec.command
        assert "exit ${_RC}" in spec.command

    # ---- Configuration injection --------------------------------------

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

    def test_prepare_default_histories_path(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "collected_histories.jsonl" in spec.command

    def test_prepare_custom_histories_path(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(histories_path="my_histories.jsonl"))
        assert "my_histories.jsonl" in spec.command

    def test_prepare_model_name_in_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(model_name_or_path="my-model"))
        assert "my-model" in spec.command

    def test_prepare_model_name_json_escaped(self):
        """Model name with special chars must be properly JSON-escaped."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(model_name_or_path='model "with" quotes'))
        # json.dumps produces escaped quotes
        assert json.dumps('model "with" quotes') in spec.command

    def test_prepare_output_dir_in_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(output_dir="/data/results"))
        assert "/data/results" in spec.command

    def test_prepare_default_max_workers(self):
        """Default max_workers should be 16."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "MAX_WORKERS = 16" in spec.command

    def test_prepare_custom_max_workers(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(max_workers=32))
        assert "MAX_WORKERS = 32" in spec.command

    # ---- Working directory --------------------------------------------

    def test_prepare_working_dir_forwarded(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(working_dir="/remote/root"))
        assert spec.working_dir == "/remote/root"

    def test_prepare_working_dir_in_command(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(working_dir="/remote/root"))
        assert "cd" in spec.command
        assert "/remote/root" in spec.command

    def test_prepare_no_cd_without_working_dir(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        # No cd when working_dir is not set
        lines = spec.command.split("\n")
        assert not any(line.startswith("cd ") for line in lines)

    # ---- Parallel processing references -------------------------------

    def test_prepare_uses_threadpool(self):
        """The generated Python script must use ThreadPoolExecutor."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "ThreadPoolExecutor" in spec.command

    def test_prepare_uses_scandir(self):
        """The generated script must use os.scandir for efficient directory walking."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "os.scandir" in spec.command

    # ---- Metadata and env ---------------------------------------------

    def test_prepare_metadata_job_name(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert spec.metadata["job_name"] == "swe_collect"

    def test_prepare_env_forwarded(self):
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(env={"FOO": "bar"}))
        assert spec.env == {"FOO": "bar"}

    def test_prepare_custom_split(self):
        """Custom split should change the DS_DIR derivation."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params(split="dev"))
        assert "__mnt__data__SWE-bench_Verified-dev" in spec.command

    # ---- History collection references --------------------------------

    def test_prepare_history_collection_in_script(self):
        """The generated script must contain history.json collection logic."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert ".history.json" in spec.command

    def test_prepare_history_output_format(self):
        """History entries should include source_file and history keys."""
        task = SWEBenchCollectTask()
        spec = task.prepare(_make_params())
        assert "source_file" in spec.command
        assert "history" in spec.command
