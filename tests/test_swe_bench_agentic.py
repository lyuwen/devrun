"""Unit tests for SWEBenchAgenticTask."""
from __future__ import annotations
import pytest
from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask


def _make_params(**kwargs):
    base = {
        "llm_config": "/fake/config.json",
        "dataset": "/fake/dataset",
        "array": "000-004",
    }
    base.update(kwargs)
    return base


class TestSWEBenchAgenticTask:
    def test_prepare_requires_dataset(self):
        task = SWEBenchAgenticTask()
        with pytest.raises(ValueError, match="dataset"):
            task.prepare({"llm_config": "/fake/config.json"})

    def test_prepare_requires_llm_config_or_model(self):
        task = SWEBenchAgenticTask()
        with pytest.raises(ValueError, match="llm_config"):
            task.prepare({"dataset": "/fake/dataset"})

    def test_prepare_array_flag_in_extra_sbatch(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        extra = spec.resources.get("extra_sbatch", [])
        assert any("--array" in e for e in extra)

    def test_prepare_concurrency_limit(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(concurrency_limit=10))
        extra = spec.resources.get("extra_sbatch", [])
        assert any("%10" in e for e in extra)

    def test_prepare_mkdir_slurm_logs_in_command(self):
        """mkdir -p slurm_logs must appear in generated command."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "mkdir -p slurm_logs" in spec.command

    def test_prepare_no_trailing_backslash(self):
        """Command must not end with a backslash."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert not spec.command.rstrip().endswith("\\")

    def test_prepare_no_oversubscribe_by_default(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        extra = spec.resources.get("extra_sbatch", [])
        assert not any("oversubscribe" in e for e in extra)

    def test_prepare_oversubscribe_opt_in(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(oversubscribe=True))
        extra = spec.resources.get("extra_sbatch", [])
        assert any("oversubscribe" in e for e in extra)

    def test_prepare_task_id_format_in_command(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(task_id_format="%04d"))
        assert "%04d" in spec.command

    def test_prepare_env_commands_prepended(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(env_commands=["source /opt/conda/etc/profile.d/conda.sh"]))
        assert spec.command.startswith("mkdir -p slurm_logs") or \
               "source /opt/conda" in spec.command

    def test_filesystem_check_skipped_for_remote(self):
        """Missing local files should not raise - just warn."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(
            llm_config="/nonexistent/config.json",
            dataset="/nonexistent/dataset",
        ))
        assert spec is not None

    def test_prepare_retry_loop_in_command(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "for attempt in" in spec.command

    def test_prepare_completion_check_in_command(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "output.jsonl" in spec.command

    def test_prepare_max_attempts_default(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "{1..5}" in spec.command

    def test_prepare_max_attempts_custom(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(max_attempts=3))
        assert "{1..3}" in spec.command

    def test_prepare_ds_dir_auto_derived(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(dataset="/fake/dataset", split="test"))
        assert "__fake__dataset-test" in spec.command

    def test_prepare_ds_dir_override(self):
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(ds_dir="custom_ds_dir"))
        assert "custom_ds_dir" in spec.command

    def test_prepare_set_e_false_in_metadata(self):
        """The agentic task must set set_e=False so retry loop works."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert spec.metadata.get("set_e") is False

    def test_prepare_failed_run_archiving(self):
        """Command should include logic to archive failed previous runs."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "/old/" in spec.command
        assert "date" in spec.command
