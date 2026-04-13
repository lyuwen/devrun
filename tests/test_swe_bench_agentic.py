"""Unit tests for SWEBenchAgenticTask."""
from __future__ import annotations
import pytest
import yaml
from pathlib import Path
from devrun.tasks.swe_bench_agentic import SWEBenchAgenticTask
from devrun.models import TaskConfig, TaskSpec


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


class TestRunInferMaxAttempts:
    """Tests for the --max-attempts flag passed to run_infer.py."""

    def test_default_run_infer_max_attempts_in_command(self):
        """Default run_infer_max_attempts=5 renders as --max-attempts 5."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "--max-attempts 5" in spec.command

    def test_custom_run_infer_max_attempts(self):
        """Custom run_infer_max_attempts renders correctly."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(run_infer_max_attempts=10))
        assert "--max-attempts 10" in spec.command

    def test_run_infer_max_attempts_independent_of_retry_loop(self):
        """run_infer_max_attempts and max_attempts (retry loop) are independent."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(max_attempts=2, run_infer_max_attempts=8))
        assert "{1..2}" in spec.command  # retry loop
        assert "--max-attempts 8" in spec.command  # run_infer flag


class TestGitSafeDirs:
    """Tests for git safe.directory configuration in task prepare."""

    def test_no_git_safe_dirs_by_default(self):
        """No git config lines when git_safe_dirs is not provided."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "git config --global --add safe.directory" not in spec.command

    def test_git_safe_dirs_empty_list(self):
        """No git config lines when git_safe_dirs is an empty list."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(git_safe_dirs=[]))
        assert "git config --global --add safe.directory" not in spec.command

    def test_git_safe_dirs_single_entry(self):
        """Single git_safe_dirs entry renders one config line."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(git_safe_dirs=["/opt/repo"]))
        assert "git config --global --add safe.directory" in spec.command
        assert "/opt/repo" in spec.command

    def test_git_safe_dirs_multiple_entries(self):
        """Multiple git_safe_dirs entries render multiple config lines."""
        task = SWEBenchAgenticTask()
        dirs = ["/opt/repo1", "/opt/repo2"]
        spec = task.prepare(_make_params(git_safe_dirs=dirs))
        for d in dirs:
            assert d in spec.command
        # Count occurrences of the git config line
        count = spec.command.count("git config --global --add safe.directory")
        assert count == 2


class TestTaskIdFormatDefault:
    """Tests for task_id_format default value."""

    def test_default_task_id_format_is_03d(self):
        """Default task_id_format should be %03d, not %05d."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params())
        assert "%03d" in spec.command

    def test_custom_task_id_format(self):
        """Custom task_id_format is respected."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(task_id_format="%05d"))
        assert "%05d" in spec.command
        assert "%03d" not in spec.command


class TestPrepareMany:
    """Tests for SWEBenchAgenticTask.prepare_many (multi-shard support)."""

    def test_prepare_many_no_shards_returns_single_spec(self):
        """prepare_many without shards returns a single-element list."""
        task = SWEBenchAgenticTask()
        specs = task.prepare_many(_make_params())
        assert isinstance(specs, list)
        assert len(specs) == 1
        assert isinstance(specs[0], TaskSpec)

    def test_prepare_many_with_shards_returns_n_specs(self):
        """prepare_many with N shards returns N TaskSpecs."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array=None,
            shards=[
                {"array": "000-249", "env": {"JOB_ID": "shard1"}},
                {"array": "250-499", "env": {"JOB_ID": "shard2"}},
            ],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 2
        assert all(isinstance(s, TaskSpec) for s in specs)

    def test_prepare_many_shard_array_ranges(self):
        """Each shard gets the correct array range in extra_sbatch."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array=None,
            shards=[
                {"array": "000-249"},
                {"array": "250-499"},
            ],
        )
        specs = task.prepare_many(params)
        extra0 = specs[0].resources.get("extra_sbatch", [])
        extra1 = specs[1].resources.get("extra_sbatch", [])
        assert any("000-249" in e for e in extra0)
        assert any("250-499" in e for e in extra1)

    def test_prepare_many_shard_env_overrides(self):
        """Each shard merges its env dict into the rendered command."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array=None,
            shards=[
                {"array": "000-249", "env": {"JOB_ID": "shard1"}},
                {"array": "250-499", "env": {"JOB_ID": "shard2"}},
            ],
        )
        specs = task.prepare_many(params)
        assert "shard1" in specs[0].command
        assert "shard2" in specs[1].command

    def test_prepare_many_shards_do_not_mutate_original(self):
        """prepare_many must deep-copy params — original should be unmodified."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array=None,
            shards=[
                {"array": "000-249", "env": {"JOB_ID": "s1"}},
                {"array": "250-499", "env": {"JOB_ID": "s2"}},
            ],
        )
        original_shards = params["shards"].copy()
        task.prepare_many(params)
        assert params["shards"] == original_shards

    def test_prepare_many_three_shards(self):
        """Three shards produce three TaskSpecs."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array=None,
            shards=[
                {"array": "000-166"},
                {"array": "167-333"},
                {"array": "334-499"},
            ],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 3


class TestBaseTaskPrepareMany:
    """Tests for BaseTask.prepare_many default implementation."""

    def test_base_task_prepare_many_wraps_single_spec(self):
        """BaseTask.prepare_many delegates to prepare and wraps in list."""
        from devrun.tasks.base import BaseTask

        class DummyTask(BaseTask):
            def prepare(self, params):
                return TaskSpec(command="echo hello")

        task = DummyTask()
        specs = task.prepare_many({})
        assert isinstance(specs, list)
        assert len(specs) == 1
        assert specs[0].command == "echo hello"


class TestConfigVariations:
    """Tests for type1.yaml and type2.yaml config parsing."""

    @pytest.fixture
    def configs_dir(self):
        return Path(__file__).parent.parent / "devrun" / "configs" / "swe_bench_agentic"

    def test_type1_config_parses(self, configs_dir):
        """type1.yaml should parse as valid YAML without errors."""
        path = configs_dir / "type1.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "params" in data
        assert data["params"]["array"] == "000-499"
        assert data["params"]["concurrency_limit"] == 20
        assert data["params"]["partition"] == "mini"

    def test_type2_config_parses(self, configs_dir):
        """type2.yaml should parse as valid YAML without errors."""
        path = configs_dir / "type2.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "params" in data
        assert "shards" in data["params"]
        assert len(data["params"]["shards"]) == 2

    def test_type2_shards_have_array_and_env(self, configs_dir):
        """Each shard in type2.yaml should have array and env keys."""
        path = configs_dir / "type2.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        for shard in data["params"]["shards"]:
            assert "array" in shard
            assert "env" in shard
            assert "JOB_ID" in shard["env"]

    def test_type1_config_compatible_with_task_config(self, configs_dir):
        """type1.yaml merged with default.yaml should be loadable as TaskConfig."""
        default_path = configs_dir / "default.yaml"
        type1_path = configs_dir / "type1.yaml"
        with open(default_path) as f:
            base = yaml.safe_load(f)
        with open(type1_path) as f:
            overlay = yaml.safe_load(f)
        # Simulate merge (overlay params onto base)
        merged = {**base, "params": {**base.get("params", {}), **overlay.get("params", {})}}
        config = TaskConfig(**merged)
        assert config.task == "swe_bench_agentic"
        assert config.params["partition"] == "mini"

    def test_type2_config_compatible_with_task_config(self, configs_dir):
        """type2.yaml merged with default.yaml should be loadable as TaskConfig."""
        default_path = configs_dir / "default.yaml"
        type2_path = configs_dir / "type2.yaml"
        with open(default_path) as f:
            base = yaml.safe_load(f)
        with open(type2_path) as f:
            overlay = yaml.safe_load(f)
        merged = {**base, "params": {**base.get("params", {}), **overlay.get("params", {})}}
        config = TaskConfig(**merged)
        assert config.task == "swe_bench_agentic"
        assert "shards" in config.params

    def test_default_config_parses(self, configs_dir):
        """default.yaml should parse and have run_infer_max_attempts and task_id_format."""
        path = configs_dir / "default.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["params"]["run_infer_max_attempts"] == 5
        assert data["params"]["task_id_format"] == "%03d"
