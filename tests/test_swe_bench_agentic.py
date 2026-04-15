"""Unit tests for SWEBenchAgenticTask."""
from __future__ import annotations
import json
import pytest
import yaml
from pathlib import Path
from devrun.tasks.swe_bench_agentic import (
    SWEBenchAgenticTask,
    _parse_array_range,
    _compute_shard_ranges,
    _format_llm_config,
)
from devrun.models import TaskConfig, TaskSpec


def _make_params(**kwargs):
    base = {
        "llm_config": "/fake/config.json",
        "dataset": "/fake/dataset",
        "array": "000-004",
    }
    base.update(kwargs)
    return base


def _extract_heredoc_json(command: str) -> dict:
    """Extract and parse JSON from the __DEVRUN_LLM_EOF__ heredoc block."""
    lines = command.split("\n")
    in_heredoc = False
    json_lines = []
    for line in lines:
        if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
            in_heredoc = True
            continue
        if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
            break
        if in_heredoc:
            json_lines.append(line)
    return json.loads("\n".join(json_lines))


class TestHelpers:
    """Tests for module-level helper functions."""

    # --- _parse_array_range ---

    def test_parse_array_range_3_digit(self):
        assert _parse_array_range("000-499") == (0, 499, 3)

    def test_parse_array_range_4_digit(self):
        assert _parse_array_range("0000-0099") == (0, 99, 4)

    def test_parse_array_range_nonzero_start(self):
        assert _parse_array_range("050-149") == (50, 149, 3)

    def test_parse_array_range_invalid_no_dash(self):
        with pytest.raises(ValueError, match="Invalid array range"):
            _parse_array_range("000499")

    def test_parse_array_range_invalid_too_many_dashes(self):
        with pytest.raises(ValueError, match="Invalid array range"):
            _parse_array_range("0-1-2")

    # --- _compute_shard_ranges ---

    def test_compute_shard_ranges_even(self):
        # 500 / 2 = 250 each
        assert _compute_shard_ranges(0, 499, 2, 3) == ["000-249", "250-499"]

    def test_compute_shard_ranges_3_way(self):
        # 500 / 3 = 166 r 2 → first 2 get 167, last gets 166
        result = _compute_shard_ranges(0, 499, 3, 3)
        assert result == ["000-166", "167-333", "334-499"]

    def test_compute_shard_ranges_uneven_7(self):
        # 100 / 7 = 14 r 2 → first 2 get 15, remaining 5 get 14
        result = _compute_shard_ranges(0, 99, 7, 3)
        assert result == [
            "000-014", "015-029", "030-043", "044-057",
            "058-071", "072-085", "086-099",
        ]

    def test_compute_shard_ranges_single(self):
        assert _compute_shard_ranges(0, 499, 1, 3) == ["000-499"]

    def test_compute_shard_ranges_n_greater_than_total_raises(self):
        with pytest.raises(ValueError, match="Cannot create"):
            _compute_shard_ranges(0, 4, 10, 3)

    def test_compute_shard_ranges_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _compute_shard_ranges(0, 99, 0, 3)

    def test_compute_shard_ranges_4_digit_padding(self):
        result = _compute_shard_ranges(0, 99, 2, 4)
        assert result == ["0000-0049", "0050-0099"]

    # --- _format_llm_config ---

    def test_format_llm_config_replaces_placeholder(self):
        config = {"base_url": "https://api.example.com/{JOB_ID}/v1"}
        result = _format_llm_config(config, {"JOB_ID": "if-abc123"})
        assert result == {"base_url": "https://api.example.com/if-abc123/v1"}

    def test_format_llm_config_nested_dict(self):
        config = {"outer": {"inner": "{JOB_ID}"}}
        result = _format_llm_config(config, {"JOB_ID": "val"})
        assert result == {"outer": {"inner": "val"}}

    def test_format_llm_config_list_values(self):
        config = {"urls": ["{JOB_ID}/a", "{JOB_ID}/b"]}
        result = _format_llm_config(config, {"JOB_ID": "x"})
        assert result == {"urls": ["x/a", "x/b"]}

    def test_format_llm_config_non_string_untouched(self):
        config = {"temp": 1.0, "top_p": 0.95, "log": True, "count": 42}
        result = _format_llm_config(config, {"JOB_ID": "x"})
        assert result == config

    def test_format_llm_config_missing_placeholder_empty_string(self):
        config = {"url": "https://{MISSING}/v1"}
        result = _format_llm_config(config, {})
        assert result == {"url": "https:///v1"}


class TestSWEBenchAgenticTask:
    def test_prepare_requires_dataset(self):
        task = SWEBenchAgenticTask()
        with pytest.raises(ValueError, match="dataset"):
            task.prepare({"llm_config": "/fake/config.json"})

    def test_prepare_requires_llm_config_or_model_name(self):
        task = SWEBenchAgenticTask()
        with pytest.raises(ValueError, match="model_name"):
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


class TestInstancesAutoSharding:
    """Tests for prepare_many with instances-based auto-sharding."""

    def test_instances_3_way_split(self):
        """3 instances over '000-499' → 3 specs with correct array ranges."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="000-499",
            instances=[
                {"JOB_ID": "if-abc123"},
                {"JOB_ID": "if-def456"},
                {"JOB_ID": "if-ghi789"},
            ],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 3
        extra0 = specs[0].resources.get("extra_sbatch", [])
        extra1 = specs[1].resources.get("extra_sbatch", [])
        extra2 = specs[2].resources.get("extra_sbatch", [])
        assert any("000-166" in e for e in extra0)
        assert any("167-333" in e for e in extra1)
        assert any("334-499" in e for e in extra2)

    def test_instances_2_way_split(self):
        """2 instances over '000-499' → '000-249', '250-499'."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="000-499",
            instances=[
                {"JOB_ID": "s1"},
                {"JOB_ID": "s2"},
            ],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 2
        extra0 = specs[0].resources.get("extra_sbatch", [])
        extra1 = specs[1].resources.get("extra_sbatch", [])
        assert any("000-249" in e for e in extra0)
        assert any("250-499" in e for e in extra1)

    def test_instances_uneven(self):
        """7 instances over '000-099' (100/7=14r2) → correct remainder distribution."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="000-099",
            instances=[{"JOB_ID": f"j{i}"} for i in range(7)],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 7
        # First 2 get 15 items, remaining 5 get 14 items
        expected_ranges = [
            "000-014", "015-029", "030-043", "044-057",
            "058-071", "072-085", "086-099",
        ]
        for spec, expected in zip(specs, expected_ranges):
            extra = spec.resources.get("extra_sbatch", [])
            assert any(expected in e for e in extra), f"Expected {expected} in {extra}"

    def test_instances_env_merged(self):
        """Each instance's JOB_ID ends up in the corresponding spec's env."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="000-499",
            instances=[
                {"JOB_ID": "if-abc123"},
                {"JOB_ID": "if-def456"},
            ],
        )
        specs = task.prepare_many(params)
        assert specs[0].env["JOB_ID"] == "if-abc123"
        assert specs[1].env["JOB_ID"] == "if-def456"

    def test_instances_pad_width(self):
        """'0000-0099' preserves 4-digit padding in shard ranges."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="0000-0099",
            instances=[
                {"JOB_ID": "s1"},
                {"JOB_ID": "s2"},
            ],
        )
        specs = task.prepare_many(params)
        extra0 = specs[0].resources.get("extra_sbatch", [])
        extra1 = specs[1].resources.get("extra_sbatch", [])
        assert any("0000-0049" in e for e in extra0)
        assert any("0050-0099" in e for e in extra1)

    def test_instances_single(self):
        """1 instance → single spec with full array range."""
        task = SWEBenchAgenticTask()
        params = _make_params(
            array="000-499",
            instances=[{"JOB_ID": "only"}],
        )
        specs = task.prepare_many(params)
        assert len(specs) == 1
        extra = specs[0].resources.get("extra_sbatch", [])
        assert any("000-499" in e for e in extra)

    def test_no_instances_returns_single_spec(self):
        """prepare_many without instances returns a single-element list."""
        task = SWEBenchAgenticTask()
        specs = task.prepare_many(_make_params())
        assert isinstance(specs, list)
        assert len(specs) == 1
        assert isinstance(specs[0], TaskSpec)

    def test_instances_require_array(self):
        """instances without array raises ValueError."""
        task = SWEBenchAgenticTask()
        params = _make_params(array=None, instances=[{"JOB_ID": "x"}])
        with pytest.raises(ValueError, match="array is required"):
            task.prepare_many(params)

    def test_job_ids_shorthand(self):
        """Comma-separated job_ids string expands into instances."""
        task = SWEBenchAgenticTask()
        params = _make_params(array="000-499", job_ids="if-abc,if-def,if-ghi")
        specs = task.prepare_many(params)
        assert len(specs) == 3
        assert specs[0].env["JOB_ID"] == "if-abc"
        assert specs[1].env["JOB_ID"] == "if-def"
        assert specs[2].env["JOB_ID"] == "if-ghi"

    def test_job_ids_whitespace_trimmed(self):
        """Whitespace around comma-separated job_ids is trimmed."""
        task = SWEBenchAgenticTask()
        params = _make_params(array="000-099", job_ids="id1 , id2")
        specs = task.prepare_many(params)
        assert specs[0].env["JOB_ID"] == "id1"
        assert specs[1].env["JOB_ID"] == "id2"


class TestInlineLlmConfig:
    """Tests for inline dict llm_config (heredoc mode)."""

    def test_inline_llm_config_heredoc(self):
        """Dict llm_config → command contains __DEVRUN_LLM_EOF__ and JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(
            llm_config={"model": "openai/my-model", "api_key": "sk-xxx"},
        ))
        assert "__DEVRUN_LLM_EOF__" in spec.command
        assert '"model"' in spec.command
        assert '"openai/my-model"' in spec.command

    def test_inline_llm_config_format_job_id(self):
        """{JOB_ID} in base_url resolved from env vars."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(
            llm_config={
                "model": "openai/my-model",
                "base_url": "https://api.example.com/{JOB_ID}/v1",
            },
            env={"JOB_ID": "if-abc123"},
        ))
        assert "https://api.example.com/if-abc123/v1" in spec.command
        # The unresolved placeholder should NOT appear
        assert "{JOB_ID}" not in spec.command

    def test_inline_llm_config_no_local_file(self, tmp_path):
        """Dict mode does NOT create local .llm_config/ directory."""
        task = SWEBenchAgenticTask()
        task.prepare(_make_params(
            llm_config={"model": "openai/my-model"},
        ))
        # No .llm_config directory should be created in cwd
        assert not (tmp_path / ".llm_config").exists()

    def test_inline_llm_config_cleanup(self):
        """Command contains rm -f for temp file cleanup."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(
            llm_config={"model": "openai/my-model"},
        ))
        assert "rm -f" in spec.command

    def test_file_llm_config_unchanged(self):
        """String path → used directly in command, no heredoc."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(llm_config="/path/to/config.json"))
        assert "__DEVRUN_LLM_EOF__" not in spec.command
        assert "rm -f" not in spec.command
        # Path appears in _LLM_CONFIG assignment (shell_quote may or may not add quotes)
        assert "_LLM_CONFIG=" in spec.command
        assert "/path/to/config.json" in spec.command

    def test_anthropic_config_passthrough(self):
        """litellm_extra_body dict preserved in serialized JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(_make_params(
            llm_config={
                "model": "anthropic/claude-opus-4-6-thinking-hz",
                "base_url": "http://10.200.95.16:30300",
                "api_key": "sk-xxx",
                "litellm_extra_body": {
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "output_config": {"effort": "max"},
                },
                "log_completions": True,
            },
        ))
        # Parse the JSON from the heredoc to verify structure
        assert '"litellm_extra_body"' in spec.command
        assert '"thinking"' in spec.command
        assert '"output_config"' in spec.command
        # Verify the JSON is valid by extracting it
        assert "__DEVRUN_LLM_EOF__" in spec.command
        # Extract JSON between heredoc markers
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed["litellm_extra_body"]["thinking"]["type"] == "adaptive"
        assert parsed["litellm_extra_body"]["output_config"]["effort"] == "max"
        assert parsed["log_completions"] is True


class TestShorthandLlmConfig:
    """Tests for shorthand llm_config auto-build from top-level params."""

    def _shorthand_params(self, **kwargs):
        """Base params without llm_config — forces shorthand path."""
        base = {"dataset": "/fake/dataset", "array": "000-004"}
        base.update(kwargs)
        return base

    def test_shorthand_model_name_only(self):
        """model_name alone → heredoc with {\"model\": \"openai/my-model\"}."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(model_name="openai/my-model"))
        assert "__DEVRUN_LLM_EOF__" in spec.command
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed == {"model": "openai/my-model"}

    def test_shorthand_model_name_and_api_key(self):
        """model_name + api_key → both in heredoc JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", api_key="sk-test123",
        ))
        assert "__DEVRUN_LLM_EOF__" in spec.command
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed["model"] == "openai/my-model"
        assert parsed["api_key"] == "sk-test123"

    def test_shorthand_model_name_and_base_url(self):
        """model_name + base_url → both in JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", base_url="http://localhost:8000/v1",
        ))
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed["model"] == "openai/my-model"
        assert parsed["base_url"] == "http://localhost:8000/v1"

    def test_shorthand_all_params(self):
        """model_name + api_key + base_url + temperature + top_p → all in JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model",
            api_key="sk-xxx",
            base_url="http://localhost:8000/v1",
            temperature="0.7",
            top_p="0.95",
        ))
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed["model"] == "openai/my-model"
        assert parsed["api_key"] == "sk-xxx"
        assert parsed["base_url"] == "http://localhost:8000/v1"
        assert parsed["temperature"] == "0.7"
        assert parsed["top_p"] == "0.95"

    def test_shorthand_does_not_override_explicit_dict(self):
        """When llm_config dict IS provided, shorthand params are ignored."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/ignored-model",
            api_key="sk-ignored",
            llm_config={"model": "openai/explicit-model", "api_key": "sk-explicit"},
        ))
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert parsed["model"] == "openai/explicit-model"
        assert parsed["api_key"] == "sk-explicit"
        assert "openai/ignored-model" not in json.dumps(parsed)

    def test_shorthand_does_not_override_explicit_path(self):
        """When llm_config is a string path, shorthand params are ignored."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/ignored-model",
            api_key="sk-ignored",
            llm_config="/path/to/config.json",
        ))
        assert "__DEVRUN_LLM_EOF__" not in spec.command
        assert "/path/to/config.json" in spec.command

    def test_no_llm_config_no_model_name_raises(self):
        """Neither llm_config nor model_name → ValueError mentioning model_name."""
        task = SWEBenchAgenticTask()
        with pytest.raises(ValueError, match="model_name"):
            task.prepare(self._shorthand_params())

    def test_shorthand_empty_api_key_excluded(self):
        """api_key=\"\" should not appear in the built dict."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", api_key="",
        ))
        lines = spec.command.split("\n")
        in_heredoc = False
        json_lines = []
        for line in lines:
            if "__DEVRUN_LLM_EOF__" in line and not in_heredoc:
                in_heredoc = True
                continue
            if "__DEVRUN_LLM_EOF__" in line and in_heredoc:
                break
            if in_heredoc:
                json_lines.append(line)
        parsed = json.loads("\n".join(json_lines))
        assert "api_key" not in parsed
        assert parsed == {"model": "openai/my-model"}

    def test_shorthand_temperature_zero_included(self):
        """temperature=0 is valid and should be included (not filtered by falsy check)."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", temperature=0,
        ))
        parsed = _extract_heredoc_json(spec.command)
        assert "temperature" in parsed
        assert parsed["temperature"] == 0

    def test_shorthand_log_completions(self):
        """model_name + log_completions=True → JSON has log_completions: true."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", log_completions=True,
        ))
        parsed = _extract_heredoc_json(spec.command)
        assert parsed["log_completions"] is True

    def test_shorthand_litellm_extra_body(self):
        """model_name + litellm_extra_body dict → nested dict in JSON."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model",
            litellm_extra_body={"thinking": {"enabled": True}},
        ))
        parsed = _extract_heredoc_json(spec.command)
        assert parsed["litellm_extra_body"] == {"thinking": {"enabled": True}}

    def test_shorthand_anthropic_style(self):
        """Full Anthropic-style shorthand with litellm_extra_body preserved."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="anthropic/claude-opus-4-6-thinking-hz",
            api_key="sk-ant-xxx",
            litellm_extra_body={
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "max"},
            },
        ))
        parsed = _extract_heredoc_json(spec.command)
        assert parsed["model"] == "anthropic/claude-opus-4-6-thinking-hz"
        assert parsed["api_key"] == "sk-ant-xxx"
        assert parsed["litellm_extra_body"]["thinking"]["type"] == "adaptive"
        assert parsed["litellm_extra_body"]["thinking"]["display"] == "summarized"
        assert parsed["litellm_extra_body"]["output_config"]["effort"] == "max"

    def test_shorthand_log_completions_false_included(self):
        """log_completions=False is included (not filtered by falsy check)."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", log_completions=False,
        ))
        parsed = _extract_heredoc_json(spec.command)
        assert "log_completions" in parsed
        assert parsed["log_completions"] is False

    def test_shorthand_numeric_temperature_no_crash(self):
        """Float temperature=0.7 doesn't crash shell_quote; appears in JSON and shell var."""
        task = SWEBenchAgenticTask()
        spec = task.prepare(self._shorthand_params(
            model_name="openai/my-model", temperature=0.7,
        ))
        # Heredoc JSON preserves the float
        parsed = _extract_heredoc_json(spec.command)
        assert parsed["temperature"] == 0.7
        # Template exports TEMPERATURE as a string shell var
        assert "TEMPERATURE=" in spec.command


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
        # Verify inline llm_config is present
        assert isinstance(data["params"]["llm_config"], dict)
        assert "model" in data["params"]["llm_config"]

    def test_type2_config_parses(self, configs_dir):
        """type2.yaml should parse as valid YAML without errors."""
        path = configs_dir / "type2.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "params" in data
        assert "instances" in data["params"]
        assert len(data["params"]["instances"]) == 2

    def test_type2_instances_have_job_id(self, configs_dir):
        """Each instance in type2.yaml should have a JOB_ID key."""
        path = configs_dir / "type2.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        for instance in data["params"]["instances"]:
            assert "JOB_ID" in instance

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
        assert "instances" in config.params

    def test_default_config_parses(self, configs_dir):
        """default.yaml should parse and have run_infer_max_attempts and task_id_format."""
        path = configs_dir / "default.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["params"]["run_infer_max_attempts"] == 5
        assert data["params"]["task_id_format"] == "%03d"
