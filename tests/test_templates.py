"""Unit tests for Jinja2 template rendering utility."""
from __future__ import annotations

import jinja2
import pytest
from devrun.utils.templates import render_template, _get_jinja_env


class TestRenderTemplate:
    def test_render_nonexistent_template_raises(self):
        with pytest.raises(Exception):
            render_template("nonexistent.j2")

    def test_shell_quote_filter_available(self):
        """The shell_quote filter must be registered."""
        env = _get_jinja_env()
        assert "shell_quote" in env.filters

    def test_shell_quote_filter_quotes_spaces(self):
        env = _get_jinja_env()
        tmpl = env.from_string("{{ val | shell_quote }}")
        result = tmpl.render(val="hello world")
        assert result == "'hello world'"

    def test_shell_quote_filter_quotes_special_chars(self):
        env = _get_jinja_env()
        tmpl = env.from_string("{{ val | shell_quote }}")
        result = tmpl.render(val="it's a test")
        assert "'" in result  # shlex.quote wraps in quotes

    def test_strict_undefined_raises_on_missing(self):
        """StrictUndefined should raise when a variable is missing."""
        env = _get_jinja_env()
        tmpl = env.from_string("Hello {{ missing_var }}")
        with pytest.raises(jinja2.UndefinedError):
            tmpl.render()


class TestSWEBenchAgenticTemplate:
    """Tests for the swe_bench_agentic.sh.j2 template."""

    @pytest.fixture
    def base_params(self):
        return {
            "working_dir": "/remote/project",
            "env_commands": ["source /opt/conda/bin/activate", "conda activate swebench"],
            "dataset": "/mnt/data/SWE-bench_Verified",
            "model_name": "test-model",
            "run_name": "run1",
            "max_iterations": 100,
            "ds_dir": "__mnt__data__SWE-bench_Verified-test",
            "task_id_format": "%03d",
            "output_dir": "logs/run1",
            "max_attempts": 5,
            "run_infer_max_attempts": 5,
            "script": "benchmarks/swebench/run_infer.py",
            "llm_config": ".llm_config/test-model.json",
            "split": "test",
            "select_dir": "job_array",
            "workspace": "docker",
            "extra_flags": ["--use-legacy-tools", "--bind-dev-sdk"],
            "env_vars": {"OH_SEND_REASONING_CONTENT": "yes"},
            "git_safe_dirs": [],
            "llm_config_content": None,
        }

    def test_contains_retry_loop(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "for attempt in {1..5}" in result

    def test_max_attempts_configurable(self, base_params):
        base_params["max_attempts"] = 3
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "for attempt in {1..3}" in result

    def test_contains_completion_check(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "output.jsonl" in result
        assert "LINE_COUNT" in result

    def test_contains_failed_run_archiving(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "/old/" in result
        assert "date" in result

    def test_contains_ds_dir(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "__mnt__data__SWE-bench_Verified-test" in result

    def test_template_does_not_emit_cd(self, base_params):
        """cd is handled by generate_sbatch_script, not the template."""
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        for line in result.strip().splitlines():
            assert not line.strip().startswith("cd ")

    def test_set_x_present_set_e_absent(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "set -x" in result
        assert "set -e" not in result

    def test_exit_zero_on_failure(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "exit 0" in result

    def test_env_commands_rendered(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "source /opt/conda/bin/activate" in result
        assert "conda activate swebench" in result

    def test_python_command_rendered(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "python" in result
        assert "run_infer.py" in result
        assert "--max-iterations 100" in result
        assert "--workspace docker" in result

    def test_extra_flags_rendered(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "--use-legacy-tools" in result
        assert "--bind-dev-sdk" in result

    def test_printf_format(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "%03d" in result

    def test_env_vars_rendered(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "OH_SEND_REASONING_CONTENT" in result

    # ------------------------------------------------------------------
    # --max-attempts flag (run_infer_max_attempts)
    # ------------------------------------------------------------------

    def test_run_infer_max_attempts_default(self, base_params):
        """--max-attempts flag should render with the default value of 5."""
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "--max-attempts 5" in result

    def test_run_infer_max_attempts_custom(self, base_params):
        """--max-attempts flag should render with a custom value."""
        base_params["run_infer_max_attempts"] = 10
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "--max-attempts 10" in result

    def test_run_infer_max_attempts_distinct_from_retry_loop(self, base_params):
        """run_infer_max_attempts and max_attempts are independent."""
        base_params["max_attempts"] = 3
        base_params["run_infer_max_attempts"] = 7
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "for attempt in {1..3}" in result
        assert "--max-attempts 7" in result

    # ------------------------------------------------------------------
    # git safe.directory
    # ------------------------------------------------------------------

    def test_git_safe_dirs_empty(self, base_params):
        """No git config lines when git_safe_dirs is empty."""
        base_params["git_safe_dirs"] = []
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "git config --global --add safe.directory" not in result

    def test_git_safe_dirs_single(self, base_params):
        """A single git safe.directory entry renders one config line."""
        base_params["git_safe_dirs"] = ["/opt/repo"]
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "git config --global --add safe.directory" in result
        assert "/opt/repo" in result

    def test_git_safe_dirs_multiple(self, base_params):
        """Multiple git safe.directory entries render multiple config lines."""
        base_params["git_safe_dirs"] = ["/opt/repo1", "/opt/repo2", "/home/user/project"]
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        for d in base_params["git_safe_dirs"]:
            assert d in result
        count = result.count("git config --global --add safe.directory")
        assert count == 3

    def test_git_safe_dirs_before_exports(self, base_params):
        """git safe.directory lines must appear before export statements."""
        base_params["git_safe_dirs"] = ["/opt/repo"]
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        git_line_pos = result.index("git config --global --add safe.directory")
        export_pos = result.index("export DATASET=")
        assert git_line_pos < export_pos

    def test_git_safe_dirs_after_env_commands(self, base_params):
        """git safe.directory lines must appear after env_commands."""
        base_params["git_safe_dirs"] = ["/opt/repo"]
        base_params["env_commands"] = ["source /opt/conda/bin/activate"]
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        env_cmd_pos = result.index("source /opt/conda/bin/activate")
        git_line_pos = result.index("git config --global --add safe.directory")
        assert env_cmd_pos < git_line_pos
