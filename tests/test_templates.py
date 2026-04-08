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
            "base_url": "http://localhost:8000",
            "api_key": "sk-test",
            "temperature": "0.7",
            "top_p": "0.95",
            "run_name": "run1",
            "max_iterations": 100,
            "ds_dir": "__mnt__data__SWE-bench_Verified-test",
            "task_id_format": "%03d",
            "output_dir": "logs/run1",
            "max_attempts": 5,
            "script": "benchmarks/swebench/run_infer.py",
            "llm_config": ".llm_config/test-model.json",
            "split": "test",
            "select_dir": "job_array",
            "workspace": "docker",
            "extra_flags": ["--use-legacy-tools", "--bind-dev-sdk"],
            "env_vars": {"OH_SEND_REASONING_CONTENT": "yes"},
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

    def test_contains_cd_working_dir(self, base_params):
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        assert "cd " in result
        assert "/remote/project" in result

    def test_no_working_dir_omits_cd(self, base_params):
        base_params["working_dir"] = None
        result = render_template("swe_bench_agentic.sh.j2", **base_params)
        lines = result.strip().splitlines()
        first_real = next(l for l in lines if l.strip())
        assert not first_real.strip().startswith("cd ")

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
