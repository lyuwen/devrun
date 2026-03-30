"""Tests for PythonEnv model and BaseExecutor helper methods."""

from __future__ import annotations

import pytest

from devrun.models import PythonEnv
from devrun.executors.base import BaseExecutor


# ---------------------------------------------------------------------------
# Helpers — use BaseExecutor's static methods directly
# ---------------------------------------------------------------------------

resolve = BaseExecutor._resolve_python_env
to_lines = BaseExecutor._env_to_shell_lines


# ---------------------------------------------------------------------------
# PythonEnv model
# ---------------------------------------------------------------------------

class TestPythonEnvModel:
    def test_all_defaults_are_empty(self):
        env = PythonEnv()
        assert env.venv is None
        assert env.conda is None
        assert env.modules == []
        assert env.setup_commands == []

    def test_full_construction(self):
        env = PythonEnv(
            venv="/opt/venv",
            conda="myenv",
            modules=["python/3.11", "cuda/12.1"],
            setup_commands=["export FOO=bar"],
        )
        assert env.venv == "/opt/venv"
        assert env.conda == "myenv"
        assert env.modules == ["python/3.11", "cuda/12.1"]
        assert env.setup_commands == ["export FOO=bar"]


# ---------------------------------------------------------------------------
# _env_to_shell_lines
# ---------------------------------------------------------------------------

class TestEnvToShellLines:
    def test_empty_env_produces_no_lines(self):
        assert to_lines(PythonEnv()) == []

    def test_venv_root_path(self):
        lines = to_lines(PythonEnv(venv="/opt/venv"))
        assert lines == ["source /opt/venv/bin/activate"]

    def test_venv_direct_activate_path(self):
        lines = to_lines(PythonEnv(venv="/opt/venv/bin/activate"))
        assert lines == ["source /opt/venv/bin/activate"]

    def test_conda(self):
        lines = to_lines(PythonEnv(conda="myenv"))
        assert lines == ["conda activate myenv"]

    def test_single_module(self):
        lines = to_lines(PythonEnv(modules=["python/3.11"]))
        assert lines == ["module load python/3.11"]

    def test_multiple_modules(self):
        lines = to_lines(PythonEnv(modules=["python/3.11", "cuda/12.1"]))
        assert lines == ["module load python/3.11", "module load cuda/12.1"]

    def test_setup_commands(self):
        lines = to_lines(PythonEnv(setup_commands=["export FOO=bar", "ulimit -n 65536"]))
        assert lines == ["export FOO=bar", "ulimit -n 65536"]

    def test_order_modules_venv_conda_setup(self):
        """modules → venv → conda → setup_commands."""
        env = PythonEnv(
            modules=["python/3.11"],
            venv="/opt/venv",
            conda="myenv",
            setup_commands=["export PYTHONUNBUFFERED=1"],
        )
        lines = to_lines(env)
        assert lines == [
            "module load python/3.11",
            "source /opt/venv/bin/activate",
            "conda activate myenv",
            "export PYTHONUNBUFFERED=1",
        ]


# ---------------------------------------------------------------------------
# _resolve_python_env (merge logic)
# ---------------------------------------------------------------------------

class TestResolvePythonEnv:
    def test_both_none_returns_none(self):
        assert resolve(None, None) is None

    def test_executor_only(self):
        base = PythonEnv(conda="base_env")
        result = resolve(base, None)
        assert result is not None
        assert result.conda == "base_env"

    def test_task_only(self):
        result = resolve(None, PythonEnv(venv="/task/venv"))
        assert result is not None
        assert result.venv == "/task/venv"

    def test_task_conda_overrides_executor(self):
        result = resolve(
            PythonEnv(conda="executor_env"),
            PythonEnv(conda="task_env"),
        )
        assert result is not None
        assert result.conda == "task_env"

    def test_task_venv_overrides_executor(self):
        result = resolve(
            PythonEnv(venv="/executor/venv"),
            PythonEnv(venv="/task/venv"),
        )
        assert result is not None
        assert result.venv == "/task/venv"

    def test_executor_conda_kept_when_task_has_none(self):
        result = resolve(
            PythonEnv(conda="executor_env"),
            PythonEnv(venv="/task/venv"),  # no conda
        )
        assert result is not None
        assert result.conda == "executor_env"
        assert result.venv == "/task/venv"

    def test_task_modules_replace_executor_modules(self):
        result = resolve(
            PythonEnv(modules=["python/3.10"]),
            PythonEnv(modules=["python/3.11", "cuda/12.1"]),
        )
        assert result is not None
        assert result.modules == ["python/3.11", "cuda/12.1"]

    def test_executor_modules_kept_when_task_has_none(self):
        result = resolve(
            PythonEnv(modules=["python/3.11"]),
            PythonEnv(conda="task_env"),  # no modules
        )
        assert result is not None
        assert result.modules == ["python/3.11"]

    def test_setup_commands_concatenated(self):
        result = resolve(
            PythonEnv(setup_commands=["export FOO=1"]),
            PythonEnv(setup_commands=["export BAR=2"]),
        )
        assert result is not None
        assert result.setup_commands == ["export FOO=1", "export BAR=2"]

    def test_executor_setup_commands_only(self):
        result = resolve(
            PythonEnv(setup_commands=["export FOO=1"]),
            PythonEnv(),  # empty task
        )
        assert result is not None
        assert result.setup_commands == ["export FOO=1"]

    def test_task_setup_commands_only(self):
        result = resolve(
            PythonEnv(),  # empty executor
            PythonEnv(setup_commands=["export BAR=2"]),
        )
        assert result is not None
        assert result.setup_commands == ["export BAR=2"]

    def test_full_merge(self):
        executor_env = PythonEnv(
            conda="base_env",
            modules=["python/3.10"],
            setup_commands=["export PYTHONUNBUFFERED=1"],
        )
        task_env = PythonEnv(
            conda="task_env",
            modules=["python/3.11", "cuda/12.1"],
            setup_commands=["export MY_VAR=hello"],
        )
        result = resolve(executor_env, task_env)
        assert result is not None
        assert result.conda == "task_env"
        assert result.modules == ["python/3.11", "cuda/12.1"]
        assert result.setup_commands == ["export PYTHONUNBUFFERED=1", "export MY_VAR=hello"]
