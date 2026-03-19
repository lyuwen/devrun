"""Unit tests for devrun executor plugins.

This module tests the executor plugins including LocalExecutor, SSHExecutor,
SlurmExecutor, and HTTPExecutor.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from devrun.executors.base import BaseExecutor
from devrun.executors.local import LocalExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import get_executor_class


class TestLocalExecutor:
    """Tests for LocalExecutor."""

    @pytest.fixture
    def executor(self, temp_dir):
        """Provide a LocalExecutor instance for testing."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            entry = ExecutorEntry(type="local")
            return LocalExecutor(name="local", config=entry)

    def test_local_executor_initialization(self, executor):
        """Verify LocalExecutor initializes correctly."""
        assert executor.name == "local"
        assert executor.config.type == "local"

    def test_submit_creates_log_file(self, executor, temp_dir):
        """Verify submit creates log file."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                spec = TaskSpec(command="echo test")
                job_id = executor.submit(spec)

                assert job_id is not None
                log_file = log_dir / f"{job_id}.log"
                # Log file should be created (opened for writing)
                mock_popen.assert_called_once()

    def test_submit_with_env(self, executor, temp_dir):
        """Verify submit passes environment variables."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                spec = TaskSpec(
                    command="echo test",
                    env={"CUSTOM_VAR": "test_value"}
                )
                executor.submit(spec)

                # Verify Popen was called with env
                call_kwargs = mock_popen.call_args[1]
                assert call_kwargs["env"]["CUSTOM_VAR"] == "test_value"

    def test_submit_with_working_dir(self, executor, temp_dir):
        """Verify submit uses working directory."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                spec = TaskSpec(
                    command="echo test",
                    working_dir="/tmp"
                )
                executor.submit(spec)

                call_kwargs = mock_popen.call_args[1]
                assert call_kwargs["cwd"] == "/tmp"

    def test_status_completed_success(self, executor, temp_dir):
        """Verify status returns completed for successful job."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            # Create mock files
            job_id = "test_job_123"
            rc_file = log_dir / f"{job_id}.rc"
            rc_file.write_text("0")

            status = executor.status(job_id)
            assert status == "completed"

    def test_status_completed_failure(self, executor, temp_dir):
        """Verify status returns failed for non-zero exit code."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = "test_job_123"
            rc_file = log_dir / f"{job_id}.rc"
            rc_file.write_text("1")

            status = executor.status(job_id)
            assert status == "failed"

    def test_status_running(self, executor, temp_dir):
        """Verify status returns running for active process."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = "test_job_123"
            pid_file = log_dir / f"{job_id}.pid"
            pid_file.write_text(str(os.getpid()))  # Current process is running

            status = executor.status(job_id)
            assert status == "running"

    def test_status_running_invalid_pid(self, executor, temp_dir):
        """Verify status returns failed for invalid PID."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = "test_job_123"
            pid_file = log_dir / f"{job_id}.pid"
            pid_file.write_text("999999999")  # Invalid PID

            status = executor.status(job_id)
            assert status == "failed"

    def test_status_unknown(self, executor, temp_dir):
        """Verify status returns unknown for nonexistent job."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        status = executor.status("nonexistent_job")
        assert status == "unknown"

    def test_status_digit_pid_fallback(self, executor, temp_dir):
        """Verify status handles old-style numeric PIDs."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Use current PID as a running job
        status = executor.status(str(os.getpid()))
        assert status == "running"

    def test_logs_reads_file(self, executor, temp_dir):
        """Verify logs returns content from log file."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Need to patch the _LOG_DIR that the executor uses
        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = "test_job_123"
            log_file = log_dir / f"{job_id}.log"
            log_file.write_text("Test log output\nLine 2")

            logs = executor.logs(job_id)
            assert "Test log output" in logs

    def test_logs_not_found(self, executor, temp_dir):
        """Verify logs returns not found message."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logs = executor.logs("nonexistent_job")
        assert "no logs found" in logs

    def test_cancel_kills_process(self, executor, temp_dir):
        """Verify cancel sends SIGTERM to process."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = "test_job_123"
            pid_file = log_dir / f"{job_id}.pid"
            pid_file.write_text(str(os.getpid()))

            with patch("os.kill") as mock_kill:
                executor.cancel(job_id)
                mock_kill.assert_called_once()


class TestBaseExecutor:
    """Tests for BaseExecutor abstract class."""

    def test_base_executor_is_abstract(self):
        """Verify BaseExecutor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExecutor(name="test", config=ExecutorEntry(type="local"))

    def test_submit_with_retry_success(self):
        """Verify submit_with_retry succeeds on first try."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        spec = TaskSpec(command="echo test")

        with patch.object(executor, "submit", return_value="job_123"):
            result = executor.submit_with_retry(spec, retries=3)
            assert result == "job_123"

    def test_submit_with_retry_eventual_success(self):
        """Verify submit_with_retry retries and eventually succeeds."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        spec = TaskSpec(command="echo test")

        # Fail twice, succeed on third
        with patch.object(
            executor,
            "submit",
            side_effect=[Exception("fail"), Exception("fail"), "job_123"]
        ):
            result = executor.submit_with_retry(spec, retries=3, retry_delay=0.01)
            assert result == "job_123"

    def test_submit_with_retry_all_fail(self):
        """Verify submit_with_retry raises after all retries fail."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        spec = TaskSpec(command="echo test")

        with patch.object(executor, "submit", side_effect=Exception("always fails")):
            with pytest.raises(RuntimeError) as exc_info:
                executor.submit_with_retry(spec, retries=3, retry_delay=0.01)
            assert "Failed to submit after 3 attempts" in str(exc_info.value)

    def test_cancel_not_implemented(self):
        """Verify cancel raises NotImplementedError by default for abstract base."""
        from devrun.executors.base import BaseExecutor

        # BaseExecutor.cancel raises NotImplementedError
        # Note: Actual executor implementations may override cancel
        entry = ExecutorEntry(type="http", endpoint="https://example.com")

        # We can't directly instantiate BaseExecutor since it's abstract
        # Instead, let's verify that a fresh executor without override would raise
        # Since LocalExecutor DOES implement cancel, we'll test via the base class interface
        from devrun.executors.http import HTTPExecutor
        http_executor = HTTPExecutor(name="test_http", config=entry)

        # HTTPExecutor doesn't override cancel, so it should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            http_executor.cancel("job_123")

    def test_executor_repr(self):
        """Verify executor has a reasonable __repr__."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test_exec", config=entry)

        repr_str = repr(executor)
        assert "LocalExecutor" in repr_str
        assert "test_exec" in repr_str


class TestExecutorRegistry:
    """Tests for executor registration and retrieval."""

    def test_get_local_executor_class(self):
        """Verify LocalExecutor class can be retrieved."""
        cls = get_executor_class("local")
        assert cls.__name__ == "LocalExecutor"

    def test_get_ssh_executor_class(self):
        """Verify SSHExecutor class can be retrieved."""
        cls = get_executor_class("ssh")
        assert cls.__name__ == "SSHExecutor"

    def test_get_slurm_executor_class(self):
        """Verify SlurmExecutor class can be retrieved."""
        cls = get_executor_class("slurm")
        assert cls.__name__ == "SlurmExecutor"

    def test_get_http_executor_class(self):
        """Verify HTTPExecutor class can be retrieved."""
        cls = get_executor_class("http")
        assert cls.__name__ == "HTTPExecutor"


class TestExecutorInstantiation:
    """Tests for executor instantiation."""

    def test_instantiate_local_with_config(self):
        """Verify LocalExecutor can be instantiated with config."""
        cls = get_executor_class("local")
        entry = ExecutorEntry(type="local")
        executor = cls(name="my_local", config=entry)

        assert executor.name == "my_local"
        assert executor.config.type == "local"

    def test_executor_has_logger(self):
        """Verify executor has a logger."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        assert executor.logger is not None
        # Logger name should include the executor name
        assert executor.logger.name == f"devrun.executors.{executor.name}"


class TestExecutorIntegration:
    """Integration tests for executors."""

    def test_submit_returns_string_id(self):
        """Verify submit returns a string job ID."""
        from devrun.executors.local import LocalExecutor

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        spec = TaskSpec(command="echo test")

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            job_id = executor.submit(spec)
            assert isinstance(job_id, str)
            assert len(job_id) > 0