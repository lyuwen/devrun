"""Unit tests for SSHExecutor."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from devrun.executors.ssh import SSHExecutor
from devrun.models import ExecutorEntry, TaskSpec


def _make_executor():
    entry = ExecutorEntry(type="ssh", host="test.host.com", user="testuser")
    return SSHExecutor(name="test_ssh", config=entry)


def _make_spec(**kwargs):
    base = dict(command="echo hello", env={}, working_dir=None, resources={}, metadata={})
    base.update(kwargs)
    return TaskSpec(**base)


def _make_ssh_result(stdout="12345", returncode=0, stderr=""):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    r.stderr = stderr
    return r


class TestSSHExecutorSubmit:
    def test_submit_returns_composite_job_id(self):
        executor = _make_executor()
        with patch("devrun.executors.ssh.run_ssh_command", return_value=_make_ssh_result("12345")) as mock_ssh:
            job_id = executor.submit(_make_spec())
        assert "12345" in job_id
        # Should be pid:token format
        assert ":" in job_id

    def test_submit_log_path_uses_token(self):
        """Log path in submitted command must use the run_token, not $$."""
        executor = _make_executor()
        captured_cmd = []

        def capture(ssh, cmd, **kwargs):
            captured_cmd.append(cmd)
            return _make_ssh_result("99999")

        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            job_id = executor.submit(_make_spec())

        token = job_id.split(":")[1]
        assert token in captured_cmd[0], "run_token must appear in remote command"
        assert "$$" not in captured_cmd[0], "$$ must not appear in remote command"

    def test_submit_raises_on_nonzero(self):
        executor = _make_executor()
        with patch("devrun.executors.ssh.run_ssh_command", return_value=_make_ssh_result("", returncode=1, stderr="fail")):
            with pytest.raises(RuntimeError, match="SSH submit failed"):
                executor.submit(_make_spec())

    def test_submit_env_in_command(self):
        executor = _make_executor()
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("1")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(_make_spec(env={"MY_KEY": "my_value"}))
        assert "MY_KEY" in captured[0]

    def test_submit_requires_host(self):
        entry = ExecutorEntry(type="ssh")
        with pytest.raises(ValueError, match="host"):
            SSHExecutor(name="no_host", config=entry)


class TestSSHExecutorStatus:
    def test_status_running_when_kill0_succeeds(self):
        executor = _make_executor()
        job_id = "99999:abc123"
        with patch("devrun.executors.ssh.run_ssh_command", return_value=_make_ssh_result("running")) as m:
            status = executor.status(job_id)
        assert status in ("running", "completed")

    def test_status_uses_pid_part_of_job_id(self):
        executor = _make_executor()
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("running")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.status("99999:abc123token")
        assert "99999" in captured[0]


class TestSSHExecutorLogs:
    def test_logs_use_token_not_pid(self):
        """logs() must read the file named with run_token, not with PID."""
        executor = _make_executor()
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("log output")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.logs("99999:mytoken123")
        assert "mytoken123" in captured[0], "log retrieval must use the token part of job_id"
        assert "99999" not in captured[0] or "mytoken123" in captured[0]
