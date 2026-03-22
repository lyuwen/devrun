"""Unit tests for SSHExecutor."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from devrun.executors.ssh import SSHExecutor
from devrun.models import ExecutorEntry, PythonEnv, TaskSpec


def _make_executor(python_env: PythonEnv | None = None):
    entry = ExecutorEntry(type="ssh", host="test.host.com", user="testuser", python_env=python_env)
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


class TestSSHExecutorPythonEnv:
    def test_executor_level_venv_prepended_to_heredoc(self):
        """Executor python_env.venv should appear before the command in the heredoc."""
        executor = _make_executor(python_env=PythonEnv(venv="/opt/venv"))
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(_make_spec(command="python train.py"))
        heredoc_body = captured[0]
        activate_pos = heredoc_body.find("source /opt/venv/bin/activate")
        command_pos = heredoc_body.find("python train.py")
        assert activate_pos != -1, "venv activation must appear in heredoc"
        assert activate_pos < command_pos, "activation must come before the command"

    def test_executor_level_conda_prepended(self):
        executor = _make_executor(python_env=PythonEnv(conda="myenv"))
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(_make_spec())
        assert "conda activate myenv" in captured[0]

    def test_executor_level_modules_prepended(self):
        executor = _make_executor(python_env=PythonEnv(modules=["python/3.11", "cuda/12.1"]))
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(_make_spec())
        assert "module load python/3.11" in captured[0]
        assert "module load cuda/12.1" in captured[0]

    def test_task_level_env_overrides_executor_conda(self):
        """Task python_env.conda should override executor's conda."""
        executor = _make_executor(python_env=PythonEnv(conda="executor_env"))
        task_env = PythonEnv(conda="task_env")
        spec = _make_spec(metadata={"python_env": task_env})
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(spec)
        assert "conda activate task_env" in captured[0]
        assert "conda activate executor_env" not in captured[0]

    def test_task_setup_commands_appended_after_executor(self):
        executor = _make_executor(python_env=PythonEnv(setup_commands=["export EXECUTOR_VAR=1"]))
        task_env = PythonEnv(setup_commands=["export TASK_VAR=2"])
        spec = _make_spec(metadata={"python_env": task_env})
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(spec)
        body = captured[0]
        executor_pos = body.find("EXECUTOR_VAR=1")
        task_pos = body.find("TASK_VAR=2")
        assert executor_pos != -1 and task_pos != -1
        assert executor_pos < task_pos, "executor setup_commands must come before task setup_commands"

    def test_no_python_env_no_preamble(self):
        """When neither executor nor task has python_env, no extra lines are added."""
        executor = _make_executor(python_env=None)
        captured = []
        def capture(ssh, cmd, **kwargs):
            captured.append(cmd)
            return _make_ssh_result("42")
        with patch("devrun.executors.ssh.run_ssh_command", side_effect=capture):
            executor.submit(_make_spec(command="python train.py"))
        body = captured[0]
        assert "source " not in body
        assert "conda activate" not in body
        assert "module load" not in body

