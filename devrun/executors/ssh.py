"""SSHExecutor — runs commands on remote hosts via the ssh binary."""

from __future__ import annotations

import shlex
import uuid

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor
from devrun.utils.ssh import SSHConfig, run_ssh_command


@register_executor("ssh")
class SSHExecutor(BaseExecutor):
    """Execute commands on a remote host over SSH."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        if not config.host:
            raise ValueError(f"SSHExecutor '{name}' requires a 'host' in config")
        self._ssh = SSHConfig(
            host=config.host,
            user=config.user,
            key_file=config.key_file,
        )

    def submit(self, task_spec: TaskSpec) -> str:
        # Bug 3 fix: shell-quote env values
        env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in task_spec.env.items())
        full_cmd = f"{env_prefix} {task_spec.command}".strip()

        # Bug 4 fix: shell-quote the working_dir path
        if task_spec.working_dir:
            full_cmd = f"cd {shlex.quote(task_spec.working_dir)} && {full_cmd}"

        # Bug 1 fix: use a uuid token so the log file name is deterministic
        run_token = uuid.uuid4().hex[:12]
        remote_log = f"/tmp/devrun_ssh_{run_token}.log"

        # Bug 2 fix: use a heredoc so single quotes inside full_cmd are safe
        remote_cmd = (
            f"nohup bash << 'DEVRUN_EOF' > {remote_log} 2>&1 &\n"
            f"{full_cmd}\n"
            f"DEVRUN_EOF\n"
            f"echo $!"
        )

        self.logger.info("SSH submit to %s: %s", self._ssh.host, task_spec.command)
        result = run_ssh_command(self._ssh, remote_cmd)

        if result.returncode != 0:
            raise RuntimeError(f"SSH submit failed: {result.stderr}")

        remote_pid = result.stdout.strip()
        self.logger.info("Remote PID: %s", remote_pid)

        # Bug 1 fix: return composite job_id so logs/status/cancel can find the file
        return f"{remote_pid}:{run_token}"

    def status(self, job_id: str) -> str:
        # Bug 1 fix: parse composite job_id
        pid = job_id.split(":")[0]

        # Bug 7 fix: use timeout=30 for status checks
        # Bug 6 fix: return "completed" instead of "done"
        result = run_ssh_command(
            self._ssh,
            f"kill -0 {pid} 2>/dev/null && echo running || echo completed",
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def logs(self, job_id: str, log_path: str | None = None) -> str:
        # Bug 1 fix: parse composite job_id to get the token for the log path
        parts = job_id.split(":")
        run_token = parts[1] if len(parts) > 1 else parts[0]
        remote_log = f"/tmp/devrun_ssh_{run_token}.log"

        # Bug 7 fix: use timeout=60 for log retrieval
        result = run_ssh_command(
            self._ssh,
            f"cat {remote_log} 2>/dev/null || echo '(no logs)'",
            timeout=60,
        )
        return result.stdout

    def cancel(self, job_id: str) -> None:
        # Bug 1 fix: parse composite job_id to get the PID
        pid = job_id.split(":")[0]
        run_ssh_command(self._ssh, f"kill {pid} 2>/dev/null")
        self.logger.info("Sent kill to remote PID %s", pid)
