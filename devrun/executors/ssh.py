"""SSHExecutor — runs commands on remote hosts via the ssh binary."""

from __future__ import annotations

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
        env_prefix = " ".join(f"{k}={v}" for k, v in task_spec.env.items())
        full_cmd = f"{env_prefix} {task_spec.command}".strip()
        if task_spec.working_dir:
            full_cmd = f"cd {task_spec.working_dir} && {full_cmd}"

        # Launch in background, capture PID
        remote_cmd = f"nohup bash -c '{full_cmd}' > /tmp/devrun_ssh_$$.log 2>&1 & echo $!"
        self.logger.info("SSH submit to %s: %s", self._ssh.host, task_spec.command)
        result = run_ssh_command(self._ssh, remote_cmd)

        if result.returncode != 0:
            raise RuntimeError(f"SSH submit failed: {result.stderr}")

        remote_pid = result.stdout.strip()
        self.logger.info("Remote PID: %s", remote_pid)
        return remote_pid

    def status(self, job_id: str) -> str:
        result = run_ssh_command(self._ssh, f"kill -0 {job_id} 2>/dev/null && echo running || echo done")
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def logs(self, job_id: str) -> str:
        result = run_ssh_command(self._ssh, f"cat /tmp/devrun_ssh_{job_id}.log 2>/dev/null || echo '(no logs)'")
        return result.stdout

    def cancel(self, job_id: str) -> None:
        run_ssh_command(self._ssh, f"kill {job_id} 2>/dev/null")
        self.logger.info("Sent kill to remote PID %s", job_id)
