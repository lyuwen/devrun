"""SlurmExecutor — submits batch jobs to SLURM, locally or via SSH."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor
from devrun.utils.slurm import generate_sbatch_script, parse_sbatch_output, parse_squeue_status

_SCRIPT_DIR = Path.home() / ".devrun" / "slurm_scripts"


def _run_local(cmd: str) -> subprocess.CompletedProcess:
    """Run a shell command locally."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


@register_executor("slurm")
class SlurmExecutor(BaseExecutor):
    """Submit and manage SLURM batch jobs, locally or on a remote cluster via SSH."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        self._partition = config.partition
        self._remote = bool(config.host)

        # Extra config options
        self._setup_commands: list[str] = config.extra.get("setup_commands", [])
        self._extra_sbatch: list[str] = config.extra.get("extra_sbatch", [])
        self._mem: str | None = config.extra.get("mem")
        self._cpus_per_task: int | None = config.extra.get("cpus_per_task")

        if self._remote:
            from devrun.utils.ssh import SSHConfig
            self._ssh = SSHConfig(
                host=config.host,
                user=config.user,
                key_file=config.key_file,
            )
        else:
            self._ssh = None

        _SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # -- helpers for local vs remote execution ----------------------------

    def _run_cmd(self, cmd: str) -> subprocess.CompletedProcess:
        if self._remote:
            from devrun.utils.ssh import run_ssh_command
            return run_ssh_command(self._ssh, cmd)
        return _run_local(cmd)

    def _upload_script(self, local_path: str, remote_path: str) -> str:
        """Upload the script to a target and return the path to use for sbatch."""
        if self._remote:
            from devrun.utils.ssh import scp_upload
            scp_upload(self._ssh, local_path, remote_path)
            self.logger.info("Uploaded SLURM script to %s:%s", self._ssh.host, remote_path)
            return remote_path
        # Local mode: just use the local path directly
        return local_path

    # -- executor interface -----------------------------------------------

    def submit(self, task_spec: TaskSpec) -> str:
        resources = task_spec.resources

        script = generate_sbatch_script(
            command=task_spec.command,
            job_name=task_spec.metadata.get("job_name", "devrun_job"),
            nodes=resources.get("nodes", 1),
            gpus_per_node=resources.get("gpus_per_node") or resources.get("gpus"),
            cpus_per_task=resources.get("cpus_per_task") or self._cpus_per_task,
            mem=resources.get("mem") or self._mem,
            partition=resources.get("partition") or self._partition,
            walltime=resources.get("walltime", "04:00:00"),
            env=task_spec.env,
            extra_sbatch=self._extra_sbatch,
            working_dir=task_spec.working_dir,
            setup_commands=self._setup_commands,
        )

        # Write script to a persistent location for debugging
        job_name = task_spec.metadata.get("job_name", "devrun")
        script_path = _SCRIPT_DIR / f"sbatch_{job_name}.sh"
        script_path.write_text(script)
        self.logger.info("Wrote SLURM script to %s", script_path)

        submit_path = self._upload_script(str(script_path), f"/tmp/devrun_sbatch_{job_name}.sh")

        result = self._run_cmd(f"sbatch {submit_path}")
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        slurm_job_id = parse_sbatch_output(result.stdout)
        self.logger.info("SLURM job submitted: %s", slurm_job_id)
        return slurm_job_id

    def status(self, job_id: str) -> str:
        result = self._run_cmd(f"squeue --job {job_id} -h -o %T 2>/dev/null")
        if result.returncode == 0 and result.stdout.strip():
            return parse_squeue_status(result.stdout, job_id)

        # Fallback to sacct
        result = self._run_cmd(
            f"sacct -j {job_id} --parsable2 --noheader --format=JobID,State,ExitCode 2>/dev/null",
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 2 and parts[0] == job_id:
                    return parts[1].lower()
        return "unknown"

    def logs(self, job_id: str) -> str:
        result = self._run_cmd(
            f"cat devrun_{job_id}.out 2>/dev/null || echo '(no output file found)'",
        )
        return result.stdout

    def cancel(self, job_id: str) -> None:
        self._run_cmd(f"scancel {job_id}")
        self.logger.info("Cancelled SLURM job %s", job_id)
