"""SlurmExecutor — submits batch jobs to SLURM clusters via SSH."""

from __future__ import annotations

import tempfile
from pathlib import Path

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor
from devrun.utils.slurm import generate_sbatch_script, parse_sbatch_output, parse_squeue_status
from devrun.utils.ssh import SSHConfig, run_ssh_command, scp_upload


@register_executor("slurm")
class SlurmExecutor(BaseExecutor):
    """Submit and manage SLURM batch jobs on a remote cluster."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        if not config.host:
            raise ValueError(f"SlurmExecutor '{name}' requires a 'host' in config")
        self._ssh = SSHConfig(
            host=config.host,
            user=config.user,
            key_file=config.key_file,
        )
        self._partition = config.partition

    def submit(self, task_spec: TaskSpec) -> str:
        resources = task_spec.resources
        script = generate_sbatch_script(
            command=task_spec.command,
            job_name=task_spec.metadata.get("job_name", "devrun_job"),
            nodes=resources.get("nodes", 1),
            gpus_per_node=resources.get("gpus_per_node") or resources.get("gpus"),
            cpus_per_task=resources.get("cpus_per_task") or resources.get("cpus"),
            partition=resources.get("partition") or self._partition,
            walltime=resources.get("walltime", "04:00:00"),
            env=task_spec.env,
        )

        # Write script to a temp file and upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
            tmp.write(script)
            local_script = tmp.name

        remote_script = "/tmp/devrun_sbatch.sh"
        scp_upload(self._ssh, local_script, remote_script)
        self.logger.info("Uploaded SLURM script to %s:%s", self._ssh.host, remote_script)

        result = run_ssh_command(self._ssh, f"sbatch {remote_script}")
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        slurm_job_id = parse_sbatch_output(result.stdout)
        self.logger.info("SLURM job submitted: %s", slurm_job_id)
        return slurm_job_id

    def status(self, job_id: str) -> str:
        result = run_ssh_command(self._ssh, f"squeue --job {job_id} -h -o %T 2>/dev/null")
        if result.returncode == 0 and result.stdout.strip():
            return parse_squeue_status(result.stdout, job_id)

        # Fallback to sacct
        result = run_ssh_command(
            self._ssh,
            f"sacct -j {job_id} --parsable2 --noheader --format=JobID,State,ExitCode 2>/dev/null",
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 2 and parts[0] == job_id:
                    return parts[1].lower()
        return "unknown"

    def logs(self, job_id: str) -> str:
        result = run_ssh_command(
            self._ssh,
            f"cat devrun_{job_id}.out 2>/dev/null || echo '(no output file found)'",
        )
        return result.stdout

    def cancel(self, job_id: str) -> None:
        run_ssh_command(self._ssh, f"scancel {job_id}")
        self.logger.info("Cancelled SLURM job %s", job_id)
