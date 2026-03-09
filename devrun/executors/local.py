"""LocalExecutor — runs commands via subprocess on the local machine."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor

_LOG_DIR = Path.home() / ".devrun" / "logs"


@register_executor("local")
class LocalExecutor(BaseExecutor):
    """Execute commands locally via ``subprocess``."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        # pid → subprocess.Popen
        self._processes: dict[str, subprocess.Popen] = {}

    def submit(self, task_spec: TaskSpec) -> str:
        log_file = _LOG_DIR / f"local_{id(task_spec) & 0xFFFFFF:06x}.log"
        self.logger.info("Local exec: %s", task_spec.command)

        env = None
        if task_spec.env:
            import os
            env = {**os.environ, **task_spec.env}

        with open(log_file, "w") as fh:
            proc = subprocess.Popen(
                task_spec.command,
                shell=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=task_spec.working_dir,
            )
        job_id = str(proc.pid)
        self._processes[job_id] = proc
        self.logger.info("Started local process pid=%s, log=%s", job_id, log_file)
        return job_id

    def status(self, job_id: str) -> str:
        proc = self._processes.get(job_id)
        if proc is None:
            return "unknown"
        rc = proc.poll()
        if rc is None:
            return "running"
        return "completed" if rc == 0 else "failed"

    def logs(self, job_id: str) -> str:
        # Find the most recent log whose name contains the pid-derived hex
        # For simplicity, scan log dir
        for f in sorted(_LOG_DIR.glob("local_*.log"), reverse=True):
            return f.read_text(errors="replace")
        return "(no logs found)"

    def cancel(self, job_id: str) -> None:
        proc = self._processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            self.logger.info("Terminated local process pid=%s", job_id)
