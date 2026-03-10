"""LocalExecutor — runs commands via subprocess on the local machine."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor

_LOG_DIR = Path.home() / ".devrun" / "logs"


@register_executor("local")
class LocalExecutor(BaseExecutor):
    """Execute commands locally via ``subprocess`` with persistent state."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    def submit(self, task_spec: TaskSpec) -> str:
        # Use a unique identifier for this job's files
        ts = int(time.time() * 1000)
        job_id = f"local_{ts}_{id(task_spec) & 0xFFFFFF:06x}"
        
        log_file = _LOG_DIR / f"{job_id}.log"
        rc_file = _LOG_DIR / f"{job_id}.rc"
        pid_file = _LOG_DIR / f"{job_id}.pid"

        self.logger.info("Local exec: %s", task_spec.command)

        env = None
        if task_spec.env:
            env = {**os.environ, **task_spec.env}

        # Wrap command so it writes its exit code to the rc_file when done
        wrapped_cmd = f"({task_spec.command}) ; echo $? > '{rc_file}'"

        with open(log_file, "w") as fh:
            proc = subprocess.Popen(
                wrapped_cmd,
                shell=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=task_spec.working_dir,
                start_new_session=True,  # Daemonize so it survives CLI exit
            )
        
        # Save PID robustly
        pid_file.write_text(str(proc.pid))
        self.logger.info("Started local process pid=%s, job_id=%s, log=%s", proc.pid, job_id, log_file)
        return job_id

    def status(self, job_id: str) -> str:
        # 1. Did it finish normally and write its exit code?
        rc_file = _LOG_DIR / f"{job_id}.rc"
        if rc_file.exists():
            try:
                rc = int(rc_file.read_text().strip())
                return "completed" if rc == 0 else "failed"
            except ValueError:
                pass

        # 2. Check if the PID is still running
        pid_file = _LOG_DIR / f"{job_id}.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                return "running"
            except ProcessLookupError:
                return "failed"  # Not running and no rc file -> killed/failed
            except OSError:
                pass # permission denied -> it's running but belongs to someone else?

        # 3. Fallback for older jobs that just used bare PIDs as job_ids
        if job_id.isdigit():
            try:
                os.kill(int(job_id), 0)
                return "running"
            except OSError:
                # If an old bare-PID job isn't running, assume it completed
                return "completed"

        return "unknown"

    def logs(self, job_id: str) -> str:
        # Check standard format
        log_file = _LOG_DIR / f"{job_id}.log"
        if log_file.exists():
            return log_file.read_text(errors="replace")
            
        # Fallback to old format scanning
        for f in sorted(_LOG_DIR.glob(f"local_*{job_id}*.log"), reverse=True):
            return f.read_text(errors="replace")
            
        return f"(no logs found for {job_id})"

    def cancel(self, job_id: str) -> None:
        try:
            pid = None
            pid_file = _LOG_DIR / f"{job_id}.pid"
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
            elif job_id.isdigit():
                pid = int(job_id)

            if pid is not None:
                import signal
                os.kill(pid, signal.SIGTERM)
                self.logger.info("Terminated local process pid=%s", pid)
        except Exception as e:
            self.logger.warning("Failed to cancel job %s: %s", job_id, e)
