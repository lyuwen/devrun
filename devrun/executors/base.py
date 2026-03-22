"""Abstract base class for executor plugins."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from devrun.models import ExecutorEntry, PythonEnv, TaskSpec

logger = logging.getLogger("devrun.executors.base")


class BaseExecutor(ABC):
    """All executor plugins must subclass this and implement the three core methods."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"devrun.executors.{name}")

    # ---- abstract interface ----------------------------------------------

    @abstractmethod
    def submit(self, task_spec: TaskSpec) -> str:
        """Submit a job and return a job/process identifier (string)."""

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Query the current status of a previously submitted job."""

    @abstractmethod
    def logs(self, job_id: str, log_path: str | None = None) -> str:
        """Retrieve log output for a job."""

    # ---- optional hooks --------------------------------------------------

    def cancel(self, job_id: str) -> None:  # pragma: no cover
        """Cancel a running job (optional)."""
        raise NotImplementedError(f"cancel() not implemented for {self.__class__.__name__}")

    # ---- retry / timeout helpers -----------------------------------------

    def submit_with_retry(
        self,
        task_spec: TaskSpec,
        *,
        retries: int = 2,
        retry_delay: float = 5.0,
    ) -> str:
        """Submit with automatic retries on failure."""
        import time

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self.submit(task_spec)
            except Exception as exc:
                last_err = exc
                self.logger.warning(
                    "Submit attempt %d/%d failed: %s", attempt, retries, exc
                )
                if attempt < retries:
                    time.sleep(retry_delay)
        raise RuntimeError(
            f"Failed to submit after {retries} attempts"
        ) from last_err

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

    # ---- python environment helpers -------------------------------------

    @staticmethod
    def _resolve_python_env(
        executor_env: PythonEnv | None,
        task_env: PythonEnv | None,
    ) -> PythonEnv | None:
        """Merge executor-level and task-level PythonEnv.

        Merge semantics:
        - ``venv`` / ``conda``: task value wins if set, else executor value.
        - ``modules``: task list replaces executor list entirely if non-empty.
        - ``setup_commands``: executor list prepended to task list (both kept).
        """
        if not executor_env and not task_env:
            return None
        base = executor_env or PythonEnv()
        override = task_env or PythonEnv()
        return PythonEnv(
            venv=override.venv or base.venv,
            conda=override.conda or base.conda,
            modules=override.modules if override.modules else base.modules,
            setup_commands=base.setup_commands + override.setup_commands,
        )

    @staticmethod
    def _env_to_shell_lines(env: PythonEnv) -> list[str]:
        """Convert a PythonEnv into ordered shell preamble lines.

        Order: module loads → venv activation → conda activation → setup_commands.
        """
        lines: list[str] = []
        for mod in env.modules:
            lines.append(f"module load {mod}")
        if env.venv:
            # Accept either a venv root dir or a direct path to the activate script
            activate = env.venv if env.venv.endswith("activate") else f"{env.venv}/bin/activate"
            lines.append(f"source {activate}")
        if env.conda:
            lines.append(f"conda activate {env.conda}")
        lines.extend(env.setup_commands)
        return lines
