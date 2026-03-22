"""Abstract base class for executor plugins."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from devrun.models import ExecutorEntry, TaskSpec

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
