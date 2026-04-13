"""Abstract base class for task plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from devrun.models import TaskSpec


class BaseTask(ABC):
    """All task plugins must subclass this and implement ``prepare``."""

    @abstractmethod
    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        """Convert user-supplied *params* into a concrete :class:`TaskSpec`."""

    def prepare_many(self, params: dict[str, Any]) -> list[TaskSpec]:
        """Return one or more :class:`TaskSpec` objects for the given *params*.

        The default implementation delegates to :meth:`prepare` and wraps the
        result in a single-element list.  Subclasses may override this to
        expand a single param dict into multiple jobs (e.g. sharded runs).
        """
        return [self.prepare(params)]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
