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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
