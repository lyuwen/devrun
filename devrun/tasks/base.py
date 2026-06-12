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

    @classmethod
    def import_from_job(
        cls, source_task: str, source_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate params from a previous job into this task's param schema.

        Subclasses override this to enable ``devrun run <task> --from-job <id>``.
        The returned dict is shallow-merged into the YAML defaults before CLI
        overrides are applied.  Return an empty dict if *source_task* is not
        recognised; raise :class:`ValueError` to signal an explicit mismatch.
        """
        return {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
