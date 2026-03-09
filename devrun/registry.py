"""Plugin registry — auto-discovery of executor and task classes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from devrun.executors.base import BaseExecutor
    from devrun.tasks.base import BaseTask

logger = logging.getLogger("devrun.registry")

# ---------------------------------------------------------------------------
# Internal stores
# ---------------------------------------------------------------------------

_EXECUTOR_REGISTRY: dict[str, Type[BaseExecutor]] = {}
_TASK_REGISTRY: dict[str, Type[BaseTask]] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------


def register_executor(name: str):
    """Class decorator that registers an executor plugin under *name*."""

    def _wrapper(cls: Type[BaseExecutor]):
        if name in _EXECUTOR_REGISTRY:
            logger.warning("Overwriting executor '%s' (%s → %s)", name, _EXECUTOR_REGISTRY[name], cls)
        _EXECUTOR_REGISTRY[name] = cls
        logger.debug("Registered executor '%s' → %s", name, cls.__qualname__)
        return cls

    return _wrapper


def register_task(name: str):
    """Class decorator that registers a task plugin under *name*."""

    def _wrapper(cls: Type[BaseTask]):
        if name in _TASK_REGISTRY:
            logger.warning("Overwriting task '%s' (%s → %s)", name, _TASK_REGISTRY[name], cls)
        _TASK_REGISTRY[name] = cls
        logger.debug("Registered task '%s' → %s", name, cls.__qualname__)
        return cls

    return _wrapper


# ---------------------------------------------------------------------------
# Look-ups
# ---------------------------------------------------------------------------


def get_executor_class(name: str) -> Type[BaseExecutor]:
    """Return the executor class registered under *name*."""
    # Ensure plugins are imported so decorators fire.
    import devrun.executors  # noqa: F401

    if name not in _EXECUTOR_REGISTRY:
        available = ", ".join(sorted(_EXECUTOR_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown executor type '{name}'. Available: {available}")
    return _EXECUTOR_REGISTRY[name]


def get_task_class(name: str) -> Type[BaseTask]:
    """Return the task class registered under *name*."""
    import devrun.tasks  # noqa: F401

    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return _TASK_REGISTRY[name]


def list_executors() -> list[str]:
    import devrun.executors  # noqa: F401
    return sorted(_EXECUTOR_REGISTRY)


def list_tasks() -> list[str]:
    import devrun.tasks  # noqa: F401
    return sorted(_TASK_REGISTRY)
