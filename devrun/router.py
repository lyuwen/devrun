"""Executor router — resolves executor names to configured instances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry
from devrun.registry import get_executor_class

logger = logging.getLogger("devrun.router")

def _find_executors_file() -> Path:
    """Search for executors.yaml in common locations in order of precedence: CWD > Home > Repo Root."""
    devrun_repo_root = Path(__file__).parent.parent
    candidates = [
        Path.cwd() / ".devrun" / "configs" / "executors.yaml",
        Path.home() / ".devrun" / "configs" / "executors.yaml",
        devrun_repo_root / "devrun" / "configs" / "executors.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find executors.yaml. Searched: {[str(c) for c in candidates]}"
    )


def load_executor_configs(path: str | Path | None = None) -> dict[str, ExecutorEntry]:
    """Load and validate all executor entries from a YAML file."""
    cfg_path = Path(path) if path else _find_executors_file()
    logger.info("Loading executor configs from %s", cfg_path)
    with open(cfg_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    entries: dict[str, ExecutorEntry] = {}
    for name, data in raw.items():
        if not isinstance(data, dict):
            logger.warning("Skipping non-dict entry '%s'", name)
            continue
        entries[name] = ExecutorEntry(**data)
    return entries


def resolve_executor(
    executor_name: str,
    configs: dict[str, ExecutorEntry] | None = None,
    executors_path: str | Path | None = None,
) -> BaseExecutor:
    """Instantiate the executor identified by *executor_name*.

    1. Look up the executor entry in ``executors.yaml``.
    2. Find the registered class for the entry's ``type``.
    3. Return an instance.
    """
    if configs is None:
        configs = load_executor_configs(executors_path)

    if executor_name not in configs:
        available = ", ".join(sorted(configs)) or "(none)"
        raise KeyError(f"Unknown executor '{executor_name}'. Available: {available}")

    entry = configs[executor_name]
    cls = get_executor_class(entry.type)
    instance = cls(name=executor_name, config=entry)
    logger.info("Resolved executor '%s' → %s", executor_name, instance)
    return instance


class ExecutorRouter:
    """Thin facade used by the heartbeat to resolve and cache executors by name.

    Loads ``executors.yaml`` once on first ``get()`` and reuses the resulting
    config map; each ``get(name)`` instantiates a fresh executor via
    :func:`resolve_executor` so that backends can hold per-job mutable state
    (e.g. SSH log tokens, sbatch script paths) without leaking across jobs.
    """

    def __init__(self, executors_path: str | Path | None = None) -> None:
        self._executors_path = executors_path
        self._configs: dict[str, ExecutorEntry] | None = None

    def _ensure_configs(self) -> dict[str, ExecutorEntry]:
        if self._configs is None:
            self._configs = load_executor_configs(self._executors_path)
        return self._configs

    def get(self, name: str) -> BaseExecutor:
        """Return an executor instance for *name* (raises KeyError if unknown)."""
        return resolve_executor(name, self._ensure_configs(), self._executors_path)
