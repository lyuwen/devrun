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

_DEFAULT_EXECUTORS_FILE = "configs/executors.yaml"


def _find_executors_file() -> Path:
    """Search for executors.yaml in common locations."""
    candidates = [
        Path(_DEFAULT_EXECUTORS_FILE),
        Path.cwd() / "executors.yaml",
        Path.home() / ".devrun" / "executors.yaml",
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
