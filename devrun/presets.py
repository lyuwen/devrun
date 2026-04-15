"""Field-namespaced preset store for devrun configs.

Stores named presets in ``~/.devrun/presets.yaml`` and exposes them to OmegaConf
via a ``${preset:field,name}`` resolver so configs can reference reusable
parameter blocks without hard-coding them.

Values can be any YAML-representable type (dict, list, str, int, etc.) and are
stored as-is — no encoding or file-permission hardening is applied.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class PresetStore:
    """Manage field-namespaced presets stored in a YAML file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".devrun" / "presets.yaml")

    # -- public API ----------------------------------------------------------

    def set(self, field: str, name: str, value: Any) -> None:
        """Store *value* under ``data[field][name]``."""
        _validate_name(field)
        _validate_name(name)
        data = self._load()
        data.setdefault(field, {})[name] = value
        self._save(data)
        logger.info("Preset '%s.%s' stored.", field, name)

    def get(self, field: str, name: str) -> Any:
        """Return the value for ``data[field][name]``."""
        data = self._load()
        if field not in data or name not in data[field]:
            raise KeyError(f"{field}.{name}")
        return data[field][name]

    def delete(self, field: str, name: str) -> None:
        """Remove ``data[field][name]``; clean up empty field dicts."""
        data = self._load()
        if field not in data or name not in data[field]:
            raise KeyError(f"{field}.{name}")
        del data[field][name]
        if not data[field]:
            del data[field]
        self._save(data)
        logger.info("Preset '%s.%s' deleted.", field, name)

    def list_presets(self, field: str | None = None) -> dict[str, list[str]]:
        """Return ``{field: [sorted names]}``; optionally filter by *field*."""
        data = self._load()
        if field is not None:
            if field not in data:
                return {}
            return {field: sorted(data[field])}
        return {f: sorted(names) for f, names in sorted(data.items())}

    # -- internal ------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        with open(self._path) as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._ensure_dir()
        with open(self._path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False)

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid preset name {name!r}: must match [a-zA-Z0-9_-]+"
        )


# ---------------------------------------------------------------------------
# OmegaConf resolver — activated on import
# ---------------------------------------------------------------------------

def register_resolver() -> None:
    """Register the ``${preset:field,name}`` OmegaConf resolver."""
    store = PresetStore()
    OmegaConf.register_new_resolver(
        "preset", lambda field, name: store.get(field, name), replace=True,
    )


register_resolver()
