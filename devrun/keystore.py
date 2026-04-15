"""Key-value store for secrets referenced in devrun configs.

Stores named secrets in ``~/.devrun/keys.yaml`` and exposes them to OmegaConf
via a ``${key:name}`` resolver so configs can reference secrets without
hard-coding them.

**Security note**: values are base64-encoded, **not encrypted**.  This provides
obfuscation (avoids plain-text secrets in the YAML file) but does NOT protect
against an attacker with filesystem access.  Protect ``keys.yaml`` with
appropriate file permissions (0600, enforced automatically) and consider
full-disk encryption for sensitive environments.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class KeyStore:
    """Manage base64-encoded secrets stored in a YAML file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".devrun" / "keys.yaml")

    # -- public API ----------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        """Store *value* under *name* (base64-encoded on disk)."""
        _validate_name(name)
        data = self._load()
        data[name] = base64.b64encode(value.encode()).decode()
        self._save(data)
        logger.info("Key '%s' stored.", name)

    def get(self, name: str) -> str:
        """Return the plain-text value for *name*."""
        data = self._load()
        if name not in data:
            raise KeyError(name)
        return base64.b64decode(data[name].encode()).decode()

    def delete(self, name: str) -> None:
        """Remove *name* from the store."""
        data = self._load()
        if name not in data:
            raise KeyError(name)
        del data[name]
        self._save(data)
        logger.info("Key '%s' deleted.", name)

    def list_keys(self) -> list[str]:
        """Return sorted key names."""
        return sorted(self._load())

    # -- internal ------------------------------------------------------------

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        with open(self._path) as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, str]) -> None:
        self._ensure_dir()
        with open(self._path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False)
        self._path.chmod(0o600)

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid key name {name!r}: must match [a-zA-Z0-9_-]+"
        )


# ---------------------------------------------------------------------------
# OmegaConf resolver — activated on import
# ---------------------------------------------------------------------------

def register_resolver() -> None:
    """Register the ``${key:name}`` OmegaConf resolver."""
    store = KeyStore()
    OmegaConf.register_new_resolver("key", lambda name: store.get(name), replace=True)


register_resolver()
