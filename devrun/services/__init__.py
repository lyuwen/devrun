"""Cross-platform service management for the devrun heartbeat scheduler.

``get_service()`` dispatches on ``sys.platform`` to a concrete backend:
``SystemdUserService`` on Linux, ``LaunchdService`` on macOS. Both implement
the :class:`HeartbeatService` Protocol.
"""

from __future__ import annotations

import sys
from typing import Protocol


class HeartbeatService(Protocol):
    """Cross-platform service-manager facade for the devrun heartbeat."""

    def install(self, *, python_path: str, db_path: str) -> None: ...
    def uninstall(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def is_active(self) -> bool: ...


def get_service() -> HeartbeatService:
    """Return the appropriate HeartbeatService backend for the host OS."""
    if sys.platform == "darwin":
        from devrun.services.darwin import LaunchdService

        return LaunchdService()
    if sys.platform.startswith("linux"):
        from devrun.services.linux import SystemdUserService

        return SystemdUserService()
    raise RuntimeError(f"Unsupported platform for heartbeat service: {sys.platform}")
