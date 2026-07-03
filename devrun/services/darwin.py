"""launchd backend for the devrun heartbeat service (macOS)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from devrun.utils.templates import render_template

logger = logging.getLogger(__name__)


class LaunchdService:
    """Manage the devrun heartbeat via a LaunchAgent plist + ``launchctl``."""

    LABEL = "com.devrun.heartbeat"
    PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

    def install(self, *, python_path: str, db_path: str) -> None:
        body = render_template(
            "com.devrun.heartbeat.plist.j2",
            python_path=python_path,
            db_path=db_path,
        )
        self.PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PLIST_PATH.write_text(body)
        logger.info("Wrote launchd plist to %s", self.PLIST_PATH)
        subprocess.run(["launchctl", "load", str(self.PLIST_PATH)], check=True)

    def uninstall(self) -> None:
        subprocess.run(["launchctl", "unload", str(self.PLIST_PATH)], check=False)
        self.PLIST_PATH.unlink(missing_ok=True)
        logger.info("Removed launchd plist %s", self.PLIST_PATH)

    def start(self) -> None:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{_uid()}/{self.LABEL}"], check=True)

    def stop(self) -> None:
        subprocess.run(["launchctl", "stop", self.LABEL], check=True)

    def restart(self) -> None:
        self.stop()
        self.start()

    def is_active(self) -> bool:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{_uid()}/{self.LABEL}"],
            capture_output=True,
        )
        return result.returncode == 0


def _uid() -> int:
    import os

    return os.getuid()
