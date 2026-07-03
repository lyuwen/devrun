"""systemd ``--user`` backend for the devrun heartbeat service."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from devrun.utils.templates import render_template

logger = logging.getLogger(__name__)


class SystemdUserService:
    """Manage the devrun heartbeat via a ``systemctl --user`` unit file."""

    UNIT_NAME = "devrun-heartbeat.service"
    UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / UNIT_NAME

    def install(self, *, python_path: str, db_path: str) -> None:
        body = render_template(
            "devrun-heartbeat.service.j2",
            python_path=python_path,
            db_path=db_path,
        )
        self.UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.UNIT_PATH.write_text(body)
        logger.info("Wrote systemd unit to %s", self.UNIT_PATH)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", self.UNIT_NAME], check=True)

    def uninstall(self) -> None:
        subprocess.run(["systemctl", "--user", "stop", self.UNIT_NAME], check=False)
        subprocess.run(["systemctl", "--user", "disable", self.UNIT_NAME], check=False)
        self.UNIT_PATH.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        logger.info("Removed systemd unit %s", self.UNIT_PATH)

    def start(self) -> None:
        subprocess.run(["systemctl", "--user", "start", self.UNIT_NAME], check=True)

    def stop(self) -> None:
        subprocess.run(["systemctl", "--user", "stop", self.UNIT_NAME], check=True)

    def restart(self) -> None:
        subprocess.run(["systemctl", "--user", "restart", self.UNIT_NAME], check=True)

    def is_active(self) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", self.UNIT_NAME]
        )
        return result.returncode == 0
