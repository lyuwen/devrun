"""SSH helpers — thin wrappers around the ssh / scp binaries."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("devrun.utils.ssh")


@dataclass
class SSHConfig:
    """Connection parameters for an SSH host."""

    host: str
    user: str | None = None
    key_file: str | None = None
    port: int = 22

    @property
    def target(self) -> str:
        prefix = f"{self.user}@" if self.user else ""
        return f"{prefix}{self.host}"

    def _base_args(self) -> list[str]:
        args: list[str] = []
        if self.key_file:
            args += ["-i", self.key_file]
        if self.port != 22:
            args += ["-p", str(self.port)]
        args += ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        return args


def run_ssh_command(
    cfg: SSHConfig,
    command: str,
    *,
    timeout: int | None = 300,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute *command* on the remote host and return the result."""
    cmd = ["ssh"] + cfg._base_args() + [cfg.target, command]
    logger.debug("SSH exec: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def scp_upload(cfg: SSHConfig, local_path: str, remote_path: str) -> None:
    """Copy a local file to the remote host."""
    port_args = ["-P", str(cfg.port)] if cfg.port != 22 else []
    key_args = ["-i", cfg.key_file] if cfg.key_file else []
    cmd = ["scp"] + key_args + port_args + [local_path, f"{cfg.target}:{remote_path}"]
    logger.debug("SCP upload: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def scp_download(cfg: SSHConfig, remote_path: str, local_path: str) -> None:
    """Copy a remote file to the local host."""
    port_args = ["-P", str(cfg.port)] if cfg.port != 22 else []
    key_args = ["-i", cfg.key_file] if cfg.key_file else []
    cmd = ["scp"] + key_args + port_args + [f"{cfg.target}:{remote_path}", local_path]
    logger.debug("SCP download: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
