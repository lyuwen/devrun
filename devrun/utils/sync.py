"""File-sync utilities wrapping rsync."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("devrun.utils.sync")


def rsync(
    source: str,
    destination: str,
    *,
    delete: bool = False,
    exclude: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``rsync -avz`` between *source* and *destination*.

    Both *source* and *destination* accept ``host:path`` notation for remote
    targets just like regular rsync.
    """
    cmd = ["rsync", "-avz", "--progress"]
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dry-run")
    for pat in exclude or []:
        cmd += ["--exclude", pat]
    cmd += [source, destination]

    logger.info("rsync: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("rsync failed (rc=%d): %s", result.returncode, result.stderr)
    return result


def sync_to_remote(local_dir: str, remote_spec: str, **kwargs) -> subprocess.CompletedProcess[str]:
    """Upload *local_dir* to a remote target (``host:/path``)."""
    return rsync(local_dir, remote_spec, **kwargs)


def fetch_from_remote(remote_spec: str, local_dir: str, **kwargs) -> subprocess.CompletedProcess[str]:
    """Download from a remote target to *local_dir*."""
    return rsync(remote_spec, local_dir, **kwargs)
