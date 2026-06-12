"""Global heartbeat scheduler. See spec for phase ordering."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devrun.db.jobs import JobStore

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def instance_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def tick(db: JobStore, executor_router: Any) -> None:
    """Run one heartbeat tick.

    Phases run in fixed order: stale-lease reclaim, workflow-deadline expiry,
    cascade-skip of dependents, promotion of ready ``QUEUED`` jobs, and poll
    of active jobs. Each phase is a separate function for isolation testing.
    """
    _reclaim_stale_leases(db)
    _expire_workflow_deadlines(db)
    _cascade_skip_failed(db)
    _promote_ready_queued(db, executor_router)
    _poll_active_jobs(db, executor_router)


def _reclaim_stale_leases(db: JobStore) -> None:
    reclaimed = db.reclaim_expired_leases(now=_now())
    if reclaimed:
        logger.warning("Reclaimed %d stale lease(s): %s", len(reclaimed), reclaimed)


def _expire_workflow_deadlines(db: JobStore) -> None:
    for wf_id in db.fetch_expired_workflows(now=_now()):
        logger.warning("Workflow %s exceeded deadline; expiring", wf_id)
        db.expire_workflow(wf_id)


def _cascade_skip_failed(db: JobStore) -> None:
    skipped = db.cascade_skip_dependents()
    if skipped:
        logger.info("Cascade-skipped %d dependent(s): %s", len(skipped), skipped)


def _promote_ready_queued(db: JobStore, executor_router: Any) -> None:
    """Promote QUEUED jobs whose dependencies are satisfied. Implemented in Task 3."""
    return None


def _poll_active_jobs(db: JobStore, executor_router: Any) -> None:
    """Poll SUBMITTED/RUNNING/CANCELING jobs. Implemented in Task 5."""
    return None


def run_loop(
    db_path: Path,
    interval: float = 10.0,
    tick_file: Path | None = None,
) -> None:
    """Foreground loop. Catches SIGTERM/SIGINT; exits cleanly after the current tick."""
    from devrun.router import resolve_executor  # local import to avoid cycles  # noqa: F401

    db = JobStore(db_path)
    router: Any = None  # PR2 placeholder; replaced once promotion/poll phases land
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_event.set())
    signal.signal(signal.SIGINT, lambda *_: _shutdown_event.set())
    logger.info(
        "Heartbeat starting (interval=%ss, instance=%s)", interval, instance_id()
    )
    while not _shutdown_event.is_set():
        try:
            tick(db, router)
        except Exception:
            logger.exception("Heartbeat tick failed; continuing")
        if tick_file is not None:
            tick_file.write_text(_now().isoformat())
        _shutdown_event.wait(interval)
    logger.info("Heartbeat shut down cleanly")
