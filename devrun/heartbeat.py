"""Global heartbeat scheduler. See spec for phase ordering."""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devrun.db.jobs import JobStore

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()

_PROMOTION_LEASE_SECONDS = 20
_PROMOTION_LIMIT = 100

_REQUIRED_RE = re.compile(r"^<REQUIRED(?::\s*.*?)?>$")


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


def _has_required_placeholder(obj: Any) -> bool:
    """Recursively detect a residual ``<REQUIRED:...>`` marker in a resolved config."""
    if isinstance(obj, str):
        return bool(_REQUIRED_RE.match(obj))
    if isinstance(obj, dict):
        return any(_has_required_placeholder(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_required_placeholder(v) for v in obj)
    return False


def _promote_ready_queued(db: JobStore, executor_router: Any) -> None:
    """Promote QUEUED jobs whose dependencies are satisfied."""
    if executor_router is None:
        return
    from omegaconf import OmegaConf

    from devrun.jobref import (
        JobRefContext,
        clear_jobref_context,
        install_jobref_context,
    )
    from devrun.registry import get_task_class

    inst = instance_id()
    for cand in db.fetch_ready_queued(limit=_PROMOTION_LIMIT):
        if not db.claim_for_submit(
            job_id=cand.job_id,
            instance_id=inst,
            lease_seconds=_PROMOTION_LEASE_SECONDS,
        ):
            continue
        try:
            cfg = OmegaConf.create(cand.params_template or "{}")
            ctx = JobRefContext(
                allowed_parents=db.get_parent_parameters(cand.job_id),
                calling_job_id=cand.job_id,
            )
            install_jobref_context(ctx)
            try:
                resolved = OmegaConf.to_container(cfg, resolve=True)
            finally:
                clear_jobref_context()
            if not isinstance(resolved, dict):
                resolved = {}
            if _has_required_placeholder(resolved):
                db.fail_promotion(
                    job_id=cand.job_id,
                    skip_reason="unfilled <REQUIRED:...> placeholder",
                )
                continue
            task = get_task_class(cand.task_name)()
            spec = task.prepare(resolved)
            executor = executor_router.get(cand.executor)
            remote_job_id = executor.submit_with_retry(spec)
            log_path = spec.metadata.get("log_path")
            db.finalize_submit(
                job_id=cand.job_id,
                remote_job_id=remote_job_id,
                log_path=log_path,
                resolved_parameters=resolved,
            )
            logger.info(
                "Promoted job %s -> SUBMITTED (remote=%s)",
                cand.job_id, remote_job_id,
            )
        except Exception as exc:
            logger.exception("Promotion failed for %s", cand.job_id)
            db.fail_promotion(job_id=cand.job_id, skip_reason=str(exc))


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
