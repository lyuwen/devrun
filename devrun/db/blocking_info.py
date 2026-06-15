"""Helper to explain why a QUEUED job hasn't been promoted yet."""

from __future__ import annotations

from dataclasses import dataclass

from devrun.db.jobs import JobStore
from devrun.models import JobStatus


@dataclass
class BlockingInfo:
    """Explains why a QUEUED job is not ready for promotion."""

    is_blocked: bool
    blocking_parents: list[tuple[str, str]]  # (job_id, status)
    heartbeat_running: bool | None = None  # None = unknown

    def explain(self) -> str:
        """Return a human-readable explanation."""
        if not self.is_blocked:
            if self.heartbeat_running is False:
                return "Ready for promotion, but heartbeat scheduler is not running."
            elif self.heartbeat_running is True:
                return "Ready for promotion. Heartbeat will pick it up on next tick."
            else:
                return "Ready for promotion (heartbeat status unknown)."

        lines = ["Blocked by parent dependencies:"]
        for parent_id, parent_status in self.blocking_parents:
            lines.append(f"  • {parent_id}: {parent_status}")
        return "\n".join(lines)


def get_blocking_info(db: JobStore, job_id: str) -> BlockingInfo | None:
    """Return BlockingInfo if job is QUEUED, else None."""
    rec = db.get(job_id)
    if not rec or JobStatus(rec.status) != JobStatus.QUEUED:
        return None

    # Check dependencies
    deps = db.list_dependencies(job_id)
    blocking = []

    for dep in deps:
        parent = db.get(dep.parent_job_id)
        if not parent:
            blocking.append((dep.parent_job_id, "NOT_FOUND"))
            continue

        parent_status = JobStatus(parent.status)

        # Parent is satisfied if:
        # - COMPLETED, OR
        # - (FAILED/SKIPPED/CANCELLED) AND allow_failure=True
        if parent_status == JobStatus.COMPLETED:
            continue
        if dep.allow_failure and parent_status in (
            JobStatus.FAILED,
            JobStatus.SKIPPED,
            JobStatus.CANCELLED,
        ):
            continue

        # Otherwise, it's blocking
        blocking.append((dep.parent_job_id, parent_status.value))

    # Check heartbeat status (best effort)
    heartbeat_running = None
    try:
        from devrun.services import get_service
        service = get_service()
        heartbeat_running = service.is_active()
    except Exception:
        pass

    return BlockingInfo(
        is_blocked=len(blocking) > 0,
        blocking_parents=blocking,
        heartbeat_running=heartbeat_running,
    )
