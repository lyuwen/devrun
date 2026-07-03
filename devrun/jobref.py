"""OmegaConf resolver for cross-job parameter references."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class JobRefContext:
    """Context for job reference resolution during promotion."""

    allowed_parents: dict[str, dict]
    calling_job_id: str


_jobref_context: contextvars.ContextVar[JobRefContext | None] = contextvars.ContextVar(
    "jobref_context", default=None
)


def install_jobref_context(ctx: JobRefContext) -> None:
    """Install context for the current async context / thread."""
    _jobref_context.set(ctx)


def clear_jobref_context() -> None:
    """Clear context after promotion completes."""
    _jobref_context.set(None)


def _jobs_resolver(job_id: str, dotted_path: str) -> Any:
    """OmegaConf resolver: ${jobs:<job_id>,<dotted.path>}"""
    ctx = _jobref_context.get()
    if ctx is None:
        raise RuntimeError(
            f"jobs resolver called outside promotion context for job_id={job_id}"
        )

    if job_id not in ctx.allowed_parents:
        raise ValueError(
            f"Job {ctx.calling_job_id} references job {job_id} which is not in allowed parents. "
            f"Declare an explicit dependency edge."
        )

    params = ctx.allowed_parents[job_id]
    current: Any = params

    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(
                f"Job {ctx.calling_job_id} references missing key '{dotted_path}' from job {job_id}"
            )
        current = current[part]

    return current


OmegaConf.register_new_resolver("jobs", _jobs_resolver, replace=True)
