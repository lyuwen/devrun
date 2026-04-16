"""SLURM utilities — script generation and job-status parsing."""

from __future__ import annotations

import json as _json
import logging
import re
import shlex
from collections import Counter
from typing import Any

logger = logging.getLogger("devrun.utils.slurm")


def generate_sbatch_script(
    command: str,
    *,
    job_name: str = "devrun_job",
    nodes: int = 1,
    gpus_per_node: int | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    partition: str | None = None,
    walltime: str = "04:00:00",
    env: dict[str, str] | None = None,
    extra_sbatch: list[str] | None = None,
    working_dir: str | None = None,
    setup_commands: list[str] | None = None,
    output_dir: str | None = None,
    set_e: bool = True,
) -> str:
    """Return a complete ``#!/bin/bash`` SLURM batch script."""
    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --nodes={nodes}")
    if gpus_per_node:
        lines.append(f"#SBATCH --gres=gpu:{gpus_per_node}")
    if cpus_per_task:
        lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --time={walltime}")

    # Only emit default --output/--error if extra_sbatch doesn't already provide them
    has_custom_output = any("--output" in e for e in (extra_sbatch or []))
    has_custom_error = any("--error" in e for e in (extra_sbatch or []))
    if not has_custom_output:
        if output_dir:
            lines.append(f"#SBATCH --output={output_dir}/devrun_%j.out")
        else:
            lines.append("#SBATCH --output=devrun_%j.out")
    if not has_custom_error:
        if output_dir:
            lines.append(f"#SBATCH --error={output_dir}/devrun_%j.err")
        else:
            lines.append("#SBATCH --error=devrun_%j.err")

    for extra in extra_sbatch or []:
        lines.append(f"#SBATCH {extra}")

    lines.append("")

    # Setup commands (conda init, etc.)
    for cmd in setup_commands or []:
        lines.append(cmd)
    if setup_commands:
        lines.append("")

    # Export env vars
    for key, val in (env or {}).items():
        lines.append(f"export {key}={shlex.quote(str(val))}")
    if env:
        lines.append("")

    # Change to working directory if specified
    if working_dir:
        lines.append(f"cd {shlex.quote(working_dir)}")
        lines.append("")

    lines.append("set -ex" if set_e else "set -x")
    lines.append(command)
    lines.append("")
    return "\n".join(lines)


_JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")


def parse_sbatch_output(output: str) -> str:
    """Extract the numeric job ID from sbatch stdout."""
    m = _JOB_ID_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {output!r}")
    return m.group(1)


def parse_squeue_status(output: str, job_id: str) -> str:
    """Parse ``squeue --job <id> -h -o %T`` and return status string.

    .. deprecated:: Use :func:`parse_squeue_json` for JSON-based parsing.
    """
    status = output.strip()
    if not status or "Invalid" in status:
        return "unknown"
    return status.lower()


# ---------------------------------------------------------------------------
# Failure-like Slurm states (terminal states that indicate a problem)
# ---------------------------------------------------------------------------

_FAILURE_STATES = frozenset({
    "failed", "timeout", "out_of_memory", "node_fail",
    "preempted", "boot_fail", "deadline", "stopped",
})

_ACTIVE_STATES = frozenset({"running", "pending", "completing", "suspended", "requeued", "resizing"})


# ---------------------------------------------------------------------------
# sacct / squeue JSON parsing
# ---------------------------------------------------------------------------


def aggregate_array_status(task_counts: dict[str, int]) -> str:
    """Derive an overall job status from per-task state counts.

    Priority:
    1. Any active state (running, pending, …) → ``"running"``
    2. All completed → ``"completed"``
    3. All cancelled → ``"cancelled"``
    4. Any failure-like state among terminal tasks → ``"failed"``
    5. Otherwise → ``"unknown"``
    """
    if not task_counts:
        return "unknown"

    has_active = any(task_counts.get(s, 0) > 0 for s in _ACTIVE_STATES)
    if has_active:
        return "running"

    has_failure = any(task_counts.get(s, 0) > 0 for s in _FAILURE_STATES)
    total = sum(task_counts.values())
    completed = task_counts.get("completed", 0)
    cancelled = task_counts.get("cancelled", 0)

    if completed == total:
        return "completed"
    if cancelled == total:
        return "cancelled"
    if has_failure:
        return "failed"
    return "unknown"


def merge_array_counts(
    sacct_counts: dict[str, int],
    squeue_counts: dict[str, int],
) -> dict[str, int]:
    """Merge sacct and squeue task-state counts for a complete picture.

    - **Terminal states** (completed, failed, cancelled, …) come from sacct
      — squeue does not reliably track finished tasks.
    - **Active states** (running, pending, …) prefer squeue — it catches
      pending tasks that sacct may not have recorded yet.
    """
    merged: dict[str, int] = {}

    # Terminal states: sacct is authoritative
    for state, count in sacct_counts.items():
        if state not in _ACTIVE_STATES and count > 0:
            merged[state] = count

    # Active states: prefer squeue (catches pending tasks sacct misses)
    for state in _ACTIVE_STATES:
        sq = squeue_counts.get(state, 0)
        sa = sacct_counts.get(state, 0)
        best = max(sq, sa)  # take the higher count
        if best > 0:
            merged[state] = best

    return merged


def _make_unknown_info() -> dict[str, Any]:
    """Return the canonical 'unknown' result dict."""
    return {"status": "unknown", "is_array": False, "task_counts": None, "total_tasks": None}


def parse_sacct_json(raw_json: str, job_id: str) -> dict[str, Any]:
    """Parse ``sacct -j <id> --json`` output into an aggregate status dict.

    Returns ``{"status", "is_array", "task_counts", "total_tasks"}``.
    """
    try:
        data = _json.loads(raw_json)
    except (ValueError, TypeError):
        logger.warning("Failed to parse sacct JSON for job %s", job_id)
        return _make_unknown_info()

    jobs = data.get("jobs", [])
    if not jobs:
        return _make_unknown_info()

    # Separate parent aggregate record from actual task records.
    # Parent: array.task_id.set == false (or missing array entirely)
    # Tasks: array.task_id.set == true
    tasks: list[dict] = []
    parent: dict | None = None
    for entry in jobs:
        array_info = entry.get("array", {})
        task_id_info = array_info.get("task_id", {})
        if task_id_info.get("set", False):
            tasks.append(entry)
        else:
            parent = entry

    if not tasks:
        # Non-array job: use the single parent/entry directly
        if parent:
            state_list = parent.get("state", {}).get("current", [])
            status = state_list[0].lower() if state_list else "unknown"
            return {"status": status, "is_array": False, "task_counts": None, "total_tasks": None}
        return _make_unknown_info()

    # Array job: aggregate task states
    counts: Counter[str] = Counter()
    for task in tasks:
        state_list = task.get("state", {}).get("current", [])
        state = state_list[0].lower() if state_list else "unknown"
        counts[state] += 1

    task_counts = dict(counts)
    total = sum(counts.values())
    status = aggregate_array_status(task_counts)

    return {"status": status, "is_array": True, "task_counts": task_counts, "total_tasks": total}


def parse_squeue_json(raw_json: str, job_id: str) -> dict[str, Any]:
    """Parse ``squeue --job <id> --json`` output into an aggregate status dict.

    Returns the same structure as :func:`parse_sacct_json`.
    """
    try:
        data = _json.loads(raw_json)
    except (ValueError, TypeError):
        logger.warning("Failed to parse squeue JSON for job %s", job_id)
        return _make_unknown_info()

    jobs = data.get("jobs", [])
    if not jobs:
        return _make_unknown_info()

    # Separate parent from tasks using array_task_id.set
    tasks: list[dict] = []
    parent: dict | None = None
    for entry in jobs:
        task_id_info = entry.get("array_task_id", {})
        if task_id_info.get("set", False):
            tasks.append(entry)
        else:
            parent = entry

    if not tasks:
        # Non-array job: use parent/entry directly
        if parent:
            state_list = parent.get("job_state", [])
            status = state_list[0].lower() if state_list else "unknown"
            return {"status": status, "is_array": False, "task_counts": None, "total_tasks": None}
        return _make_unknown_info()

    # Array job: aggregate task states
    counts: Counter[str] = Counter()
    for task in tasks:
        state_list = task.get("job_state", [])
        state = state_list[0].lower() if state_list else "unknown"
        counts[state] += 1

    task_counts = dict(counts)
    total = sum(counts.values())
    status = aggregate_array_status(task_counts)

    return {"status": status, "is_array": True, "task_counts": task_counts, "total_tasks": total}
