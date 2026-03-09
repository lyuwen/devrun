"""SLURM utilities — script generation and job-status parsing."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("devrun.utils.slurm")


def generate_sbatch_script(
    command: str,
    *,
    job_name: str = "devrun_job",
    nodes: int = 1,
    gpus_per_node: int | None = None,
    cpus_per_task: int | None = None,
    partition: str | None = None,
    walltime: str = "04:00:00",
    env: dict[str, str] | None = None,
    extra_sbatch: list[str] | None = None,
) -> str:
    """Return a complete ``#!/bin/bash`` SLURM batch script."""
    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --nodes={nodes}")
    if gpus_per_node:
        lines.append(f"#SBATCH --gres=gpu:{gpus_per_node}")
    if cpus_per_task:
        lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --time={walltime}")
    lines.append("#SBATCH --output=devrun_%j.out")
    lines.append("#SBATCH --error=devrun_%j.err")

    for extra in extra_sbatch or []:
        lines.append(f"#SBATCH {extra}")

    lines.append("")

    # Export env vars
    for key, val in (env or {}).items():
        lines.append(f"export {key}={val}")
    if env:
        lines.append("")

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
    """Parse ``squeue --job <id> -h -o %T`` and return status string."""
    status = output.strip()
    if not status or "Invalid" in status:
        return "unknown"
    return status.lower()


def parse_sacct_status(output: str) -> dict[str, Any]:
    """Parse sacct output (``--parsable2 --format=JobID,State,ExitCode``)."""
    result: dict[str, Any] = {}
    for line in output.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            result["job_id"] = parts[0]
            result["state"] = parts[1].lower()
            if len(parts) >= 3:
                result["exit_code"] = parts[2]
    return result
