"""SLURM utilities — script generation and job-status parsing."""

from __future__ import annotations

import logging
import re
import shlex
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
        lines.append(f"cd {working_dir}")
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
    """Parse ``squeue --job <id> -h -o %T`` and return status string."""
    status = output.strip()
    if not status or "Invalid" in status:
        return "unknown"
    return status.lower()
