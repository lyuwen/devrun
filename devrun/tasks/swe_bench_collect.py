"""SWEBenchCollectTask — aggregate inference outputs into predictions.jsonl."""
from __future__ import annotations

import shlex
from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask
from devrun.utils.swebench import derive_ds_dir
from devrun.utils.templates import render_template


@register_task("swe_bench_collect")
class SWEBenchCollectTask(BaseTask):
    """Scan inference output directories and produce predictions.jsonl + collected_histories.jsonl.

    Uses a parallel Python collector script (ThreadPoolExecutor for both
    discovery and processing) to handle directories with tens of thousands
    of entries without exceeding shell wildcard limits.

    The generated command writes the Python script to a temp file and
    executes it, ensuring compatibility with all executor backends
    (SSH heredoc wrapping, Slurm sbatch scripts, etc.).
    """

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        output_dir = params.get("output_dir")
        dataset = params.get("dataset")
        model_name_or_path = params.get("model_name_or_path")

        if not output_dir:
            raise ValueError("params.output_dir is required")
        if not dataset:
            raise ValueError("params.dataset is required")
        if not model_name_or_path:
            raise ValueError("params.model_name_or_path is required")

        split = params.get("split", "test")
        predictions_path = params.get("predictions_path", "predictions.jsonl")
        histories_path = params.get("histories_path", "collected_histories.jsonl")
        working_dir = params.get("working_dir")
        max_workers = params.get("max_workers", 16)

        ds_dir = derive_ds_dir(dataset, split)

        # Render the Python collector script via Jinja2
        collector_script = render_template(
            "swe_bench_collect.py.j2",
            output_dir=output_dir,
            ds_dir=ds_dir,
            model_name_or_path=model_name_or_path,
            predictions_path=predictions_path,
            histories_path=histories_path,
            max_workers=max_workers,
        )

        # Build the shell command: write script to temp file and execute.
        # Uses a quoted heredoc ('__DEVRUN_COLLECT_EOF__') so that no bash
        # expansion happens inside the Python script body.  The delimiter
        # is intentionally different from SSHExecutor's DEVRUN_EOF to allow
        # safe nesting.
        command_lines: list[str] = []
        if working_dir:
            command_lines.append(f"cd {shlex.quote(working_dir)}")
        command_lines.extend([
            "set -x",
            "_DEVRUN_COLLECT=$(mktemp /tmp/devrun_collect_XXXXXX.py)",
            "cat > \"${_DEVRUN_COLLECT}\" << '__DEVRUN_COLLECT_EOF__'",
            collector_script,
            "__DEVRUN_COLLECT_EOF__",
            "python3 \"${_DEVRUN_COLLECT}\"",
            "_RC=$?",
            "rm -f \"${_DEVRUN_COLLECT}\"",
            "exit ${_RC}",
        ])

        command = "\n".join(command_lines)

        return TaskSpec(
            command=command,
            working_dir=working_dir,
            env=params.get("env", {}),
            metadata={"job_name": "swe_collect"},
        )
