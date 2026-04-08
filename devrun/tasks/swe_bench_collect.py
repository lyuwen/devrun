"""SWEBenchCollectTask — aggregate inference outputs into predictions.jsonl."""
from __future__ import annotations

import json
import shlex
from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask
from devrun.utils.swebench import derive_ds_dir


@register_task("swe_bench_collect")
class SWEBenchCollectTask(BaseTask):
    """Scan inference output directories and produce a predictions.jsonl file."""

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
        working_dir = params.get("working_dir")
        array = params.get("array")

        ds_dir = derive_ds_dir(dataset, split)

        # Build the glob pattern for output files
        glob_pattern = f"{shlex.quote(output_dir)}/*/{shlex.quote(ds_dir)}/*/*/output.jsonl"

        # Use json.dumps for the model name inside jq expressions (not shlex.quote)
        model_jq_str = json.dumps(model_name_or_path)
        pred_escaped = shlex.quote(predictions_path)

        command_lines = []
        if working_dir:
            command_lines.append(f"cd {shlex.quote(working_dir)}")

        command_lines.extend([
            "set -x",
            f"PRED_FILE={pred_escaped}",
            f'> "${{PRED_FILE}}"',  # truncate/create
            "TOTAL=0",
            "SKIPPED=0",
            f"for f in {glob_pattern}; do",
            '    if [[ ! -f "$f" ]]; then continue; fi',
            '    PATCH=$(jq -r ".test_result.git_patch // null" "$f" 2>/dev/null)',
            '    INSTANCE_ID=$(jq -r ".instance_id // null" "$f" 2>/dev/null)',
            '    if [[ "$PATCH" == "null" || -z "$PATCH" || "$INSTANCE_ID" == "null" || -z "$INSTANCE_ID" ]]; then',
            '        echo "WARNING: Skipping $f — missing instance_id or git_patch" >&2',
            '        SKIPPED=$((SKIPPED + 1))',
            "        continue",
            "    fi",
            f'    jq -c \'{{instance_id: .instance_id, model_name_or_path: {model_jq_str}, model_patch: .test_result.git_patch}}\' "$f" >> "${{PRED_FILE}}"',
            "    TOTAL=$((TOTAL + 1))",
            "done",
            'echo "Collected ${TOTAL} predictions, skipped ${SKIPPED} (missing patches)"',
            'echo "Output written to ${PRED_FILE}"',
        ])

        command = "\n".join(command_lines)

        return TaskSpec(
            command=command,
            working_dir=working_dir,
            env=params.get("env", {}),
            metadata={"job_name": "swe_collect"},
        )
