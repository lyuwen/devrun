"""SWEBenchEvalTask — formulate evaluation commands for SWE-bench harness."""

from __future__ import annotations

import datetime
from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask


@register_task("swe_bench_eval")
class SWEBenchEvalTask(BaseTask):
    """Prepare a SWE-bench evaluation job."""

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        dataset_name = params.get("dataset_name")
        split = params.get("split", "test")
        max_workers = params.get("max_workers", 8)
        run_id = params.get("run_id")
        predictions_path = params.get("predictions_path", "predictions.jsonl")
        namespace = params.get("namespace")

        if not dataset_name:
            raise ValueError("params.dataset_name is required")

        # Determine working directory if specified, else use None.
        working_dir = params.get("working_dir")

        if not run_id:
            # Generate run_id based on datetime if not provided.
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        command_parts = [
            f"python -m swebench.harness.run_evaluation",
            f"    --dataset_name {dataset_name}",
            f"    --split {split}",
            f"    --max_workers {max_workers}",
            f"    --run_id {run_id}",
            f"    --predictions_path {predictions_path}",
        ]
        if namespace:
            command_parts.append(f"    --namespace {namespace}")

        command = " \\\n".join(command_parts)

        resources = {}
        # Allow passing down arbitrary resources
        for k in ["nodes", "gpus_per_node", "gpus", "walltime", "partition", "mem", "cpus_per_task"]:
            if k in params:
                resources[k] = params[k]

        return TaskSpec(
            command=command,
            resources=resources,
            env=params.get("env", {}),
            working_dir=working_dir,
            metadata={"job_name": f"swe_eval_{run_id}_{dataset_name.split('/')[-1]}"},
        )
