"""EvalTask — builds evaluation commands for LLM benchmarks."""

from __future__ import annotations

from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask


@register_task("eval")
class EvalTask(BaseTask):
    """Prepare an LLM evaluation job."""

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        model = params.get("model", "default-model")
        dataset = params.get("dataset", "default-dataset")
        batch_size = params.get("batch_size", 8)
        extra_args = params.get("extra_args", "")

        command = (
            f"python eval.py "
            f"--model {model} "
            f"--dataset {dataset} "
            f"--batch-size {batch_size}"
        )
        if extra_args:
            command += f" {extra_args}"

        resources = {}
        if "nodes" in params:
            resources["nodes"] = params["nodes"]
        if "gpus_per_node" in params:
            resources["gpus_per_node"] = params["gpus_per_node"]
        if "gpus" in params:
            resources["gpus"] = params["gpus"]
        if "walltime" in params:
            resources["walltime"] = params["walltime"]
        if "partition" in params:
            resources["partition"] = params["partition"]

        return TaskSpec(
            command=command,
            resources=resources,
            env=params.get("env", {}),
            metadata={"job_name": f"eval_{model}_{dataset}"},
        )
