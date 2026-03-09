"""InferenceTask — builds inference run commands."""

from __future__ import annotations

from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask


@register_task("inference")
class InferenceTask(BaseTask):
    """Prepare an inference job."""

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        input_file = params.get("input_file", "prompts.jsonl")
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        model = params.get("model", "")
        output_file = params.get("output_file", "results.jsonl")

        command = (
            f"python inference.py "
            f"--input {input_file} "
            f"--output {output_file} "
            f"--temperature {temperature} "
            f"--max-tokens {max_tokens}"
        )
        if model:
            command += f" --model {model}"

        return TaskSpec(
            command=command,
            resources=params.get("resources", {}),
            env=params.get("env", {}),
            metadata={"job_name": f"inference_{input_file}"},
        )
