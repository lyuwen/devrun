"""DeployRayTask — builds Ray Serve deployment commands."""

from __future__ import annotations

from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask


@register_task("deploy_ray")
class DeployRayTask(BaseTask):
    """Prepare a Ray Serve deployment."""

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        app_module = params.get("app_module", "serve_app:app")
        num_replicas = params.get("num_replicas", 1)
        num_gpus = params.get("num_gpus", 1)
        ray_address = params.get("ray_address", "auto")
        port = params.get("port", 8000)

        command = (
            f"serve run {app_module} "
            f"--address={ray_address} "
            f"--host=0.0.0.0 --port={port}"
        )

        resources = {
            "num_replicas": num_replicas,
            "num_gpus": num_gpus,
        }

        env = params.get("env", {})
        env.setdefault("RAY_ADDRESS", ray_address)

        return TaskSpec(
            command=command,
            resources=resources,
            env=env,
            metadata={"job_name": f"deploy_ray_{app_module}"},
        )
