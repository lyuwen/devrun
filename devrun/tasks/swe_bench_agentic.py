"""SWEBenchAgenticTask — formulate agentic evaluation commands via Slurm Arrays."""

from __future__ import annotations

import logging as _logging
from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask
from devrun.utils.swebench import derive_ds_dir
from devrun.utils.templates import render_template

_agentic_logger = _logging.getLogger("devrun.tasks.swe_bench_agentic")


@register_task("swe_bench_agentic")
class SWEBenchAgenticTask(BaseTask):
    """Prepare an Agentic SWE-bench evaluation job using an OpenHands run_infer script.

    This class is designed to be inherited by other synthesis or evaluation tasks.
    It expects the executor (like SlurmExecutor) to provide an array ID via
    environment variables (e.g. SLURM_ARRAY_TASK_ID).
    """

    def _get_run_script(self, params: dict[str, Any]) -> str:
        """Override in subclasses to change the executed script."""
        return params.get("run_script", "benchmarks/swebench/run_infer.py")

    def _get_default_flags(self, params: dict[str, Any]) -> list[str]:
        """Override in subclasses to provide extra default flags."""
        return params.get("extra_flags", ["--use-legacy-tools", "--bind-dev-sdk"])

    def prepare(self, params: dict[str, Any]) -> TaskSpec:
        model_name = params.get("model_name")
        run_name = params.get("run_name")

        # --- resolve llm_config ---
        llm_config = params.get("llm_config")
        if isinstance(llm_config, dict):
            import json
            from pathlib import Path

            config_name = model_name or run_name or "custom_llm"
            llm_config_dir = Path(params.get("llm_config_dir", ".llm_config"))
            llm_config_dir.mkdir(parents=True, exist_ok=True)

            generated_path = llm_config_dir / f"{config_name}.json"
            with open(generated_path, "w", encoding="utf-8") as f:
                json.dump(llm_config, f, indent=2)

            llm_config = str(generated_path)

        if not llm_config:
            if model_name:
                llm_config_dir = params.get("llm_config_dir", ".llm_config")
                llm_config = f"{llm_config_dir}/{model_name}.json"
            else:
                raise ValueError("Either params.llm_config or params.model_name is required")

        from pathlib import Path
        if not Path(llm_config).is_file():
            _agentic_logger.warning(
                "llm_config file not found locally: %s — assuming it exists on the remote host.",
                llm_config,
            )

        # --- resolve output_dir and run_name ---
        output_dir = params.get("output_dir")
        if not output_dir:
            logs_dir = params.get("logs_dir", "logs")
            if not run_name:
                import datetime
                run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = f"{logs_dir}/{run_name}"

        dataset = params.get("dataset")
        if not dataset:
            raise ValueError("params.dataset is required")

        if not Path(dataset).exists():
            _agentic_logger.warning(
                "dataset path not found locally: %s — assuming it exists on the remote host.",
                dataset,
            )

        split = params.get("split", "test")
        max_iterations = params.get("max_iterations", 100)
        max_attempts = params.get("max_attempts", 5)
        select_dir = params.get("select_dir", "job_array")
        workspace = params.get("workspace", "docker")
        task_id_format = params.get("task_id_format", "%05d")
        working_dir = params.get("working_dir")

        # Derive DS_DIR (shared utility ensures consistency with collect task)
        ds_dir = params.get("ds_dir") or derive_ds_dir(dataset, split)

        script = self._get_run_script(params)
        flags = self._get_default_flags(params)
        env_commands = params.get("env_commands", [])
        env_vars = params.get("env", {})

        # Render the bash command via Jinja2 template
        command = render_template(
            "swe_bench_agentic.sh.j2",
            working_dir=working_dir,
            env_commands=env_commands,
            dataset=dataset,
            model_name=model_name or "",
            base_url=params.get("base_url", ""),
            api_key=params.get("api_key", ""),
            temperature=params.get("temperature", ""),
            top_p=params.get("top_p", ""),
            run_name=run_name or "",
            max_iterations=max_iterations,
            ds_dir=ds_dir,
            task_id_format=task_id_format,
            output_dir=output_dir,
            max_attempts=max_attempts,
            script=script,
            llm_config=llm_config,
            split=split,
            select_dir=select_dir,
            workspace=workspace,
            extra_flags=flags,
            env_vars=env_vars,
        )

        # --- resources and extra_sbatch ---
        resources = {}
        for k in ["nodes", "gpus_per_node", "gpus", "walltime", "partition", "mem", "cpus_per_task", "job_name"]:
            if k in params:
                resources[k] = params[k]

        array = params.get("array")
        concurrency_limit = params.get("concurrency_limit")
        extra_sbatch = []
        if array:
            array_str = str(array)
            if concurrency_limit:
                array_str += f"%{concurrency_limit}"
            extra_sbatch.append(f"--array {array_str}")
            extra_sbatch.append("--output=slurm_logs/slurm-%A_%a.out")
            extra_sbatch.append("--error=slurm_logs/slurm-%A_%a.err")
            command = "mkdir -p slurm_logs\n" + command

        if params.get("oversubscribe", False):
            extra_sbatch.append("--oversubscribe")

        if extra_sbatch:
            resources["extra_sbatch"] = extra_sbatch

        return TaskSpec(
            command=command,
            resources=resources,
            env=env_vars,
            working_dir=working_dir,
            metadata={
                "job_name": resources.get("job_name", "swe_agentic"),
                "set_e": False,  # retry loop requires set -x only, not set -e
            },
        )
