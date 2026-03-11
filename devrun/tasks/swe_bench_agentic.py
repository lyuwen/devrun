"""SWEBenchAgenticTask — formulate agentic evaluation commands via Slurm Arrays."""

from __future__ import annotations

from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask


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
        llm_config = params.get("llm_config")
        dataset = params.get("dataset")
        split = params.get("split", "test")
        max_iterations = params.get("max_iterations", 100)
        select_dir = params.get("select_dir", "job_array")
        workspace = params.get("workspace", "docker")
        output_dir = params.get("output_dir", "logs/eval_run")
        task_id_format = params.get("task_id_format", "%03d")
        array = params.get("array")
        concurrency_limit = params.get("concurrency_limit")
        
        if not llm_config:
            raise ValueError("params.llm_config is required")
        if not dataset:
            raise ValueError("params.dataset is required")

        script = self._get_run_script(params)
        flags = self._get_default_flags(params)
        
        env_commands = params.get("env_commands", [])
        
        # Bash automation to resolve the SLURM_ARRAY_TASK_ID into the formatted number
        command_lines = []
        for cmd in env_commands:
            command_lines.append(cmd)
            
        command_lines.extend([
            "",
            f"DATASET={dataset}",
            "",
            "set -x",
            "",
            "cd /mnt/huawei/users/lfu/projects/openhands/benchmarks",
            "",
            f'num=$(printf "{task_id_format}\\n" ${{SLURM_ARRAY_TASK_ID:-0}})',
            f'OUTPUT_PATH="{output_dir}/${{num}}"',
            f'mkdir -p "${{OUTPUT_PATH}}"',
            f'echo "Processing ${{num}} -> ${{OUTPUT_PATH}}"',
            f'python {script} {llm_config} \\',
            f'    --dataset ${{DATASET}} \\',
            f'    --split {split} \\',
            f'    --max-iterations {max_iterations} \\',
            f'    --select {select_dir}/${{num}}.txt \\',
            f'    --workspace {workspace} \\',
            f'    --output-dir "${{OUTPUT_PATH}}" \\'
        ])
        
        # Append the extra flags
        for flag in flags:
            command_lines.append(f'    {flag} \\')
            
        # Strip trailing slashes safely
        command = "\n".join(command_lines).rstrip(" \\")
        
        resources = {}
        # Parse standard slurm resources
        for k in ["nodes", "gpus_per_node", "gpus", "walltime", "partition", "mem", "cpus_per_task", "job_name"]:
            if k in params:
                resources[k] = params[k]
                
        # Forward the --array flag to Slurm via extra_sbatch
        extra_sbatch = []
        if array:
            array_str = str(array)
            if concurrency_limit:
                array_str += f"%{concurrency_limit}"
            extra_sbatch.append(f"--array {array_str}")
            extra_sbatch.append("--oversubscribe")
            extra_sbatch.append("--output=slurm_logs/slurm-%A_%a.out")
            extra_sbatch.append("--error=slurm_logs/slurm-%A_%a.err")
            
        if extra_sbatch:
            # We assume SlurmExecutor supports extra_sbatch injected via resources
            resources["extra_sbatch"] = extra_sbatch
            
        working_dir = params.get("working_dir")
        
        return TaskSpec(
            command=command,
            resources=resources,
            env=params.get("env", {}),
            working_dir=working_dir,
            metadata={"job_name": resources.get("job_name", "swe_agentic")},
        )
