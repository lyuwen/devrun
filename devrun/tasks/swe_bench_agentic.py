"""SWEBenchAgenticTask — formulate agentic evaluation commands via Slurm Arrays."""

from __future__ import annotations

import copy
import json
import logging as _logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from devrun.models import TaskSpec
from devrun.registry import register_task
from devrun.tasks.base import BaseTask
from devrun.utils.swebench import derive_ds_dir
from devrun.utils.templates import render_template

_agentic_logger = _logging.getLogger("devrun.tasks.swe_bench_agentic")


# --------------- Helpers ---------------


def _parse_array_range(array_str: str) -> tuple[int, int, int]:
    """Parse a zero-padded array range string like ``"000-499"``.

    Returns ``(start, end, pad_width)`` where *pad_width* is the number of
    digits in the original string (e.g. 3 for ``"000"``).
    """
    parts = array_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid array range: {array_str!r} (expected 'START-END')")
    pad_width = len(parts[0])
    return int(parts[0]), int(parts[1]), pad_width


def _compute_shard_ranges(
    start: int, end: int, n: int, pad_width: int
) -> list[str]:
    """Divide ``[start, end]`` into *n* contiguous chunks.

    Remainder items are distributed one-per-chunk to the earlier chunks.
    Returns a list of zero-padded range strings.
    """
    total = end - start + 1
    if n <= 0:
        raise ValueError("Number of shards must be positive")
    if n > total:
        raise ValueError(
            f"Cannot create {n} shards from {total} items (array {start}-{end})"
        )
    chunk_size, remainder = divmod(total, n)
    ranges: list[str] = []
    offset = start
    for i in range(n):
        size = chunk_size + (1 if i < remainder else 0)
        chunk_end = offset + size - 1
        ranges.append(f"{offset:0{pad_width}d}-{chunk_end:0{pad_width}d}")
        offset = chunk_end + 1
    return ranges


def _format_llm_config(config: Any, env_vars: dict[str, str]) -> Any:
    """Recursively format string values in *config* using *env_vars*.

    Uses :meth:`str.format_map` with a :class:`defaultdict` so that
    unknown placeholders resolve to empty strings instead of raising.
    """
    fmt = defaultdict(str, env_vars)
    if isinstance(config, dict):
        return {k: _format_llm_config(v, env_vars) for k, v in config.items()}
    if isinstance(config, list):
        return [_format_llm_config(v, env_vars) for v in config]
    if isinstance(config, str):
        return config.format_map(fmt)
    return config


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
        llm_config_content: str | None = None  # None = file-path mode

        if isinstance(llm_config, dict):
            # Inline dict: resolve format strings ({JOB_ID}, etc.) using
            # env vars, then serialize to JSON.  The template will write
            # this to a temp file on the remote machine.
            env_vars = params.get("env", {})
            resolved = _format_llm_config(llm_config, env_vars)
            llm_config_content = json.dumps(resolved, indent=2)
            llm_config = ""  # not used when content is set
        elif not llm_config:
            if model_name:
                llm_config_dir = params.get("llm_config_dir", ".llm_config")
                llm_config = f"{llm_config_dir}/{model_name}.json"
            else:
                raise ValueError("Either params.llm_config or params.model_name is required")

        if llm_config_content is None and llm_config:
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
        run_infer_max_attempts = params.get("run_infer_max_attempts", 5)
        select_dir = params.get("select_dir", "job_array")
        workspace = params.get("workspace", "docker")
        task_id_format = params.get("task_id_format", "%03d")
        working_dir = params.get("working_dir")

        # Derive DS_DIR (shared utility ensures consistency with collect task)
        ds_dir = params.get("ds_dir") or derive_ds_dir(dataset, split)

        script = self._get_run_script(params)
        flags = self._get_default_flags(params)
        env_commands = params.get("env_commands", [])
        env_vars = params.get("env", {})
        git_safe_dirs = params.get("git_safe_dirs", [])

        # Render the bash command via Jinja2 template
        command = render_template(
            "swe_bench_agentic.sh.j2",
            env_commands=env_commands,
            git_safe_dirs=git_safe_dirs,
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
            run_infer_max_attempts=run_infer_max_attempts,
            script=script,
            llm_config=llm_config,
            llm_config_content=llm_config_content,
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

    def prepare_many(self, params: dict[str, Any]) -> list[TaskSpec]:
        """Expand ``instances`` into multiple :class:`TaskSpec` objects.

        When *params* contains an ``instances`` list, each entry is a dict
        of environment variables (e.g. ``{JOB_ID: "if-abc123"}``).  The
        top-level ``array`` range is divided evenly among the instances,
        and each instance gets its own :class:`TaskSpec` via :meth:`prepare`.

        Without ``instances``, falls back to the default single-spec behaviour.

        Shorthand: ``job_ids`` (comma-separated string) is expanded into
        ``instances`` automatically, e.g. ``"id1,id2,id3"`` becomes
        ``[{JOB_ID: "id1"}, {JOB_ID: "id2"}, {JOB_ID: "id3"}]``.
        """
        instances = params.get("instances")
        if not instances:
            job_ids = params.get("job_ids")
            if job_ids:
                instances = [{"JOB_ID": jid.strip()} for jid in str(job_ids).split(",")]
        if not instances:
            return [self.prepare(params)]

        array_str = params.get("array")
        if not array_str:
            raise ValueError("params.array is required when using instances")

        start, end, pad_width = _parse_array_range(str(array_str))
        ranges = _compute_shard_ranges(start, end, len(instances), pad_width)

        specs: list[TaskSpec] = []
        for instance, array_range in zip(instances, ranges):
            merged = copy.deepcopy(params)
            merged.pop("instances", None)
            merged["array"] = array_range
            merged.setdefault("env", {}).update(instance)
            specs.append(self.prepare(merged))
        return specs
