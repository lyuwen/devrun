"""Task runner — the central orchestration engine for devrun."""

from __future__ import annotations

import itertools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from devrun.db.jobs import JobStore
from devrun.models import JobStatus, PythonEnv, TaskConfig, TaskSpec
from devrun.registry import get_task_class
from devrun.router import resolve_executor, load_executor_configs

logger = logging.getLogger("devrun.runner")


def get_config_dirs() -> list[Path]:
    """Return the ordered list of config search directories (lowest → highest priority)."""
    devrun_repo_root = Path(__file__).parent.parent
    return [
        devrun_repo_root / "devrun" / "configs",
        Path.home() / ".devrun" / "configs",
        Path.cwd() / ".devrun" / "configs",
    ]


class TaskRunner:
    """Loads task configs, expands sweeps, prepares tasks, and dispatches to executors."""

    def __init__(self, executors_path: str | Path | None = None, db_path: str | Path | None = None) -> None:
        self._executors_path = executors_path
        self._executor_configs = None  # lazy
        self._db = JobStore(db_path)
        self._config_dirs = get_config_dirs()

    # ---- lazy config loading ---------------------------------------------

    @property
    def executor_configs(self):
        if self._executor_configs is None:
            self._executor_configs = load_executor_configs(self._executors_path)
        return self._executor_configs

    # ---- public API ------------------------------------------------------

    def _find_configs(self, target: str) -> list[Path]:
        """Resolve a target name to config file paths across all search directories.

        Returns all matching configs in priority order (first = lowest priority).
        This merge strategy cascades configuration definitions: it searches for 
        'default.yaml' across all layers (repo, user, project) and then overlays
        '{variation}.yaml' across all layers.
        """
        p = Path(target)
        if p.is_file():
            return [p]

        # Parse target into task_name and variation
        parts = target.split("/", 1)
        task_name = parts[0]
        variation = parts[1] if len(parts) > 1 else "default"

        found: list[Path] = []
        variations_to_check = ["default"]
        if variation != "default":
            variations_to_check.append(variation)

        # Merge strategy: load all 'default' files first, then variation layer
        for v in variations_to_check:
            filename = f"{v}.yaml"
            for search_dir in self._config_dirs:
                candidate = search_dir / task_name / filename
                if candidate.is_file():
                    found.append(candidate)

        if not found:
            raise FileNotFoundError(
                f"Config for '{target}' not found. Searched for "
                f"{variations_to_check} under '{task_name}' in: "
                + ", ".join(str(d) for d in self._config_dirs)
            )

        return found

    def run(self, target: str, overrides: list[str] | None = None, dry_run: bool = False) -> list[str]:
        """Parse a task YAML, apply overrides, expand sweeps, submit all jobs. Returns list of job_ids."""
        cfg = self._load_config(target, overrides)
        param_combos = self._expand_sweep(cfg)
        job_ids: list[str] = []

        for params in param_combos:
            if dry_run:
                task_cls = get_task_class(cfg.task)
                task = task_cls()
                specs: list[TaskSpec] = task.prepare_many(params)
                for i, spec in enumerate(specs):
                    label = f"spec {i + 1}/{len(specs)}" if len(specs) > 1 else "spec"
                    logger.info(
                        "DRY RUN [%s]: task='%s', executor='%s', working_dir=%s",
                        label, cfg.task, cfg.executor, spec.working_dir,
                    )
                    if spec.resources:
                        logger.info("  resources: %s", spec.resources)
                    if spec.env:
                        logger.info("  env: %s", spec.env)
                    logger.info("  command:\n%s", spec.command)
            else:
                job_ids.extend(self._submit_single(cfg.task, cfg.executor, params, python_env=cfg.python_env))

        return job_ids

    def status(self, job_id: str) -> dict[str, Any]:
        """Return live status for a job, refreshing from the executor if needed."""
        record = self._db.get(job_id)
        if not record:
            return {"error": f"Job {job_id} not found"}

        # If still active (or unknown), query executor for live status
        if record.status in (JobStatus.PENDING, JobStatus.SUBMITTED, JobStatus.RUNNING, JobStatus.UNKNOWN):
            try:
                executor = resolve_executor(record.executor, self.executor_configs)
                remote_id = record.remote_job_id or job_id
                live_status = executor.status(remote_id)
                mapped = self._map_status(live_status)
                if mapped != record.status:
                    completed_at = datetime.now(timezone.utc) if mapped in (JobStatus.COMPLETED, JobStatus.FAILED) else None
                    self._db.update_status(job_id, mapped, completed_at=completed_at)
                    record = self._db.get(job_id)
            except Exception as exc:
                logger.warning("Could not refresh status for %s: %s", job_id, exc)

        return record.model_dump(mode="json") if record else {}

    def logs(self, job_id: str) -> str:
        """Retrieve logs for a job."""
        record = self._db.get(job_id)
        if not record:
            return f"Job {job_id} not found"
        try:
            executor = resolve_executor(record.executor, self.executor_configs)
            remote_id = record.remote_job_id or job_id
            log_path = record.log_path
            return executor.logs(remote_id, log_path=log_path)
        except Exception as exc:
            return f"Error fetching logs: {exc}"

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent job records, refreshing active ones first."""
        records = self._db.list_all(limit)
        results = []
        for r in records:
            if r.status in (JobStatus.PENDING, JobStatus.SUBMITTED, JobStatus.RUNNING, JobStatus.UNKNOWN):
                results.append(self.status(r.job_id))
            else:
                results.append(r.model_dump(mode="json"))
        return results

    def rerun(self, job_id: str) -> list[str]:
        """Re-submit a previous job with the same parameters."""
        record = self._db.get(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found")
        params = record.params_dict
        return self._submit_single(record.task_name, record.executor, params)

    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        record = self._db.get(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found")

        # Forcefully update the job status from the executor before deciding
        self.status(job_id)
        record = self._db.get(job_id)

        # Do not cancel if it's already in a terminal state
        if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            logger.info("Job %s is already %s, skipping cancellation.", job_id, record.status)
            raise ValueError(f"Job {job_id} is already {record.status}.")

        try:
            executor = resolve_executor(record.executor, self.executor_configs)
            remote_id = record.remote_job_id or job_id
            executor.cancel(remote_id)
            self._db.update_status(job_id, JobStatus.CANCELLED, completed_at=datetime.now(timezone.utc))
            logger.info("Job %s cancelled successfully.", job_id)
        except Exception as exc:
            logger.error("Failed to cancel job %s: %s", job_id, exc)
            raise

    # ---- internal --------------------------------------------------------

    def _load_config(self, target: str, overrides: list[str] | None = None) -> TaskConfig:
        from omegaconf import OmegaConf

        config_paths = self._find_configs(target)
        logger.debug("Config merge chain: %s", [str(p) for p in config_paths])

        # Merge all configs in order (first = base, last = highest priority)
        merged_cfg = OmegaConf.load(config_paths[0])
        for extra_path in config_paths[1:]:
            merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(extra_path))

        # CLI overrides have the highest priority
        if overrides:
            merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.from_dotlist(overrides))

        raw = OmegaConf.to_container(merged_cfg, resolve=True)
        return TaskConfig(**raw)

    @staticmethod
    def _expand_sweep(cfg: TaskConfig) -> list[dict[str, Any]]:
        """Expand a sweep config into a list of concrete param dicts."""
        if not cfg.sweep:
            return [cfg.params.copy()]

        keys = list(cfg.sweep.keys())
        value_lists = [cfg.sweep[k] for k in keys]
        combos: list[dict[str, Any]] = []
        for values in itertools.product(*value_lists):
            merged = cfg.params.copy()
            for k, v in zip(keys, values):
                merged[k] = v
            combos.append(merged)

        logger.info("Sweep expanded to %d combinations", len(combos))
        return combos

    def _submit_single(self, task_name: str, executor_name: str, params: dict[str, Any], *, python_env: PythonEnv | None = None) -> list[str]:
        """Prepare and submit one or more jobs (multi-shard aware).

        Returns a list of job IDs — one per :class:`TaskSpec` returned by
        the task's ``prepare_many`` method.
        """
        # 1. Resolve task plugin and expand shards
        task_cls = get_task_class(task_name)
        task = task_cls()
        specs: list[TaskSpec] = task.prepare_many(params)

        job_ids: list[str] = []
        for task_spec in specs:
            # 2. Propagate task-level python_env into metadata for executors to consume
            if python_env is not None:
                task_spec.metadata["python_env"] = python_env

            # 3. Record in DB
            job_id = self._db.insert(task_name, executor_name, params)

            # 4. Resolve executor and submit
            try:
                executor = resolve_executor(executor_name, self.executor_configs)
                self._db.update_status(job_id, JobStatus.SUBMITTED)
                remote_job_id = executor.submit_with_retry(task_spec, retries=3, retry_delay=5.0)
                log_path = task_spec.metadata.get("log_path")
                self._db.update_status(job_id, JobStatus.RUNNING, remote_job_id=remote_job_id, log_path=log_path)
                logger.info(
                    "Job %s submitted → executor=%s, remote_id=%s",
                    job_id, executor_name, remote_job_id,
                )
            except Exception as exc:
                self._db.update_status(job_id, JobStatus.FAILED, completed_at=datetime.now(timezone.utc))
                logger.error("Job %s failed to submit: %s", job_id, exc)
                raise

            job_ids.append(job_id)

        return job_ids

    @staticmethod
    def _map_status(raw: str) -> JobStatus:
        """Map executor-reported status strings to :class:`JobStatus`."""
        mapping = {
            "running": JobStatus.RUNNING,
            "pending": JobStatus.PENDING,
            "completed": JobStatus.COMPLETED,
            "done": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED,
            "cancelled": JobStatus.CANCELLED,
            "timeout": JobStatus.FAILED,
            "completing": JobStatus.RUNNING,
            "node_fail": JobStatus.FAILED,
            "out_of_memory": JobStatus.FAILED,
            "preempted": JobStatus.FAILED,
            "boot_fail": JobStatus.FAILED,
            "deadline": JobStatus.FAILED,
            "stopped": JobStatus.FAILED,
            "suspended": JobStatus.RUNNING,
            "requeued": JobStatus.PENDING,
            "resizing": JobStatus.RUNNING,
        }
        return mapping.get(raw.lower(), JobStatus.UNKNOWN)
