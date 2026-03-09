"""Task runner — the central orchestration engine for devrun."""

from __future__ import annotations

import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from devrun.db.jobs import JobStore
from devrun.models import JobStatus, TaskConfig, TaskSpec
from devrun.registry import get_task_class
from devrun.router import resolve_executor, load_executor_configs

logger = logging.getLogger("devrun.runner")


class TaskRunner:
    """Loads task configs, expands sweeps, prepares tasks, and dispatches to executors."""

    def __init__(self, executors_path: str | Path | None = None, db_path: str | Path | None = None) -> None:
        self._executors_path = executors_path
        self._executor_configs = None  # lazy
        self._db = JobStore(db_path)

    # ---- lazy config loading ---------------------------------------------

    @property
    def executor_configs(self):
        if self._executor_configs is None:
            self._executor_configs = load_executor_configs(self._executors_path)
        return self._executor_configs

    # ---- public API ------------------------------------------------------

    def run(self, config_path: str) -> list[str]:
        """Parse a task YAML, expand sweeps, submit all jobs. Returns list of job_ids."""
        cfg = self._load_config(config_path)
        param_combos = self._expand_sweep(cfg)
        job_ids: list[str] = []

        for params in param_combos:
            job_id = self._submit_single(cfg.task, cfg.executor, params)
            job_ids.append(job_id)

        return job_ids

    def status(self, job_id: str) -> dict[str, Any]:
        """Return live status for a job, refreshing from the executor if needed."""
        record = self._db.get(job_id)
        if not record:
            return {"error": f"Job {job_id} not found"}

        # If still active, query executor for live status
        if record.status in (JobStatus.PENDING, JobStatus.SUBMITTED, JobStatus.RUNNING):
            try:
                executor = resolve_executor(record.executor, self.executor_configs)
                remote_id = record.remote_job_id or job_id
                live_status = executor.status(remote_id)
                mapped = self._map_status(live_status)
                if mapped != record.status:
                    completed_at = datetime.utcnow() if mapped in (JobStatus.COMPLETED, JobStatus.FAILED) else None
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
            return executor.logs(remote_id)
        except Exception as exc:
            return f"Error fetching logs: {exc}"

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent job records."""
        return [r.model_dump(mode="json") for r in self._db.list_all(limit)]

    def rerun(self, job_id: str) -> list[str]:
        """Re-submit a previous job with the same parameters."""
        record = self._db.get(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found")
        params = record.params_dict
        new_id = self._submit_single(record.task_name, record.executor, params)
        return [new_id]

    # ---- internal --------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> TaskConfig:
        with open(config_path) as fh:
            raw = yaml.safe_load(fh)
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

    def _submit_single(self, task_name: str, executor_name: str, params: dict[str, Any]) -> str:
        """Prepare and submit one job."""
        # 1. Resolve task plugin
        task_cls = get_task_class(task_name)
        task = task_cls()
        task_spec: TaskSpec = task.prepare(params)

        # 2. Record in DB
        job_id = self._db.insert(task_name, executor_name, params)

        # 3. Resolve executor and submit
        try:
            executor = resolve_executor(executor_name, self.executor_configs)
            self._db.update_status(job_id, JobStatus.SUBMITTED)
            remote_job_id = executor.submit(task_spec)
            self._db.update_status(job_id, JobStatus.RUNNING, remote_job_id=remote_job_id)
            logger.info(
                "Job %s submitted → executor=%s, remote_id=%s",
                job_id, executor_name, remote_job_id,
            )
        except Exception as exc:
            self._db.update_status(job_id, JobStatus.FAILED, completed_at=datetime.utcnow())
            logger.error("Job %s failed to submit: %s", job_id, exc)
            raise

        return job_id

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
        }
        return mapping.get(raw.lower(), JobStatus.UNKNOWN)
