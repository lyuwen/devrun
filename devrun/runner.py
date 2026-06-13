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


def _is_negative_index(s: str) -> bool:
    """True iff *s* is a string like ``"-1"``, ``"-12"`` (negative integer)."""
    return len(s) > 1 and s[0] == "-" and s[1:].isdigit() and int(s) < 0


def get_config_dirs() -> list[Path]:
    """Return the ordered list of config search directories (lowest → highest priority)."""
    devrun_repo_root = Path(__file__).parent.parent
    return [
        devrun_repo_root / "devrun" / "configs",
        Path.home() / ".devrun" / "configs",
        Path.cwd() / ".devrun" / "configs",
    ]


def find_configs(target: str, config_dirs: list[Path] | None = None) -> list[Path]:
    """Resolve a target name to config file paths across search directories.

    Returns all matching configs in priority order (first = lowest priority).
    If *target* is a file path that exists on disk, returns ``[path]`` directly.
    Otherwise parses ``name`` or ``name/variation`` and searches the config
    directory hierarchy for ``default.yaml`` (all layers) then
    ``<variation>.yaml`` (all layers).
    """
    if config_dirs is None:
        config_dirs = get_config_dirs()

    p = Path(target)
    if p.is_file():
        return [p]

    parts = target.split("/", 1)
    config_name = parts[0]
    variation = parts[1] if len(parts) > 1 else "default"

    found: list[Path] = []
    variations_to_check = ["default"]
    if variation != "default":
        variations_to_check.append(variation)

    for v in variations_to_check:
        filename = f"{v}.yaml"
        for search_dir in config_dirs:
            candidate = search_dir / config_name / filename
            if candidate.is_file():
                found.append(candidate)

    if not found:
        raise FileNotFoundError(
            f"Config for '{target}' not found. Searched for "
            f"{variations_to_check} under '{config_name}' in: "
            + ", ".join(str(d) for d in config_dirs)
        )

    return found


def load_merged_omegaconf(
    target: str,
    overrides: list[str] | None = None,
    config_dirs: list[Path] | None = None,
):
    """Load config files for *target*, deep-merge via OmegaConf, apply overrides.

    Returns the merged ``DictConfig`` without resolving interpolations. Use
    this when you need to preserve ``${...}`` references for later resolution.
    """
    from omegaconf import OmegaConf
    import devrun.keystore  # noqa: F401  — registers ${key:…} resolver
    import devrun.presets  # noqa: F401  — registers ${preset:…} resolver
    import devrun.jobref  # noqa: F401  — registers ${jobs:…} resolver

    config_paths = find_configs(target, config_dirs)
    logger.debug("Config merge chain: %s", [str(p) for p in config_paths])

    merged_cfg = OmegaConf.load(config_paths[0])
    for extra_path in config_paths[1:]:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(extra_path))

    if overrides:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.from_dotlist(overrides))

    return merged_cfg


def load_merged_config(
    target: str,
    overrides: list[str] | None = None,
    config_dirs: list[Path] | None = None,
) -> dict:
    """Load config files for *target*, deep-merge via OmegaConf, apply overrides.

    Returns the resolved config as a plain dict (not model-validated).
    """
    from omegaconf import OmegaConf

    merged_cfg = load_merged_omegaconf(target, overrides, config_dirs)
    return OmegaConf.to_container(merged_cfg, resolve=True)


def _warn_if_no_heartbeat() -> None:
    """Log a one-line warning when the heartbeat service is not active.

    The producer path enqueues jobs in ``QUEUED`` state; without a running
    heartbeat they will sit there indefinitely. Best-effort only — never
    raises (a missing service module or unsupported platform must not block
    job enqueue).
    """
    try:
        from devrun.services import get_service

        if not get_service().is_active():
            logger.warning(
                "Heartbeat service is not active. Jobs queued but will not "
                "promote until 'devrun heartbeat start' runs."
            )
    except Exception:
        logger.debug("Heartbeat status check skipped", exc_info=True)


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
        Delegates to the module-level :func:`find_configs`.
        """
        return find_configs(target, self._config_dirs)

    def run(
        self,
        target: str,
        overrides: list[str] | None = None,
        dry_run: bool = False,
        *,
        after: list[str] | None = None,
        allow_failure_from: set[str] | None = None,
    ) -> list[str]:
        """Parse a task YAML, apply overrides, expand sweeps, enqueue all jobs. Returns list of job_ids.

        ``after`` is a list of parent job IDs the new job depends on; the
        heartbeat will not promote it until each parent reaches a terminal
        state. ``allow_failure_from`` is the subset of ``after`` whose failure
        should be tolerated (mapped to ``allow_failure=1`` on the edge).
        """
        after = list(after or [])
        allow_failure_from = set(allow_failure_from or set())
        if after:
            for parent_id in after:
                if self._db.get(parent_id) is None:
                    raise ValueError(f"--after references unknown job id: {parent_id}")
            unknown_lenient = allow_failure_from - set(after)
            if unknown_lenient:
                raise ValueError(
                    "--allow-failure-from ids must also appear in --after: "
                    f"{sorted(unknown_lenient)}"
                )
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
                    header = (
                        f"# DRY RUN [{label}]: task={cfg.task}, executor={cfg.executor}"
                    )
                    if spec.working_dir:
                        header += f", working_dir={spec.working_dir}"
                    lines = [header]
                    if spec.resources:
                        lines.append(f"# resources: {spec.resources}")
                    if spec.env:
                        lines.append(f"# env: {spec.env}")
                    lines.append("")
                    lines.append(spec.command)
                    print("\n".join(lines))
            else:
                job_ids.extend(
                    self._submit_single(
                        cfg.task, cfg.executor, params,
                        python_env=cfg.python_env,
                        after=after,
                        allow_failure_from=allow_failure_from,
                    )
                )

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

        result = record.model_dump(mode="json") if record else {}

        # Fetch live progress info (e.g. array task counts) — best-effort
        try:
            executor = resolve_executor(record.executor, self.executor_configs)
            remote_id = record.remote_job_id or job_id
            progress = executor.progress(remote_id)
            if progress:
                result["progress"] = progress
        except Exception as exc:
            logger.warning("Could not fetch progress for %s: %s", job_id, exc)

        return result

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

    def history(self, limit: int | None = 50) -> list[dict[str, Any]]:
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

    def extract_task_params(
        self, job_id: str, target_task: str
    ) -> tuple[dict[str, Any], str, str]:
        """Translate params from an existing job into ``target_task``'s schema.

        *job_id* may be either a literal job ID or a negative integer string
        ``"-N"`` selecting the N-th most recent job whose task type is
        accepted by ``target_task.import_from_job`` (newest = ``-1``).
        Delegates to the target task class's :meth:`BaseTask.import_from_job`
        hook.  Returns ``(imported_params, source_task_name, resolved_job_id)``.
        Raises :class:`ValueError` when the job is unknown or the target task
        does not support importing from the source task type.
        """
        task_cls = get_task_class(target_task)

        if _is_negative_index(job_id):
            n = -int(job_id)
            matches: list[tuple[Any, dict[str, Any]]] = []
            for rec in self._db.list_all():
                try:
                    imported = task_cls.import_from_job(rec.task_name, rec.params_dict)
                except Exception as exc:
                    logger.debug(
                        "Skipping job %s while resolving %s: import_from_job raised %s",
                        rec.job_id, job_id, exc,
                    )
                    continue
                if imported:
                    matches.append((rec, imported))
                    if len(matches) >= n:
                        break
            if len(matches) < n:
                raise ValueError(
                    f"Cannot resolve --from-job {job_id}: only {len(matches)} "
                    f"job(s) in history can be imported into '{target_task}'."
                )
            record, imported = matches[n - 1]
            logger.info(
                "Resolved --from-job %s → %s (%s)",
                job_id, record.job_id, record.task_name,
            )
        else:
            record = self._db.get(job_id)
            if record is None:
                raise ValueError(
                    f"Job '{job_id}' not found. Use `devrun history` to find job IDs."
                )
            imported = task_cls.import_from_job(record.task_name, record.params_dict)
            if not imported:
                raise ValueError(
                    f"Task '{target_task}' does not support importing from "
                    f"source task '{record.task_name}' (job {job_id})."
                )

        logger.info(
            "Imported %d params from job %s (%s → %s): %s",
            len(imported), record.job_id, record.task_name, target_task,
            list(imported.keys()),
        )
        return imported, record.task_name, record.job_id

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
        raw = load_merged_config(target, overrides, self._config_dirs)
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

    def _submit_single(
        self,
        task_name: str,
        executor_name: str,
        params: dict[str, Any],
        *,
        python_env: PythonEnv | None = None,
        after: list[str] | None = None,
        allow_failure_from: set[str] | None = None,
    ) -> list[str]:
        """Enqueue one or more jobs in QUEUED state (heartbeat does the actual submit).

        Returns a list of job IDs — one per :class:`TaskSpec` returned by
        the task's ``prepare_many`` method. Sweeps therefore produce one
        QUEUED row per shard. Each new job receives one ``job_dependencies``
        edge per id in ``after``; ids in ``allow_failure_from`` get
        ``allow_failure=1`` on their edge.
        """
        task_cls = get_task_class(task_name)
        task = task_cls()
        specs: list[TaskSpec] = task.prepare_many(params)

        after = list(after or [])
        allow_failure_from = set(allow_failure_from or set())
        params_template = yaml.safe_dump(params, sort_keys=False)

        job_ids: list[str] = []
        for task_spec in specs:
            if python_env is not None:
                task_spec.metadata["python_env"] = python_env

            job_id = self._db.enqueue(
                task_name=task_name,
                executor=executor_name,
                params_template=params_template,
                parameters=params,
            )
            for parent_id in after:
                self._db.insert_dependency(
                    child_job_id=job_id,
                    parent_job_id=parent_id,
                    allow_failure=(parent_id in allow_failure_from),
                )
            logger.info(
                "Job %s queued (task=%s, executor=%s, parents=%d)",
                job_id, task_name, executor_name, len(after),
            )
            job_ids.append(job_id)

        _warn_if_no_heartbeat()
        return job_ids

    @staticmethod
    def _map_status(raw: str) -> JobStatus:
        """Map executor-reported status strings to :class:`JobStatus`."""
        from devrun.heartbeat import map_status

        return map_status(raw)
