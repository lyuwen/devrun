"""WorkflowRunner — orchestrates multi-stage workflows with heartbeat polling."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devrun.db.jobs import JobStore
from devrun.models import JobStatus, WorkflowConfig, WorkflowStage
from devrun.registry import get_task_class
from devrun.router import resolve_executor

logger = logging.getLogger("devrun.workflow")


class WorkflowRunner:
    """Orchestrate multi-stage workflows with dependency resolution and heartbeat polling."""

    def __init__(
        self,
        executors_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._db = JobStore(db_path)
        self._executors_path = executors_path

    # Statuses that satisfy downstream dependency checks.
    _SATISFIED_STATUSES = frozenset({"completed", "skipped_by_user"})

    def run(
        self,
        config: WorkflowConfig,
        dry_run: bool = False,
        start_after: str | None = None,
    ) -> str:
        """Execute (or dry-run) a workflow.

        Returns the workflow_id for real runs, or a plan string for dry_run.
        """
        stages_by_name = {s.name: s for s in config.stages}

        # Validate start_after early (applies to both dry-run and real runs)
        skip_set: set[str] = set()
        if start_after:
            skip_set = self._compute_skip_set(start_after, stages_by_name)

        if dry_run:
            return self._dry_run(config, stages_by_name, skip_set=skip_set)

        # Fail fast on unfilled <REQUIRED:…> placeholders
        self._validate_no_placeholders(config)

        # Initialise per-stage state
        stages_state: dict[str, dict[str, Any]] = {}
        for stage in config.stages:
            stages_state[stage.name] = {"status": "pending", "remote_job_id": None, "db_job_id": None}

        # Pre-mark skipped stages
        if skip_set:
            for name in skip_set:
                stages_state[name] = {"status": "skipped_by_user", "remote_job_id": None, "db_job_id": None}
                logger.info("Stage %s skipped (--start-after %s)", name, start_after)

        wf_id = self._db.insert_workflow(config.workflow, stages_state)
        self._db.update_workflow(wf_id, status="running")
        logger.info("Workflow %s started: %s", wf_id, config.workflow)

        return self._heartbeat_loop(wf_id, config, stages_state)

    def run_detached(
        self,
        config: WorkflowConfig,
        start_after: str | None = None,
    ) -> str:
        """Start a workflow in a background process and return immediately.

        Returns the workflow_id.  The background process runs the heartbeat
        loop and updates the DB.  Use ``status()`` / ``logs()`` to monitor.
        """
        stages_by_name = {s.name: s for s in config.stages}

        # Validate early so the user gets errors before we fork
        self._validate_no_placeholders(config)
        skip_set: set[str] = set()
        if start_after:
            skip_set = self._compute_skip_set(start_after, stages_by_name)

        # Create workflow record so the caller can query it immediately
        stages_state: dict[str, dict[str, Any]] = {}
        for stage in config.stages:
            stages_state[stage.name] = {"status": "pending", "remote_job_id": None, "db_job_id": None}
        if skip_set:
            for name in skip_set:
                stages_state[name] = {"status": "skipped_by_user", "remote_job_id": None, "db_job_id": None}

        wf_id = self._db.insert_workflow(config.workflow, stages_state)
        self._db.update_workflow(wf_id, status="pending")

        # Serialise state for the child process.
        # skip_set is already applied to stages_state above, so the child
        # process does not need start_after — it just drives the heartbeat.
        state: dict[str, Any] = {
            "config": config.model_dump(mode="json"),
            "workflow_id": wf_id,
            "db_path": str(self._db._db_path),
        }
        if self._executors_path is not None:
            state["executors_path"] = str(self._executors_path)
        fd, state_path = tempfile.mkstemp(prefix="devrun_wf_", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh)

        # Ensure log directory exists
        log_dir = Path.home() / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"workflow_{wf_id}.log"

        with open(log_file, "w") as log_fh:
            subprocess.Popen(
                [sys.executable, "-m", "devrun.workflow", "--state-file", state_path],
                start_new_session=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )

        logger.info(
            "Workflow %s detached (PID forked). Log: %s", wf_id, log_file,
        )
        return wf_id

    def _run_existing(
        self,
        wf_id: str,
        config: WorkflowConfig,
    ) -> str:
        """Drive the heartbeat loop on an already-created workflow record.

        Used by the detached background process.  The workflow record and
        initial stages_state already exist in the DB (created by
        ``run_detached``).  The skip_set has already been applied to
        stages_state before serialisation, so no ``start_after`` is needed here.
        """
        record = self._db.get_workflow(wf_id)
        if not record:
            raise ValueError(f"Workflow {wf_id} not found in DB")
        stages_state: dict[str, dict[str, Any]] = json.loads(record["stages_state"])

        self._db.update_workflow(wf_id, status="running")
        logger.info("Workflow %s resumed (detached): %s", wf_id, config.workflow)

        return self._heartbeat_loop(wf_id, config, stages_state)

    def _heartbeat_loop(
        self,
        wf_id: str,
        config: WorkflowConfig,
        stages_state: dict[str, dict[str, Any]],
    ) -> str:
        """Core heartbeat polling loop shared by ``run()`` and ``_run_existing()``."""
        start_time = time.monotonic()

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > config.timeout:
                    logger.warning("Workflow %s timed out after %.0fs", wf_id, elapsed)
                    self._db.update_workflow(
                        wf_id,
                        status="timed_out",
                        stages_state=stages_state,
                        completed_at=datetime.now(timezone.utc),
                    )
                    return wf_id

                all_done = True
                any_failed = False

                for stage in config.stages:
                    state = stages_state[stage.name]

                    if state["status"] in ("completed", "skipped", "skipped_by_user"):
                        continue
                    if state["status"] in ("failed", "cancelled"):
                        any_failed = True
                        continue

                    all_done = False

                    if state["status"] == "pending":
                        deps = stage.depends_on or []
                        if isinstance(deps, str):
                            deps = [deps]
                        deps_met = all(
                            stages_state[d]["status"] in self._SATISFIED_STATUSES
                            for d in deps
                        )
                        if not deps_met:
                            if any(
                                stages_state[d]["status"] == "failed" for d in deps
                            ):
                                state["status"] = "skipped"
                                logger.info(
                                    "Stage %s skipped: dependency failed", stage.name
                                )
                            continue

                        try:
                            db_job_id, remote_id = self._submit_stage(stage.name, stage)
                            state["status"] = "submitted"
                            state["remote_job_id"] = remote_id
                            state["db_job_id"] = db_job_id
                            state["executor"] = stage.executor
                            logger.info(
                                "Stage %s submitted: remote_job_id=%s, db_job_id=%s",
                                stage.name, remote_id, db_job_id,
                            )
                        except Exception:
                            logger.exception("Stage %s failed to submit", stage.name)
                            state["status"] = "failed"
                            any_failed = True

                    elif state["status"] in ("submitted", "running"):
                        remote_id = state["remote_job_id"]
                        poll_status = self._poll_job_status(remote_id, stage.executor)

                        if poll_status == "completed":
                            state["status"] = "completed"
                            logger.info("Stage %s completed", stage.name)
                            db_jid = state.get("db_job_id")
                            if db_jid:
                                self._db.update_status(
                                    db_jid, JobStatus.COMPLETED,
                                    completed_at=datetime.now(timezone.utc),
                                )
                        elif poll_status == "failed":
                            state["status"] = "failed"
                            any_failed = True
                            logger.error("Stage %s failed", stage.name)
                            db_jid = state.get("db_job_id")
                            if db_jid:
                                self._db.update_status(
                                    db_jid, JobStatus.FAILED,
                                    completed_at=datetime.now(timezone.utc),
                                )
                        elif poll_status == "running":
                            state["status"] = "running"

                self._db.update_workflow(wf_id, stages_state=stages_state)

                if all_done:
                    self._db.update_workflow(
                        wf_id,
                        status="completed",
                        stages_state=stages_state,
                        completed_at=datetime.now(timezone.utc),
                    )
                    logger.info("Workflow %s completed successfully", wf_id)
                    return wf_id

                if any_failed:
                    self._db.update_workflow(
                        wf_id,
                        status="failed",
                        stages_state=stages_state,
                        completed_at=datetime.now(timezone.utc),
                    )
                    logger.error("Workflow %s failed", wf_id)
                    return wf_id

                time.sleep(config.heartbeat_interval)

        except Exception:
            logger.exception("Workflow %s encountered an unexpected error", wf_id)
            self._db.update_workflow(
                wf_id,
                status="failed",
                stages_state=stages_state,
                completed_at=datetime.now(timezone.utc),
            )
            return wf_id

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _compute_skip_set(
        start_after: str, stages_by_name: dict[str, WorkflowStage]
    ) -> set[str]:
        """Return the set of stage names to skip: *start_after* plus its transitive deps."""
        if start_after not in stages_by_name:
            raise ValueError(
                f"--start-after stage '{start_after}' does not exist. "
                f"Available stages: {sorted(stages_by_name)}"
            )
        skip: set[str] = set()
        queue = [start_after]
        while queue:
            name = queue.pop()
            if name in skip:
                continue
            skip.add(name)
            deps = stages_by_name[name].depends_on or []
            if isinstance(deps, str):
                deps = [deps]
            queue.extend(deps)
        return skip

    @staticmethod
    def _validate_no_placeholders(config: WorkflowConfig) -> None:
        """Raise ``ValueError`` if any param still contains a ``<REQUIRED:…>`` marker."""
        pattern = re.compile(r"^<REQUIRED(?::\s*.*?)?>$")
        unfilled: list[str] = []

        def _check(prefix: str, mapping: dict[str, Any]) -> None:
            for key, val in mapping.items():
                if isinstance(val, str) and pattern.match(val):
                    unfilled.append(f"  {prefix}.{key}: {val}")
                elif isinstance(val, dict):
                    _check(f"{prefix}.{key}", val)

        _check("params", config.params)
        for stage in config.stages:
            _check(f"stages.{stage.name}.params", stage.params)

        if unfilled:
            lines = ["Workflow config has unfilled required parameters:"]
            lines.extend(unfilled)
            lines.append("")
            lines.append("Set them via CLI overrides:")
            lines.append(
                "  devrun workflow run config.yaml params.model_name=mymodel params.dataset=/path/to/data"
            )
            raise ValueError("\n".join(lines))

    def _submit_stage(self, stage_name: str, stage: WorkflowStage) -> tuple[str, str]:
        """Submit a single stage: resolve task + executor, prepare, submit.

        Returns (db_job_id, remote_job_id).
        """
        task_cls = get_task_class(stage.task)
        task = task_cls()
        task_spec = task.prepare(stage.params)

        executor = resolve_executor(stage.executor, executors_path=self._executors_path)
        remote_id = executor.submit(task_spec)

        # Record in jobs table
        db_job_id = self._db.insert(
            task_name=stage.task,
            executor=stage.executor,
            parameters=stage.params,
        )
        self._db.update_status(
            db_job_id,
            JobStatus.SUBMITTED,
            remote_job_id=remote_id,
            log_path=task_spec.metadata.get("log_path"),
        )
        return db_job_id, remote_id

    def _poll_job_status(self, job_id: str, executor_name: str) -> str:
        """Check the live status of a submitted job."""
        executor = resolve_executor(executor_name, executors_path=self._executors_path)
        raw_status = executor.status(job_id)

        status_lower = raw_status.lower()
        if status_lower in ("completed", "complete", "done"):
            return "completed"
        if status_lower in ("failed", "error", "timeout", "out_of_memory"):
            return "failed"
        if status_lower in ("running", "pending", "submitted", "configuring"):
            return "running"
        return "running"  # treat unknown as still running

    def _dry_run(
        self,
        config: WorkflowConfig,
        stages_by_name: dict[str, WorkflowStage],
        skip_set: set[str] | None = None,
    ) -> str:
        """Print the full execution plan without submitting anything."""
        skip_set = skip_set or set()
        timeout_h = config.timeout / 3600
        lines = [
            f"Workflow: {config.workflow}",
            f"Timeout: {config.timeout:.0f}s ({timeout_h:.0f}h)",
            "",
        ]
        will_run_count = 0
        for i, stage in enumerate(config.stages, 1):
            skipped = stage.name in skip_set
            tag = " [SKIPPED — start-after]" if skipped else " [WILL RUN]"
            lines.append(f"Stage {i}: {stage.name}{tag}")
            lines.append(f"  Task: {stage.task}")
            lines.append(f"  Executor: {stage.executor}")
            deps = stage.depends_on or []
            if isinstance(deps, str):
                deps = [deps]
            lines.append(f"  Depends on: {', '.join(deps) if deps else '(none)'}")
            if skipped:
                lines.append("")
                continue
            will_run_count += 1
            task_cls = get_task_class(stage.task)
            task = task_cls()
            task_spec = task.prepare(stage.params)
            lines.append(f"  Working dir: {task_spec.working_dir or '(default)'}")
            # Show key params (up to 5)
            if stage.params:
                param_items = list(stage.params.items())[:5]
                param_str = ", ".join(f"{k}={v}" for k, v in param_items)
                if len(stage.params) > 5:
                    param_str += f", ... (+{len(stage.params) - 5} more)"
                lines.append(f"  Params: {param_str}")
            lines.append(f"  Command preview (first 500 chars):")
            lines.append(f"    {task_spec.command[:500]}")
            lines.append("")
        if skip_set:
            lines.append(
                f"Summary: {len(skip_set)} stage(s) skipped, "
                f"{will_run_count} stage(s) will run"
            )
        plan = "\n".join(lines)
        logger.info("Dry-run plan:\n%s", plan)
        return plan

    # -- public query methods ------------------------------------------------

    def extract_workflow_params(self, job_id: str) -> tuple[dict[str, str], str]:
        """Extract workflow-level params from an existing job record.

        Returns (dotlist_dict, task_name) where dotlist_dict has keys like
        ``"params.model_name"`` suitable for OmegaConf merging.

        Mapping priority: explicit _PARAM_MAPPING entries take precedence
        (allowing key renaming), then any remaining job params are mapped
        generically as ``params.{key}``.
        """
        record = self._db.get(job_id)
        if record is None:
            raise ValueError(
                f"Job '{job_id}' not found. Use `devrun history` to find job IDs."
            )
        job_params = record.params_dict

        # Map task-specific param names → workflow-level param names
        _PARAM_MAPPING: dict[str, str] = {
            "model_name": "params.model_name",
            "dataset": "params.dataset",
            "split": "params.split",
            "output_dir": "params.output_dir",
            "working_dir": "params.working_dir",
            "run_name": "params.run_name",
        }

        dotlist: dict[str, str] = {}
        mapped_keys: set[str] = set()
        for job_key, workflow_key in _PARAM_MAPPING.items():
            if job_key in job_params and job_params[job_key]:
                dotlist[workflow_key] = str(job_params[job_key])
                mapped_keys.add(job_key)

        # Generic fallback for unmapped params (skip known-sensitive keys)
        _SENSITIVE_KEYS = frozenset({"api_key", "token", "secret", "password", "credentials"})
        for key, val in job_params.items():
            wf_key = f"params.{key}"
            if key not in mapped_keys and wf_key not in dotlist and val:
                if key in _SENSITIVE_KEYS:
                    logger.debug("Skipping sensitive param %s from generic fallback", key)
                    continue
                dotlist[wf_key] = str(val)

        logger.info(
            "Extracted %d params from job %s (%s): %s",
            len(dotlist), job_id, record.task_name, list(dotlist.keys()),
        )
        return dotlist, record.task_name

    def detect_stage_for_task(
        self, task_name: str, config: WorkflowConfig
    ) -> str | None:
        """Find the stage name whose task type matches *task_name*."""
        for stage in config.stages:
            if stage.task == task_name:
                return stage.name
        return None

    def status(self, workflow_id: str) -> dict[str, Any] | None:
        """Return workflow record or None."""
        return self._db.get_workflow(workflow_id)

    def list_workflows(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_workflows(limit=limit)

    def cancel(self, workflow_id: str) -> None:
        """Cancel all active stages of a workflow, including remote executor jobs."""
        record = self._db.get_workflow(workflow_id)
        if not record:
            raise ValueError(f"Workflow {workflow_id} not found")
        stages_state = json.loads(record["stages_state"])
        for name, state in stages_state.items():
            remote_id = state.get("remote_job_id")
            if state["status"] in ("submitted", "running") and remote_id:
                logger.info("Cancelling stage %s (remote_job_id %s)", name, remote_id)
                executor_name = state.get("executor")
                if executor_name:
                    try:
                        executor = resolve_executor(
                            executor_name, executors_path=self._executors_path
                        )
                        executor.cancel(remote_id)
                    except Exception:
                        logger.warning(
                            "Failed to cancel remote job %s for stage %s",
                            remote_id,
                            name,
                            exc_info=True,
                        )
                state["status"] = "cancelled"
        self._db.update_workflow(
            workflow_id,
            status="cancelled",
            stages_state=stages_state,
            completed_at=datetime.now(timezone.utc),
        )

    def logs(self, workflow_id: str, stage: str | None = None) -> str:
        """Retrieve logs for a workflow or specific stage.

        For specific stages with executor info, delegates to the executor's
        ``logs()`` method.  Falls back to a status summary otherwise.
        For detached workflows, appends the background process log.
        """
        record = self._db.get_workflow(workflow_id)
        if not record:
            raise ValueError(f"Workflow {workflow_id} not found")
        stages_state = json.loads(record["stages_state"])

        if stage:
            state = stages_state.get(stage)
            if not state or not state.get("remote_job_id"):
                return f"No logs available for stage '{stage}'"
            remote_id = state["remote_job_id"]
            # Try to delegate to executor for real logs
            executor_name = state.get("executor")
            if executor_name:
                try:
                    executor = resolve_executor(
                        executor_name, executors_path=self._executors_path
                    )
                    return executor.logs(remote_id)
                except Exception:
                    logger.debug(
                        "Could not fetch executor logs for stage %s, falling back",
                        stage, exc_info=True,
                    )
            return f"Stage {stage}: remote_job_id={remote_id}, status={state['status']}"

        lines = []
        for name, state in stages_state.items():
            lines.append(
                f"{name}: status={state['status']}, remote_job_id={state.get('remote_job_id', 'N/A')}"
            )

        # Append background process log for detached workflows
        bg_log = Path.home() / ".devrun" / "logs" / f"workflow_{workflow_id}.log"
        if bg_log.exists():
            log_text = bg_log.read_text().strip()
            if log_text:
                lines.append("")
                lines.append(f"--- Background process log ({bg_log}) ---")
                lines.append(log_text)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Background process entry point (used by run_detached)
# ---------------------------------------------------------------------------


def _run_from_state_file(state_path: str) -> None:
    """Entry point for ``python -m devrun.workflow --state-file <path>``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    path = Path(state_path)
    state = json.loads(path.read_text())
    # Clean up state file now that we've read it
    path.unlink(missing_ok=True)

    config = WorkflowConfig(**state["config"])
    wf_id: str = state["workflow_id"]

    runner = WorkflowRunner(
        db_path=state.get("db_path"),
        executors_path=state.get("executors_path"),
    )
    runner._run_existing(wf_id, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detached workflow runner")
    parser.add_argument("--state-file", required=True, help="Path to serialised state JSON")
    args = parser.parse_args()
    _run_from_state_file(args.state_file)
