"""WorkflowRunner — orchestrates multi-stage workflows with heartbeat polling."""
from __future__ import annotations

import json
import logging
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

    def run(self, config: WorkflowConfig, dry_run: bool = False) -> str:
        """Execute (or dry-run) a workflow.

        Returns the workflow_id for real runs, or a plan string for dry_run.
        """
        stages_by_name = {s.name: s for s in config.stages}

        if dry_run:
            return self._dry_run(config, stages_by_name)

        # Initialise per-stage state
        stages_state: dict[str, dict[str, Any]] = {}
        for stage in config.stages:
            stages_state[stage.name] = {"status": "pending", "job_id": None}

        wf_id = self._db.insert_workflow(config.workflow, stages_state)
        self._db.update_workflow(wf_id, status="running")
        logger.info("Workflow %s started: %s", wf_id, config.workflow)

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

                    if state["status"] in ("completed", "skipped"):
                        continue
                    if state["status"] == "failed":
                        any_failed = True
                        continue

                    all_done = False

                    if state["status"] == "pending":
                        deps = stage.depends_on or []
                        if isinstance(deps, str):
                            deps = [deps]
                        deps_met = all(
                            stages_state[d]["status"] == "completed" for d in deps
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
                            job_id = self._submit_stage(stage.name, stage)
                            state["status"] = "submitted"
                            state["job_id"] = job_id
                            state["executor"] = stage.executor
                            logger.info(
                                "Stage %s submitted: job_id=%s", stage.name, job_id
                            )
                        except Exception:
                            logger.exception("Stage %s failed to submit", stage.name)
                            state["status"] = "failed"
                            any_failed = True

                    elif state["status"] in ("submitted", "running"):
                        job_id = state["job_id"]
                        poll_status = self._poll_job_status(job_id, stage.executor)

                        if poll_status == "completed":
                            state["status"] = "completed"
                            logger.info("Stage %s completed", stage.name)
                        elif poll_status == "failed":
                            state["status"] = "failed"
                            any_failed = True
                            logger.error("Stage %s failed", stage.name)
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

    def _submit_stage(self, stage_name: str, stage: WorkflowStage) -> str:
        """Submit a single stage: resolve task + executor, prepare, submit."""
        task_cls = get_task_class(stage.task)
        task = task_cls()
        task_spec = task.prepare(stage.params)

        executor = resolve_executor(stage.executor, executors_path=self._executors_path)
        remote_id = executor.submit(task_spec)

        # Record in jobs table
        job_id = self._db.insert(
            task_name=stage.task,
            executor=stage.executor,
            parameters=stage.params,
        )
        self._db.update_status(
            job_id,
            JobStatus.SUBMITTED,
            remote_job_id=remote_id,
            log_path=task_spec.metadata.get("log_path"),
        )
        return remote_id

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
        self, config: WorkflowConfig, stages_by_name: dict[str, WorkflowStage]
    ) -> str:
        """Print the full execution plan without submitting anything."""
        lines = [f"Workflow: {config.workflow}", f"Timeout: {config.timeout}s", ""]
        for i, stage in enumerate(config.stages, 1):
            deps = stage.depends_on or []
            if isinstance(deps, str):
                deps = [deps]
            task_cls = get_task_class(stage.task)
            task = task_cls()
            task_spec = task.prepare(stage.params)
            lines.append(f"Stage {i}: {stage.name}")
            lines.append(f"  Task: {stage.task}")
            lines.append(f"  Executor: {stage.executor}")
            lines.append(f"  Depends on: {deps or '(none)'}")
            lines.append(f"  Working dir: {task_spec.working_dir or '(default)'}")
            lines.append(f"  Command preview (first 200 chars):")
            lines.append(f"    {task_spec.command[:200]}...")
            lines.append("")
        plan = "\n".join(lines)
        logger.info("Dry-run plan:\n%s", plan)
        return plan

    # -- public query methods ------------------------------------------------

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
            if state["status"] in ("submitted", "running") and state.get("job_id"):
                logger.info("Cancelling stage %s (job %s)", name, state["job_id"])
                # Cancel the actual remote job if executor info is available
                executor_name = state.get("executor")
                if executor_name:
                    try:
                        executor = resolve_executor(
                            executor_name, executors_path=self._executors_path
                        )
                        executor.cancel(state["job_id"])
                    except Exception:
                        logger.warning(
                            "Failed to cancel remote job %s for stage %s",
                            state["job_id"],
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
        """Retrieve logs summary for a workflow or specific stage."""
        record = self._db.get_workflow(workflow_id)
        if not record:
            raise ValueError(f"Workflow {workflow_id} not found")
        stages_state = json.loads(record["stages_state"])
        if stage:
            state = stages_state.get(stage)
            if not state or not state.get("job_id"):
                return f"No logs available for stage '{stage}'"
            return f"Stage {stage}: job_id={state['job_id']}, status={state['status']}"
        lines = []
        for name, state in stages_state.items():
            lines.append(
                f"{name}: status={state['status']}, job_id={state.get('job_id', 'N/A')}"
            )
        return "\n".join(lines)
