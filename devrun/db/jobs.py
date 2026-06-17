"""SQLite-backed job store for devrun."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from devrun.models import JobRecord, JobStatus

logger = logging.getLogger("devrun.db.jobs")

_DEFAULT_DB_PATH = Path.home() / ".devrun" / "jobs.db"


def default_db_path() -> Path:
    """Return the default SQLite path used by ``JobStore`` when none is supplied."""
    return _DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    task_name     TEXT NOT NULL,
    executor      TEXT NOT NULL,
    parameters    TEXT DEFAULT '',
    remote_job_id TEXT,
    status        TEXT DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    completed_at  TEXT,
    log_path      TEXT
);
"""

_WORKFLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id   TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    stages_state  TEXT DEFAULT '{}',
    status        TEXT DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    completed_at  TEXT
);
"""

_JOB_DEPENDENCIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_dependencies (
    child_job_id  TEXT NOT NULL,
    parent_job_id TEXT NOT NULL,
    allow_failure INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (child_job_id, parent_job_id),
    FOREIGN KEY (child_job_id)  REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobdeps_child  ON job_dependencies(child_job_id);
CREATE INDEX IF NOT EXISTS idx_jobdeps_parent ON job_dependencies(parent_job_id);
"""

_JOBS_MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN params_template TEXT",
    "ALTER TABLE jobs ADD COLUMN skip_reason TEXT",
    "ALTER TABLE jobs ADD COLUMN claimed_by TEXT",
    "ALTER TABLE jobs ADD COLUMN claimed_at TEXT",
    "ALTER TABLE jobs ADD COLUMN claim_expires_at TEXT",
]

_WORKFLOW_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_jobs (
    workflow_id    TEXT NOT NULL,
    stage_name     TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    job_id         TEXT,
    source_job_id  TEXT,
    PRIMARY KEY (workflow_id, stage_name),
    FOREIGN KEY (workflow_id)   REFERENCES workflows(workflow_id)  ON DELETE CASCADE,
    FOREIGN KEY (job_id)        REFERENCES jobs(job_id)            ON DELETE SET NULL,
    FOREIGN KEY (source_job_id) REFERENCES jobs(job_id)            ON DELETE SET NULL,
    CHECK (job_id IS NOT NULL OR source_job_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_wfjobs_workflow ON workflow_jobs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wfjobs_job      ON workflow_jobs(job_id);
"""

_WORKFLOWS_MIGRATIONS = [
    "ALTER TABLE workflows ADD COLUMN deadline_at TEXT",
]


@dataclass
class WorkflowStageRow:
    """One row of a workflow plan handed to ``JobStore.enqueue_workflow``."""

    stage_name: str
    ordinal: int
    job_id: str | None
    source_job_id: str | None
    task_name: str | None
    executor: str | None
    params_template: str | None
    parameters: dict[str, Any] | None


@dataclass
class Dependency:
    """A persisted parent edge for a child job."""

    child_job_id: str
    parent_job_id: str
    allow_failure: int


class JobStore:
    """Thin wrapper around an SQLite database for job records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.execute(_WORKFLOW_SCHEMA)
        self._conn.executescript(_JOB_DEPENDENCIES_SCHEMA)
        self._conn.executescript(_WORKFLOW_JOBS_SCHEMA)
        for migration in _JOBS_MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        for migration in _WORKFLOWS_MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        self._conn.commit()
        logger.debug("Job store initialised at %s", self._db_path)

    # ---- mutations -------------------------------------------------------

    def insert(
        self,
        task_name: str,
        executor: str,
        parameters: dict[str, Any] | None = None,
        log_path: str | None = None,
    ) -> str:
        """Insert a new job and return its ``job_id``."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO jobs (job_id, task_name, executor, parameters, status, created_at, log_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, task_name, executor, json.dumps(parameters or {}), JobStatus.PENDING, now, log_path),
        )
        self._conn.commit()
        logger.info("Inserted job %s (task=%s, executor=%s)", job_id, task_name, executor)
        return job_id

    def enqueue(
        self,
        *,
        task_name: str,
        executor: str,
        params_template: str,
        parameters: dict[str, Any] | None = None,
        initial_status: JobStatus = JobStatus.QUEUED,
    ) -> str:
        """Enqueue a new job with unresolved params_template."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        status_value = initial_status.value if isinstance(initial_status, JobStatus) else initial_status
        self._conn.execute(
            "INSERT INTO jobs (job_id, task_name, executor, params_template, parameters, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                task_name,
                executor,
                params_template,
                json.dumps(parameters or {}),
                status_value,
                now,
            ),
        )
        self._conn.commit()
        logger.info(
            "Enqueued job %s (task=%s, executor=%s, status=%s)",
            job_id, task_name, executor, status_value,
        )
        return job_id

    def insert_dependency(
        self,
        *,
        child_job_id: str,
        parent_job_id: str,
        allow_failure: bool,
    ) -> None:
        """Insert a job dependency edge."""
        self._conn.execute(
            "INSERT INTO job_dependencies (child_job_id, parent_job_id, allow_failure) VALUES (?, ?, ?)",
            (child_job_id, parent_job_id, 1 if allow_failure else 0),
        )
        self._conn.commit()

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        remote_job_id: str | None = None,
        completed_at: datetime | None = None,
        log_path: str | None = None,
    ) -> None:
        """Update the status (and optional fields) of an existing job."""
        fields = ["status = ?"]
        values: list[Any] = [status.value if isinstance(status, JobStatus) else status]
        if remote_job_id is not None:
            fields.append("remote_job_id = ?")
            values.append(remote_job_id)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at.isoformat())
        if log_path is not None:
            fields.append("log_path = ?")
            values.append(log_path)
        values.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
        self._conn.execute(sql, values)
        self._conn.commit()

    # ---- queries ---------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self, limit: int | None = None) -> list[JobRecord]:
        if limit is None:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_remote_id(self, remote_job_id: str) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE remote_job_id = ?", (remote_job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        d = dict(row)
        return JobRecord(
            job_id=d["job_id"],
            task_name=d["task_name"],
            executor=d["executor"],
            parameters=d.get("parameters", ""),
            params_template=d.get("params_template"),
            remote_job_id=d.get("remote_job_id"),
            status=d.get("status", "pending"),
            created_at=d["created_at"],
            completed_at=d.get("completed_at"),
            log_path=d.get("log_path"),
            skip_reason=d.get("skip_reason"),
            claimed_by=d.get("claimed_by"),
            claimed_at=d.get("claimed_at"),
            claim_expires_at=d.get("claim_expires_at"),
        )

    # ---- workflow mutations -------------------------------------------------

    def insert_workflow(self, workflow_name: str, stages_state: dict[str, Any]) -> str:
        """Insert a new workflow and return its ``workflow_id``."""
        workflow_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO workflows (workflow_id, workflow_name, stages_state, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (workflow_id, workflow_name, json.dumps(stages_state), "pending", now),
        )
        self._conn.commit()
        logger.info("Inserted workflow %s (name=%s)", workflow_id, workflow_name)
        return workflow_id

    def update_workflow(
        self,
        workflow_id: str,
        *,
        status: str | None = None,
        stages_state: dict[str, Any] | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """Update workflow status and/or stages_state."""
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if stages_state is not None:
            fields.append("stages_state = ?")
            values.append(json.dumps(stages_state))
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at.isoformat())
        if not fields:
            return
        values.append(workflow_id)
        self._conn.execute(f"UPDATE workflows SET {', '.join(fields)} WHERE workflow_id = ?", values)
        self._conn.commit()

    # ---- workflow queries ---------------------------------------------------

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        """Return a workflow record as a dict, or None."""
        row = self._conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
        return dict(row) if row else None

    def list_workflows(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent workflows ordered by creation time."""
        rows = self._conn.execute("SELECT * FROM workflows ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- workflow enqueue (atomic) -----------------------------------------

    def enqueue_workflow(
        self,
        *,
        workflow_name: str,
        deadline_at: datetime | None,
        stage_rows: list[WorkflowStageRow],
        edges: list[tuple[str, str, bool]],
    ) -> str:
        """Atomically insert a workflow plan: workflow row + per-stage rows + edges."""
        workflow_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO workflows (workflow_id, workflow_name, stages_state, status, "
                "created_at, deadline_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    workflow_id,
                    workflow_name,
                    "{}",
                    JobStatus.QUEUED.value,
                    now,
                    deadline_at.isoformat() if deadline_at else None,
                ),
            )
            for r in stage_rows:
                if r.job_id is not None:
                    self._conn.execute(
                        "INSERT INTO jobs (job_id, task_name, executor, params_template, "
                        "parameters, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            r.job_id,
                            r.task_name,
                            r.executor,
                            r.params_template,
                            json.dumps(r.parameters or {}),
                            JobStatus.QUEUED.value,
                            now,
                        ),
                    )
                self._conn.execute(
                    "INSERT INTO workflow_jobs (workflow_id, stage_name, ordinal, "
                    "job_id, source_job_id) VALUES (?, ?, ?, ?, ?)",
                    (workflow_id, r.stage_name, r.ordinal, r.job_id, r.source_job_id),
                )
            for child, parent, allow_fail in edges:
                self._conn.execute(
                    "INSERT INTO job_dependencies (child_job_id, parent_job_id, allow_failure) "
                    "VALUES (?, ?, ?)",
                    (child, parent, 1 if allow_fail else 0),
                )
        logger.info(
            "Enqueued workflow %s (name=%s, stages=%d, edges=%d)",
            workflow_id, workflow_name, len(stage_rows), len(edges),
        )
        return workflow_id

    def get_workflow_stages(self, workflow_id: str) -> list[WorkflowStageRow]:
        """Return stage rows for *workflow_id* ordered by ordinal."""
        rows = self._conn.execute(
            "SELECT wj.stage_name, wj.ordinal, wj.job_id, wj.source_job_id, "
            "       j.task_name, j.executor, j.params_template, j.parameters "
            "FROM workflow_jobs wj "
            "LEFT JOIN jobs j ON j.job_id = wj.job_id "
            "WHERE wj.workflow_id = ? ORDER BY wj.ordinal",
            (workflow_id,),
        ).fetchall()
        result: list[WorkflowStageRow] = []
        for r in rows:
            params = json.loads(r["parameters"]) if r["parameters"] else None
            result.append(
                WorkflowStageRow(
                    stage_name=r["stage_name"],
                    ordinal=r["ordinal"],
                    job_id=r["job_id"],
                    source_job_id=r["source_job_id"],
                    task_name=r["task_name"],
                    executor=r["executor"],
                    params_template=r["params_template"],
                    parameters=params,
                )
            )
        return result

    def list_dependencies(self, child_job_id: str) -> list[Dependency]:
        """Return all parent edges for *child_job_id*."""
        rows = self._conn.execute(
            "SELECT child_job_id, parent_job_id, allow_failure FROM job_dependencies "
            "WHERE child_job_id = ?",
            (child_job_id,),
        ).fetchall()
        return [
            Dependency(
                child_job_id=r["child_job_id"],
                parent_job_id=r["parent_job_id"],
                allow_failure=r["allow_failure"],
            )
            for r in rows
        ]

    # ---- atomic claim & recovery -------------------------------------------

    def claim_for_submit(
        self,
        *,
        job_id: str,
        instance_id: str,
        lease_seconds: int,
    ) -> bool:
        """Compare-and-set QUEUED -> SUBMITTING. Returns True only for the winner."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=lease_seconds)
        cursor = self._conn.execute(
            "UPDATE jobs SET status = ?, claimed_by = ?, claimed_at = ?, claim_expires_at = ? "
            "WHERE job_id = ? AND status = ?",
            (
                JobStatus.SUBMITTING.value,
                instance_id,
                now.isoformat(),
                expires.isoformat(),
                job_id,
                JobStatus.QUEUED.value,
            ),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def finalize_submit(
        self,
        *,
        job_id: str,
        remote_job_id: str | None,
        log_path: str | None,
        resolved_parameters: dict[str, Any],
    ) -> None:
        """Transition SUBMITTING -> SUBMITTED and persist resolved fields + clear claim."""
        self._conn.execute(
            "UPDATE jobs SET status = ?, remote_job_id = ?, log_path = ?, parameters = ?, "
            "claimed_by = NULL, claimed_at = NULL, claim_expires_at = NULL, skip_reason = NULL "
            "WHERE job_id = ? AND status = ?",
            (
                JobStatus.SUBMITTED.value,
                remote_job_id,
                log_path,
                json.dumps(resolved_parameters),
                job_id,
                JobStatus.SUBMITTING.value,
            ),
        )
        self._conn.commit()

    def fail_promotion(self, *, job_id: str, skip_reason: str) -> None:
        """Mark job FAILED with *skip_reason*, clearing claim columns."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE jobs SET status = ?, skip_reason = COALESCE(skip_reason || ' ; ', '') || ?, "
            "claimed_by = NULL, claimed_at = NULL, claim_expires_at = NULL, "
            "completed_at = ? WHERE job_id = ?",
            (JobStatus.FAILED.value, skip_reason, now, job_id),
        )
        self._conn.commit()

    def reclaim_expired_leases(self, *, now: datetime) -> list[str]:
        """Reclaim SUBMITTING rows whose lease has expired and have no remote_job_id."""
        now_iso = now.isoformat()
        rows = self._conn.execute(
            "SELECT job_id FROM jobs WHERE status = ? AND claim_expires_at < ? "
            "AND remote_job_id IS NULL",
            (JobStatus.SUBMITTING.value, now_iso),
        ).fetchall()
        reclaimed = [r["job_id"] for r in rows]
        if not reclaimed:
            return []
        self._conn.execute(
            "UPDATE jobs SET status = ?, claimed_by = NULL, claimed_at = NULL, "
            "claim_expires_at = NULL, "
            "skip_reason = COALESCE(skip_reason || ' ; ', '') || 'reclaimed after stale lease' "
            "WHERE status = ? AND claim_expires_at < ? AND remote_job_id IS NULL",
            (JobStatus.QUEUED.value, JobStatus.SUBMITTING.value, now_iso),
        )
        self._conn.commit()
        logger.info("Reclaimed %d expired lease(s): %s", len(reclaimed), reclaimed)
        return reclaimed

    # ---- heartbeat read/write helpers --------------------------------------

    def cascade_skip_dependents(self) -> list[str]:
        """Transition QUEUED children whose blocking parent is failed/skipped/cancelled/timed_out."""
        rows = self._conn.execute(
            "SELECT DISTINCT j.job_id, p.job_id AS parent_id, p.status AS parent_status "
            "FROM jobs j "
            "JOIN job_dependencies d ON d.child_job_id = j.job_id "
            "JOIN jobs p ON p.job_id = d.parent_job_id "
            "WHERE j.status = ? AND d.allow_failure = 0 AND p.status IN (?, ?, ?, ?)",
            (
                JobStatus.QUEUED.value,
                JobStatus.FAILED.value,
                JobStatus.SKIPPED.value,
                JobStatus.CANCELLED.value,
                JobStatus.TIMED_OUT.value,
            ),
        ).fetchall()
        if not rows:
            return []
        now = datetime.now(timezone.utc).isoformat()
        skipped: list[str] = []
        for r in rows:
            child_id = r["job_id"]
            reason = f"parent {r['parent_id']} {r['parent_status']}"
            cursor = self._conn.execute(
                "UPDATE jobs SET status = ?, skip_reason = ?, completed_at = ? "
                "WHERE job_id = ? AND status = ?",
                (JobStatus.SKIPPED.value, reason, now, child_id, JobStatus.QUEUED.value),
            )
            if cursor.rowcount == 1:
                skipped.append(child_id)
        self._conn.commit()
        if skipped:
            logger.info("Cascade-skipped %d job(s): %s", len(skipped), skipped)
        return skipped

    def fetch_ready_queued(self, limit: int = 100) -> list[JobRecord]:
        """Return QUEUED jobs whose every parent edge is satisfied."""
        rows = self._conn.execute(
            "SELECT * FROM jobs j WHERE j.status = ? AND NOT EXISTS ("
            "  SELECT 1 FROM job_dependencies d "
            "  JOIN jobs p ON p.job_id = d.parent_job_id "
            "  WHERE d.child_job_id = j.job_id "
            "    AND NOT (p.status = ? OR (d.allow_failure = 1 AND p.status IN (?, ?, ?)))"
            ") ORDER BY j.created_at LIMIT ?",
            (
                JobStatus.QUEUED.value,
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.SKIPPED.value,
                JobStatus.CANCELLED.value,
                limit,
            ),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def fetch_active_jobs(self) -> list[JobRecord]:
        """Return jobs in submitted / running / canceling states."""
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?, ?)",
            (
                JobStatus.SUBMITTED.value,
                JobStatus.RUNNING.value,
                JobStatus.CANCELING.value,
            ),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_parent_parameters(self, child_job_id: str) -> dict[str, dict[str, Any]]:
        """Return {parent_job_id: parsed_parameters_dict} for every dep edge of *child_job_id*."""
        rows = self._conn.execute(
            "SELECT p.job_id AS pid, p.parameters AS params "
            "FROM job_dependencies d "
            "JOIN jobs p ON p.job_id = d.parent_job_id "
            "WHERE d.child_job_id = ?",
            (child_job_id,),
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for r in rows:
            raw = r["params"]
            result[r["pid"]] = json.loads(raw) if raw else {}
        return result

    # ---- workflow expiry & cancel ------------------------------------------

    _TERMINAL_WORKFLOW_STATUSES = (
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.TIMED_OUT.value,
    )

    _TERMINAL_JOB_STATUSES = frozenset(
        {
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.SKIPPED.value,
            JobStatus.TIMED_OUT.value,
        }
    )

    def fetch_expired_workflows(self, *, now: datetime) -> list[str]:
        """Return workflow_ids whose deadline_at is past and status is non-terminal."""
        placeholders = ",".join("?" * len(self._TERMINAL_WORKFLOW_STATUSES))
        sql = (
            f"SELECT workflow_id FROM workflows "
            f"WHERE status NOT IN ({placeholders}) "
            f"AND deadline_at IS NOT NULL AND deadline_at < ?"
        )
        rows = self._conn.execute(
            sql, (*self._TERMINAL_WORKFLOW_STATUSES, now.isoformat())
        ).fetchall()
        return [r["workflow_id"] for r in rows]

    def expire_workflow(self, workflow_id: str) -> None:
        """Transition a workflow's non-terminal stage jobs and mark workflow timed_out."""
        now = datetime.now(timezone.utc).isoformat()
        rows = self._conn.execute(
            "SELECT job_id FROM workflow_jobs WHERE workflow_id = ? AND job_id IS NOT NULL",
            (workflow_id,),
        ).fetchall()
        with self._conn:
            for r in rows:
                jid = r["job_id"]
                job = self._conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?", (jid,)
                ).fetchone()
                if job is None:
                    continue
                status = job["status"]
                if status in self._TERMINAL_JOB_STATUSES:
                    continue
                if status == JobStatus.QUEUED.value:
                    self._conn.execute(
                        "UPDATE jobs SET status = ?, skip_reason = ?, completed_at = ? "
                        "WHERE job_id = ?",
                        (JobStatus.SKIPPED.value, "workflow deadline", now, jid),
                    )
                else:
                    self._conn.execute(
                        "UPDATE jobs SET status = ? WHERE job_id = ?",
                        (JobStatus.CANCELING.value, jid),
                    )
            self._conn.execute(
                "UPDATE workflows SET status = ?, completed_at = ? WHERE workflow_id = ?",
                (JobStatus.TIMED_OUT.value, now, workflow_id),
            )
        logger.info("Expired workflow %s (jobs touched=%d)", workflow_id, len(rows))

    def aggregate_workflow_statuses(self) -> list[str]:
        """Aggregate stage job states into workflow status for non-terminal workflows.

        Returns list of workflow_ids that were updated.

        Logic:
        - If any job is failed/cancelled/timed_out -> workflow failed
        - If all jobs are completed/skipped -> workflow completed
        - Otherwise -> workflow remains queued/running
        """
        # Fetch non-terminal workflows
        placeholders = ",".join("?" * len(self._TERMINAL_WORKFLOW_STATUSES))
        sql = f"SELECT workflow_id FROM workflows WHERE status NOT IN ({placeholders})"
        rows = self._conn.execute(sql, self._TERMINAL_WORKFLOW_STATUSES).fetchall()

        if not rows:
            return []

        updated: list[str] = []
        now = datetime.now(timezone.utc).isoformat()

        for r in rows:
            wf_id = r["workflow_id"]

            # Get all job statuses for this workflow
            job_rows = self._conn.execute(
                "SELECT j.status FROM workflow_jobs wj "
                "JOIN jobs j ON j.job_id = wj.job_id "
                "WHERE wj.workflow_id = ? AND wj.job_id IS NOT NULL",
                (wf_id,),
            ).fetchall()

            if not job_rows:
                # No jobs means workflow is still queued (e.g., all stages skipped at enqueue)
                continue

            statuses = [row["status"] for row in job_rows]

            # Determine workflow status
            new_status = None
            if any(s in (JobStatus.FAILED.value, JobStatus.CANCELLED.value, JobStatus.TIMED_OUT.value) for s in statuses):
                new_status = JobStatus.FAILED.value
            elif all(s in (JobStatus.COMPLETED.value, JobStatus.SKIPPED.value) for s in statuses):
                new_status = JobStatus.COMPLETED.value

            if new_status:
                self._conn.execute(
                    "UPDATE workflows SET status = ?, completed_at = ? WHERE workflow_id = ?",
                    (new_status, now, wf_id),
                )
                updated.append(wf_id)

        if updated:
            self._conn.commit()
            logger.info("Aggregated workflow status for %d workflow(s): %s", len(updated), updated)

        return updated

    def request_cancel(self, job_id: str) -> JobStatus:
        """User-initiated cancel. Returns the new status; raises if already terminal."""
        row = self._conn.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Job {job_id} not found")
        current = row["status"]
        if current in self._TERMINAL_JOB_STATUSES:
            raise ValueError(f"Job {job_id} is already {current}; cannot cancel")
        # QUEUED and legacy PENDING can transition directly to CANCELLED
        if current in (JobStatus.QUEUED.value, "pending"):
            new_status = JobStatus.CANCELLED
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ? WHERE job_id = ?",
                (new_status.value, now, job_id),
            )
        else:
            # SUBMITTED, RUNNING, SUBMITTING -> CANCELING (heartbeat will finish)
            new_status = JobStatus.CANCELING
            self._conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (new_status.value, job_id),
            )
        self._conn.commit()
        logger.info("Cancel requested for job %s: %s -> %s", job_id, current, new_status.value)
        return new_status

    def close(self) -> None:
        self._conn.close()

    def status_counts(self) -> dict[str, int]:
        """Return ``{status: count}`` across all jobs (heartbeat status view)."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
