"""SQLite-backed job store for devrun."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devrun.models import JobRecord, JobStatus

logger = logging.getLogger("devrun.db.jobs")

_DEFAULT_DB_PATH = Path.home() / ".devrun" / "jobs.db"

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


class JobStore:
    """Thin wrapper around an SQLite database for job records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.execute(_WORKFLOW_SCHEMA)
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
            remote_job_id=d.get("remote_job_id"),
            status=d.get("status", "pending"),
            created_at=d["created_at"],
            completed_at=d.get("completed_at"),
            log_path=d.get("log_path"),
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

    def close(self) -> None:
        self._conn.close()
