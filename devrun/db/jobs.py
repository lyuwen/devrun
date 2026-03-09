"""SQLite-backed job store for devrun."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
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


class JobStore:
    """Thin wrapper around an SQLite database for job records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
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
        now = datetime.utcnow().isoformat()
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

    def list_all(self, limit: int = 50) -> list[JobRecord]:
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

    def close(self) -> None:
        self._conn.close()
