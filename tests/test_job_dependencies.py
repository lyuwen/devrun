"""Tests for job_dependencies table schema (PR1 Task 2)."""

import tempfile
from pathlib import Path

from devrun.db.jobs import JobStore


def test_job_dependencies_table_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='job_dependencies'"
        )
        assert cursor.fetchone() is not None

        cursor = store._conn.execute("PRAGMA table_info(job_dependencies)")
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {"child_job_id", "parent_job_id", "allow_failure"}


def test_job_dependencies_indexes_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='job_dependencies'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_jobdeps_child" in indexes
        assert "idx_jobdeps_parent" in indexes
