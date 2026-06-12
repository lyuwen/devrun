"""Tests for job_dependencies table schema (PR1 Task 2)."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

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


def test_jobs_new_columns_added():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "params_template" in columns
        assert "skip_reason" in columns
        assert "claimed_by" in columns
        assert "claimed_at" in columns
        assert "claim_expires_at" in columns


def test_jobs_migrations_are_idempotent():
    """Re-opening the same DB file must not raise on duplicate-column errors."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store1 = JobStore(db_path)
        store1.close()

        store2 = JobStore(db_path)
        cursor = store2._conn.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "params_template" in columns
        assert "claim_expires_at" in columns
        store2.close()


def test_workflow_jobs_table_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_jobs'"
        )
        assert cursor.fetchone() is not None

        cursor = store._conn.execute("PRAGMA table_info(workflow_jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {
            "workflow_id",
            "stage_name",
            "ordinal",
            "job_id",
            "source_job_id",
        }


def test_workflow_jobs_indexes_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='workflow_jobs'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_wfjobs_workflow" in indexes
        assert "idx_wfjobs_job" in indexes


def test_workflow_jobs_check_constraint():
    """job_id and source_job_id cannot both be NULL."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        store._conn.execute(
            "INSERT INTO workflows (workflow_id, workflow_name, status, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("wf1", "test", "pending", "2026-01-01T00:00:00Z"),
        )

        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO workflow_jobs "
                "(workflow_id, stage_name, ordinal, job_id, source_job_id) "
                "VALUES (?, ?, ?, ?, ?)",
                ("wf1", "stage1", 0, None, None),
            )


def test_workflows_deadline_at_column_added():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)

        cursor = store._conn.execute("PRAGMA table_info(workflows)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "deadline_at" in columns
