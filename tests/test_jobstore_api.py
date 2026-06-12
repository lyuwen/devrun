"""Tests for JobStore typed API surface (PR1 Tasks 7+)."""

import tempfile
from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.models import JobStatus


def test_enqueue_creates_queued_job():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        job_id = store.enqueue(
            task_name="test_task",
            executor="local",
            params_template='{"param": "${jobs:abc,val}"}',
            parameters={"param": "placeholder"},
            initial_status=JobStatus.QUEUED,
        )

        assert isinstance(job_id, str) and len(job_id) > 0

        record = store.get(job_id)
        assert record is not None
        assert record.task_name == "test_task"
        assert record.executor == "local"
        assert JobStatus(record.status) == JobStatus.QUEUED

        # params_template lives on the row directly (not on JobRecord), so
        # verify via direct SQL.
        row = store._conn.execute(
            "SELECT params_template, parameters FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        assert row is not None
        assert "${jobs:abc,val}" in row[0]
        assert '"param"' in row[1]
        assert '"placeholder"' in row[1]


def test_enqueue_default_status_is_queued():
    """initial_status defaults to QUEUED when omitted."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        job_id = store.enqueue(
            task_name="t",
            executor="local",
            params_template="x: 1",
            parameters={"x": 1},
        )
        record = store.get(job_id)
        assert record is not None
        assert JobStatus(record.status) == JobStatus.QUEUED


def test_enqueue_generates_unique_ids():
    """Successive calls return distinct job_ids."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        ids = {
            store.enqueue(
                task_name="t",
                executor="local",
                params_template="",
                parameters={},
            )
            for _ in range(5)
        }
        assert len(ids) == 5


def test_insert_dependency():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        parent_id = store.insert("parent_task", "local")
        child_id = store.insert("child_task", "local")

        store.insert_dependency(
            child_job_id=child_id,
            parent_job_id=parent_id,
            allow_failure=False,
        )

        row = store._conn.execute(
            "SELECT allow_failure FROM job_dependencies "
            "WHERE child_job_id=? AND parent_job_id=?",
            (child_id, parent_id),
        ).fetchone()
        assert row is not None
        assert row[0] == 0


def test_insert_dependency_allow_failure_true():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        parent_id = store.insert("parent_task", "local")
        child_id = store.insert("child_task", "local")

        store.insert_dependency(
            child_job_id=child_id,
            parent_job_id=parent_id,
            allow_failure=True,
        )

        row = store._conn.execute(
            "SELECT allow_failure FROM job_dependencies "
            "WHERE child_job_id=? AND parent_job_id=?",
            (child_id, parent_id),
        ).fetchone()
        assert row is not None
        assert row[0] == 1
