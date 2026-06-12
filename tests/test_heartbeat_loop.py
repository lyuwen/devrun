"""Heartbeat module loop tests (PR2 Task 1)."""

from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.heartbeat import tick


def test_tick_empty_db_is_noop(tmp_path: Path):
    """A tick against a fresh empty DB does nothing and does not raise."""
    db = JobStore(tmp_path / "jobs.db")
    tick(db, executor_router=None)


def test_tick_returns_none(tmp_path: Path):
    """tick() returns None (side-effect only API)."""
    db = JobStore(tmp_path / "jobs.db")
    assert tick(db, executor_router=None) is None
