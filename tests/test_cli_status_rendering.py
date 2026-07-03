"""CLI rendering tests for new statuses + `status --with-deps` (PR3 Task 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from devrun.cli import app
from devrun.db.jobs import JobStore
from devrun.models import JobStatus


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch):
    """Real JobStore on tmp_path wired into the CLI's _runner() factory."""
    db_path = tmp_path / "jobs.db"
    db = JobStore(db_path)

    executors_yaml = tmp_path / "executors.yaml"
    executors_yaml.write_text(yaml.safe_dump({"local": {"type": "local"}}))

    from devrun.runner import TaskRunner

    def _factory():
        return TaskRunner(executors_path=str(executors_yaml), db_path=db_path)

    monkeypatch.setattr("devrun.cli._runner", _factory)
    return db


@pytest.fixture
def runner_cli() -> CliRunner:
    return CliRunner()


def _enqueue(db: JobStore, status: JobStatus = JobStatus.QUEUED) -> str:
    jid = db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    if status != JobStatus.QUEUED:
        db.update_status(jid, status)
    return jid


# ============================================================================
# History rendering — new statuses surface in the table
# ============================================================================


def test_history_renders_queued_status(cli_db, runner_cli):
    """A QUEUED row is included in `devrun history` output."""
    _enqueue(cli_db, JobStatus.QUEUED)
    result = runner_cli.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "queued" in result.stdout.lower()


def test_history_renders_canceling_status(cli_db, runner_cli):
    _enqueue(cli_db, JobStatus.CANCELING)
    result = runner_cli.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "canceling" in result.stdout.lower()


def test_history_renders_skipped_status(cli_db, runner_cli):
    _enqueue(cli_db, JobStatus.SKIPPED)
    result = runner_cli.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "skipped" in result.stdout.lower()


def test_history_renders_timed_out_status(cli_db, runner_cli):
    _enqueue(cli_db, JobStatus.TIMED_OUT)
    result = runner_cli.invoke(app, ["history"])
    assert result.exit_code == 0
    # "timed_out" or "timed out" — be tolerant of formatting.
    out = result.stdout.lower().replace("_", " ")
    assert "timed out" in out


# ============================================================================
# `devrun status --with-deps` — parent listing
# ============================================================================


def test_status_with_deps_lists_parents(cli_db, runner_cli):
    """`--with-deps` prints the child's parents and their statuses."""
    parent = _enqueue(cli_db, JobStatus.COMPLETED)
    child = cli_db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    cli_db.insert_dependency(
        child_job_id=child, parent_job_id=parent, allow_failure=False
    )

    result = runner_cli.invoke(app, ["status", child, "--with-deps"])
    assert result.exit_code == 0, result.output
    # The parent's job id should appear in the output.
    assert parent in result.stdout
    # And the parent's status should appear too.
    assert "completed" in result.stdout.lower()


def test_status_with_deps_shows_allow_failure_flag(cli_db, runner_cli):
    """The allow_failure flag is surfaced alongside each parent."""
    parent = _enqueue(cli_db, JobStatus.FAILED)
    child = cli_db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    cli_db.insert_dependency(
        child_job_id=child, parent_job_id=parent, allow_failure=True
    )

    result = runner_cli.invoke(app, ["status", child, "--with-deps"])
    assert result.exit_code == 0, result.output
    # The allow_failure flag value should be visible (True/true/1).
    assert "allow_failure" in result.stdout.lower() or "allow failure" in result.stdout.lower()


def test_status_without_with_deps_does_not_show_parents(cli_db, runner_cli):
    """Default `devrun status` does NOT include the parent list (backwards-compatible)."""
    parent = _enqueue(cli_db, JobStatus.COMPLETED)
    child = cli_db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    cli_db.insert_dependency(
        child_job_id=child, parent_job_id=parent, allow_failure=False
    )

    result = runner_cli.invoke(app, ["status", child])
    assert result.exit_code == 0, result.output
    # The parent's job id must NOT appear when --with-deps is absent.
    assert parent not in result.stdout


def test_status_with_deps_missing_job_errors(cli_db, runner_cli):
    """Unknown job id with --with-deps → non-zero exit."""
    result = runner_cli.invoke(app, ["status", "nonexistent_job", "--with-deps"])
    assert result.exit_code != 0
