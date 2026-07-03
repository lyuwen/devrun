"""Tests for `devrun rerun` as dependency-free producer (PR3 Task 8)."""

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
    """Real JobStore on tmp_path wired into the CLI."""
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


def test_rerun_creates_dependency_free_copy(cli_db, runner_cli):
    """A rerun of a job that had `--after` edges produces a NEW job with NO edges."""
    parent = cli_db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )
    cli_db.update_status(parent, JobStatus.COMPLETED)

    # Original job: an eval that depended on `parent`.
    original = cli_db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\n",
        parameters={"model": "gpt-4", "dataset": "ds"},
    )
    cli_db.insert_dependency(
        child_job_id=original, parent_job_id=parent, allow_failure=False
    )
    cli_db.update_status(original, JobStatus.COMPLETED)

    pre_ids = {j.job_id for j in cli_db.list_all()}
    result = runner_cli.invoke(app, ["rerun", original])
    assert result.exit_code == 0, result.output

    post_ids = {j.job_id for j in cli_db.list_all()}
    new_ids = post_ids - pre_ids
    assert len(new_ids) == 1, f"expected 1 new job, got {new_ids}"
    new_id = next(iter(new_ids))

    # New job has the same task and executor as the original.
    new_rec = cli_db.get(new_id)
    assert new_rec is not None
    assert new_rec.task_name == "eval"
    assert new_rec.executor == "local"

    # Critical: NO dependency edges were copied over.
    deps = cli_db.list_dependencies(new_id)
    assert deps == [], f"rerun must not copy dependencies, got {deps}"


def test_rerun_new_job_is_queued(cli_db, runner_cli):
    """The rerun copy must be QUEUED (heartbeat will promote it), not SUBMITTED."""
    original = cli_db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\n",
        parameters={"model": "gpt-4"},
    )
    cli_db.update_status(original, JobStatus.COMPLETED)

    pre_ids = {j.job_id for j in cli_db.list_all()}
    result = runner_cli.invoke(app, ["rerun", original])
    assert result.exit_code == 0, result.output

    new_ids = {j.job_id for j in cli_db.list_all()} - pre_ids
    assert len(new_ids) == 1
    new_rec = cli_db.get(next(iter(new_ids)))
    assert new_rec is not None
    assert JobStatus(new_rec.status) == JobStatus.QUEUED


def test_rerun_copies_resolved_parameters(cli_db, runner_cli):
    """The rerun copy carries forward the original's resolved parameters."""
    original = cli_db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\ndataset: ds\n",
        parameters={"model": "gpt-4", "dataset": "ds"},
    )
    cli_db.update_status(original, JobStatus.COMPLETED)

    pre_ids = {j.job_id for j in cli_db.list_all()}
    result = runner_cli.invoke(app, ["rerun", original])
    assert result.exit_code == 0, result.output

    new_ids = {j.job_id for j in cli_db.list_all()} - pre_ids
    new_rec = cli_db.get(next(iter(new_ids)))
    assert new_rec is not None
    assert new_rec.params_dict.get("model") == "gpt-4"
    assert new_rec.params_dict.get("dataset") == "ds"


def test_rerun_unknown_job_errors(cli_db, runner_cli):
    """Unknown job id → non-zero exit."""
    result = runner_cli.invoke(app, ["rerun", "nonexistent_job"])
    assert result.exit_code != 0


def test_rerun_does_not_modify_original(cli_db, runner_cli):
    """The original job's row is untouched by a rerun."""
    original = cli_db.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\n",
        parameters={"model": "gpt-4"},
    )
    cli_db.update_status(original, JobStatus.COMPLETED)
    orig_before = cli_db.get(original)
    assert orig_before is not None

    runner_cli.invoke(app, ["rerun", original])

    orig_after = cli_db.get(original)
    assert orig_after is not None
    assert orig_after.status == orig_before.status
    assert orig_after.parameters == orig_before.parameters
