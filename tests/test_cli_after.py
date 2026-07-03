"""CLI tests for `devrun run --after` and `--allow-failure-from` (PR3 Task 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from devrun.cli import app
from devrun.db.jobs import JobStore


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch):
    """A real JobStore on tmp_path, wired into the CLI's TaskRunner.

    The CLI builds a TaskRunner via ``devrun.cli._runner()``; we replace that
    factory so each test gets a fresh DB without touching the user's
    ~/.devrun/jobs.db.
    """
    db_path = tmp_path / "jobs.db"
    db = JobStore(db_path)

    # Minimal executors.yaml: only `local` is needed for these tests.
    executors_yaml = tmp_path / "executors.yaml"
    executors_yaml.write_text(yaml.safe_dump({"local": {"type": "local"}}))

    from devrun.runner import TaskRunner

    def _factory():
        return TaskRunner(executors_path=str(executors_yaml), db_path=db_path)

    monkeypatch.setattr("devrun.cli._runner", _factory)
    return db


@pytest.fixture
def eval_config(tmp_path: Path) -> Path:
    """A minimal task config that the CLI can load via a literal file path."""
    cfg_path = tmp_path / "after_eval.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "task": "eval",
                "executor": "local",
                "params": {"model": "gpt-4", "dataset": "ds"},
            }
        )
    )
    return cfg_path


@pytest.fixture
def runner_cli() -> CliRunner:
    return CliRunner()


def _enqueue_parent(db: JobStore) -> str:
    return db.enqueue(
        task_name="t", executor="local", params_template="", parameters={}
    )


def test_run_after_writes_dependency_edge(cli_db, eval_config, runner_cli):
    """`devrun run <config> --after <parent>` enqueues a child with one parent edge."""
    parent = _enqueue_parent(cli_db)

    result = runner_cli.invoke(
        app, ["run", str(eval_config), "--after", parent]
    )

    assert result.exit_code == 0, result.output
    new_jobs = [j for j in cli_db.list_all() if j.job_id != parent]
    assert len(new_jobs) == 1

    child_id = new_jobs[0].job_id
    deps = cli_db.list_dependencies(child_id)
    assert len(deps) == 1
    assert deps[0].parent_job_id == parent
    assert bool(deps[0].allow_failure) is False


def test_run_after_multiple_parents(cli_db, eval_config, runner_cli):
    """`--after` is repeatable: every parent gets its own dep edge."""
    p1 = _enqueue_parent(cli_db)
    p2 = _enqueue_parent(cli_db)

    result = runner_cli.invoke(
        app, ["run", str(eval_config), "--after", p1, "--after", p2]
    )

    assert result.exit_code == 0, result.output
    new_jobs = [j for j in cli_db.list_all() if j.job_id not in {p1, p2}]
    assert len(new_jobs) == 1

    child_id = new_jobs[0].job_id
    deps = cli_db.list_dependencies(child_id)
    parent_ids = {d.parent_job_id for d in deps}
    assert parent_ids == {p1, p2}


def test_run_allow_failure_from_marks_edge(cli_db, eval_config, runner_cli):
    """`--allow-failure-from <id>` sets allow_failure=True on that parent's edge."""
    p_strict = _enqueue_parent(cli_db)
    p_lenient = _enqueue_parent(cli_db)

    result = runner_cli.invoke(
        app,
        [
            "run", str(eval_config),
            "--after", p_strict,
            "--after", p_lenient,
            "--allow-failure-from", p_lenient,
        ],
    )

    assert result.exit_code == 0, result.output
    new_jobs = [
        j for j in cli_db.list_all() if j.job_id not in {p_strict, p_lenient}
    ]
    assert len(new_jobs) == 1

    deps = {d.parent_job_id: bool(d.allow_failure)
            for d in cli_db.list_dependencies(new_jobs[0].job_id)}
    assert deps[p_strict] is False
    assert deps[p_lenient] is True


def test_run_unknown_after_id_errors(cli_db, eval_config, runner_cli):
    """An unknown `--after <id>` must abort with a non-zero exit code and clear message."""
    result = runner_cli.invoke(
        app, ["run", str(eval_config), "--after", "nonexistent_job_id"]
    )

    assert result.exit_code != 0
    # All jobs that exist in the DB must be unrelated to the failed run.
    # In particular, no QUEUED row was created for the eval task.
    all_jobs = cli_db.list_all()
    assert all(j.task_name != "eval" for j in all_jobs)


def test_run_after_enqueues_in_queued_state(cli_db, eval_config, runner_cli):
    """The child job is QUEUED (not SUBMITTED) — promotion is the heartbeat's job."""
    from devrun.models import JobStatus

    parent = _enqueue_parent(cli_db)

    result = runner_cli.invoke(
        app, ["run", str(eval_config), "--after", parent]
    )
    assert result.exit_code == 0, result.output

    new_jobs = [j for j in cli_db.list_all() if j.job_id != parent]
    assert len(new_jobs) == 1
    assert JobStatus(new_jobs[0].status) == JobStatus.QUEUED


def test_run_without_after_still_works(cli_db, eval_config, runner_cli):
    """No `--after` flags → child is enqueued with no parent edges."""
    result = runner_cli.invoke(app, ["run", str(eval_config)])
    assert result.exit_code == 0, result.output

    new_jobs = cli_db.list_all()
    assert len(new_jobs) == 1
    assert cli_db.list_dependencies(new_jobs[0].job_id) == []
