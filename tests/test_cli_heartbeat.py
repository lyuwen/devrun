"""Smoke tests for the ``devrun heartbeat`` CLI subapp (PR2 Task 11)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from devrun.cli import app


def test_heartbeat_help_smoke():
    """`devrun heartbeat --help` runs and exits 0."""
    result = CliRunner().invoke(app, ["heartbeat", "--help"])
    assert result.exit_code == 0
    assert "Heartbeat scheduler control" in result.output


def test_heartbeat_status_runs_with_empty_db(tmp_path: Path, monkeypatch):
    """`devrun heartbeat status` exits 0 against an empty DB when service is inactive."""
    fake_db = tmp_path / "jobs.db"
    monkeypatch.setattr(
        "devrun.cli_heartbeat.default_db_path", lambda: fake_db, raising=True
    )
    fake_service = MagicMock(is_active=MagicMock(return_value=False))
    with patch("devrun.services.get_service", return_value=fake_service):
        result = CliRunner().invoke(app, ["heartbeat", "status"])
    assert result.exit_code == 0, result.output
    assert "Service: inactive" in result.output
    assert "no jobs" in result.output or "Job status counts" in result.output


def test_heartbeat_foreground_refuses_when_service_active(tmp_path: Path, monkeypatch):
    """`devrun heartbeat` (default) bails out with non-zero exit when service is active."""
    fake_db = tmp_path / "jobs.db"
    monkeypatch.setattr(
        "devrun.cli_heartbeat.default_db_path", lambda: fake_db, raising=True
    )
    fake_service = MagicMock(is_active=MagicMock(return_value=True))
    runner = CliRunner(mix_stderr=True)
    with patch("devrun.services.get_service", return_value=fake_service):
        result = runner.invoke(app, ["heartbeat"])
    assert result.exit_code != 0
    assert "already running" in result.output


def test_heartbeat_install_invokes_service(tmp_path: Path, monkeypatch):
    """`devrun heartbeat install` calls ``get_service().install(...)`` then ``.start()``."""
    fake_db = tmp_path / "jobs.db"
    monkeypatch.setattr(
        "devrun.cli_heartbeat.default_db_path", lambda: fake_db, raising=True
    )
    fake_service = MagicMock()
    with patch("devrun.services.get_service", return_value=fake_service):
        result = CliRunner().invoke(app, ["heartbeat", "install"])
    assert result.exit_code == 0
    fake_service.install.assert_called_once()
    kwargs = fake_service.install.call_args.kwargs
    assert kwargs["db_path"] == str(fake_db)
    assert kwargs["python_path"]
    fake_service.start.assert_called_once()


def test_heartbeat_start_stop_restart_uninstall_invoke_service(tmp_path: Path, monkeypatch):
    """`start/stop/restart/uninstall` all delegate to the matching service method."""
    fake_db = tmp_path / "jobs.db"
    monkeypatch.setattr(
        "devrun.cli_heartbeat.default_db_path", lambda: fake_db, raising=True
    )
    fake_service = MagicMock()
    runner = CliRunner()
    with patch("devrun.services.get_service", return_value=fake_service):
        for cmd in ("start", "stop", "restart", "uninstall"):
            result = runner.invoke(app, ["heartbeat", cmd])
            assert result.exit_code == 0, (cmd, result.output)
    assert fake_service.start.called
    assert fake_service.stop.called
    assert fake_service.restart.called
    assert fake_service.uninstall.called
