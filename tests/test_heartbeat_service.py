"""Tests for HeartbeatService backends with mocked subprocess (PR2 Task 10)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


def test_get_service_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    from devrun.services import get_service
    from devrun.services.linux import SystemdUserService

    assert isinstance(get_service(), SystemdUserService)


def test_get_service_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    from devrun.services import get_service
    from devrun.services.darwin import LaunchdService

    assert isinstance(get_service(), LaunchdService)


def test_get_service_unsupported(monkeypatch):
    monkeypatch.setattr(sys, "platform", "freebsd")
    from devrun.services import get_service

    with pytest.raises(RuntimeError, match="Unsupported platform"):
        get_service()


# ============================================================================
# SystemdUserService — Linux backend
# ============================================================================


def _patched_unit_path(tmp_path: Path):
    """Helper: patch SystemdUserService.UNIT_PATH to a path under tmp_path."""
    from devrun.services.linux import SystemdUserService

    new_path = (
        tmp_path / ".config" / "systemd" / "user" / SystemdUserService.UNIT_NAME
    )
    return patch.object(SystemdUserService, "UNIT_PATH", new_path), new_path


def test_systemd_install_writes_unit_and_runs_systemctl(tmp_path):
    """install() writes the unit file and runs systemctl daemon-reload + enable."""
    from devrun.services.linux import SystemdUserService

    unit_patch, unit_path = _patched_unit_path(tmp_path)

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with unit_patch, patch("subprocess.run", side_effect=_fake_run):
        SystemdUserService().install(
            python_path="/usr/bin/python3", db_path="/x/jobs.db"
        )

    assert unit_path.exists()
    body = unit_path.read_text()
    assert "/usr/bin/python3" in body
    assert "/x/jobs.db" in body

    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", SystemdUserService.UNIT_NAME] in calls


def test_systemd_start_runs_systemctl(tmp_path):
    from devrun.services.linux import SystemdUserService

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=_fake_run):
        SystemdUserService().start()

    assert ["systemctl", "--user", "start", SystemdUserService.UNIT_NAME] in calls


def test_systemd_stop_runs_systemctl():
    from devrun.services.linux import SystemdUserService

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=_fake_run):
        SystemdUserService().stop()

    assert ["systemctl", "--user", "stop", SystemdUserService.UNIT_NAME] in calls


def test_systemd_restart_runs_systemctl():
    from devrun.services.linux import SystemdUserService

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=_fake_run):
        SystemdUserService().restart()

    assert ["systemctl", "--user", "restart", SystemdUserService.UNIT_NAME] in calls


def test_systemd_uninstall_removes_unit_and_runs_systemctl(tmp_path):
    from devrun.services.linux import SystemdUserService

    unit_patch, unit_path = _patched_unit_path(tmp_path)
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("dummy unit")

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with unit_patch, patch("subprocess.run", side_effect=_fake_run):
        SystemdUserService().uninstall()

    assert not unit_path.exists()
    assert ["systemctl", "--user", "disable", SystemdUserService.UNIT_NAME] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls


def test_systemd_is_active_returns_true_on_returncode_zero():
    from devrun.services.linux import SystemdUserService

    with patch(
        "subprocess.run",
        return_value=types.SimpleNamespace(returncode=0),
    ):
        assert SystemdUserService().is_active() is True


def test_systemd_is_active_returns_false_on_nonzero():
    from devrun.services.linux import SystemdUserService

    with patch(
        "subprocess.run",
        return_value=types.SimpleNamespace(returncode=3),
    ):
        assert SystemdUserService().is_active() is False


# ============================================================================
# LaunchdService — macOS backend
# ============================================================================


def _patched_plist_path(tmp_path: Path):
    from devrun.services.darwin import LaunchdService

    new_path = tmp_path / "Library" / "LaunchAgents" / f"{LaunchdService.LABEL}.plist"
    return patch.object(LaunchdService, "PLIST_PATH", new_path), new_path


def test_launchd_install_writes_plist_and_loads(tmp_path):
    from devrun.services.darwin import LaunchdService

    plist_patch, plist_path = _patched_plist_path(tmp_path)

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with plist_patch, patch("subprocess.run", side_effect=_fake_run):
        LaunchdService().install(
            python_path="/usr/local/bin/python3", db_path="/x/jobs.db"
        )

    assert plist_path.exists()
    body = plist_path.read_text()
    assert "/usr/local/bin/python3" in body
    assert "/x/jobs.db" in body

    assert ["launchctl", "load", str(plist_path)] in calls


def test_launchd_uninstall_unloads_and_removes(tmp_path):
    from devrun.services.darwin import LaunchdService

    plist_patch, plist_path = _patched_plist_path(tmp_path)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text("<plist/>")

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with plist_patch, patch("subprocess.run", side_effect=_fake_run):
        LaunchdService().uninstall()

    assert not plist_path.exists()
    assert ["launchctl", "unload", str(plist_path)] in calls


def test_launchd_stop_runs_launchctl():
    from devrun.services.darwin import LaunchdService

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=_fake_run):
        LaunchdService().stop()

    assert ["launchctl", "stop", LaunchdService.LABEL] in calls


def test_launchd_start_uses_kickstart_with_uid():
    from devrun.services.darwin import LaunchdService

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    with patch("subprocess.run", side_effect=_fake_run), patch(
        "devrun.services.darwin._uid", return_value=1234
    ):
        LaunchdService().start()

    assert ["launchctl", "kickstart", "-k", f"gui/1234/{LaunchdService.LABEL}"] in calls


def test_launchd_is_active_returns_true_on_returncode_zero():
    from devrun.services.darwin import LaunchdService

    with patch(
        "subprocess.run",
        return_value=types.SimpleNamespace(returncode=0),
    ), patch("devrun.services.darwin._uid", return_value=1234):
        assert LaunchdService().is_active() is True


def test_launchd_is_active_returns_false_on_nonzero():
    from devrun.services.darwin import LaunchdService

    with patch(
        "subprocess.run",
        return_value=types.SimpleNamespace(returncode=1),
    ), patch("devrun.services.darwin._uid", return_value=1234):
        assert LaunchdService().is_active() is False
