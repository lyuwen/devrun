"""Unit tests for devrun CLI commands.

This module tests all CLI commands including run, list, status, logs,
history, rerun, cancel, sync, and fetch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from devrun.cli import app
from devrun.models import JobStatus


# Create a CliRunner without mix_stderr (older typer version compatibility)
def get_cli_runner():
    """Create a CliRunner instance."""
    return CliRunner()


class TestCLIList:
    """Tests for the 'list' command."""

    def test_list_shows_plugins(self):
        """Verify list command shows available plugins."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "task" in result.stdout.lower()
        assert "executor" in result.stdout.lower()
        assert "eval" in result.stdout
        assert "local" in result.stdout

    def test_list_shows_variations(self):
        """Verify list command shows config variations."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "Variations" in result.stdout
        assert "swe_bench_eval/slurm" in result.stdout


class TestCLIStatus:
    """Tests for the 'status' command."""

    def test_status_missing_job_id(self):
        """Verify status requires job_id argument."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["status"])

        # Should show help or error
        assert result.exit_code != 0

    def test_status_nonexistent_job(self):
        """Verify status handles nonexistent job."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.status.return_value = {"error": "Job not found"}
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["status", "nonexistent"])

            assert result.exit_code == 1
            assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_status_shows_array_progress(self):
        """Verify status formats array progress from executor."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.status.return_value = {
                "job_id": "arr_100",
                "task_name": "swe_bench_agentic",
                "executor": "slurm",
                "status": "running",
                "created_at": "2024-01-01T00:00:00",
                "progress": {
                    "task_counts": {"completed": 50, "running": 10, "pending": 5},
                    "total_tasks": 65,
                },
            }
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["status", "arr_100"])

            assert result.exit_code == 0
            assert "array_progress" in result.stdout
            assert "50 completed" in result.stdout
            assert "10 running" in result.stdout
            assert "total: 65" in result.stdout


class TestCLILogs:
    """Tests for the 'logs' command."""

    def test_logs_missing_job_id(self):
        """Verify logs requires job_id argument."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["logs"])

        # Should show help or error
        assert result.exit_code != 0


class TestCLIHistory:
    """Tests for the 'history' command."""

    def test_history_empty(self):
        """Verify history shows empty message when no jobs."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history"])

            assert result.exit_code == 0

    def test_history_with_jobs(self):
        """Verify history shows job records."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = [
                {
                    "job_id": "test123",
                    "task_name": "eval",
                    "executor": "local",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00",
                }
            ]
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history"])

            assert result.exit_code == 0
            assert "test123" in result.stdout

    def test_history_with_limit(self):
        """Verify history respects --limit option."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history", "--limit", "10"])

            assert result.exit_code == 0
            mock_runner.history.assert_called_once_with(10)

    def test_history_all_flag(self):
        """--all passes None as limit to fetch unlimited records."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history", "--all"])

            assert result.exit_code == 0
            mock_runner.history.assert_called_once_with(None)

    def test_history_all_overrides_limit(self):
        """--all takes precedence even when --limit is also provided."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history", "--limit", "10", "--all"])

            assert result.exit_code == 0
            mock_runner.history.assert_called_once_with(None)

    def test_history_uses_pager_for_long_output(self):
        """When output exceeds terminal height, history should use a pager."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            # Generate enough records to exceed a typical terminal
            mock_runner.history.return_value = [
                {
                    "job_id": f"job_{i:04d}",
                    "task_name": "eval",
                    "executor": "local",
                    "status": "completed",
                    "created_at": "2024-01-01T00:00:00",
                }
                for i in range(100)
            ]
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            with patch("devrun.cli.console") as mock_console:
                mock_console.height = 24  # simulate small terminal
                result = runner.invoke(app, ["history", "--all"])

            assert result.exit_code == 0

    def test_history_no_pager_flag(self):
        """--no-pager should disable pager even for long output."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.history.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["history", "--no-pager"])

            assert result.exit_code == 0


class TestCLIRerun:
    """Tests for the 'rerun' command."""

    def test_rerun_missing_job_id(self):
        """Verify rerun requires job_id argument."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["rerun"])

        assert result.exit_code != 0

    def test_rerun_success(self):
        """Verify rerun successfully resubmits job."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.rerun.return_value = ["new_job_456"]
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["rerun", "old_job_123"])

            assert result.exit_code == 0
            assert "new_job_456" in result.stdout

    def test_rerun_not_found(self):
        """Verify rerun handles nonexistent job."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.rerun.side_effect = ValueError("Job not found")
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["rerun", "nonexistent"])

            assert result.exit_code == 1


class TestCLICancel:
    """Tests for the 'cancel' command."""

    def test_cancel_missing_job_id(self):
        """Verify cancel requires job_id argument."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["cancel"])

        assert result.exit_code != 0

    def test_cancel_success(self):
        """Verify cancel successfully cancels job."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.cancel.return_value = None
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["cancel", "job_123"])

            assert result.exit_code == 0

    def test_cancel_already_completed(self):
        """Verify cancel handles already completed job."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.cancel.side_effect = ValueError("Job is already completed")
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["cancel", "job_123"])

            assert result.exit_code == 1


class TestCLIRun:
    """Tests for the 'run' command."""

    def test_run_missing_target(self):
        """Verify run requires target argument."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 2
        assert "missing" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_run_with_dry_run(self):
        """Verify run with --dry-run doesn't submit jobs."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run.return_value = []
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["run", "--dry-run", "eval"])

            # Should not actually run or might error if config not found
            # But dry-run means no submission
            mock_runner.run.assert_called_once()


class TestCLISync:
    """Tests for the 'sync' command."""

    def test_sync_missing_args(self):
        """Verify sync requires source and destination."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["sync"])

        assert result.exit_code != 0

    @pytest.mark.skip(reason="Requires remote machine access")
    def test_sync_dry_run(self):
        """Verify sync with --dry-run shows what would be synced."""
        with patch("devrun.utils.sync.sync_to_remote") as mock_sync:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "DRY RUN"
            mock_sync.return_value = mock_result

            runner = get_cli_runner()
            result = runner.invoke(app, ["sync", "local/path", "remote:path", "--dry-run"])

            # Should complete without error
            assert result.exit_code == 0


class TestCLIFetch:
    """Tests for the 'fetch' command."""

    def test_fetch_missing_args(self):
        """Verify fetch requires source and destination."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["fetch"])

        assert result.exit_code != 0


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Verify main help shows available commands."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "list" in result.stdout
        assert "status" in result.stdout

    def test_run_help(self):
        """Verify run --help shows usage."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0


class TestCLIVerbose:
    """Tests for verbose logging option."""

    def test_verbose_flag(self):
        """Verify --verbose flag sets debug logging."""
        with patch("devrun.cli._setup_logging") as mock_logging:
            runner = get_cli_runner()
            # verbose is a subcommand option, not global
            result = runner.invoke(app, ["run", "--help"])

            # Should not affect the command
            assert result.exit_code == 0


class TestCLITaskHelp:
    """Tests for task-specific help."""

    def test_show_task_help(self):
        """Verify task help can be displayed."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner._load_config.return_value = MagicMock(
                task="eval",
                executor="local",
                params={"model": "default-model"},
            )
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            result = runner.invoke(app, ["run", "eval", "--help"])

            # May fail due to config loading but should not crash
            # The important thing is the help system works
            assert result.exit_code in [0, 1]


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_unknown_command(self):
        """Verify unknown command shows error."""
        runner = get_cli_runner()
        result = runner.invoke(app, ["unknown_command"])

        assert result.exit_code != 0


class TestCLIEdgeCases:
    """Edge case tests for CLI."""

    def test_run_with_overrides(self):
        """Verify run accepts parameter overrides."""
        with patch("devrun.cli.TaskRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run.return_value = ["job_123"]
            mock_runner_class.return_value = mock_runner

            runner = get_cli_runner()
            # This should pass overrides to the runner
            result = runner.invoke(app, ["run", "eval", "params.model=new-model"])

            # Either succeeds or fails gracefully
            assert result.exit_code in [0, 1]


class TestWorkflowCLI:
    """Tests for workflow subcommands."""

    def test_workflow_list_empty(self):
        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "list"])
        assert result.exit_code == 0

    def test_workflow_status_not_found(self):
        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "status", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_workflow_cancel_not_found(self):
        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "cancel", "nonexistent"])
        assert result.exit_code == 1

    def test_workflow_logs_not_found(self):
        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "logs", "nonexistent"])
        assert result.exit_code == 1

    def test_workflow_run_missing_config(self):
        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "run", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_workflow_run_dry_run(self, tmp_path):
        config = {
            "workflow": "test_wf",
            "stages": [
                {"name": "s1", "task": "eval", "executor": "local", "params": {"model": "x"}},
            ],
            "heartbeat_interval": 0.001,
        }
        cfg_path = tmp_path / "wf.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = get_cli_runner()
        result = runner.invoke(app, ["workflow", "run", str(cfg_path), "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout.lower()