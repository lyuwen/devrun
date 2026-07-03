"""Test heartbeat warning in CLI after successful enqueue."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from devrun.cli import app


@pytest.fixture
def cli_runner():
    """Create a CliRunner instance."""
    return CliRunner()


def test_run_shows_heartbeat_warning_when_service_not_active(cli_runner, tmp_path):
    """devrun run should warn if heartbeat service is not running."""
    # Mock TaskRunner to avoid actual execution
    with patch("devrun.cli.TaskRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ["test_job_123"]
        mock_runner_class.return_value = mock_runner
        
        # Mock get_service at the import location
        with patch("devrun.services.get_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_active.return_value = False
            mock_get_service.return_value = mock_service
            
            result = cli_runner.invoke(app, ["run", "eval"])
            
            assert result.exit_code == 0
            assert "test_job_123" in result.stdout
            assert "Warning" in result.stdout or "warning" in result.stdout.lower()
            assert "Heartbeat scheduler is not running" in result.stdout
            assert "devrun heartbeat start" in result.stdout


def test_run_no_warning_when_service_active(cli_runner, tmp_path):
    """devrun run should NOT warn if heartbeat service is running."""
    with patch("devrun.cli.TaskRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ["test_job_456"]
        mock_runner_class.return_value = mock_runner
        
        # Mock get_service to return active service
        with patch("devrun.services.get_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_active.return_value = True
            mock_get_service.return_value = mock_service
            
            result = cli_runner.invoke(app, ["run", "eval"])
            
            assert result.exit_code == 0
            assert "test_job_456" in result.stdout
            # Should NOT show warning
            assert "Heartbeat scheduler is not running" not in result.stdout


def test_workflow_run_shows_heartbeat_warning_when_service_not_active(cli_runner, tmp_path):
    """workflow run should warn if heartbeat service is not running."""
    # Create a minimal workflow config
    workflow_config = tmp_path / "test_wf.yaml"
    workflow_config.write_text("""
workflow: test_workflow
stages:
  - name: s1
    task: eval
    executor: local
    params:
      model: test
""")
    
    with patch("devrun.workflow.WorkflowRunner") as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.run.return_value = "wf_test_789"
        mock_runner_class.return_value = mock_runner
        
        # Mock get_service at the import location
        with patch("devrun.services.get_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.is_active.return_value = False
            mock_get_service.return_value = mock_service
            
            result = cli_runner.invoke(app, ["workflow", "run", str(workflow_config)])
            
            assert result.exit_code == 0
            assert "wf_test_789" in result.stdout
            assert "Warning" in result.stdout or "warning" in result.stdout.lower()
            assert "Heartbeat scheduler is not running" in result.stdout
            assert "devrun heartbeat start" in result.stdout
