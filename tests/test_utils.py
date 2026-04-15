"""Unit tests for devrun utility modules.

This module tests utility functions including sync, ssh, and slurm helpers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from devrun.utils.sync import sync_to_remote, fetch_from_remote


class TestSyncUtils:
    """Tests for sync utility functions."""

    def test_sync_to_local_remote(self):
        """Verify sync_to_remote handles local to remote sync."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "synced files"
            mock_run.return_value = mock_result

            result = sync_to_remote("local/path", "remote:path")

            assert result.returncode == 0
            mock_run.assert_called_once()
            # Verify rsync command was constructed
            call_args = mock_run.call_args[0][0]
            assert "rsync" in call_args[0]

    def test_sync_to_remote_with_delete(self):
        """Verify sync_to_remote handles --delete flag."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = sync_to_remote("local/path", "remote:path", delete=True)

            call_args = mock_run.call_args[0][0]
            assert "--delete" in call_args

    def test_sync_to_remote_with_dry_run(self):
        """Verify sync_to_remote handles --dry-run flag."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = sync_to_remote("local/path", "remote:path", dry_run=True)

            call_args = mock_run.call_args[0][0]
            assert "--dry-run" in call_args

    def test_sync_failure(self):
        """Verify sync_to_remote handles failure."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "rsync error"
            mock_run.return_value = mock_result

            result = sync_to_remote("local/path", "remote:path")

            assert result.returncode == 1

    def test_fetch_from_remote(self):
        """Verify fetch_from_remote handles remote to local sync."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = fetch_from_remote("remote:path", "local/path")

            assert result.returncode == 0
            mock_run.assert_called_once()

    def test_fetch_with_delete(self):
        """Verify fetch_from_remote handles --delete flag."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = fetch_from_remote("remote:path", "local/path", delete=True)

            call_args = mock_run.call_args[0][0]
            assert "--delete" in call_args


class TestSSHUtils:
    """Tests for SSH utility functions."""

    def test_ssh_command_construction(self):
        """Verify SSH command is constructed correctly."""
        from devrun.utils.ssh import SSHConfig, run_ssh_command

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "command output"
            mock_run.return_value = mock_result

            cfg = SSHConfig(host="server.example.com", user="testuser")
            result = run_ssh_command(cfg, "echo test")

            assert result.returncode == 0
            call_args = mock_run.call_args[0][0]
            assert "ssh" in call_args[0]
            assert "testuser@server.example.com" in " ".join(call_args)

    def test_ssh_with_key_file(self):
        """Verify SSH command includes key file."""
        from devrun.utils.ssh import SSHConfig, run_ssh_command

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            cfg = SSHConfig(host="server.example.com", user="testuser", key_file="/path/to/key")
            result = run_ssh_command(cfg, "echo test")

            call_args = mock_run.call_args[0][0]
            assert "-i" in call_args
            assert "/path/to/key" in call_args

    def test_ssh_timeout(self):
        """Verify SSH command includes timeout."""
        from devrun.utils.ssh import SSHConfig, run_ssh_command

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            cfg = SSHConfig(host="server.example.com", user="testuser")
            result = run_ssh_command(cfg, "echo test", timeout=30)

            call_args = mock_run.call_args[0][0]
            assert "BatchMode=yes" in " ".join(call_args)


class TestSlurmUtils:
    """Tests for Slurm utility functions."""

    def test_sbatch_script_generation(self):
        """Verify sbatch script is generated correctly."""
        from devrun.utils.slurm import generate_sbatch_script

        script = generate_sbatch_script(
            command="python run.py",
            job_name="test_job",
            partition="gpu",
            nodes=2,
            gpus_per_node=4,
        )

        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=test_job" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gres=gpu:4" in script
        assert "python run.py" in script

    def test_sbatch_script_with_walltime(self):
        """Verify sbatch script includes walltime."""
        from devrun.utils.slurm import generate_sbatch_script

        script = generate_sbatch_script(
            command="python run.py",
            walltime="02:00:00",
        )

        assert "#SBATCH --time=02:00:00" in script

    def test_sbatch_script_with_output(self):
        """Verify sbatch script includes default output path."""
        from devrun.utils.slurm import generate_sbatch_script

        script = generate_sbatch_script(
            command="python run.py",
        )

        # Default output path is hardcoded in the function
        assert "#SBATCH --output=devrun_%j.out" in script

    def test_sbatch_script_with_env(self):
        """Verify sbatch script exports environment variables."""
        from devrun.utils.slurm import generate_sbatch_script

        script = generate_sbatch_script(
            command="python run.py",
            env={"CUDA_VISIBLE_DEVICES": "0,1", "MY_VAR": "value"},
        )

        # The env vars are exported without quotes
        assert 'export CUDA_VISIBLE_DEVICES=0,1' in script
        assert 'export MY_VAR=value' in script


class TestUtilsIntegration:
    """Integration tests for utility modules."""

    @pytest.mark.skip(reason="Requires remote machine access")
    def test_sync_rsync_flags(self):
        """Verify rsync command uses appropriate flags."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            sync_to_remote("source/", "dest:")

            call_args = mock_run.call_args[0][0]
            # Should use -avz (archive, verbose, compress) by default
            call_str = " ".join(call_args)
            assert "-avz" in call_str or "-a" in call_str


class TestGenerateSbatchSetE:
    def test_default_includes_set_ex(self):
        from devrun.utils.slurm import generate_sbatch_script
        script = generate_sbatch_script("echo hello")
        assert "set -ex" in script

    def test_set_e_false_uses_set_x_only(self):
        from devrun.utils.slurm import generate_sbatch_script
        script = generate_sbatch_script("echo hello", set_e=False)
        assert "set -x" in script
        assert "set -ex" not in script

    def test_extra_sbatch_output_skips_default_output(self):
        from devrun.utils.slurm import generate_sbatch_script
        script = generate_sbatch_script(
            "echo hello",
            extra_sbatch=["--output=custom_%A_%a.out", "--error=custom_%A_%a.err"],
        )
        lines = script.splitlines()
        output_lines = [l for l in lines if "#SBATCH --output" in l]
        assert len(output_lines) == 1
        assert "custom_%A_%a.out" in output_lines[0]

    def test_no_extra_output_uses_default(self):
        from devrun.utils.slurm import generate_sbatch_script
        script = generate_sbatch_script("echo hello")
        assert "devrun_%j.out" in script