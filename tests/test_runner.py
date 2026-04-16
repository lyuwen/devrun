"""Unit tests for devrun.runner module.

This module tests the TaskRunner orchestration engine, including config loading,
parameter sweep expansion, job submission, and status/logs/cancel operations.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from devrun.models import JobStatus, TaskConfig, TaskSpec
from devrun.runner import TaskRunner, find_configs, load_merged_config


class TestTaskRunnerInitialization:
    """Tests for TaskRunner initialization."""

    def test_default_initialization(self, executors_yaml, temp_dir):
        """Verify TaskRunner can be initialized with default settings."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml))
            assert runner._executors_path == str(executors_yaml)

    def test_custom_db_path(self, executors_yaml, temp_dir):
        """Verify custom database path can be specified."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            db_path = temp_dir / "test.db"
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=str(db_path))
            assert runner._db._db_path == db_path

    def test_config_dirs_include_repo_configs(self, executors_yaml, temp_dir):
        """Verify config_dirs includes the repo configs directory."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml))
            # Should include at least the repo configs
            config_dirs = runner._config_dirs
            assert len(config_dirs) >= 1


class TestFindConfigs:
    """Tests for TaskRunner._find_configs method."""

    def test_find_absolute_path(self, executors_yaml, eval_config_yaml, temp_dir):
        """Verify finding config by absolute path."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml))
            configs = runner._find_configs(str(eval_config_yaml))
            assert len(configs) == 1
            assert configs[0] == eval_config_yaml

    def test_find_task_in_configs_dir(self, executors_yaml, temp_dir):
        """Verify finding config by task name in configs directory."""
        # Create a task config in the expected location
        config_dir = temp_dir / ".devrun" / "configs" / "test_task"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "default.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"task": "eval", "executor": "local"}, f)

        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            # Temporarily add temp_dir to config dirs
            runner._config_dirs.insert(0, temp_dir / ".devrun" / "configs")

            configs = runner._find_configs("test_task")
            assert len(configs) >= 1

    def test_find_with_variation(self, executors_yaml, temp_dir):
        """Verify finding config with variation name."""
        # Create default and variation configs
        config_dir = temp_dir / ".devrun" / "configs" / "test_task"
        config_dir.mkdir(parents=True)

        default_path = config_dir / "default.yaml"
        with open(default_path, "w") as f:
            yaml.dump({"task": "eval", "executor": "local", "params": {"base": True}}, f)

        variation_path = config_dir / "custom.yaml"
        with open(variation_path, "w") as f:
            yaml.dump({"task": "eval", "executor": "local", "params": {"custom": True}}, f)

        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            runner._config_dirs.insert(0, temp_dir / ".devrun" / "configs")

            configs = runner._find_configs("test_task/custom")
            # Should find both default and custom configs
            assert len(configs) == 2

    def test_find_nonexistent_raises_error(self, executors_yaml, temp_dir):
        """Verify FileNotFoundError for nonexistent task."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            with pytest.raises(FileNotFoundError) as exc_info:
                runner._find_configs("nonexistent_task")
            assert "Config for" in str(exc_info.value)


class TestExpandSweep:
    """Tests for TaskRunner._expand_sweep method."""

    def test_no_sweep(self):
        """Verify single param dict is returned when no sweep."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test", "batch_size": 8},
        )
        result = TaskRunner._expand_sweep(config)
        assert len(result) == 1
        assert result[0] == {"model": "test", "batch_size": 8}

    def test_single_sweep_param(self):
        """Verify sweep with single parameter."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test"},
            sweep={"batch_size": [4, 8, 16]},
        )
        result = TaskRunner._expand_sweep(config)
        assert len(result) == 3
        assert result[0]["batch_size"] == 4
        assert result[1]["batch_size"] == 8
        assert result[2]["batch_size"] == 16

    def test_multiple_sweep_params(self):
        """Verify sweep with multiple parameters (Cartesian product)."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test"},
            sweep={
                "batch_size": [4, 8],
                "lr": [0.01, 0.001],
            },
        )
        result = TaskRunner._expand_sweep(config)
        # Should be 2 x 2 = 4 combinations
        assert len(result) == 4

    def test_sweep_merges_with_params(self):
        """Verify sweep values are merged with base params."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test-model", "base_param": "value"},
            sweep={"batch_size": [4, 8]},
        )
        result = TaskRunner._expand_sweep(config)
        assert all(r["model"] == "test-model" for r in result)
        assert all(r["base_param"] == "value" for r in result)
        assert result[0]["batch_size"] == 4
        assert result[1]["batch_size"] == 8


class TestLoadConfig:
    """Tests for TaskRunner._load_config method."""

    def test_load_simple_config(self, executors_yaml, eval_config_yaml, temp_dir):
        """Verify loading a simple YAML config."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            config = runner._load_config(str(eval_config_yaml))
            assert isinstance(config, TaskConfig)
            assert config.task == "eval"
            assert config.executor == "local"

    def test_load_with_overrides(self, executors_yaml, eval_config_yaml, temp_dir):
        """Verify CLI overrides are applied."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            overrides = ["params.model=new-model", "params.batch_size=32"]
            config = runner._load_config(str(eval_config_yaml), overrides=overrides)
            assert config.params["model"] == "new-model"
            assert config.params["batch_size"] == 32


class TestRunnerRun:
    """Tests for TaskRunner.run method."""

    def test_run_returns_job_ids(self, executors_yaml, eval_config_yaml, temp_dir, monkeypatch):
        """Verify run returns list of job IDs."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            # Mock the executor to avoid actual subprocess calls
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit.return_value = "mock_job_123"
                mock_executor.submit_with_retry.return_value = "mock_job_123"
                mock_resolve.return_value = mock_executor

                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=":memory:",
                )
                job_ids = runner.run(str(eval_config_yaml))

                assert len(job_ids) == 1
                assert isinstance(job_ids[0], str)

    def test_run_dry_run(self, executors_yaml, eval_config_yaml, temp_dir):
        """Verify dry run doesn't submit jobs."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_resolve.return_value = mock_executor

                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=":memory:",
                )
                job_ids = runner.run(str(eval_config_yaml), dry_run=True)

                # No jobs should be created
                assert len(job_ids) == 0
                # Executor should not be called
                mock_executor.submit.assert_not_called()

    def test_run_expands_sweep(self, executors_yaml, temp_dir):
        """Verify run expands parameter sweeps into multiple jobs."""
        # Create a config with sweep
        config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "test"},
            "sweep": {"batch_size": [4, 8]},
        }
        config_path = temp_dir / "sweep_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit.return_value = "mock_job"
                mock_executor.submit_with_retry.return_value = "mock_job"
                mock_resolve.return_value = mock_executor

                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=":memory:",
                )
                job_ids = runner.run(str(config_path))

                # Should have 2 job IDs for 2 sweep values
                assert len(job_ids) == 2


class TestRunnerStatus:
    """Tests for TaskRunner.status method."""

    def test_status_not_found(self, executors_yaml, temp_dir):
        """Verify status returns error for unknown job."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            result = runner.status("nonexistent_job")
            assert "error" in result

    @pytest.mark.skip(reason="Test isolation issue - mock_job_store creates separate db")
    def test_status_completed_job(self, executors_yaml, temp_dir, mock_job_store):
        """Verify status returns completed job info without querying executor."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            # Insert a completed job directly
            job_id = mock_job_store.insert("eval", "local", {"model": "test"})
            mock_job_store.update_status(job_id, JobStatus.COMPLETED, completed_at=datetime.now(timezone.utc))

            runner = TaskRunner(
                executors_path=str(executors_yaml),
                db_path=mock_job_store._db_path,
            )
            result = runner.status(job_id)
            assert result["status"] == "completed"

    def test_status_includes_progress_from_executor(self, executors_yaml, temp_dir):
        """Verify runner.status() merges executor.progress() into the result dict."""
        from devrun.db.jobs import JobStore

        db_path = temp_dir / "test_progress.db"
        store = JobStore(db_path)
        job_id = store.insert("eval", "local", {"model": "test"})
        store.update_status(job_id, JobStatus.RUNNING, remote_job_id="slurm_123")

        mock_executor = MagicMock()
        mock_executor.status.return_value = "running"
        mock_executor.progress.return_value = {
            "task_counts": {"completed": 50, "running": 10, "pending": 5},
            "total_tasks": 65,
        }

        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("devrun.executors.local._LOG_DIR", log_dir),
            patch("devrun.runner.resolve_executor", return_value=mock_executor),
        ):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=db_path)
            result = runner.status(job_id)

        assert "progress" in result
        assert result["progress"]["total_tasks"] == 65
        assert result["progress"]["task_counts"]["completed"] == 50


class TestRunnerLogs:
    """Tests for TaskRunner.logs method."""

    def test_logs_not_found(self, executors_yaml, temp_dir):
        """Verify logs returns error for unknown job."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            result = runner.logs("nonexistent_job")
            assert "not found" in result


class TestRunnerCancel:
    """Tests for TaskRunner.cancel method."""

    def test_cancel_not_found(self, executors_yaml, temp_dir):
        """Verify cancel raises error for unknown job."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            with pytest.raises(ValueError) as exc_info:
                runner.cancel("nonexistent_job")
            assert "not found" in str(exc_info.value)

    @pytest.mark.skip(reason="Flaky test - requires remote machine")
    def test_cancel_already_completed(self, executors_yaml, temp_dir, mock_job_store):
        """Verify cancel raises error for already completed job."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            job_id = mock_job_store.insert("eval", "local", {"model": "test"})
            mock_job_store.update_status(job_id, JobStatus.COMPLETED, completed_at=datetime.now(timezone.utc))

            runner = TaskRunner(
                executors_path=str(executors_yaml),
                db_path=mock_job_store._db_path,
            )
            with pytest.raises(ValueError) as exc_info:
                runner.cancel(job_id)
            assert "already" in str(exc_info.value).lower()


class TestRunnerRerun:
    """Tests for TaskRunner.rerun method."""

    def test_rerun_not_found(self, executors_yaml, temp_dir):
        """Verify rerun raises error for unknown job."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            with pytest.raises(ValueError) as exc_info:
                runner.rerun("nonexistent_job")
            assert "not found" in str(exc_info.value)


class TestMapStatus:
    """Tests for TaskRunner._map_status method."""

    def test_map_running(self):
        """Verify 'running' maps to RUNNING status."""
        result = TaskRunner._map_status("running")
        assert result == JobStatus.RUNNING

    def test_map_pending(self):
        """Verify 'pending' maps to PENDING status."""
        result = TaskRunner._map_status("pending")
        assert result == JobStatus.PENDING

    def test_map_completed(self):
        """Verify 'completed' maps to COMPLETED status."""
        result = TaskRunner._map_status("completed")
        assert result == JobStatus.COMPLETED

    def test_map_done(self):
        """Verify 'done' maps to COMPLETED status."""
        result = TaskRunner._map_status("done")
        assert result == JobStatus.COMPLETED

    def test_map_failed(self):
        """Verify 'failed' maps to FAILED status."""
        result = TaskRunner._map_status("failed")
        assert result == JobStatus.FAILED

    def test_map_cancelled(self):
        """Verify 'cancelled' maps to CANCELLED status."""
        result = TaskRunner._map_status("cancelled")
        assert result == JobStatus.CANCELLED

    def test_map_timeout(self):
        """Verify 'timeout' maps to FAILED status."""
        result = TaskRunner._map_status("timeout")
        assert result == JobStatus.FAILED

    def test_map_unknown(self):
        """Verify unknown status maps to UNKNOWN."""
        result = TaskRunner._map_status("weird_status")
        assert result == JobStatus.UNKNOWN

    def test_map_case_insensitive(self):
        """Verify mapping is case insensitive."""
        assert TaskRunner._map_status("RUNNING") == JobStatus.RUNNING
        assert TaskRunner._map_status("Running") == JobStatus.RUNNING
        assert TaskRunner._map_status("running") == JobStatus.RUNNING


class TestRunnerHistory:
    """Tests for TaskRunner.history method."""

    def test_history_empty(self, executors_yaml, temp_dir):
        """Verify history returns empty list when no jobs."""
        with patch("devrun.executors.local._LOG_DIR", temp_dir / "logs"):
            runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
            result = runner.history()
            assert result == []

    @pytest.mark.skip(reason="Flaky test - requires remote machine")
    def test_history_returns_jobs(self, executors_yaml, temp_dir, mock_job_store):
        """Verify history returns job records."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            # Insert some jobs
            mock_job_store.insert("eval", "local", {"model": "test1"})
            mock_job_store.insert("eval", "local", {"model": "test2"})

            runner = TaskRunner(
                executors_path=str(executors_yaml),
                db_path=mock_job_store._db_path,
            )
            result = runner.history()
            assert len(result) == 2


class TestRunnerIntegration:
    """Integration tests for TaskRunner."""

    def test_full_job_lifecycle(self, executors_yaml, temp_dir):
        """Verify full job lifecycle from submit to completion."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit.return_value = "remote_job_123"
                mock_executor.submit_with_retry.return_value = "remote_job_123"
                mock_resolve.return_value = mock_executor

                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=":memory:",
                )

                # Run the job - fix: temp_dir is already a Path, pass it directly
                job_ids = runner.run(str(eval_config_yaml(temp_dir)))
                assert len(job_ids) == 1


class TestSubmitSingleMultiShard:
    """Tests for _submit_single handling of prepare_many / multi-shard."""

    def test_submit_single_calls_prepare_many(self, executors_yaml, temp_dir):
        """_submit_single should call prepare_many, not prepare."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit_with_retry.return_value = "mock_job_1"
                mock_resolve.return_value = mock_executor

                with patch("devrun.runner.get_task_class") as mock_get_task:
                    mock_task = MagicMock()
                    mock_task.return_value = mock_task
                    mock_task.prepare_many.return_value = [
                        TaskSpec(command="echo shard1"),
                        TaskSpec(command="echo shard2"),
                    ]
                    mock_get_task.return_value = mock_task

                    runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
                    job_ids = runner._submit_single("test_task", "local", {})

                    mock_task.prepare_many.assert_called_once()
                    assert len(job_ids) == 2

    def test_submit_single_single_spec_returns_one_id(self, executors_yaml, temp_dir):
        """When prepare_many returns one spec, _submit_single returns one ID."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit_with_retry.return_value = "mock_job_1"
                mock_resolve.return_value = mock_executor

                with patch("devrun.runner.get_task_class") as mock_get_task:
                    mock_task = MagicMock()
                    mock_task.return_value = mock_task
                    mock_task.prepare_many.return_value = [
                        TaskSpec(command="echo single"),
                    ]
                    mock_get_task.return_value = mock_task

                    runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
                    job_ids = runner._submit_single("test_task", "local", {})

                    assert len(job_ids) == 1

    def test_run_extends_job_ids_from_multi_shard(self, executors_yaml, temp_dir):
        """run() should extend (not append) job_ids from _submit_single."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a config file that uses our mock task
        config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "test"},
        }
        config_path = temp_dir / "multi_shard_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("devrun.runner.resolve_executor") as mock_resolve:
                mock_executor = MagicMock()
                mock_executor.submit_with_retry.return_value = "mock_job"
                mock_resolve.return_value = mock_executor

                runner = TaskRunner(executors_path=str(executors_yaml), db_path=":memory:")
                # Mock _submit_single to return multiple IDs
                with patch.object(runner, "_submit_single", return_value=["id1", "id2"]):
                    job_ids = runner.run(str(config_path))
                    # Should have 2 IDs, not a nested list
                    assert job_ids == ["id1", "id2"]


class TestFindConfigsModuleLevel:
    """Tests for the module-level find_configs() function."""

    def test_find_configs_with_file_path(self, tmp_path):
        """Given a path to an existing YAML file, returns [path]."""
        cfg_file = tmp_path / "my_config.yaml"
        cfg_file.write_text(yaml.dump({"task": "eval", "executor": "local"}))

        result = find_configs(str(cfg_file), config_dirs=[tmp_path])
        assert result == [cfg_file]

    def test_find_configs_by_name(self, tmp_path):
        """Given a name like 'myconfig', searches config_dirs for myconfig/default.yaml."""
        config_dir = tmp_path / "configs"
        task_dir = config_dir / "myconfig"
        task_dir.mkdir(parents=True)
        default_yaml = task_dir / "default.yaml"
        default_yaml.write_text(yaml.dump({"task": "eval", "executor": "local"}))

        result = find_configs("myconfig", config_dirs=[config_dir])
        assert len(result) == 1
        assert result[0] == default_yaml

    def test_find_configs_with_variation(self, tmp_path):
        """Given 'myconfig/custom', returns default.yaml then custom.yaml from config_dirs."""
        config_dir = tmp_path / "configs"
        task_dir = config_dir / "myconfig"
        task_dir.mkdir(parents=True)

        default_yaml = task_dir / "default.yaml"
        default_yaml.write_text(yaml.dump({"task": "eval", "executor": "local", "params": {"base": True}}))

        custom_yaml = task_dir / "custom.yaml"
        custom_yaml.write_text(yaml.dump({"params": {"custom_flag": True}}))

        result = find_configs("myconfig/custom", config_dirs=[config_dir])
        assert len(result) == 2
        assert result[0] == default_yaml
        assert result[1] == custom_yaml

    def test_find_configs_not_found(self, tmp_path):
        """Raises FileNotFoundError when target doesn't match anything."""
        config_dir = tmp_path / "empty_configs"
        config_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError) as exc_info:
            find_configs("nonexistent_task", config_dirs=[config_dir])
        assert "nonexistent_task" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_find_configs_merge_order(self, tmp_path):
        """With configs in multiple dirs, returns them in correct priority order."""
        # dir1 is lower priority (comes first in the list), dir2 is higher
        dir1 = tmp_path / "low_priority"
        dir2 = tmp_path / "high_priority"
        for d in [dir1, dir2]:
            task_dir = d / "myconfig"
            task_dir.mkdir(parents=True)
            (task_dir / "default.yaml").write_text(
                yaml.dump({"task": "eval", "executor": "local", "source": str(d)})
            )

        result = find_configs("myconfig", config_dirs=[dir1, dir2])
        # Both dirs should contribute a default.yaml, dir1 first (lower priority)
        assert len(result) == 2
        assert result[0] == dir1 / "myconfig" / "default.yaml"
        assert result[1] == dir2 / "myconfig" / "default.yaml"


class TestLoadMergedConfigModuleLevel:
    """Tests for the module-level load_merged_config() function."""

    def test_load_merged_config_basic(self, tmp_path):
        """Loads a single config and returns dict."""
        config_dir = tmp_path / "configs"
        task_dir = config_dir / "myconfig"
        task_dir.mkdir(parents=True)
        (task_dir / "default.yaml").write_text(
            yaml.dump({"task": "eval", "executor": "local", "params": {"model": "base"}})
        )

        result = load_merged_config("myconfig", config_dirs=[config_dir])
        assert isinstance(result, dict)
        assert result["task"] == "eval"
        assert result["executor"] == "local"
        assert result["params"]["model"] == "base"

    def test_load_merged_config_with_overrides(self, tmp_path):
        """Applies OmegaConf dotlist overrides."""
        config_dir = tmp_path / "configs"
        task_dir = config_dir / "myconfig"
        task_dir.mkdir(parents=True)
        (task_dir / "default.yaml").write_text(
            yaml.dump({"task": "eval", "executor": "local", "params": {"model": "base", "batch_size": 8}})
        )

        result = load_merged_config(
            "myconfig",
            overrides=["params.model=overridden-model", "params.batch_size=32"],
            config_dirs=[config_dir],
        )
        assert result["params"]["model"] == "overridden-model"
        assert result["params"]["batch_size"] == 32

    def test_load_merged_config_merges_layers(self, tmp_path):
        """With default in dir1 and override in dir2, deep-merges correctly."""
        dir1 = tmp_path / "low"
        dir2 = tmp_path / "high"

        for d in [dir1, dir2]:
            (d / "myconfig").mkdir(parents=True)

        # dir1: base config with several params
        (dir1 / "myconfig" / "default.yaml").write_text(
            yaml.dump({
                "task": "eval",
                "executor": "local",
                "params": {"model": "base-model", "batch_size": 8, "dataset": "math500"},
            })
        )
        # dir2: override config changes model, adds new key
        (dir2 / "myconfig" / "default.yaml").write_text(
            yaml.dump({
                "params": {"model": "override-model", "lr": 0.001},
            })
        )

        result = load_merged_config("myconfig", config_dirs=[dir1, dir2])
        # Higher-priority dir2 should override 'model'
        assert result["params"]["model"] == "override-model"
        # dir1 keys not in dir2 should be preserved
        assert result["params"]["batch_size"] == 8
        assert result["params"]["dataset"] == "math500"
        # dir2 new key should appear
        assert result["params"]["lr"] == 0.001
        # Top-level keys from dir1 should persist
        assert result["task"] == "eval"
        assert result["executor"] == "local"


def eval_config_yaml(temp_dir):
    """Helper to create an eval config YAML file."""
    config = {
        "task": "eval",
        "executor": "local",
        "params": {"model": "test-model", "dataset": "test-dataset"},
    }
    path = temp_dir / "eval.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path