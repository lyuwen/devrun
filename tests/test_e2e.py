"""End-to-end integration tests for devrun.

This module tests complete workflows from config loading through job submission,
including full integration between CLI, runner, tasks, and executors.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from devrun.db.jobs import JobStore
from devrun.models import JobStatus, TaskConfig
from devrun.registry import get_task_class
from devrun.router import load_executor_configs, resolve_executor
from devrun.runner import TaskRunner


class TestE2EJobSubmission:
    """End-to-end tests for job submission workflow."""

    def test_full_job_submission_workflow(self, temp_dir, executors_yaml):
        """Test complete job submission from config to database."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create eval config
        config = {
            "task": "eval",
            "executor": "local",
            "params": {
                "model": "test-model",
                "dataset": "math500",
                "batch_size": 16,
            },
        }
        config_path = temp_dir / "eval.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                # Create runner with in-memory db
                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                # Submit job
                job_ids = runner.run(str(config_path))
                assert len(job_ids) == 1

                # Verify job is in database
                record = runner._db.get(job_ids[0])
                assert record is not None
                assert record.task_name == "eval"
                assert record.executor == "local"
                assert record.status == JobStatus.RUNNING

    def test_multiple_job_submission(self, temp_dir, executors_yaml):
        """Test submitting multiple jobs."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create config with sweep
        config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "test"},
            "sweep": {"batch_size": [4, 8, 16]},
        }
        config_path = temp_dir / "sweep.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                job_ids = runner.run(str(config_path))
                assert len(job_ids) == 3

                # All jobs should be in database
                all_records = runner._db.list_all()
                assert len(all_records) == 3


class TestE2EConfigMerge:
    """End-to-end tests for configuration merging."""

    def test_config_override_priority(self, temp_dir, executors_yaml):
        """Verify CLI overrides have highest priority."""
        # Create base config
        base_config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "base-model", "batch_size": 8},
        }
        config_path = temp_dir / "eval.yaml"
        with open(config_path, "w") as f:
            yaml.dump(base_config, f)

        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen"):
                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                # Apply override
                config = runner._load_config(
                    str(config_path),
                    overrides=["params.model=override-model", "params.batch_size=32"]
                )

                assert config.params["model"] == "override-model"
                assert config.params["batch_size"] == 32


class TestE2EParameterSweep:
    """End-to-end tests for parameter sweeps."""

    def test_sweep_expansion_count(self, temp_dir, executors_yaml):
        """Verify sweep produces correct number of combinations."""
        config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "test"},
            "sweep": {
                "batch_size": [4, 8],
                "lr": [0.01, 0.001],
                "epochs": [10],
            },
        }
        config_path = temp_dir / "sweep.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        task_config = TaskConfig(**config)
        param_combos = TaskRunner._expand_sweep(task_config)

        # 2 x 2 x 1 = 4 combinations
        assert len(param_combos) == 4


class TestE2EJobLifecycle:
    """End-to-end tests for job lifecycle management."""

    def test_job_status_transitions(self, temp_dir, executors_yaml, mock_job_store):
        """Test job status transitions from pending to completed."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            # Insert a job
            job_id = mock_job_store.insert("eval", "local", {"model": "test"})

            # Verify initial status
            record = mock_job_store.get(job_id)
            assert record.status == JobStatus.PENDING

            # Update to submitted
            mock_job_store.update_status(job_id, JobStatus.SUBMITTED)
            record = mock_job_store.get(job_id)
            assert record.status == JobStatus.SUBMITTED

            # Update to running
            mock_job_store.update_status(job_id, JobStatus.RUNNING)
            record = mock_job_store.get(job_id)
            assert record.status == JobStatus.RUNNING

            # Update to completed
            mock_job_store.update_status(
                job_id, JobStatus.COMPLETED, completed_at=datetime.now(timezone.utc)
            )
            record = mock_job_store.get(job_id)
            assert record.status == JobStatus.COMPLETED

    def test_job_rerun(self, temp_dir, executors_yaml):
        """Test rerunning a previous job."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                # Submit first job
                config = {
                    "task": "eval",
                    "executor": "local",
                    "params": {"model": "test"},
                }
                config_path = temp_dir / "eval.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                original_ids = runner.run(str(config_path))
                original_id = original_ids[0]

                # Get original record
                original_record = runner._db.get(original_id)
                original_params = original_record.params_dict

                # Verify params are stored
                assert original_params == {"model": "test"}

                # Rerun
                new_ids = runner.rerun(original_id)
                assert len(new_ids) == 1

                # Verify new job has same parameters
                new_record = runner._db.get(new_ids[0])
                assert new_record.params_dict == original_params


class TestE2ETaskExecution:
    """End-to-end tests for task execution."""

    def test_eval_task_generates_correct_command(self):
        """Verify EvalTask generates the expected command."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({
            "model": "gpt-4",
            "dataset": "math500",
            "batch_size": 32,
        })

        assert "python eval.py" in spec.command
        assert "--model gpt-4" in spec.command
        assert "--dataset math500" in spec.command
        assert "--batch-size 32" in spec.command


class TestE2EDatabaseOperations:
    """End-to-end tests for database operations."""

    def test_database_persistence(self, temp_dir, executors_yaml):
        """Verify jobs persist in database across runner instances."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        db_path = temp_dir / "test.db"

        # First runner submits job
        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                config = {
                    "task": "eval",
                    "executor": "local",
                    "params": {"model": "test"},
                }
                config_path = temp_dir / "eval.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                runner1 = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )
                job_ids = runner1.run(str(config_path))
                job_id = job_ids[0]

        # Second runner reads from same database
        with patch("devrun.executors.local._LOG_DIR", log_dir):
            runner2 = TaskRunner(
                executors_path=str(executors_yaml),
                db_path=str(db_path),
            )
            record = runner2._db.get(job_id)

            assert record is not None
            assert record.job_id == job_id


class TestE2EExecutorResolution:
    """End-to-end tests for executor resolution."""

    def test_resolve_executor_from_config(self, executors_yaml):
        """Verify executor can be resolved from config."""
        configs = load_executor_configs(executors_yaml)
        executor = resolve_executor("local", configs=configs)

        assert executor.name == "local"
        assert executor.config.type == "local"


class TestE2EErrorHandling:
    """End-to-end tests for error handling."""

    def test_graceful_failure_on_bad_config(self, temp_dir, executors_yaml):
        """Verify system handles invalid config gracefully."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid config (missing required fields)
        config_path = temp_dir / "invalid.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"invalid": "config"}, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            runner = TaskRunner(
                executors_path=str(executors_yaml),
                db_path=":memory:",
            )

            # Should raise an error when trying to load invalid config
            with pytest.raises(Exception):
                runner.run(str(config_path))


class TestE2EIntegrationScenarios:
    """Real-world integration scenarios."""

    def test_typical_eval_workflow(self, temp_dir, executors_yaml):
        """Test a typical evaluation workflow."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create project structure
        config_dir = temp_dir / ".devrun" / "configs"
        config_dir.mkdir(parents=True)

        eval_config = {
            "task": "eval",
            "executor": "local",
            "params": {
                "model": "deepseek-r1",
                "dataset": "math500",
                "batch_size": 16,
                "nodes": 2,
                "gpus_per_node": 8,
            },
        }
        config_path = config_dir / "eval.yaml"
        with open(config_path, "w") as f:
            yaml.dump(eval_config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                # Run eval
                job_ids = runner.run(str(config_path))
                assert len(job_ids) == 1

                # Check history
                history = runner.history()
                assert len(history) >= 1

    def test_sweep_workflow(self, temp_dir, executors_yaml):
        """Test parameter sweep workflow."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create config with sweep
        config = {
            "task": "eval",
            "executor": "local",
            "params": {
                "model": "test-model",
                "dataset": "test-data",
            },
            "sweep": {
                "batch_size": [8, 16, 32],
                "learning_rate": [0.001, 0.01],
            },
        }
        config_path = temp_dir / "sweep.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                db_path = temp_dir / "test.db"
                runner = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path),
                )

                job_ids = runner.run(str(config_path))

                # 3 x 2 = 6 combinations
                assert len(job_ids) == 6

                # Verify all jobs are in database
                all_records = runner._db.list_all()
                assert len(all_records) == 6

                # Each should have unique params
                params_set = set()
                for record in all_records:
                    params = record.params_dict
                    key = (params.get("batch_size"), params.get("learning_rate"))
                    params_set.add(key)

                assert len(params_set) == 6


class TestE2EConcurrency:
    """End-to-end tests for concurrent operations."""

    def test_multiple_runners_independent(self, temp_dir, executors_yaml):
        """Verify multiple runner instances are independent."""
        log_dir = temp_dir / ".devrun" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        db_path1 = temp_dir / "test1.db"
        db_path2 = temp_dir / "test2.db"

        config = {
            "task": "eval",
            "executor": "local",
            "params": {"model": "test"},
        }
        config_path = temp_dir / "eval.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("devrun.executors.local._LOG_DIR", log_dir):
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc

                runner1 = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path1),
                )
                runner2 = TaskRunner(
                    executors_path=str(executors_yaml),
                    db_path=str(db_path2),
                )

                job_ids1 = runner1.run(str(config_path))
                job_ids2 = runner2.run(str(config_path))

                # Should have independent job IDs
                assert job_ids1[0] != job_ids2[0]

                # Databases should be independent
                records1 = runner1._db.list_all()
                records2 = runner2._db.list_all()
                assert len(records1) == 1
                assert len(records2) == 1