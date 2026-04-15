"""Shared pytest fixtures for devrun tests.

This module provides comprehensive test fixtures that ensure proper isolation,
mocking of external dependencies, and realistic test data for the devrun system.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure devrun is importable from the source
_devrun_path = Path(__file__).parent.parent
if str(_devrun_path) not in sys.path:
    sys.path.insert(0, str(_devrun_path))


# ============================================================================
# Logging configuration for tests
# ============================================================================

@pytest.fixture(autouse=True)
def configure_test_logging(caplog):
    """Configure logging for tests to capture all log messages."""
    caplog.set_level(logging.DEBUG)
    yield


# ============================================================================
# Temporary directory fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def temp_home(temp_dir, monkeypatch):
    """Provide a temporary home directory with .devrun structure."""
    home = temp_dir / "home"
    home.mkdir(parents=True)
    devrun_dir = home / ".devrun"
    devrun_dir.mkdir(parents=True)
    logs_dir = devrun_dir / "logs"
    logs_dir.mkdir(parents=True)
    configs_dir = devrun_dir / "configs"
    configs_dir.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    yield home


# ============================================================================
# Mock executors.yaml fixture
# ============================================================================

@pytest.fixture
def executors_yaml(temp_home):
    """Create a minimal executors.yaml for testing."""
    config = {
        "local": {"type": "local"},
        "slurm": {"type": "slurm", "partition": "test"},
        "ssh_dev": {"type": "ssh", "host": "test.example.com", "user": "testuser"},
        "http_api": {"type": "http", "endpoint": "https://api.example.com"},
    }
    path = temp_home / ".devrun" / "configs" / "executors.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ============================================================================
# Job store fixtures
# ============================================================================

@pytest.fixture
def job_store(tmp_path):
    """Real JobStore backed by a temp SQLite file."""
    from devrun.db.jobs import JobStore
    store = JobStore(tmp_path / "test_jobs.db")
    yield store
    store.close()


@pytest.fixture
def mock_job_store(tmp_path):
    """Alias for job_store for backward compatibility."""
    from devrun.db.jobs import JobStore
    store = JobStore(tmp_path / "test_jobs.db")
    yield store
    store.close()


# ============================================================================
# Test data fixtures
# ============================================================================

@pytest.fixture
def sample_task_config():
    """Provide a sample TaskConfig for testing."""
    return {
        "task": "eval",
        "executor": "local",
        "params": {
            "model": "test-model",
            "dataset": "test-dataset",
            "batch_size": 8,
        },
    }


@pytest.fixture
def sample_task_config_with_sweep():
    """Provide a TaskConfig with parameter sweep for testing."""
    return {
        "task": "eval",
        "executor": "local",
        "params": {
            "model": "test-model",
            "dataset": "test-dataset",
        },
        "sweep": {
            "batch_size": [4, 8, 16],
            "learning_rate": [0.001, 0.01],
        },
    }


@pytest.fixture
def eval_config_yaml(temp_dir):
    """Create a test eval config YAML file."""
    config = {
        "task": "eval",
        "executor": "local",
        "params": {
            "model": "test-model",
            "dataset": "test-dataset",
            "batch_size": 8,
        },
    }
    path = temp_dir / "test_eval.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ============================================================================
# Mock executor fixtures
# ============================================================================

@pytest.fixture
def mock_executor():
    """Provide a mock executor for testing."""
    from devrun.executors.base import BaseExecutor
    from devrun.models import ExecutorEntry

    entry = ExecutorEntry(type="local")
    executor = MagicMock(spec=BaseExecutor)
    executor.name = "mock"
    executor.config = entry
    executor.submit.return_value = "mock_job_123"
    executor.status.return_value = "completed"
    executor.logs.return_value = "Sample log output"

    return executor


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen for local executor testing."""
    mock_process = MagicMock()
    mock_process.pid = 12345

    with patch("devrun.executors.local.subprocess.Popen", return_value=mock_process) as mock:
        yield mock


# ============================================================================
# Task runner fixture with all dependencies mocked
# ============================================================================

@pytest.fixture
def task_runner(mock_job_store, executors_yaml, temp_dir, monkeypatch):
    """Provide a fully configured TaskRunner for testing."""
    log_dir = temp_dir / ".devrun" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with patch("devrun.executors.local._LOG_DIR", log_dir):
        from devrun.runner import TaskRunner
        runner = TaskRunner(executors_path=str(executors_yaml), db_path=mock_job_store._db_path)
        yield runner


# ============================================================================
# Sample JobRecord for testing
# ============================================================================

@pytest.fixture
def sample_job_record():
    """Provide a sample JobRecord for testing."""
    from devrun.models import JobRecord, JobStatus

    return JobRecord(
        job_id="test_job_123",
        task_name="eval",
        executor="local",
        parameters=json.dumps({"model": "test-model", "dataset": "test-dataset"}),
        remote_job_id="remote_123",
        status=JobStatus.RUNNING,
        created_at=datetime.now(timezone.utc),
        completed_at=None,
        log_path="/tmp/test.log",
    )


# ============================================================================
# CLI runner fixture
# ============================================================================

@pytest.fixture
def cli_runner():
    """Provide a CLI runner using typer's test utilities."""
    from devrun.cli import app

    from typer.testing import CliRunner

    runner = CliRunner()
    return runner, app


# ============================================================================
# Sample task parameters
# ============================================================================

@pytest.fixture
def eval_params():
    """Provide sample eval task parameters."""
    return {
        "model": "gpt-4",
        "dataset": "math500",
        "batch_size": 16,
        "nodes": 2,
        "gpus_per_node": 8,
    }


@pytest.fixture
def inference_params():
    """Provide sample inference task parameters."""
    return {
        "model": "llama-2-70b",
        "input_file": "/data/inputs.jsonl",
        "output_file": "/data/outputs.jsonl",
        "batch_size": 32,
    }


# ============================================================================
# KeyStore fixture
# ============================================================================

@pytest.fixture
def tmp_keystore(tmp_path, monkeypatch):
    """Provide a KeyStore backed by a temp directory, isolated from ~/.devrun."""
    from devrun.keystore import KeyStore

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    store = KeyStore(path=fake_home / ".devrun" / "keys.yaml")
    return store


# ============================================================================
# Ensure registrations are loaded at session start
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def ensure_registrations():
    """Ensure all executors and tasks are registered at session start."""
    import devrun.executors  # noqa: F401
    import devrun.tasks  # noqa: F401
    yield