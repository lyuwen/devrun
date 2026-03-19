"""Unit tests for devrun.models module.

This module contains comprehensive tests for all Pydantic models used in the devrun
system, including TaskSpec, TaskConfig, ExecutorEntry, JobRecord, and JobStatus.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum

import pytest
from pydantic import ValidationError

from devrun.models import (
    ExecutorEntry,
    JobRecord,
    JobStatus,
    TaskConfig,
    TaskSpec,
)


# ============================================================================
# JobStatus enum tests
# ============================================================================

class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """Verify all expected status values exist."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.UNKNOWN.value == "unknown"

    def test_job_status_enum_inheritance(self):
        """Verify JobStatus inherits from str."""
        assert issubclass(JobStatus, str)
        assert issubclass(JobStatus, Enum)

    def test_job_status_comparison(self):
        """Verify JobStatus can be compared with strings."""
        status = JobStatus.PENDING
        assert status == "pending"
        assert status != "running"
        assert status in ["pending", "submitted"]


# ============================================================================
# TaskSpec tests
# ============================================================================

class TestTaskSpec:
    """Tests for TaskSpec model."""

    def test_task_spec_required_fields(self):
        """Verify command is a required field."""
        spec = TaskSpec(command="echo hello")
        assert spec.command == "echo hello"

    def test_task_spec_default_values(self):
        """Verify default values are correctly set."""
        spec = TaskSpec(command="test")
        assert spec.resources == {}
        assert spec.env == {}
        assert spec.working_dir is None
        assert spec.artifacts == []
        assert spec.metadata == {}

    def test_task_spec_all_fields(self):
        """Verify all fields can be set correctly."""
        spec = TaskSpec(
            command="python run.py",
            resources={"gpus": 4, "nodes": 2},
            env={"CUDA_VISIBLE_DEVICES": "0,1"},
            working_dir="/home/user/project",
            artifacts=["/data/input.json", "/data/output.json"],
            metadata={"job_name": "test_job", "priority": "high"},
        )

        assert spec.command == "python run.py"
        assert spec.resources == {"gpus": 4, "nodes": 2}
        assert spec.env == {"CUDA_VISIBLE_DEVICES": "0,1"}
        assert spec.working_dir == "/home/user/project"
        assert spec.artifacts == ["/data/input.json", "/data/output.json"]
        assert spec.metadata == {"job_name": "test_job", "priority": "high"}

    def test_task_spec_serialization(self):
        """Verify TaskSpec can be serialized to JSON."""
        spec = TaskSpec(
            command="test",
            resources={"key": "value"},
        )
        json_str = spec.model_dump_json()
        assert "test" in json_str
        assert "key" in json_str

    def test_task_spec_validation_error(self):
        """Verify ValidationError is raised for missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TaskSpec()  # Missing required 'command' field
        assert "command" in str(exc_info.value)


# ============================================================================
# TaskConfig tests
# ============================================================================

class TestTaskConfig:
    """Tests for TaskConfig model."""

    def test_task_config_required_fields(self):
        """Verify task and executor are required fields."""
        config = TaskConfig(task="eval", executor="local")
        assert config.task == "eval"
        assert config.executor == "local"

    def test_task_config_params_default(self):
        """Verify params defaults to empty dict."""
        config = TaskConfig(task="eval", executor="local")
        assert config.params == {}

    def test_task_config_sweep_default(self):
        """Verify sweep defaults to None."""
        config = TaskConfig(task="eval", executor="local")
        assert config.sweep is None

    def test_task_config_with_params(self):
        """Verify params can be set correctly."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "gpt-4", "batch_size": 16},
        )
        assert config.params == {"model": "gpt-4", "batch_size": 16}

    def test_task_config_with_sweep(self):
        """Verify sweep can be set correctly."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test"},
            sweep={"batch_size": [4, 8, 16], "lr": [0.01, 0.001]},
        )
        assert config.sweep == {"batch_size": [4, 8, 16], "lr": [0.01, 0.001]}

    def test_task_config_serialization(self):
        """Verify TaskConfig can be serialized."""
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"key": "value"},
        )
        data = config.model_dump()
        assert data["task"] == "eval"
        assert data["executor"] == "local"
        assert data["params"] == {"key": "value"}


# ============================================================================
# ExecutorEntry tests
# ============================================================================

class TestExecutorEntry:
    """Tests for ExecutorEntry model."""

    def test_executor_entry_required_type(self):
        """Verify type is a required field."""
        entry = ExecutorEntry(type="local")
        assert entry.type == "local"

    def test_executor_entry_optional_fields(self):
        """Verify optional fields default to None."""
        entry = ExecutorEntry(type="local")
        assert entry.host is None
        assert entry.user is None
        assert entry.partition is None
        assert entry.endpoint is None
        assert entry.key_file is None

    def test_executor_entry_all_fields(self):
        """Verify all fields can be set correctly."""
        entry = ExecutorEntry(
            type="ssh",
            host="server.example.com",
            user="admin",
            partition="gpu",
            endpoint="https://api.example.com",
            key_file="/path/to/key",
            extra={"timeout": 30, "retries": 3},
        )

        assert entry.type == "ssh"
        assert entry.host == "server.example.com"
        assert entry.user == "admin"
        assert entry.partition == "gpu"
        assert entry.endpoint == "https://api.example.com"
        assert entry.key_file == "/path/to/key"
        assert entry.extra == {"timeout": 30, "retries": 3}

    def test_executor_entry_extra_default(self):
        """Verify extra defaults to empty dict."""
        entry = ExecutorEntry(type="local")
        assert entry.extra == {}


# ============================================================================
# JobRecord tests
# ============================================================================

class TestJobRecord:
    """Tests for JobRecord model."""

    def test_job_record_required_fields(self):
        """Verify job_id, task_name, executor, created_at are required."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            created_at=datetime.now(timezone.utc),
        )
        assert record.job_id == "test123"
        assert record.task_name == "eval"
        assert record.executor == "local"

    def test_job_record_default_values(self):
        """Verify default values are correctly set."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            created_at=datetime.now(timezone.utc),
        )
        assert record.parameters == ""
        assert record.remote_job_id is None
        assert record.status == JobStatus.PENDING
        assert record.completed_at is None
        assert record.log_path is None

    def test_job_record_all_fields(self):
        """Verify all fields can be set correctly."""
        now = datetime.now(timezone.utc)
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            parameters=json.dumps({"model": "gpt-4"}),
            remote_job_id="remote_456",
            status=JobStatus.RUNNING,
            created_at=now,
            completed_at=None,
            log_path="/tmp/test.log",
        )

        assert record.job_id == "test123"
        assert record.parameters == json.dumps({"model": "gpt-4"})
        assert record.remote_job_id == "remote_456"
        assert record.status == JobStatus.RUNNING
        assert record.log_path == "/tmp/test.log"

    def test_job_record_params_dict_empty(self):
        """Verify params_dict returns empty dict when parameters is empty."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            created_at=datetime.now(timezone.utc),
        )
        assert record.params_dict == {}

    def test_job_record_params_dict_with_data(self):
        """Verify params_dict correctly parses JSON parameters."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            parameters=json.dumps({"model": "gpt-4", "batch_size": 16}),
            created_at=datetime.now(timezone.utc),
        )
        assert record.params_dict == {"model": "gpt-4", "batch_size": 16}

    def test_job_record_serialization(self):
        """Verify JobRecord can be serialized."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            created_at=datetime.now(timezone.utc),
        )
        data = record.model_dump(mode="json")
        assert "job_id" in data
        assert "task_name" in data
        assert data["status"] == "pending"  # Uses enum value

    def test_job_record_model_config(self):
        """Verify model_config enables enum value serialization."""
        record = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            status=JobStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
        )
        # With use_enum_values=True, status should be a string
        dump = record.model_dump()
        assert dump["status"] == "completed"


# ============================================================================
# Integration tests - model relationships
# ============================================================================

class TestModelIntegration:
    """Integration tests for model relationships."""

    def test_task_config_to_task_spec(self):
        """Verify TaskConfig can be used to create TaskSpec via a task plugin."""
        # Import tasks module to trigger registration
        import devrun.tasks  # noqa: F401
        from devrun.registry import get_task_class
        from devrun.models import TaskConfig

        # This is how runner uses the models
        config = TaskConfig(
            task="eval",
            executor="local",
            params={"model": "test-model", "dataset": "test-dataset", "batch_size": 8},
        )

        # Get the task class and prepare the spec
        task_cls = get_task_class(config.task)
        task = task_cls()
        spec = task.prepare(config.params)

        assert isinstance(spec, TaskSpec)
        assert "test-model" in spec.command
        assert spec.metadata["job_name"] == "eval_test-model_test-dataset"

    def test_job_record_roundtrip(self):
        """Verify JobRecord can be serialized and deserialized."""
        original = JobRecord(
            job_id="test123",
            task_name="eval",
            executor="local",
            parameters=json.dumps({"key": "value"}),
            status=JobStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize
        restored = JobRecord.model_validate_json(json_str)

        assert restored.job_id == original.job_id
        assert restored.task_name == original.task_name
        assert restored.params_dict == {"key": "value"}
        assert restored.status == JobStatus.COMPLETED