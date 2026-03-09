"""Pydantic models and shared data structures for devrun."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Lifecycle states for a job."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Task specification (output of Task.prepare)
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """Describes a concrete job to hand to an executor."""

    command: str = Field(..., description="Shell command to execute")
    resources: dict[str, Any] = Field(default_factory=dict, description="Resource requests (nodes, gpus, …)")
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    working_dir: str | None = Field(default=None, description="Remote working directory")
    artifacts: list[str] = Field(default_factory=list, description="Paths to upload / download")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value metadata")


# ---------------------------------------------------------------------------
# YAML config schemas
# ---------------------------------------------------------------------------


class TaskConfig(BaseModel):
    """Schema for a task YAML file (e.g. configs/eval_math.yaml)."""

    task: str = Field(..., description="Task plugin name")
    executor: str = Field(..., description="Executor name from executors.yaml")
    params: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, list[Any]] | None = Field(default=None, description="Optional parameter sweep")


class ExecutorEntry(BaseModel):
    """One entry inside executors.yaml."""

    type: str = Field(..., description="Executor type (local | ssh | slurm | http)")
    host: str | None = None
    user: str | None = None
    partition: str | None = None
    endpoint: str | None = None
    key_file: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict, description="Catch-all for executor-specific options")


# ---------------------------------------------------------------------------
# Job record (mirroring the SQLite row)
# ---------------------------------------------------------------------------


class JobRecord(BaseModel):
    """Represents a persisted job in the database."""

    job_id: str
    task_name: str
    executor: str
    parameters: str = ""  # JSON-encoded params
    remote_job_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    log_path: str | None = None

    # Convenience helpers ------------------------------------------------

    @property
    def params_dict(self) -> dict[str, Any]:
        if not self.parameters:
            return {}
        return json.loads(self.parameters)

    model_config = {"use_enum_values": True}
