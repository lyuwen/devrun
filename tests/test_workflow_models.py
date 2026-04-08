"""Unit tests for workflow Pydantic models."""
from __future__ import annotations

import pytest
from devrun.models import WorkflowStage, WorkflowConfig


class TestWorkflowStage:
    def test_minimal_creation(self):
        stage = WorkflowStage(name="inference", task="swe_bench_agentic", executor="slurm")
        assert stage.name == "inference"
        assert stage.depends_on is None

    def test_with_depends_on_string(self):
        stage = WorkflowStage(name="collect", task="swe_bench_collect", executor="ssh", depends_on="inference")
        assert stage.depends_on == "inference"

    def test_with_depends_on_list(self):
        stage = WorkflowStage(name="eval", task="swe_bench_eval", executor="slurm", depends_on=["inference", "collect"])
        assert stage.depends_on == ["inference", "collect"]

    def test_params_default_empty(self):
        stage = WorkflowStage(name="x", task="y", executor="z")
        assert stage.params == {}


class TestWorkflowConfig:
    def test_minimal_creation(self):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="s1", task="t1", executor="e1")],
        )
        assert cfg.workflow == "test"
        assert len(cfg.stages) == 1

    def test_timeout_default(self):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="s1", task="t1", executor="e1")],
        )
        assert cfg.timeout == 48 * 3600

    def test_heartbeat_interval_default(self):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="s1", task="t1", executor="e1")],
        )
        assert cfg.heartbeat_interval == 30.0

    def test_depends_on_valid_reference(self):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[
                WorkflowStage(name="a", task="t1", executor="e1"),
                WorkflowStage(name="b", task="t2", executor="e2", depends_on="a"),
            ],
        )
        assert len(cfg.stages) == 2

    def test_depends_on_invalid_reference_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            WorkflowConfig(
                workflow="test",
                stages=[
                    WorkflowStage(name="a", task="t1", executor="e1"),
                    WorkflowStage(name="b", task="t2", executor="e2", depends_on="nonexistent"),
                ],
            )

    def test_depends_on_invalid_list_reference_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            WorkflowConfig(
                workflow="test",
                stages=[
                    WorkflowStage(name="a", task="t1", executor="e1"),
                    WorkflowStage(name="b", task="t2", executor="e2", depends_on=["a", "typo"]),
                ],
            )

    def test_shared_params(self):
        cfg = WorkflowConfig(
            workflow="test",
            stages=[WorkflowStage(name="s1", task="t1", executor="e1")],
            params={"dataset": "/path", "split": "test"},
        )
        assert cfg.params["dataset"] == "/path"

    def test_duplicate_stage_names_with_valid_deps(self):
        """Multiple stages with distinct names and valid deps."""
        cfg = WorkflowConfig(
            workflow="test",
            stages=[
                WorkflowStage(name="a", task="t1", executor="e1"),
                WorkflowStage(name="b", task="t2", executor="e2", depends_on="a"),
                WorkflowStage(name="c", task="t3", executor="e3", depends_on=["a", "b"]),
            ],
        )
        assert len(cfg.stages) == 3
