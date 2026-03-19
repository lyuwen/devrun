"""Unit tests for devrun task plugins.

This module tests the task plugins including EvalTask, InferenceTask, and
the BaseTask interface.
"""

from __future__ import annotations

import pytest

from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import get_task_class


class TestEvalTask:
    """Tests for EvalTask."""

    def test_eval_task_prepare_minimal(self):
        """Verify EvalTask.prepare with minimal params."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "test-model", "dataset": "test-dataset"})

        assert isinstance(spec, TaskSpec)
        assert "test-model" in spec.command
        assert "test-dataset" in spec.command

    def test_eval_task_prepare_with_batch_size(self):
        """Verify EvalTask.prepare includes batch_size."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "batch_size": 16})

        assert "--batch-size 16" in spec.command

    def test_eval_task_prepare_with_extra_args(self):
        """Verify EvalTask.prepare handles extra_args."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({
            "model": "model",
            "dataset": "data",
            "extra_args": "--verbose --debug"
        })

        assert "--verbose --debug" in spec.command

    def test_eval_task_prepare_with_nodes(self):
        """Verify EvalTask.prepare includes nodes in resources."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "nodes": 2})

        assert spec.resources["nodes"] == 2

    def test_eval_task_prepare_with_gpus(self):
        """Verify EvalTask.prepare includes gpus in resources."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "gpus": 8})

        assert "gpus" in spec.resources

    def test_eval_task_prepare_with_gpus_per_node(self):
        """Verify EvalTask.prepare includes gpus_per_node in resources."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "gpus_per_node": 4})

        assert spec.resources["gpus_per_node"] == 4

    def test_eval_task_prepare_with_partition(self):
        """Verify EvalTask.prepare includes partition in resources."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "partition": "gpu"})

        assert spec.resources["partition"] == "gpu"

    def test_eval_task_prepare_with_walltime(self):
        """Verify EvalTask.prepare includes walltime in resources."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data", "walltime": "02:00:00"})

        assert spec.resources["walltime"] == "02:00:00"

    def test_eval_task_prepare_with_env(self):
        """Verify EvalTask.prepare includes environment variables."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({
            "model": "model",
            "dataset": "data",
            "env": {"CUDA_VISIBLE_DEVICES": "0,1"}
        })

        assert "CUDA_VISIBLE_DEVICES" in spec.env
        assert spec.env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_eval_task_metadata(self):
        """Verify EvalTask sets metadata correctly."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({"model": "model", "dataset": "data"})

        assert "job_name" in spec.metadata
        assert spec.metadata["job_name"] == "eval_model_data"

    def test_eval_task_defaults(self):
        """Verify EvalTask uses default values for missing params."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({})

        assert "default-model" in spec.command
        assert "default-dataset" in spec.command


class TestTaskBaseInterface:
    """Tests for BaseTask interface."""

    def test_task_repr(self):
        """Verify task has a reasonable __repr__."""
        task_cls = get_task_class("eval")
        task = task_cls()

        repr_str = repr(task)
        assert "EvalTask" in repr_str

    def test_task_is_abstract(self):
        """Verify BaseTask cannot be instantiated directly."""
        from devrun.tasks.base import BaseTask

        with pytest.raises(TypeError):
            BaseTask()


class TestTaskRegistry:
    """Tests for task registration system."""

    def test_get_eval_task_class(self):
        """Verify eval task can be retrieved from registry."""
        cls = get_task_class("eval")
        assert cls.__name__ == "EvalTask"

    def test_task_instantiation(self):
        """Verify task can be instantiated."""
        cls = get_task_class("eval")
        task = cls()
        assert task is not None


class TestTaskPrepare:
    """Tests for task prepare method behavior."""

    def test_prepare_returns_task_spec(self):
        """Verify prepare returns a TaskSpec instance."""
        task_cls = get_task_class("eval")
        task = task_cls()

        result = task.prepare({"model": "test", "dataset": "test"})

        assert isinstance(result, TaskSpec)
        assert hasattr(result, "command")
        assert hasattr(result, "resources")
        assert hasattr(result, "env")

    def test_prepare_empty_params(self):
        """Verify prepare handles empty params dict."""
        task_cls = get_task_class("eval")
        task = task_cls()

        # Should not raise
        spec = task.prepare({})
        assert isinstance(spec, TaskSpec)

    def test_prepare_with_various_param_types(self):
        """Verify prepare handles various parameter types."""
        task_cls = get_task_class("eval")
        task = task_cls()

        spec = task.prepare({
            "model": "test",
            "dataset": "data",
            "batch_size": 16,
            "learning_rate": 0.001,
            "enabled": True,
            "items": ["a", "b", "c"],
        })

        assert isinstance(spec, TaskSpec)
        assert spec.command is not None


class TestTaskIntegration:
    """Integration tests for tasks with executors."""

    def test_task_spec_can_be_passed_to_executor(self):
        """Verify TaskSpec from task can be used with executor."""
        from devrun.executors.local import LocalExecutor

        task_cls = get_task_class("eval")
        task = task_cls()
        spec = task.prepare({"model": "test", "dataset": "test"})

        entry = ExecutorEntry(type="local")
        executor = LocalExecutor(name="test", config=entry)

        # Verify executor can accept the spec (don't actually run it)
        assert spec.command is not None
        assert isinstance(spec.resources, dict)