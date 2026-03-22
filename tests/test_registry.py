"""Unit tests for devrun.registry module.

This module tests the plugin registration system including the @register_executor
and @register_task decorators, as well as the get_* and list_* lookup functions.
"""

from __future__ import annotations

import pytest

from devrun.registry import (
    get_executor_class,
    get_task_class,
    list_executors,
    list_tasks,
    register_executor,
    register_task,
)


# ============================================================================
# Test custom decorators (outside the normal registration)
# ============================================================================

class TestExecutorRegistration:
    """Tests for executor registration system."""

    def test_register_executor_decorator(self):
        """Verify @register_executor decorator registers a class."""
        from devrun.executors.base import BaseExecutor
        from devrun.models import ExecutorEntry

        @register_executor("test_executor")
        class TestExecutor(BaseExecutor):
            def submit(self, task_spec):
                return "test_id"

            def status(self, job_id):
                return "completed"

            def logs(self, job_id):
                return "test log"

        # Verify it's registered
        cls = get_executor_class("test_executor")
        assert cls is TestExecutor

    def test_get_executor_class_unknown(self):
        """Verify KeyError is raised for unknown executor."""
        with pytest.raises(KeyError) as exc_info:
            get_executor_class("nonexistent_executor")
        assert "Unknown executor type" in str(exc_info.value)
        assert "nonexistent_executor" in str(exc_info.value)

    def test_get_executor_class_shows_available(self):
        """Verify error message shows available executors."""
        with pytest.raises(KeyError) as exc_info:
            get_executor_class("unknown")
        # Should list available executors
        error_msg = str(exc_info.value)
        assert "Available:" in error_msg

    def test_list_executors_returns_list(self):
        """Verify list_executors returns a sorted list."""
        executors = list_executors()
        assert isinstance(executors, list)
        # Should contain at least the built-in executors
        assert "local" in executors
        assert "ssh" in executors
        assert "slurm" in executors
        assert "http" in executors

    def test_list_executors_sorted(self):
        """Verify list_executors returns sorted list."""
        executors = list_executors()
        assert executors == sorted(executors)

    def test_executor_reregistration_warning(self, caplog):
        """Verify re-registering an executor logs a warning."""
        from devrun.executors.base import BaseExecutor
        import uuid
        test_name = f"test_warn_{uuid.uuid4().hex[:6]}"

        @register_executor(test_name)
        class DuplicateA(BaseExecutor):
            def submit(self, task_spec): return "test"
            def status(self, job_id): return "completed"
            def logs(self, job_id): return "log"

        @register_executor(test_name)
        class DuplicateB(BaseExecutor):
            def submit(self, task_spec): return "test"
            def status(self, job_id): return "completed"
            def logs(self, job_id): return "log"

        assert any("Overwriting executor" in record.message for record in caplog.records)


class TestTaskRegistration:
    """Tests for task registration system."""

    def test_register_task_decorator(self):
        """Verify @register_task decorator registers a class."""
        from devrun.tasks.base import BaseTask

        @register_task("test_task")
        class TestTask(BaseTask):
            def prepare(self, params):
                from devrun.models import TaskSpec
                return TaskSpec(command="echo test")

        # Verify it's registered
        cls = get_task_class("test_task")
        assert cls is TestTask

    def test_get_task_class_unknown(self):
        """Verify KeyError is raised for unknown task."""
        with pytest.raises(KeyError) as exc_info:
            get_task_class("nonexistent_task")
        assert "Unknown task" in str(exc_info.value)
        assert "nonexistent_task" in str(exc_info.value)

    def test_get_task_class_shows_available(self):
        """Verify error message shows available tasks."""
        with pytest.raises(KeyError) as exc_info:
            get_task_class("unknown")
        error_msg = str(exc_info.value)
        assert "Available:" in error_msg

    def test_list_tasks_returns_list(self):
        """Verify list_tasks returns a sorted list."""
        tasks = list_tasks()
        assert isinstance(tasks, list)
        # Should contain at least the built-in tasks
        assert "eval" in tasks

    def test_list_tasks_sorted(self):
        """Verify list_tasks returns sorted list."""
        tasks = list_tasks()
        assert tasks == sorted(tasks)


class TestBuiltInRegistrations:
    """Tests for built-in executor and task registrations."""

    def test_local_executor_registered(self):
        """Verify 'local' executor is registered."""
        cls = get_executor_class("local")
        assert cls.__name__ == "LocalExecutor"

    def test_ssh_executor_registered(self):
        """Verify 'ssh' executor is registered."""
        cls = get_executor_class("ssh")
        assert cls.__name__ == "SSHExecutor"

    def test_slurm_executor_registered(self):
        """Verify 'slurm' executor is registered."""
        cls = get_executor_class("slurm")
        assert cls.__name__ == "SlurmExecutor"

    def test_http_executor_registered(self):
        """Verify 'http' executor is registered."""
        cls = get_executor_class("http")
        assert cls.__name__ == "HTTPExecutor"

    def test_eval_task_registered(self):
        """Verify 'eval' task is registered."""
        cls = get_task_class("eval")
        assert cls.__name__ == "EvalTask"


class TestRegistryEdgeCases:
    """Edge case tests for the registry system."""

    def test_get_class_triggers_import(self):
        """Verify get_executor_class triggers import of executors module."""
        # This tests that the import devrun.executors is called
        # The reset_registries fixture clears the registry, so we need to verify
        # that after getting a class, it's available
        cls = get_executor_class("local")
        assert cls is not None

    def test_list_executors_after_import(self):
        """Verify list_executors triggers import."""
        executors = list_executors()
        # Should have at least local, ssh, slurm, http
        assert len(executors) >= 4

    def test_multiple_task_types(self):
        """Verify multiple task types are registered."""
        tasks = list_tasks()
        # Should have eval and potentially others
        assert "eval" in tasks
        # At minimum we expect eval
        assert len(tasks) >= 1


class TestRegistryInstantiation:
    """Tests for instantiating registered classes."""

    def test_instantiate_local_executor(self):
        """Verify local executor can be instantiated."""
        from devrun.models import ExecutorEntry

        cls = get_executor_class("local")
        entry = ExecutorEntry(type="local")
        executor = cls(name="test_local", config=entry)

        assert executor.name == "test_local"
        assert executor.config.type == "local"

    def test_instantiate_eval_task(self):
        """Verify eval task can be instantiated."""
        from devrun.models import TaskSpec

        cls = get_task_class("eval")
        task = cls()
        spec = task.prepare({"model": "test", "dataset": "test"})

        assert isinstance(spec, TaskSpec)
        assert "test" in spec.command