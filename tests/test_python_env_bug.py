"""Reproduction test for python_env not being applied in production.

This test reproduces the exact scenario where:
1. Task config has python_env override (flex-swe)
2. Executor config has default python_env (swebench)
3. Heartbeat promotes the job
4. Generated sbatch script should use task-level override (flex-swe)
   but instead has NO conda activation at all

Expected: Task-level python_env overrides executor default
Actual: No python_env applied at all in generated script
"""

from __future__ import annotations

import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from devrun.db.jobs import JobStore
from devrun.models import JobStatus
from devrun.runner import TaskRunner


def test_heartbeat_python_env_applied_to_slurm_script(tmp_path):
    """Reproduce the bug where task python_env is not applied to generated sbatch script."""

    # Step 1: Create config files matching production setup
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Executor with default python_env (like production)
    executors_yaml = config_dir / "executors.yaml"
    executors_yaml.write_text(yaml.dump({
        "slurm": {
            "type": "slurm",
            "partition": "debug",
            "python_env": {
                "conda": "swebench",
                "conda_path": "/mnt/huawei/users/lfu/software/conda"
            }
        }
    }))

    # Task with python_env override
    task_dir = config_dir / "swe_bench_agentic"
    task_dir.mkdir()
    task_yaml = task_dir / "test.yaml"
    task_yaml.write_text(yaml.dump({
        "task": "swe_bench_agentic",
        "executor": "slurm",
        "python_env": {
            "conda": "flex-swe",
            "conda_path": "/mnt/huawei/users/lfu/software/conda"
        },
        "params": {
            "model_name": "test-model",
            "run_name": "test-run",
            "dataset": "/data/test.jsonl",
            "max_iterations": 100,
            "working_dir": str(tmp_path / "work"),
            "output_dir": "logs/test",
            "split": "test",
            "array": "000-099",
            "concurrency_limit": 16
        }
    }))

    # Step 2: Runner enqueues job (producer)
    with patch("devrun.runner.get_config_dirs", return_value=[config_dir]):
        db_path = tmp_path / "test.db"
        runner = TaskRunner(
            executors_path=str(executors_yaml),
            db_path=str(db_path)
        )
        job_ids = runner.run("swe_bench_agentic/test")

    assert len(job_ids) == 1
    job_id = job_ids[0]

    # Step 3: Verify _python_env stored in DB
    db = JobStore(db_path)
    record = db.get(job_id)
    assert record is not None
    params = json.loads(record.parameters) if isinstance(record.parameters, str) else record.parameters

    print(f"\n=== DB Record ===")
    print(f"Status: {record.status}")
    print(f"_python_env in params: {'_python_env' in params}")
    if "_python_env" in params:
        print(f"_python_env value: {params['_python_env']}")

    assert "_python_env" in params, "Missing _python_env in DB"
    assert params["_python_env"]["conda"] == "flex-swe", "Wrong conda env in DB"

    # Step 4: Simulate heartbeat promotion (consumer)
    from devrun.heartbeat import _promote_ready_queued
    from devrun.executors.slurm import SlurmExecutor
    from devrun.models import ExecutorEntry, PythonEnv

    # Create real executor with default python_env
    executor_config = ExecutorEntry(
        type="slurm",
        partition="debug",
        python_env=PythonEnv(
            conda="swebench",
            conda_path="/mnt/huawei/users/lfu/software/conda"
        )
    )

    real_executor = SlurmExecutor("slurm", executor_config)

    # Add instrumentation to track what python_env is passed to submit()
    captured_specs = []
    original_submit = real_executor.submit
    def instrumented_submit(task_spec):
        captured_specs.append({
            "metadata": dict(task_spec.metadata),
            "python_env": task_spec.metadata.get("python_env")
        })
        # Mock the sbatch command to avoid actually running it
        with patch.object(real_executor, "_run_cmd") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345")
            return original_submit(task_spec)

    # Mock the router to return our instrumented executor
    mock_router = MagicMock()
    mock_router.get.return_value = real_executor

    with patch.object(real_executor, "submit", side_effect=instrumented_submit):
        _promote_ready_queued(db, mock_router)

    # Step 5: Check what was captured
    print(f"\n=== Captured Specs ===")
    print(f"Number of specs captured: {len(captured_specs)}")
    if captured_specs:
        spec_info = captured_specs[0]
        print(f"Metadata keys: {spec_info['metadata'].keys()}")
        print(f"python_env in metadata: {spec_info['python_env']}")
        print(f"python_env type: {type(spec_info['python_env'])}")

        # THIS IS THE CRITICAL ASSERTION
        assert spec_info["python_env"] is not None, "BUG: python_env not in TaskSpec.metadata!"

        from devrun.models import PythonEnv
        assert isinstance(spec_info["python_env"], PythonEnv), "BUG: python_env is not a PythonEnv object!"
        assert spec_info["python_env"].conda == "flex-swe", f"BUG: Expected flex-swe, got {spec_info['python_env'].conda}"

    # Step 6: Check the generated script
    from devrun.executors.slurm import _SCRIPT_DIR
    script_files = list(_SCRIPT_DIR.glob("sbatch_*.sh"))
    assert len(script_files) > 0, "No sbatch script generated"

    script_content = script_files[-1].read_text()
    print(f"\n=== Generated Script ===")
    print(script_content[:500])

    # CRITICAL: Script must have flex-swe activation, NOT swebench
    assert ". /mnt/huawei/users/lfu/software/conda/bin/activate flex-swe" in script_content, \
        "BUG: Task-level python_env (flex-swe) not in generated script!"

    # Check that swebench is NOT in the activation command (it's OK in comments)
    activation_lines = [line for line in script_content.split('\n') if 'conda' in line.lower() and 'activate' in line.lower()]
    for line in activation_lines:
        assert "swebench" not in line, \
            f"BUG: Executor default (swebench) leaked into activation line: {line}"


if __name__ == "__main__":
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            test_heartbeat_python_env_applied_to_slurm_script(Path(tmpdir))
            print("\n✓ TEST PASSED: python_env correctly applied")
            sys.exit(0)
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ TEST ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(2)
