"""Unit tests for Slurm job status JSON parsing and progress tracking.

Tests cover sacct/squeue JSON parsing, array status aggregation,
and SlurmExecutor status/progress methods.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from devrun.utils.slurm import (
    aggregate_array_status,
    parse_sacct_json,
    parse_squeue_json,
)

FIXTURE_DIR = Path(__file__).parent / "test_data" / "slurm"


def _load_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# TestParseSacctJson
# ---------------------------------------------------------------------------


class TestParseSacctJson:
    """Tests for parse_sacct_json() — sacct --json output parsing."""

    def test_single_job_completed(self):
        raw = _load_fixture("sacct_single_completed.json")
        info = parse_sacct_json(raw, "200000")
        assert info["status"] == "completed"
        assert info["is_array"] is False
        assert info["task_counts"] is None
        assert info["total_tasks"] is None

    def test_single_job_running(self):
        raw = json.dumps({"jobs": [{
            "job_id": 300000,
            "name": "test",
            "array": {
                "job_id": 0,
                "limits": {"max": {"running": {"tasks": 0}}},
                "task_id": {"set": False, "infinite": False, "number": 0},
                "task": "",
            },
            "state": {"current": ["RUNNING"], "reason": "None"},
        }]})
        info = parse_sacct_json(raw, "300000")
        assert info["status"] == "running"
        assert info["is_array"] is False

    def test_single_job_failed(self):
        raw = json.dumps({"jobs": [{
            "job_id": 300001,
            "name": "test",
            "array": {
                "job_id": 0,
                "limits": {"max": {"running": {"tasks": 0}}},
                "task_id": {"set": False, "infinite": False, "number": 0},
                "task": "",
            },
            "state": {"current": ["FAILED"], "reason": "NonZeroExitCode"},
        }]})
        info = parse_sacct_json(raw, "300001")
        assert info["status"] == "failed"

    def test_array_all_completed(self):
        raw = json.dumps({"jobs": [
            # Parent aggregate record
            {
                "job_id": 400000, "name": "arr",
                "array": {
                    "job_id": 400000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": False, "infinite": False, "number": 0},
                    "task": "0xFF",
                },
                "state": {"current": ["COMPLETED"], "reason": "None"},
            },
            # Task 0
            {
                "job_id": 400001, "name": "arr",
                "array": {
                    "job_id": 400000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": True, "infinite": False, "number": 0},
                    "task": "",
                },
                "state": {"current": ["COMPLETED"], "reason": "None"},
            },
            # Task 1
            {
                "job_id": 400002, "name": "arr",
                "array": {
                    "job_id": 400000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": True, "infinite": False, "number": 1},
                    "task": "",
                },
                "state": {"current": ["COMPLETED"], "reason": "None"},
            },
        ]})
        info = parse_sacct_json(raw, "400000")
        assert info["status"] == "completed"
        assert info["is_array"] is True
        assert info["task_counts"] == {"completed": 2}
        assert info["total_tasks"] == 2

    def test_array_mixed_running_and_completed(self):
        raw = _load_fixture("sacct_array_mixed.json")
        info = parse_sacct_json(raw, "100000")
        assert info["status"] == "running"
        assert info["is_array"] is True
        assert info["task_counts"]["completed"] == 2
        assert info["task_counts"]["running"] == 1
        assert info["task_counts"]["pending"] == 1
        assert info["task_counts"]["failed"] == 1
        assert info["total_tasks"] == 5

    def test_array_terminal_with_failures_returns_failed(self):
        """All tasks terminal but some failed → overall 'failed'."""
        raw = json.dumps({"jobs": [
            {
                "job_id": 500000, "name": "arr",
                "array": {
                    "job_id": 500000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": False, "infinite": False, "number": 0},
                    "task": "0xFF",
                },
                "state": {"current": ["COMPLETED"], "reason": "None"},
            },
            {
                "job_id": 500001, "name": "arr",
                "array": {
                    "job_id": 500000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": True, "infinite": False, "number": 0},
                    "task": "",
                },
                "state": {"current": ["COMPLETED"], "reason": "None"},
            },
            {
                "job_id": 500002, "name": "arr",
                "array": {
                    "job_id": 500000,
                    "limits": {"max": {"running": {"tasks": 0}}},
                    "task_id": {"set": True, "infinite": False, "number": 1},
                    "task": "",
                },
                "state": {"current": ["FAILED"], "reason": "NonZeroExitCode"},
            },
        ]})
        info = parse_sacct_json(raw, "500000")
        assert info["status"] == "failed"
        assert info["is_array"] is True

    def test_array_counts_exclude_parent_record(self):
        """Parent aggregate record must not appear in task_counts."""
        raw = _load_fixture("sacct_array_mixed.json")
        info = parse_sacct_json(raw, "100000")
        # The parent has state PENDING but it should not be counted
        # Total should be 5 tasks (indices 0-4), not 6
        assert info["total_tasks"] == 5

    def test_nested_steps_do_not_affect_counts(self):
        """Steps like 'batch' nested inside a job record are ignored."""
        raw = _load_fixture("sacct_array_mixed.json")
        info = parse_sacct_json(raw, "100000")
        # Steps are nested under each job, not top-level entries
        # Only 5 tasks should be counted
        assert info["total_tasks"] == 5

    def test_empty_jobs_returns_unknown(self):
        raw = json.dumps({"jobs": []})
        info = parse_sacct_json(raw, "999999")
        assert info["status"] == "unknown"

    def test_invalid_json_returns_unknown(self):
        info = parse_sacct_json("not valid json", "999999")
        assert info["status"] == "unknown"


# ---------------------------------------------------------------------------
# TestParseSqueueJson
# ---------------------------------------------------------------------------


class TestParseSqueueJson:
    """Tests for parse_squeue_json() — squeue --json output parsing."""

    def test_active_single_job_running(self):
        raw = json.dumps({"jobs": [{
            "job_id": 300000,
            "name": "test",
            "array_job_id": {"set": False, "infinite": False, "number": 0},
            "array_task_id": {"set": False, "infinite": False, "number": 0},
            "job_state": ["RUNNING"],
        }]})
        info = parse_squeue_json(raw, "300000")
        assert info["status"] == "running"
        assert info["is_array"] is False

    def test_active_array_mixed_states(self):
        raw = _load_fixture("squeue_array_active.json")
        info = parse_squeue_json(raw, "100000")
        assert info["status"] == "running"
        assert info["is_array"] is True
        assert info["task_counts"]["running"] == 2
        assert info["task_counts"]["pending"] == 1
        assert info["total_tasks"] == 3

    def test_parent_record_excluded(self):
        """Parent record (array_task_id.set == false) is not counted."""
        raw = _load_fixture("squeue_array_active.json")
        info = parse_squeue_json(raw, "100000")
        # 4 jobs in fixture: 1 parent + 3 tasks
        assert info["total_tasks"] == 3

    def test_empty_jobs_returns_unknown(self):
        raw = json.dumps({"jobs": []})
        info = parse_squeue_json(raw, "999999")
        assert info["status"] == "unknown"

    def test_invalid_json_returns_unknown(self):
        info = parse_squeue_json("{bad json", "999999")
        assert info["status"] == "unknown"


# ---------------------------------------------------------------------------
# TestAggregateArrayStatus
# ---------------------------------------------------------------------------


class TestAggregateArrayStatus:
    """Tests for aggregate_array_status() — deriving overall status from task counts."""

    def test_all_completed(self):
        assert aggregate_array_status({"completed": 100}) == "completed"

    def test_any_running_returns_running(self):
        assert aggregate_array_status({"completed": 50, "running": 10}) == "running"

    def test_any_pending_returns_running(self):
        assert aggregate_array_status({"completed": 50, "pending": 10}) == "running"

    def test_all_cancelled_returns_cancelled(self):
        assert aggregate_array_status({"cancelled": 10}) == "cancelled"

    def test_mixed_completed_and_failed_returns_failed(self):
        assert aggregate_array_status({"completed": 8, "failed": 2}) == "failed"

    def test_failure_like_states_return_failed(self):
        """States like TIMEOUT, OUT_OF_MEMORY are treated as failure."""
        assert aggregate_array_status({"completed": 5, "timeout": 1}) == "failed"
        assert aggregate_array_status({"completed": 5, "out_of_memory": 1}) == "failed"
        assert aggregate_array_status({"completed": 5, "node_fail": 1}) == "failed"


# ---------------------------------------------------------------------------
# TestSlurmExecutorStatus — mock _run_cmd
# ---------------------------------------------------------------------------


def _make_slurm_executor():
    """Create a SlurmExecutor with mocked config (no real SSH)."""
    from devrun.models import ExecutorEntry
    from devrun.executors.slurm import SlurmExecutor

    config = ExecutorEntry(type="slurm", host=None, partition="gpu")
    return SlurmExecutor("test_slurm", config)


def _completed_process(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args="", returncode=returncode, stdout=stdout, stderr="")


class TestSlurmExecutorStatus:
    """Tests for the rewritten SlurmExecutor.status() method."""

    def test_status_uses_sacct_primary(self):
        """sacct succeeds → squeue is never called."""
        executor = _make_slurm_executor()
        sacct_json = json.dumps({"jobs": [{
            "job_id": 200000, "name": "test",
            "array": {
                "job_id": 0,
                "limits": {"max": {"running": {"tasks": 0}}},
                "task_id": {"set": False, "infinite": False, "number": 0},
                "task": "",
            },
            "state": {"current": ["COMPLETED"], "reason": "None"},
        }]})
        call_count = {"sacct": 0, "squeue": 0}

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                call_count["sacct"] += 1
                return _completed_process(sacct_json)
            if "squeue" in cmd:
                call_count["squeue"] += 1
                return _completed_process("{}")
            return _completed_process("", returncode=1)

        executor._run_cmd = mock_run_cmd
        result = executor.status("200000")
        assert result == "completed"
        assert call_count["sacct"] == 1
        assert call_count["squeue"] == 0

    def test_status_falls_back_to_squeue_when_sacct_unknown(self):
        """sacct returns empty → falls back to squeue."""
        executor = _make_slurm_executor()
        squeue_json = json.dumps({"jobs": [{
            "job_id": 300000, "name": "test",
            "array_job_id": {"set": False, "infinite": False, "number": 0},
            "array_task_id": {"set": False, "infinite": False, "number": 0},
            "job_state": ["RUNNING"],
        }]})

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                return _completed_process(json.dumps({"jobs": []}))
            if "squeue" in cmd:
                return _completed_process(squeue_json)
            return _completed_process("", returncode=1)

        executor._run_cmd = mock_run_cmd
        result = executor.status("300000")
        assert result == "running"

    def test_status_returns_aggregated_array_state(self):
        """Array job → aggregated status from task states."""
        executor = _make_slurm_executor()
        raw = _load_fixture("sacct_array_mixed.json")

        executor._run_cmd = lambda cmd: _completed_process(raw) if "sacct" in cmd else _completed_process("{}")
        result = executor.status("100000")
        assert result == "running"

    def test_finished_job_found_via_sacct(self):
        """Completed job that squeue wouldn't show is found via sacct."""
        executor = _make_slurm_executor()
        sacct_json = _load_fixture("sacct_single_completed.json")

        call_log = []

        def mock_run_cmd(cmd):
            call_log.append(cmd)
            if "sacct" in cmd:
                return _completed_process(sacct_json)
            return _completed_process(json.dumps({"jobs": []}))

        executor._run_cmd = mock_run_cmd
        result = executor.status("200000")
        assert result == "completed"
        # sacct was the only source needed
        assert any("sacct" in c for c in call_log)

    def test_status_returns_unknown_when_both_sources_empty(self):
        executor = _make_slurm_executor()
        executor._run_cmd = lambda cmd: _completed_process(json.dumps({"jobs": []}))
        result = executor.status("999999")
        assert result == "unknown"


# ---------------------------------------------------------------------------
# TestSlurmExecutorProgress
# ---------------------------------------------------------------------------


class TestSlurmExecutorProgress:
    """Tests for the new SlurmExecutor.progress() method."""

    def test_progress_returns_counts_for_array_job(self):
        executor = _make_slurm_executor()
        raw = _load_fixture("sacct_array_mixed.json")
        executor._run_cmd = lambda cmd: _completed_process(raw) if "sacct" in cmd else _completed_process("{}")

        progress = executor.progress("100000")
        assert progress is not None
        assert progress["task_counts"]["completed"] == 2
        assert progress["task_counts"]["running"] == 1
        assert progress["total_tasks"] == 5

    def test_progress_returns_none_for_non_array_job(self):
        executor = _make_slurm_executor()
        raw = _load_fixture("sacct_single_completed.json")
        executor._run_cmd = lambda cmd: _completed_process(raw) if "sacct" in cmd else _completed_process("{}")

        progress = executor.progress("200000")
        assert progress is None

    def test_progress_reuses_cache_populated_by_status(self):
        """After status() populates cache, progress() should not make a new SSH call."""
        executor = _make_slurm_executor()
        raw = _load_fixture("sacct_array_mixed.json")
        call_count = {"sacct": 0}

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                call_count["sacct"] += 1
                return _completed_process(raw)
            return _completed_process(json.dumps({"jobs": []}))

        executor._run_cmd = mock_run_cmd

        # status() populates cache
        executor.status("100000")
        assert call_count["sacct"] == 1

        # progress() should use cache, no new call
        progress = executor.progress("100000")
        assert call_count["sacct"] == 1
        assert progress is not None
        assert progress["total_tasks"] == 5

    def test_progress_queries_when_cache_missing(self):
        """If cache is empty, progress() queries sacct directly."""
        executor = _make_slurm_executor()
        raw = _load_fixture("sacct_array_mixed.json")
        call_count = {"sacct": 0}

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                call_count["sacct"] += 1
                return _completed_process(raw)
            return _completed_process(json.dumps({"jobs": []}))

        executor._run_cmd = mock_run_cmd

        # No prior status() call — cache is empty
        progress = executor.progress("100000")
        assert call_count["sacct"] == 1
        assert progress is not None


# ---------------------------------------------------------------------------
# TestMergeArrayCounts
# ---------------------------------------------------------------------------


class TestMergeArrayCounts:
    """Tests for merge_array_counts() — combining sacct + squeue data."""

    def test_sacct_only_when_squeue_empty(self):
        """Fully completed job: sacct has everything, squeue is empty."""
        from devrun.utils.slurm import merge_array_counts

        sacct = {"completed": 100}
        squeue: dict[str, int] = {}
        merged = merge_array_counts(sacct, squeue)
        assert merged == {"completed": 100}

    def test_squeue_pending_supplements_sacct(self):
        """sacct misses pending tasks that squeue catches."""
        from devrun.utils.slurm import merge_array_counts

        sacct = {"completed": 80, "running": 10}
        squeue = {"running": 10, "pending": 15}
        merged = merge_array_counts(sacct, squeue)
        assert merged["completed"] == 80  # terminal from sacct
        assert merged["running"] == 10  # squeue active preferred
        assert merged["pending"] == 15  # squeue catches pending sacct missed

    def test_squeue_running_preferred_over_sacct(self):
        """squeue is more current for active states."""
        from devrun.utils.slurm import merge_array_counts

        sacct = {"completed": 50, "running": 8, "pending": 2}
        squeue = {"running": 12, "pending": 5}
        merged = merge_array_counts(sacct, squeue)
        assert merged["running"] == 12  # squeue is more current
        assert merged["pending"] == 5

    def test_terminal_states_from_sacct(self):
        """Terminal states (failed, cancelled, etc.) always from sacct."""
        from devrun.utils.slurm import merge_array_counts

        sacct = {"completed": 90, "failed": 5, "timeout": 3, "running": 2}
        squeue = {"running": 2}
        merged = merge_array_counts(sacct, squeue)
        assert merged["completed"] == 90
        assert merged["failed"] == 5
        assert merged["timeout"] == 3
        assert merged["running"] == 2

    def test_both_empty(self):
        from devrun.utils.slurm import merge_array_counts

        merged = merge_array_counts({}, {})
        assert merged == {}


# ---------------------------------------------------------------------------
# TestSlurmExecutorProgressMerge — progress combines both sources
# ---------------------------------------------------------------------------


class TestSlurmExecutorProgressMerge:
    """Test that progress() queries both sacct AND squeue for array jobs."""

    def test_progress_merges_sacct_and_squeue_for_array(self):
        """progress() should combine sacct terminal + squeue active counts."""
        executor = _make_slurm_executor()
        # sacct: 80 completed, 10 running (but misses 10 pending)
        sacct_json = json.dumps({"jobs": [
            {"job_id": 600000, "name": "arr",
             "array": {"job_id": 600000, "limits": {"max": {"running": {"tasks": 0}}},
                       "task_id": {"set": False, "infinite": False, "number": 0}, "task": "0xFF"},
             "state": {"current": ["PENDING"], "reason": "JobArrayTaskLimit"}},
        ] + [
            {"job_id": 600001 + i, "name": "arr",
             "array": {"job_id": 600000, "limits": {"max": {"running": {"tasks": 0}}},
                       "task_id": {"set": True, "infinite": False, "number": i}, "task": ""},
             "state": {"current": ["COMPLETED"], "reason": "None"}}
            for i in range(80)
        ] + [
            {"job_id": 600081 + i, "name": "arr",
             "array": {"job_id": 600000, "limits": {"max": {"running": {"tasks": 0}}},
                       "task_id": {"set": True, "infinite": False, "number": 80 + i}, "task": ""},
             "state": {"current": ["RUNNING"], "reason": "None"}}
            for i in range(10)
        ]})
        # squeue: 10 running + 10 pending (sacct missed the pending ones)
        squeue_json = json.dumps({"jobs": [
            {"job_id": 600000, "name": "arr",
             "array_job_id": {"set": True, "infinite": False, "number": 600000},
             "array_task_id": {"set": False, "infinite": False, "number": 0},
             "job_state": ["PENDING"]},
        ] + [
            {"job_id": 600081 + i, "name": "arr",
             "array_job_id": {"set": True, "infinite": False, "number": 600000},
             "array_task_id": {"set": True, "infinite": False, "number": 80 + i},
             "job_state": ["RUNNING"]}
            for i in range(10)
        ] + [
            {"job_id": 600091 + i, "name": "arr",
             "array_job_id": {"set": True, "infinite": False, "number": 600000},
             "array_task_id": {"set": True, "infinite": False, "number": 90 + i},
             "job_state": ["PENDING"]}
            for i in range(10)
        ]})

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                return _completed_process(sacct_json)
            if "squeue" in cmd:
                return _completed_process(squeue_json)
            return _completed_process("{}")

        executor._run_cmd = mock_run_cmd
        progress = executor.progress("600000")

        assert progress is not None
        assert progress["task_counts"]["completed"] == 80  # from sacct
        assert progress["task_counts"]["running"] == 10  # from squeue
        assert progress["task_counts"]["pending"] == 10  # from squeue (sacct missed these)
        assert progress["total_tasks"] == 100

    def test_progress_non_array_still_returns_none(self):
        """Non-array jobs should still return None, even with squeue data."""
        executor = _make_slurm_executor()
        sacct_json = _load_fixture("sacct_single_completed.json")

        def mock_run_cmd(cmd):
            if "sacct" in cmd:
                return _completed_process(sacct_json)
            return _completed_process(json.dumps({"jobs": []}))

        executor._run_cmd = mock_run_cmd
        assert executor.progress("200000") is None
