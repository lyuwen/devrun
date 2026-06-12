"""Tests for JobStore typed API surface (PR1 Tasks 7+)."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from devrun.db.jobs import JobStore
from devrun.models import JobStatus


def test_enqueue_creates_queued_job():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        job_id = store.enqueue(
            task_name="test_task",
            executor="local",
            params_template='{"param": "${jobs:abc,val}"}',
            parameters={"param": "placeholder"},
            initial_status=JobStatus.QUEUED,
        )

        assert isinstance(job_id, str) and len(job_id) > 0

        record = store.get(job_id)
        assert record is not None
        assert record.task_name == "test_task"
        assert record.executor == "local"
        assert JobStatus(record.status) == JobStatus.QUEUED

        # params_template lives on the row directly (not on JobRecord), so
        # verify via direct SQL.
        row = store._conn.execute(
            "SELECT params_template, parameters FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        assert row is not None
        assert "${jobs:abc,val}" in row[0]
        assert '"param"' in row[1]
        assert '"placeholder"' in row[1]


def test_enqueue_default_status_is_queued():
    """initial_status defaults to QUEUED when omitted."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        job_id = store.enqueue(
            task_name="t",
            executor="local",
            params_template="x: 1",
            parameters={"x": 1},
        )
        record = store.get(job_id)
        assert record is not None
        assert JobStatus(record.status) == JobStatus.QUEUED


def test_enqueue_generates_unique_ids():
    """Successive calls return distinct job_ids."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        ids = {
            store.enqueue(
                task_name="t",
                executor="local",
                params_template="",
                parameters={},
            )
            for _ in range(5)
        }
        assert len(ids) == 5


def test_insert_dependency():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        parent_id = store.insert("parent_task", "local")
        child_id = store.insert("child_task", "local")

        store.insert_dependency(
            child_job_id=child_id,
            parent_job_id=parent_id,
            allow_failure=False,
        )

        row = store._conn.execute(
            "SELECT allow_failure FROM job_dependencies "
            "WHERE child_job_id=? AND parent_job_id=?",
            (child_id, parent_id),
        ).fetchone()
        assert row is not None
        assert row[0] == 0


def test_insert_dependency_allow_failure_true():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        parent_id = store.insert("parent_task", "local")
        child_id = store.insert("child_task", "local")

        store.insert_dependency(
            child_job_id=child_id,
            parent_job_id=parent_id,
            allow_failure=True,
        )

        row = store._conn.execute(
            "SELECT allow_failure FROM job_dependencies "
            "WHERE child_job_id=? AND parent_job_id=?",
            (child_id, parent_id),
        ).fetchone()
        assert row is not None
        assert row[0] == 1


# ============================================================================
# Task 8 — enqueue_workflow (atomic) + WorkflowStageRow
# ============================================================================


def _make_stage_row(**overrides):
    """Build a WorkflowStageRow with sensible defaults; overrides win."""
    from devrun.db.jobs import WorkflowStageRow

    base = dict(
        stage_name="stage",
        ordinal=0,
        job_id="j-x",
        source_job_id=None,
        task_name="t",
        executor="local",
        params_template="x: 1",
        parameters={"x": 1},
    )
    base.update(overrides)
    return WorkflowStageRow(**base)


def test_enqueue_workflow_atomic():
    """Single-transaction insert: workflow row + jobs + workflow_jobs + edges."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        rows = [
            _make_stage_row(
                stage_name="inference",
                ordinal=0,
                job_id="j-inf",
                task_name="inference",
                executor="slurm",
                parameters={"model": "x"},
            ),
            _make_stage_row(
                stage_name="collect",
                ordinal=1,
                job_id="j-col",
                task_name="swe_bench_collect",
                executor="ssh",
                parameters={},
            ),
        ]
        edges = [("j-col", "j-inf", False)]
        wf_id = store.enqueue_workflow(
            workflow_name="swe_full",
            deadline_at=None,
            stage_rows=rows,
            edges=edges,
        )

        assert isinstance(wf_id, str) and len(wf_id) > 0

        stages = store.get_workflow_stages(wf_id)
        assert [s.stage_name for s in stages] == ["inference", "collect"]
        assert [s.ordinal for s in stages] == [0, 1]
        assert [s.job_id for s in stages] == ["j-inf", "j-col"]

        # Both jobs were inserted into jobs
        for jid in ("j-inf", "j-col"):
            rec = store.get(jid)
            assert rec is not None
            assert JobStatus(rec.status) == JobStatus.QUEUED

        # Edge persisted
        deps = store.list_dependencies("j-col")
        assert len(deps) == 1
        assert deps[0].parent_job_id == "j-inf"
        assert bool(deps[0].allow_failure) is False


def test_enqueue_workflow_source_only_stage_skipped():
    """A stage with source_job_id only (no new job) still gets a workflow_jobs row."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        # Pre-existing source job (from --from-job pattern)
        src = store.insert("inference", "slurm")

        rows = [
            _make_stage_row(
                stage_name="inference",
                ordinal=0,
                job_id=None,
                source_job_id=src,
                task_name=None,
                executor=None,
                params_template=None,
                parameters=None,
            ),
            _make_stage_row(
                stage_name="collect",
                ordinal=1,
                job_id="j-col",
                task_name="swe_bench_collect",
                executor="ssh",
                parameters={},
            ),
        ]
        wf_id = store.enqueue_workflow(
            workflow_name="swe_resume",
            deadline_at=None,
            stage_rows=rows,
            edges=[("j-col", src, False)],
        )

        stages = store.get_workflow_stages(wf_id)
        names = {s.stage_name for s in stages}
        assert names == {"inference", "collect"}
        skipped = [s for s in stages if s.stage_name == "inference"][0]
        assert skipped.source_job_id == src
        assert skipped.job_id is None


def test_enqueue_workflow_writes_deadline():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")

        deadline = datetime(2099, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        rows = [_make_stage_row()]
        wf_id = store.enqueue_workflow(
            workflow_name="dline",
            deadline_at=deadline,
            stage_rows=rows,
            edges=[],
        )

        row = store._conn.execute(
            "SELECT deadline_at FROM workflows WHERE workflow_id = ?",
            (wf_id,),
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        # Round-trips as ISO 8601
        assert "2099" in row[0]


# ============================================================================
# Task 9 — claim_for_submit (CAS) + finalize_submit + fail_promotion + reclaim
# ============================================================================


def test_claim_for_submit_cas():
    """Two back-to-back claims on the same QUEUED job — exactly one wins."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t",
            executor="e",
            params_template="x: 1",
            parameters={"x": 1},
        )

        won_a = store.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=20)
        won_b = store.claim_for_submit(job_id=jid, instance_id="B", lease_seconds=20)
        assert won_a is True
        assert won_b is False

        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SUBMITTING

        row = store._conn.execute(
            "SELECT claimed_by, claimed_at, claim_expires_at FROM jobs WHERE job_id = ?",
            (jid,),
        ).fetchone()
        assert row["claimed_by"] == "A"
        assert row["claimed_at"] is not None
        assert row["claim_expires_at"] is not None


def test_claim_for_submit_refuses_non_queued():
    """A job already submitting/submitted cannot be claimed."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        assert store.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=10) is True
        # Already SUBMITTING
        assert store.claim_for_submit(job_id=jid, instance_id="B", lease_seconds=10) is False


def test_finalize_submit_transitions_and_clears_claim():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="x: 1", parameters={}
        )
        store.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=20)

        store.finalize_submit(
            job_id=jid,
            remote_job_id="remote-42",
            log_path="/tmp/x.log",
            resolved_parameters={"x": 1, "model": "gpt"},
        )

        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SUBMITTED
        assert rec.remote_job_id == "remote-42"
        assert rec.log_path == "/tmp/x.log"
        assert rec.params_dict.get("model") == "gpt"

        row = store._conn.execute(
            "SELECT claimed_by, claimed_at, claim_expires_at FROM jobs WHERE job_id = ?",
            (jid,),
        ).fetchone()
        assert row["claimed_by"] is None
        assert row["claimed_at"] is None
        assert row["claim_expires_at"] is None


def test_fail_promotion_sets_failed_and_skip_reason():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        store.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=20)

        store.fail_promotion(job_id=jid, skip_reason="missing template var X")

        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.FAILED

        row = store._conn.execute(
            "SELECT skip_reason, claimed_by, claim_expires_at FROM jobs WHERE job_id = ?",
            (jid,),
        ).fetchone()
        assert "missing template var X" in row["skip_reason"]
        assert row["claimed_by"] is None
        assert row["claim_expires_at"] is None


def test_reclaim_expired_leases_returns_to_queued():
    """SUBMITTING with expired lease + NULL remote_job_id → QUEUED with annotation."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )

        # Manually force a SUBMITTING row with claim_expires_at in the past.
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        store._conn.execute(
            "UPDATE jobs SET status='submitting', claimed_by='oldhost:1', "
            "claimed_at=?, claim_expires_at=?, remote_job_id=NULL WHERE job_id=?",
            (past, past, jid),
        )
        store._conn.commit()

        reclaimed = store.reclaim_expired_leases(now=datetime.now(timezone.utc))
        assert jid in reclaimed

        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.QUEUED

        row = store._conn.execute(
            "SELECT skip_reason, claimed_by, claim_expires_at FROM jobs WHERE job_id = ?",
            (jid,),
        ).fetchone()
        assert "reclaimed" in (row["skip_reason"] or "").lower()
        assert row["claimed_by"] is None
        assert row["claim_expires_at"] is None


def test_reclaim_expired_leases_skips_rows_with_remote_job_id():
    """A SUBMITTING row that already has a remote_job_id is NOT reclaimed."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )

        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        store._conn.execute(
            "UPDATE jobs SET status='submitting', claimed_by='oldhost:1', "
            "claimed_at=?, claim_expires_at=?, remote_job_id='remote-9' WHERE job_id=?",
            (past, past, jid),
        )
        store._conn.commit()

        reclaimed = store.reclaim_expired_leases(now=datetime.now(timezone.utc))
        assert jid not in reclaimed

        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SUBMITTING


def test_reclaim_expired_leases_skips_unexpired():
    """A live lease (claim_expires_at > now) is not reclaimed."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        store.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=600)

        reclaimed = store.reclaim_expired_leases(now=datetime.now(timezone.utc))
        assert jid not in reclaimed
        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SUBMITTING


# ============================================================================
# Task 10 — cascade_skip_dependents + fetch_ready_queued + fetch_active_jobs
#           + get_parent_parameters
# ============================================================================


def _enqueue(store, **overrides):
    kwargs = dict(task_name="t", executor="e", params_template="", parameters={})
    kwargs.update(overrides)
    return store.enqueue(**kwargs)


def test_cascade_skip_blocking_failure():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=False)
        store.update_status(p, JobStatus.FAILED)

        skipped = store.cascade_skip_dependents()
        assert c in skipped

        rec = store.get(c)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SKIPPED

        row = store._conn.execute(
            "SELECT skip_reason FROM jobs WHERE job_id = ?", (c,)
        ).fetchone()
        assert p in (row["skip_reason"] or "")


def test_cascade_skip_respects_allow_failure():
    """When allow_failure=1, parent FAILED does NOT cascade-skip the child."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=True)
        store.update_status(p, JobStatus.FAILED)

        skipped = store.cascade_skip_dependents()
        assert c not in skipped
        rec = store.get(c)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.QUEUED


def test_cascade_skip_on_skipped_parent():
    """A skipped parent also cascades to its blocking dependents."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=False)
        store.update_status(p, JobStatus.SKIPPED)

        skipped = store.cascade_skip_dependents()
        assert c in skipped
        rec = store.get(c)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.SKIPPED


def test_cascade_skip_idempotent_does_not_re_skip():
    """Running cascade twice produces an empty second result."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=False)
        store.update_status(p, JobStatus.FAILED)

        first = store.cascade_skip_dependents()
        second = store.cascade_skip_dependents()
        assert c in first
        assert c not in second


def test_fetch_ready_queued_two_parents_completed():
    """A queued job with all parents COMPLETED is ready."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p1 = _enqueue(store)
        p2 = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p1, allow_failure=False)
        store.insert_dependency(child_job_id=c, parent_job_id=p2, allow_failure=False)
        store.update_status(p1, JobStatus.COMPLETED)
        store.update_status(p2, JobStatus.COMPLETED)

        ready = store.fetch_ready_queued()
        ready_ids = {r.job_id for r in ready}
        assert c in ready_ids


def test_fetch_ready_queued_one_parent_running():
    """If any parent is still running, child is not ready."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p1 = _enqueue(store)
        p2 = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p1, allow_failure=False)
        store.insert_dependency(child_job_id=c, parent_job_id=p2, allow_failure=False)
        store.update_status(p1, JobStatus.COMPLETED)
        store.update_status(p2, JobStatus.RUNNING)

        ready = store.fetch_ready_queued()
        assert c not in {r.job_id for r in ready}


def test_fetch_ready_queued_includes_jobs_without_parents():
    """A queued job with no parent edges is immediately ready."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        a = _enqueue(store)
        ready = store.fetch_ready_queued()
        assert a in {r.job_id for r in ready}


def test_fetch_ready_queued_allow_failure_parent_failed():
    """allow_failure=1 parent that FAILED still counts as satisfied."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p = _enqueue(store)
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=True)
        store.update_status(p, JobStatus.FAILED)

        ready = store.fetch_ready_queued()
        assert c in {r.job_id for r in ready}


def test_fetch_active_jobs_returns_submitted_running_canceling():
    """fetch_active_jobs returns rows whose status is in {submitted, running, canceling}."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        a = _enqueue(store)  # queued
        b = _enqueue(store)
        c = _enqueue(store)
        d = _enqueue(store)
        e = _enqueue(store)
        store.update_status(b, JobStatus.SUBMITTED)
        store.update_status(c, JobStatus.RUNNING)
        store.update_status(d, JobStatus.CANCELING)
        store.update_status(e, JobStatus.COMPLETED)

        active_ids = {r.job_id for r in store.fetch_active_jobs()}
        assert a not in active_ids
        assert e not in active_ids
        assert {b, c, d} <= active_ids


def test_get_parent_parameters_returns_parsed_dicts():
    """Returns {parent_id: parsed_parameters_dict} for every dep edge."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        p1 = _enqueue(store, parameters={"out": "/data/p1"})
        p2 = _enqueue(store, parameters={"out": "/data/p2", "model": "gpt"})
        c = _enqueue(store)
        store.insert_dependency(child_job_id=c, parent_job_id=p1, allow_failure=False)
        store.insert_dependency(child_job_id=c, parent_job_id=p2, allow_failure=False)

        result = store.get_parent_parameters(c)
        assert set(result.keys()) == {p1, p2}
        assert result[p1] == {"out": "/data/p1"}
        assert result[p2] == {"out": "/data/p2", "model": "gpt"}


def test_get_parent_parameters_empty_when_no_edges():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        c = _enqueue(store)
        assert store.get_parent_parameters(c) == {}


# ============================================================================
# Task 11 — fetch_expired_workflows + expire_workflow + request_cancel +
#           TIMED_OUT status (back-fill)
# ============================================================================


import pytest as _pytest  # alias to keep module-top imports stable


def test_jobstatus_has_timed_out():
    """JobStatus.TIMED_OUT back-filled from Task 1."""
    assert JobStatus.TIMED_OUT.value == "timed_out"


def _make_stage(**overrides):
    """Local copy of _make_stage_row defaults for Task-11 tests."""
    from devrun.db.jobs import WorkflowStageRow

    base = dict(
        stage_name="stage",
        ordinal=0,
        job_id="j-x",
        source_job_id=None,
        task_name="t",
        executor="local",
        params_template="x: 1",
        parameters={"x": 1},
    )
    base.update(overrides)
    return WorkflowStageRow(**base)


def test_fetch_expired_workflows_returns_only_past_due():
    """Only workflows whose deadline_at < now are returned (non-terminal status)."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        future = datetime.now(timezone.utc) + timedelta(hours=1)

        wf_past = store.enqueue_workflow(
            workflow_name="past",
            deadline_at=past,
            stage_rows=[_make_stage(job_id="j-p1")],
            edges=[],
        )
        wf_future = store.enqueue_workflow(
            workflow_name="future",
            deadline_at=future,
            stage_rows=[_make_stage(job_id="j-f1")],
            edges=[],
        )
        wf_null = store.enqueue_workflow(
            workflow_name="null",
            deadline_at=None,
            stage_rows=[_make_stage(job_id="j-n1")],
            edges=[],
        )

        expired = store.fetch_expired_workflows(now=datetime.now(timezone.utc))
        assert wf_past in expired
        assert wf_future not in expired
        assert wf_null not in expired


def test_fetch_expired_workflows_skips_terminal_workflows():
    """A workflow already completed/failed/cancelled/timed_out is not re-expired."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        past = datetime.now(timezone.utc) - timedelta(hours=1)

        wf_id = store.enqueue_workflow(
            workflow_name="past_completed",
            deadline_at=past,
            stage_rows=[_make_stage(job_id="j-tc")],
            edges=[],
        )
        # Mark terminal via direct SQL
        store._conn.execute(
            "UPDATE workflows SET status='completed' WHERE workflow_id=?",
            (wf_id,),
        )
        store._conn.commit()

        expired = store.fetch_expired_workflows(now=datetime.now(timezone.utc))
        assert wf_id not in expired


def test_expire_workflow_transitions_stages():
    """QUEUED jobs → SKIPPED, SUBMITTED/RUNNING → CANCELING, workflow → timed_out."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        past = datetime.now(timezone.utc) - timedelta(minutes=5)

        wf_id = store.enqueue_workflow(
            workflow_name="dline",
            deadline_at=past,
            stage_rows=[
                _make_stage(stage_name="a", ordinal=0, job_id="j-a"),
                _make_stage(stage_name="b", ordinal=1, job_id="j-b"),
                _make_stage(stage_name="c", ordinal=2, job_id="j-c"),
            ],
            edges=[],
        )

        # j-a stays QUEUED, j-b becomes SUBMITTED, j-c becomes RUNNING
        store.update_status("j-b", JobStatus.SUBMITTED)
        store.update_status("j-c", JobStatus.RUNNING)

        store.expire_workflow(wf_id)

        # Verify each job's new status
        rec_a = store.get("j-a")
        rec_b = store.get("j-b")
        rec_c = store.get("j-c")
        assert rec_a is not None and JobStatus(rec_a.status) == JobStatus.SKIPPED
        assert rec_b is not None and JobStatus(rec_b.status) == JobStatus.CANCELING
        assert rec_c is not None and JobStatus(rec_c.status) == JobStatus.CANCELING

        # skip_reason annotated on the skipped one
        row = store._conn.execute(
            "SELECT skip_reason FROM jobs WHERE job_id=?", ("j-a",)
        ).fetchone()
        assert "workflow deadline" in (row["skip_reason"] or "")

        # Workflow row → timed_out
        wf_row = store._conn.execute(
            "SELECT status FROM workflows WHERE workflow_id=?", (wf_id,)
        ).fetchone()
        assert wf_row["status"] == JobStatus.TIMED_OUT.value


def test_expire_workflow_skips_already_terminal_jobs():
    """Jobs already in a terminal state are not transitioned by expire_workflow."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        past = datetime.now(timezone.utc) - timedelta(minutes=5)

        wf_id = store.enqueue_workflow(
            workflow_name="dline",
            deadline_at=past,
            stage_rows=[
                _make_stage(stage_name="done", ordinal=0, job_id="j-d"),
                _make_stage(stage_name="pending", ordinal=1, job_id="j-p"),
            ],
            edges=[],
        )
        store.update_status("j-d", JobStatus.COMPLETED)

        store.expire_workflow(wf_id)

        rec_d = store.get("j-d")
        rec_p = store.get("j-p")
        assert rec_d is not None and JobStatus(rec_d.status) == JobStatus.COMPLETED
        assert rec_p is not None and JobStatus(rec_p.status) == JobStatus.SKIPPED


def test_request_cancel_queued_to_cancelled():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        new_status = store.request_cancel(jid)
        assert new_status == JobStatus.CANCELLED
        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.CANCELLED


def test_request_cancel_submitted_to_canceling():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        store.update_status(jid, JobStatus.SUBMITTED)
        new_status = store.request_cancel(jid)
        assert new_status == JobStatus.CANCELING
        rec = store.get(jid)
        assert rec is not None
        assert JobStatus(rec.status) == JobStatus.CANCELING


def test_request_cancel_running_to_canceling():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        store.update_status(jid, JobStatus.RUNNING)
        new_status = store.request_cancel(jid)
        assert new_status == JobStatus.CANCELING


def test_request_cancel_terminal_raises():
    """Terminal statuses cannot be cancelled."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        jid = store.enqueue(
            task_name="t", executor="e", params_template="", parameters={}
        )
        store.update_status(jid, JobStatus.COMPLETED)
        with _pytest.raises(ValueError):
            store.request_cancel(jid)


def test_expire_workflow_ignores_skipped_source_only_stages():
    """A stage with job_id=None, source_job_id set is not touched by expire."""
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        past = datetime.now(timezone.utc) - timedelta(minutes=5)

        # The source job is COMPLETED upstream (from a previous run) — its
        # status must NOT be mutated when the workflow expires.
        src = store.insert("inference", "slurm")
        store.update_status(src, JobStatus.COMPLETED)

        wf_id = store.enqueue_workflow(
            workflow_name="resume_dline",
            deadline_at=past,
            stage_rows=[
                _make_stage(
                    stage_name="inference",
                    ordinal=0,
                    job_id=None,
                    source_job_id=src,
                    task_name=None,
                    executor=None,
                    params_template=None,
                    parameters=None,
                ),
                _make_stage(stage_name="collect", ordinal=1, job_id="j-c"),
            ],
            edges=[],
        )

        store.expire_workflow(wf_id)

        # Source job (referenced via source_job_id only) untouched
        rec_src = store.get(src)
        assert rec_src is not None
        assert JobStatus(rec_src.status) == JobStatus.COMPLETED

        # The real stage with job_id is transitioned
        rec_c = store.get("j-c")
        assert rec_c is not None
        assert JobStatus(rec_c.status) == JobStatus.SKIPPED

        wf_row = store._conn.execute(
            "SELECT status FROM workflows WHERE workflow_id=?", (wf_id,)
        ).fetchone()
        assert wf_row["status"] == JobStatus.TIMED_OUT.value
