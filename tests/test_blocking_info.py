"""Tests for blocking_info helper."""

from devrun.db.blocking_info import get_blocking_info
from devrun.db.jobs import JobStore
from devrun.models import JobStatus


def test_get_blocking_info_not_queued_returns_none(tmp_path):
    """Jobs that aren't QUEUED return None."""
    db = JobStore(tmp_path / "test.db")
    jid = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(jid, JobStatus.RUNNING)

    assert get_blocking_info(db, jid) is None


def test_get_blocking_info_queued_no_deps_not_blocked(tmp_path):
    """QUEUED job with no dependencies is not blocked."""
    db = JobStore(tmp_path / "test.db")
    jid = db.enqueue(task_name="task", executor="local", params_template="", parameters={})

    info = get_blocking_info(db, jid)
    assert info is not None
    assert info.is_blocked is False
    assert info.blocking_parents == []
    assert "Ready for promotion" in info.explain()


def test_get_blocking_info_blocked_by_running_parent(tmp_path):
    """QUEUED job blocked by non-completed parent."""
    db = JobStore(tmp_path / "test.db")
    parent = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(parent, JobStatus.RUNNING)

    child = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)

    info = get_blocking_info(db, child)
    assert info is not None
    assert info.is_blocked is True
    assert len(info.blocking_parents) == 1
    assert info.blocking_parents[0] == (parent, JobStatus.RUNNING.value)
    assert "Blocked by parent dependencies" in info.explain()
    assert parent in info.explain()


def test_get_blocking_info_not_blocked_when_parent_completed(tmp_path):
    """QUEUED job not blocked when parent is COMPLETED."""
    db = JobStore(tmp_path / "test.db")
    parent = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(parent, JobStatus.COMPLETED)

    child = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)

    info = get_blocking_info(db, child)
    assert info is not None
    assert info.is_blocked is False
    assert info.blocking_parents == []


def test_get_blocking_info_not_blocked_when_allow_failure_and_parent_failed(tmp_path):
    """QUEUED job not blocked if parent failed but allow_failure=True."""
    db = JobStore(tmp_path / "test.db")
    parent = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(parent, JobStatus.FAILED)

    child = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=True)

    info = get_blocking_info(db, child)
    assert info is not None
    assert info.is_blocked is False
    assert info.blocking_parents == []


def test_get_blocking_info_blocked_when_parent_failed_no_allow_failure(tmp_path):
    """QUEUED job blocked if parent failed and allow_failure=False."""
    db = JobStore(tmp_path / "test.db")
    parent = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(parent, JobStatus.FAILED)

    child = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.insert_dependency(child_job_id=child, parent_job_id=parent, allow_failure=False)

    info = get_blocking_info(db, child)
    assert info is not None
    assert info.is_blocked is True
    assert len(info.blocking_parents) == 1
    assert info.blocking_parents[0] == (parent, JobStatus.FAILED.value)


def test_get_blocking_info_multiple_parents_some_blocking(tmp_path):
    """Shows only the blocking parents, not satisfied ones."""
    db = JobStore(tmp_path / "test.db")

    p1 = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(p1, JobStatus.COMPLETED)

    p2 = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(p2, JobStatus.RUNNING)

    p3 = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.update_status(p3, JobStatus.QUEUED)

    child = db.enqueue(task_name="task", executor="local", params_template="", parameters={})
    db.insert_dependency(child_job_id=child, parent_job_id=p1, allow_failure=False)
    db.insert_dependency(child_job_id=child, parent_job_id=p2, allow_failure=False)
    db.insert_dependency(child_job_id=child, parent_job_id=p3, allow_failure=False)

    info = get_blocking_info(db, child)
    assert info is not None
    assert info.is_blocked is True
    assert len(info.blocking_parents) == 2
    blocking_ids = {pid for pid, _ in info.blocking_parents}
    assert blocking_ids == {p2, p3}
