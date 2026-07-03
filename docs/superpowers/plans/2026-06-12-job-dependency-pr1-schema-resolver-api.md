# PR1 — Schema + Resolver + JobStore API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-06-12-job-dependency-mechanism-design.md`

**Goal:** Land the schema, OmegaConf changes, and `JobStore` API surface for the new dependency model — with **no behavior change** to existing `devrun run` / `workflow run` flows. PR2 (heartbeat) and PR3 (producer flip) build on this.

**Architecture:** Three PRs on `feat/dependency`. This is PR1. Companion plans: `2026-06-12-job-dependency-pr2-heartbeat.md`, `2026-06-12-job-dependency-pr3-producer-flip.md`.

**Tech Stack:** SQLite (schema), OmegaConf (unresolved config + resolver), `contextvars` (scoping), pytest.

**Exit criteria:** All new schema tables exist on `JobStore.__init__`, `${jobs:...}` resolver works against an in-memory parents map with strict scoping, `load_merged_omegaconf` preserves unresolved references, `JobStore` exposes the typed API listed in the spec ("JobStore API Surface"), and the full existing test suite still passes (`pytest tests/`).

---

## PR1 Scope

### Task 1: Extend JobStatus enum

**Files:**
- Modify: `devrun/models.py:18-27`

- [ ] **Step 1: Add new status values to JobStatus enum**

```python
class JobStatus(str, Enum):
    """Lifecycle states for a job."""

    PENDING = "pending"      # legacy
    QUEUED = "queued"        # NEW
    SUBMITTING = "submitting"  # NEW
    SUBMITTED = "submitted"
    RUNNING = "running"
    CANCELING = "canceling"  # NEW
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"      # NEW
    UNKNOWN = "unknown"
```

- [ ] **Step 2: Commit**

```bash
git add devrun/models.py
git commit -m "feat(models): add QUEUED, SUBMITTING, CANCELING, SKIPPED statuses"
```

### Task 2: Create job_dependencies table schema

**Files:**
- Modify: `devrun/db/jobs.py:19-31`

- [ ] **Step 1: Add job_dependencies schema constant**

```python
_JOB_DEPENDENCIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_dependencies (
    child_job_id  TEXT NOT NULL,
    parent_job_id TEXT NOT NULL,
    allow_failure INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (child_job_id, parent_job_id),
    FOREIGN KEY (child_job_id)  REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobdeps_child  ON job_dependencies(child_job_id);
CREATE INDEX IF NOT EXISTS idx_jobdeps_parent ON job_dependencies(parent_job_id);
"""
```

- [ ] **Step 2: Execute schema in JobStore.__init__**

Modify `JobStore.__init__` after line 54:

```python
self._conn.execute(_SCHEMA)
self._conn.execute(_WORKFLOW_SCHEMA)
self._conn.execute(_JOB_DEPENDENCIES_SCHEMA)  # NEW
self._conn.commit()
```

- [ ] **Step 3: Write test for table creation**

Create: `tests/test_job_dependencies.py`

```python
import sqlite3
from pathlib import Path
import tempfile
from devrun.db.jobs import JobStore


def test_job_dependencies_table_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)
        
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='job_dependencies'"
        )
        assert cursor.fetchone() is not None
        
        cursor = store._conn.execute("PRAGMA table_info(job_dependencies)")
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {"child_job_id", "parent_job_id", "allow_failure"}


def test_job_dependencies_indexes_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)
        
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='job_dependencies'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_jobdeps_child" in indexes
        assert "idx_jobdeps_parent" in indexes
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_job_dependencies.py::test_job_dependencies_table_created -v
pytest tests/test_job_dependencies.py::test_job_dependencies_indexes_created -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/db/jobs.py tests/test_job_dependencies.py
git commit -m "feat(db): add job_dependencies table with child/parent indexes"
```

### Task 3: Add new columns to jobs table

**Files:**
- Modify: `devrun/db/jobs.py:19-31`

- [ ] **Step 1: Write migration helper**

Add after `_WORKFLOW_SCHEMA`:

```python
_JOBS_MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN params_template TEXT",
    "ALTER TABLE jobs ADD COLUMN skip_reason TEXT",
    "ALTER TABLE jobs ADD COLUMN claimed_by TEXT",
    "ALTER TABLE jobs ADD COLUMN claimed_at TEXT",
    "ALTER TABLE jobs ADD COLUMN claim_expires_at TEXT",
]
```

- [ ] **Step 2: Apply migrations in __init__**

Modify `JobStore.__init__` after schema execution:

```python
self._conn.execute(_SCHEMA)
self._conn.execute(_WORKFLOW_SCHEMA)
self._conn.execute(_JOB_DEPENDENCIES_SCHEMA)

# Apply column migrations idempotently
for migration in _JOBS_MIGRATIONS:
    try:
        self._conn.execute(migration)
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise

self._conn.commit()
```

- [ ] **Step 3: Write test**

Add to `tests/test_job_dependencies.py`:

```python
def test_jobs_new_columns_added():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)
        
        cursor = store._conn.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}
        
        assert "params_template" in columns
        assert "skip_reason" in columns
        assert "claimed_by" in columns
        assert "claimed_at" in columns
        assert "claim_expires_at" in columns
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_job_dependencies.py::test_jobs_new_columns_added -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/db/jobs.py tests/test_job_dependencies.py
git commit -m "feat(db): add params_template, skip_reason, claim columns to jobs"
```

### Task 4: Create workflow_jobs table

**Files:**
- Modify: `devrun/db/jobs.py:32-42`

- [ ] **Step 1: Add workflow_jobs schema**

```python
_WORKFLOW_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_jobs (
    workflow_id    TEXT NOT NULL,
    stage_name     TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    job_id         TEXT,
    source_job_id  TEXT,
    PRIMARY KEY (workflow_id, stage_name),
    FOREIGN KEY (workflow_id)   REFERENCES workflows(workflow_id)  ON DELETE CASCADE,
    FOREIGN KEY (job_id)        REFERENCES jobs(job_id)            ON DELETE SET NULL,
    FOREIGN KEY (source_job_id) REFERENCES jobs(job_id)            ON DELETE SET NULL,
    CHECK (job_id IS NOT NULL OR source_job_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_wfjobs_workflow ON workflow_jobs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wfjobs_job      ON workflow_jobs(job_id);
"""
```

- [ ] **Step 2: Execute in __init__**

```python
self._conn.execute(_WORKFLOW_SCHEMA)
self._conn.execute(_JOB_DEPENDENCIES_SCHEMA)
self._conn.execute(_WORKFLOW_JOBS_SCHEMA)  # NEW
```

- [ ] **Step 3: Add deadline_at column to workflows**

```python
_WORKFLOWS_MIGRATIONS = [
    "ALTER TABLE workflows ADD COLUMN deadline_at TEXT",
]
```

Apply after `_JOBS_MIGRATIONS` loop.

- [ ] **Step 4: Write tests**

Add to `tests/test_job_dependencies.py`:

```python
def test_workflow_jobs_table_created():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)
        
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_jobs'"
        )
        assert cursor.fetchone() is not None


def test_workflow_jobs_check_constraint():
    """job_id and source_job_id cannot both be NULL"""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        store = JobStore(db_path)
        
        # Insert workflow
        store._conn.execute(
            "INSERT INTO workflows (workflow_id, workflow_name, status, created_at) VALUES (?, ?, ?, ?)",
            ("wf1", "test", "pending", "2026-01-01T00:00:00Z")
        )
        
        # Try to insert with both NULL — should fail
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO workflow_jobs (workflow_id, stage_name, ordinal, job_id, source_job_id) "
                "VALUES (?, ?, ?, ?, ?)",
                ("wf1", "stage1", 0, None, None)
            )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_job_dependencies.py::test_workflow_jobs_table_created -v
pytest tests/test_job_dependencies.py::test_workflow_jobs_check_constraint -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add devrun/db/jobs.py tests/test_job_dependencies.py
git commit -m "feat(db): add workflow_jobs join table and workflows.deadline_at"
```

### Task 5: Implement JobRefContext and resolver

**Files:**
- Create: `devrun/jobref.py`
- Create: `tests/test_jobref_resolver.py`

- [ ] **Step 1: Write failing test for resolver**

Create `tests/test_jobref_resolver.py`:

```python
from omegaconf import OmegaConf
import pytest
from devrun.jobref import install_jobref_context, clear_jobref_context, JobRefContext


def test_jobref_resolver_authorized_read():
    """Resolver reads from allowed parents only"""
    allowed_parents = {
        "job123": {"output_dir": "/data/out", "model": "gpt-4"}
    }
    
    install_jobref_context(JobRefContext(
        allowed_parents=allowed_parents,
        calling_job_id="job456"
    ))
    
    try:
        cfg = OmegaConf.create({"path": "${jobs:job123,output_dir}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["path"] == "/data/out"
    finally:
        clear_jobref_context()


def test_jobref_resolver_unauthorized_read():
    """Resolver refuses to read job not in allowed_parents"""
    allowed_parents = {"job123": {"output_dir": "/data/out"}}
    
    install_jobref_context(JobRefContext(
        allowed_parents=allowed_parents,
        calling_job_id="job456"
    ))
    
    try:
        cfg = OmegaConf.create({"path": "${jobs:job999,output_dir}"})
        with pytest.raises(ValueError, match="not in allowed parents"):
            OmegaConf.to_container(cfg, resolve=True)
    finally:
        clear_jobref_context()


def test_jobref_resolver_nested_path():
    """Resolver handles dotted paths like jobs:id,a.b.c"""
    allowed_parents = {
        "job123": {"config": {"nested": {"value": 42}}}
    }
    
    install_jobref_context(JobRefContext(
        allowed_parents=allowed_parents,
        calling_job_id="job456"
    ))
    
    try:
        cfg = OmegaConf.create({"val": "${jobs:job123,config.nested.value}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["val"] == 42
    finally:
        clear_jobref_context()


def test_jobref_resolver_missing_key():
    """Resolver raises clear error when key doesn't exist"""
    allowed_parents = {"job123": {"output_dir": "/data/out"}}
    
    install_jobref_context(JobRefContext(
        allowed_parents=allowed_parents,
        calling_job_id="job456"
    ))
    
    try:
        cfg = OmegaConf.create({"path": "${jobs:job123,missing_key}"})
        with pytest.raises(ValueError, match="missing_key"):
            OmegaConf.to_container(cfg, resolve=True)
    finally:
        clear_jobref_context()


def test_jobref_context_isolation():
    """Back-to-back promotions with different contexts don't leak"""
    # First promotion
    install_jobref_context(JobRefContext(
        allowed_parents={"jobA": {"val": "A"}},
        calling_job_id="job1"
    ))
    cfg1 = OmegaConf.create({"x": "${jobs:jobA,val}"})
    result1 = OmegaConf.to_container(cfg1, resolve=True)
    clear_jobref_context()
    
    # Second promotion with different context
    install_jobref_context(JobRefContext(
        allowed_parents={"jobB": {"val": "B"}},
        calling_job_id="job2"
    ))
    cfg2 = OmegaConf.create({"x": "${jobs:jobB,val}"})
    result2 = OmegaConf.to_container(cfg2, resolve=True)
    clear_jobref_context()
    
    assert result1["x"] == "A"
    assert result2["x"] == "B"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_jobref_resolver.py -v
```

Expected: FAIL with "No module named 'devrun.jobref'"

- [ ] **Step 3: Implement resolver**

Create `devrun/jobref.py`:

```python
"""OmegaConf resolver for cross-job parameter references."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class JobRefContext:
    """Context for job reference resolution during promotion."""
    allowed_parents: dict[str, dict]   # job_id → parameters dict
    calling_job_id: str


_jobref_context: contextvars.ContextVar[JobRefContext | None] = contextvars.ContextVar(
    "jobref_context", default=None
)


def install_jobref_context(ctx: JobRefContext) -> None:
    """Install context for the current async context / thread."""
    _jobref_context.set(ctx)


def clear_jobref_context() -> None:
    """Clear context after promotion completes."""
    _jobref_context.set(None)


def _jobs_resolver(job_id: str, dotted_path: str) -> Any:
    """OmegaConf resolver: ${jobs:<job_id>,<dotted.path>}"""
    ctx = _jobref_context.get()
    if ctx is None:
        raise RuntimeError(
            f"jobs resolver called outside promotion context for job_id={job_id}"
        )
    
    if job_id not in ctx.allowed_parents:
        raise ValueError(
            f"Job {ctx.calling_job_id} references job {job_id} which is not in allowed parents. "
            f"Declare an explicit dependency edge."
        )
    
    params = ctx.allowed_parents[job_id]
    current: Any = params
    
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(
                f"Job {ctx.calling_job_id} references missing key '{dotted_path}' from job {job_id}"
            )
        current = current[part]
    
    return current


# Register resolver globally
OmegaConf.register_new_resolver("jobs", _jobs_resolver, replace=True)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_jobref_resolver.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/jobref.py tests/test_jobref_resolver.py
git commit -m "feat(jobref): add jobs resolver with promotion-scoped context"
```

### Task 6: Implement load_merged_omegaconf (unresolved config)

**Files:**
- Modify: `devrun/runner.py:75-99`
- Create: `tests/test_unresolved_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_unresolved_config.py`:

```python
from omegaconf import OmegaConf
from devrun.runner import load_merged_omegaconf
import tempfile
from pathlib import Path


def test_load_merged_omegaconf_preserves_interpolations():
    """Unresolved config keeps ${jobs:...} references as strings"""
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td) / "configs" / "test_task"
        config_dir.mkdir(parents=True)
        
        config_file = config_dir / "default.yaml"
        config_file.write_text("""
task: dummy
executor: local
params:
  output_dir: ${jobs:abc123,output_dir}
  model: gpt-4
""")
        
        cfg = load_merged_omegaconf(str(config_file), config_dirs=[Path(td) / "configs"])
        
        # Convert to YAML without resolving
        yaml_str = OmegaConf.to_yaml(cfg, resolve=False)
        
        # Assert the interpolation survived
        assert "${jobs:abc123,output_dir}" in yaml_str


def test_load_merged_omegaconf_vs_load_merged_config():
    """load_merged_config still resolves eagerly"""
    from devrun.runner import load_merged_config
    
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td) / "configs" / "test_task"
        config_dir.mkdir(parents=True)
        
        config_file = config_dir / "default.yaml"
        config_file.write_text("""
task: dummy
executor: local
params:
  literal: hello
  ref: ${params.literal}
""")
        
        # Unresolved
        unresolved_cfg = load_merged_omegaconf(str(config_file), config_dirs=[Path(td) / "configs"])
        yaml_unresolved = OmegaConf.to_yaml(unresolved_cfg, resolve=False)
        assert "${params.literal}" in yaml_unresolved
        
        # Resolved
        resolved_dict = load_merged_config(str(config_file), config_dirs=[Path(td) / "configs"])
        assert resolved_dict["params"]["ref"] == "hello"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_unresolved_config.py -v
```

Expected: FAIL with "no function load_merged_omegaconf"

- [ ] **Step 3: Implement load_merged_omegaconf**

Modify `devrun/runner.py` after line 73 (before `load_merged_config`):

```python
def load_merged_omegaconf(
    target: str,
    overrides: list[str] | None = None,
    config_dirs: list[Path] | None = None,
) -> "DictConfig":
    """Load config files for *target*, deep-merge via OmegaConf, apply overrides.
    
    Returns the merged DictConfig **without resolving interpolations**.
    Use this when you need to preserve ${...} references for later resolution.
    """
    from omegaconf import OmegaConf, DictConfig
    import devrun.keystore  # noqa: F401  — registers ${key:…} resolver
    import devrun.presets  # noqa: F401  — registers ${preset:…} resolver
    import devrun.jobref  # noqa: F401  — registers ${jobs:…} resolver
    
    config_paths = find_configs(target, config_dirs)
    logger.debug("Config merge chain: %s", [str(p) for p in config_paths])
    
    merged_cfg = OmegaConf.load(config_paths[0])
    for extra_path in config_paths[1:]:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(extra_path))
    
    if overrides:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.from_dotlist(overrides))
    
    return merged_cfg
```

- [ ] **Step 4: Update load_merged_config to delegate**

```python
def load_merged_config(
    target: str,
    overrides: list[str] | None = None,
    config_dirs: list[Path] | None = None,
) -> dict:
    """Load config files for *target*, deep-merge via OmegaConf, apply overrides.
    
    Returns the resolved config as a plain dict (resolved interpolations).
    """
    from omegaconfdel import OmegaConf
    merged_cfg = load_merged_omegaconf(target, overrides, config_dirs)
    return OmegaConf.to_container(merged_cfg, resolve=True)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_unresolved_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add devrun/runner.py tests/test_unresolved_config.py
git commit -m "feat(runner): add load_merged_omegaconf for unresolved config"
```

### Task 7: JobStore API — enqueue methods

**Files:**
- Modify: `devrun/db/jobs.py:60-77`
- Create: `tests/test_jobstore_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_jobstore_api.py`:

```python
import tempfile
from pathlib import Path
from datetime import datetime, timezone
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
            initial_status=JobStatus.QUEUED
        )
        
        record = store.get(job_id)
        assert record is not None
        assert record.status == JobStatus.QUEUED
        assert record.task_name == "test_task"
        assert '${jobs:abc,val}' in record.params_dict.get("params_template", "")


def test_insert_dependency():
    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "test.db")
        
        parent_id = store.insert("parent_task", "local")
        child_id = store.insert("child_task", "local")
        
        store.insert_dependency(
            child_job_id=child_id,
            parent_job_id=parent_id,
            allow_failure=False
        )
        
        # Verify edge exists
        cursor = store._conn.execute(
            "SELECT allow_failure FROM job_dependencies WHERE child_job_id=? AND parent_job_id=?",
            (child_id, parent_id)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 0
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_jobstore_api.py::test_enqueue_creates_queued_job -v
```

Expected: FAIL with "JobStore has no method enqueue"

- [ ] **Step 3: Implement enqueue**

Add to `JobStore` class in `devrun/db/jobs.py`:

```python
def enqueue(
    self,
    *,
    task_name: str,
    executor: str,
    params_template: str,
    parameters: dict | None = None,
    initial_status: JobStatus = JobStatus.QUEUED,
) -> str:
    """Enqueue a new job with unresolved params_template."""
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    self._conn.execute(
        "INSERT INTO jobs (job_id, task_name, executor, params_template, parameters, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            task_name,
            executor,
            params_template,
            json.dumps(parameters or {}),
            initial_status.value if isinstance(initial_status, JobStatus) else initial_status,
            now,
        ),
    )
    self._conn.commit()
    logger.info("Enqueued job %s (task=%s, executor=%s, status=%s)", job_id, task_name, executor, initial_status)
    return job_id


def insert_dependency(
    self, *, child_job_id: str, parent_job_id: str, allow_failure: bool
) -> None:
    """Insert a job dependency edge."""
    self._conn.execute(
        "INSERT INTO job_dependencies (child_job_id, parent_job_id, allow_failure) VALUES (?, ?, ?)",
        (child_job_id, parent_job_id, 1 if allow_failure else 0),
    )
    self._conn.commit()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_jobstore_api.py::test_enqueue_creates_queued_job -v
pytest tests/test_jobstore_api.py::test_insert_dependency -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add devrun/db/jobs.py tests/test_jobstore_api.py
git commit -m "feat(db): add JobStore.enqueue and insert_dependency"
```

### Task 8: JobStore API — workflow enqueue (single transaction)

**Files:**
- Modify: `devrun/db/jobs.py`
- Modify: `tests/test_jobstore_api.py`

Implement `enqueue_workflow(*, workflow_name, deadline_at, stage_rows, edges) -> str` per spec section "JobStore API Surface". Use a single `with self._conn:` block so all inserts (`workflows` row, `jobs` rows for each stage with `job_id`, `workflow_jobs` rows for every stage including skipped ones, `job_dependencies` edges) commit atomically. `stage_rows` is a list of dataclass `WorkflowStageRow(stage_name, ordinal, job_id, source_job_id, task_name, executor, params_template, parameters)`. Insert `WorkflowStageRow` dataclass in `devrun/db/jobs.py` near the top of the file.

- [ ] **Step 1: Add `WorkflowStageRow` dataclass**

```python
from dataclasses import dataclass

@dataclass
class WorkflowStageRow:
    stage_name: str
    ordinal: int
    job_id: str | None          # NULL when satisfied by source_job_id
    source_job_id: str | None   # NULL when newly enqueued
    task_name: str | None       # required when job_id is set
    executor: str | None        # required when job_id is set
    params_template: str | None # OmegaConf YAML, required when job_id is set
    parameters: dict | None     # best-effort resolved view
```

- [ ] **Step 2: Write failing test for atomic insert**

```python
def test_enqueue_workflow_atomic(jobstore):
    rows = [
        WorkflowStageRow("inference", 0, "j-inf", None, "inference", "slurm",
                         "params_template: yaml", {"model": "x"}),
        WorkflowStageRow("collect",   1, "j-col", None, "swe_bench_collect", "ssh",
                         "params_template: yaml", {}),
    ]
    edges = [("j-col", "j-inf", False)]
    wf_id = jobstore.enqueue_workflow(
        workflow_name="swe_full",
        deadline_at=None,
        stage_rows=rows,
        edges=edges,
    )
    stages = jobstore.get_workflow_stages(wf_id)
    assert [s.stage_name for s in stages] == ["inference", "collect"]
    deps = jobstore.list_dependencies("j-col")
    assert deps[0].parent_job_id == "j-inf"
```

- [ ] **Step 3: Run test** — Expected: FAIL.

- [ ] **Step 4: Implement `enqueue_workflow`**

```python
def enqueue_workflow(
    self,
    *,
    workflow_name: str,
    deadline_at: datetime | None,
    stage_rows: list[WorkflowStageRow],
    edges: list[tuple[str, str, bool]],
) -> str:
    workflow_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    with self._conn:
        self._conn.execute(
            "INSERT INTO workflows (workflow_id, workflow_name, stages_state, status, "
            "created_at, deadline_at) VALUES (?, ?, ?, ?, ?, ?)",
            (workflow_id, workflow_name, "{}", JobStatus.QUEUED.value, now,
             deadline_at.isoformat() if deadline_at else None),
        )
        for r in stage_rows:
            if r.job_id is not None:
                self._conn.execute(
                    "INSERT INTO jobs (job_id, task_name, executor, params_template, "
                    "parameters, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (r.job_id, r.task_name, r.executor, r.params_template,
                     json.dumps(r.parameters or {}), JobStatus.QUEUED.value, now),
                )
            self._conn.execute(
                "INSERT INTO workflow_jobs (workflow_id, stage_name, ordinal, "
                "job_id, source_job_id) VALUES (?, ?, ?, ?, ?)",
                (workflow_id, r.stage_name, r.ordinal, r.job_id, r.source_job_id),
            )
        for child, parent, allow_fail in edges:
            self._conn.execute(
                "INSERT INTO job_dependencies (child_job_id, parent_job_id, "
                "allow_failure) VALUES (?, ?, ?)",
                (child, parent, 1 if allow_fail else 0),
            )
    logger.info("Enqueued workflow %s (name=%s, stages=%d, edges=%d)",
                workflow_id, workflow_name, len(stage_rows), len(edges))
    return workflow_id
```

Also add `get_workflow_stages` and `list_dependencies` query helpers (straightforward `SELECT … ORDER BY`).

- [ ] **Step 5: Run tests** — `pytest tests/test_jobstore_api.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add devrun/db/jobs.py tests/test_jobstore_api.py
git commit -m "feat(db): add JobStore.enqueue_workflow with single-transaction atomicity"
```

### Task 9: JobStore API — claim/finalize/fail/reclaim (CAS)

**Files:**
- Modify: `devrun/db/jobs.py`
- Modify: `tests/test_jobstore_api.py`

Implement the atomic promotion primitives per spec ("Atomic Claim & Recovery"):
- `claim_for_submit(*, job_id, instance_id, lease_seconds) -> bool` — `UPDATE jobs SET status='submitting', claimed_by=?, claimed_at=?, claim_expires_at=? WHERE job_id=? AND status='queued'`. Return `cursor.rowcount == 1`.
- `finalize_submit(*, job_id, remote_job_id, log_path, resolved_parameters)` — `UPDATE jobs SET status='submitted', remote_job_id=?, log_path=?, parameters=?, claimed_by=NULL, claimed_at=NULL, claim_expires_at=NULL WHERE job_id=? AND status='submitting'`.
- `fail_promotion(*, job_id, skip_reason)` — set `status='failed'`, clear claim columns, append to `skip_reason`.
- `reclaim_expired_leases(*, now) -> list[str]` — implements the SQL in spec section "Atomic Claim & Recovery", returns reclaimed job_ids.

- [ ] **Step 1: Write failing test for CAS** — two back-to-back `claim_for_submit` calls; exactly one returns True. Verify `claimed_by` matches the winner.

```python
def test_claim_for_submit_cas(jobstore):
    jid = jobstore.enqueue(task_name="t", executor="e",
                           params_template="x: 1", parameters={"x": 1})
    won_a = jobstore.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=20)
    won_b = jobstore.claim_for_submit(job_id=jid, instance_id="B", lease_seconds=20)
    assert won_a is True
    assert won_b is False
    rec = jobstore.get(jid)
    assert rec.status == JobStatus.SUBMITTING
```

- [ ] **Step 2: Write failing test for reclaim** — populate a SUBMITTING row with `claim_expires_at` in the past, `remote_job_id IS NULL`. Call `reclaim_expired_leases(now)`. Assert row is back to QUEUED and `skip_reason` contains "reclaimed".

- [ ] **Step 3: Run tests** — Expected: FAIL.

- [ ] **Step 4: Implement methods** — see spec section "Atomic Claim & Recovery" for the SQL.

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add devrun/db/jobs.py tests/test_jobstore_api.py
git commit -m "feat(db): add CAS claim_for_submit, finalize_submit, fail_promotion, reclaim_expired_leases"
```

### Task 10: JobStore API — cascade-skip & poll queries

**Files:**
- Modify: `devrun/db/jobs.py`
- Modify: `tests/test_jobstore_api.py`

Implement read/write helpers used by heartbeat phases (no heartbeat code yet — just the DB primitives):
- `fetch_ready_queued(limit=100) -> list[JobRecord]` — runs the readiness query from spec phase 2b.
- `cascade_skip_dependents() -> list[str]` — runs the cascade SQL from spec phase 1, transitions matching QUEUED rows to SKIPPED with `skip_reason='parent <id> <status>'`. Returns affected job_ids.
- `fetch_active_jobs() -> list[JobRecord]` — `SELECT … WHERE status IN ('submitted','running','canceling')`.
- `get_parent_parameters(child_job_id) -> dict[str, dict]` — `SELECT p.job_id, p.parameters FROM job_dependencies d JOIN jobs p ON p.job_id = d.parent_job_id WHERE d.child_job_id = ?`. Returns `{parent_id: parsed_parameters_dict}`.

- [ ] **Step 1: Write failing tests** — populate a DB with a parent → child edge; mark parent FAILED with `allow_failure=0`; call `cascade_skip_dependents()`; assert child transitions to SKIPPED with `skip_reason` mentioning the parent.

```python
def test_cascade_skip_blocking_failure(jobstore):
    p = jobstore.enqueue(task_name="t", executor="e",
                         params_template="", parameters={})
    c = jobstore.enqueue(task_name="t", executor="e",
                         params_template="", parameters={})
    jobstore.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=False)
    jobstore.update_status(p, JobStatus.FAILED)
    skipped = jobstore.cascade_skip_dependents()
    assert c in skipped
    assert jobstore.get(c).status == JobStatus.SKIPPED

def test_cascade_skip_respects_allow_failure(jobstore):
    p = jobstore.enqueue(task_name="t", executor="e",
                         params_template="", parameters={})
    c = jobstore.enqueue(task_name="t", executor="e",
                         params_template="", parameters={})
    jobstore.insert_dependency(child_job_id=c, parent_job_id=p, allow_failure=True)
    jobstore.update_status(p, JobStatus.FAILED)
    jobstore.cascade_skip_dependents()
    assert jobstore.get(c).status == JobStatus.QUEUED
```

- [ ] **Step 2: Run tests** — Expected: FAIL.

- [ ] **Step 3: Implement methods** using the SQL from spec phases 1 and 2b.

- [ ] **Step 4: Write `fetch_ready_queued` test** — two parents both COMPLETED → child appears; one parent still RUNNING → child does not appear.

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add devrun/db/jobs.py tests/test_jobstore_api.py
git commit -m "feat(db): add cascade-skip and ready-queued queries"
```

### Task 11: JobStore API — workflow deadline & cancel request

**Files:**
- Modify: `devrun/db/jobs.py`
- Modify: `tests/test_jobstore_api.py`

Per spec sections "Workflow Timeout" and "`devrun cancel <job_id>`":
- `fetch_expired_workflows(*, now) -> list[str]` — `SELECT workflow_id FROM workflows WHERE status NOT IN ('completed','failed','cancelled','timed_out') AND deadline_at IS NOT NULL AND deadline_at < ?`.
- `expire_workflow(workflow_id)` — for all jobs in this workflow's `workflow_jobs.job_id` (excluding NULLs): QUEUED → SKIPPED (`skip_reason='workflow deadline'`), SUBMITTED/RUNNING → CANCELING. Set `workflows.status='timed_out'`.
- `request_cancel(job_id) -> JobStatus` — applies the state machine: QUEUED → CANCELLED, SUBMITTED/RUNNING → CANCELING, terminal raises `ValueError`. Returns the new status.

Add `'timed_out'` to `JobStatus` enum (forgot in Task 1; add it here as a small back-fill).

- [ ] **Step 1: Back-fill `TIMED_OUT` status** — add to `JobStatus`.

- [ ] **Step 2: Write failing test for `expire_workflow`** — enqueue a workflow with two stages (one QUEUED, one SUBMITTED); set `workflows.deadline_at` in the past; call `expire_workflow`. Assert QUEUED→SKIPPED, SUBMITTED→CANCELING, workflow→TIMED_OUT.

- [ ] **Step 3: Write failing test for `request_cancel`** — three cases (QUEUED → CANCELLED, SUBMITTED → CANCELING, COMPLETED → raises).

- [ ] **Step 4: Run tests** — Expected: FAIL.

- [ ] **Step 5: Implement** — straight UPDATE statements per spec.

- [ ] **Step 6: Run tests** — Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add devrun/db/jobs.py devrun/models.py tests/test_jobstore_api.py
git commit -m "feat(db): add workflow deadline expiry and request_cancel state machine"
```

### Task 12: Full-suite smoke test + AGENT.md note

**Files:**
- Modify: `AGENT.md` (Database Schema section + a short PR1 changelog line).

- [ ] **Step 1: Run full test suite** — `pytest tests/`. Expected: all previously-passing tests still pass; new tests pass.

- [ ] **Step 2: Update `AGENT.md` Database Schema section** to list the new tables/columns (do not document heartbeat behavior yet — that's PR2).

```markdown
Table: `jobs`
Columns: `job_id`, `task_name`, `executor`, `parameters`, `params_template`, `remote_job_id`,
         `status`, `created_at`, `completed_at`, `log_path`,
         `skip_reason`, `claimed_by`, `claimed_at`, `claim_expires_at`

Table: `workflows`
Columns: `workflow_id`, `workflow_name`, `stages_state`, `status`, `created_at`,
         `completed_at`, `deadline_at`

Table: `job_dependencies`
Columns: `child_job_id`, `parent_job_id`, `allow_failure`

Table: `workflow_jobs`
Columns: `workflow_id`, `stage_name`, `ordinal`, `job_id`, `source_job_id`
```

Append to "Issue to note" / development specifics:

> The `JobStore` exposes a typed API surface (`enqueue`, `enqueue_workflow`, `claim_for_submit`, `finalize_submit`, `fail_promotion`, `reclaim_expired_leases`, `cascade_skip_dependents`, `fetch_ready_queued`, `fetch_active_jobs`, `fetch_expired_workflows`, `expire_workflow`, `request_cancel`, `list_dependencies`, `get_parent_parameters`, `get_workflow_stages`). PR1 lands the schema and APIs only; no producer/consumer code uses them yet.

- [ ] **Step 3: Commit**

```bash
git add AGENT.md
git commit -m "docs(agent): document new schema tables and JobStore API surface (PR1)"
```

---

## Self-review checklist (PR1)

- [ ] All seven new `JobStatus` values present (`QUEUED`, `SUBMITTING`, `CANCELING`, `SKIPPED`, `TIMED_OUT` added; legacy `PENDING` retained).
- [ ] `jobs` columns added: `params_template`, `skip_reason`, `claimed_by`, `claimed_at`, `claim_expires_at`.
- [ ] `workflows` column added: `deadline_at`.
- [ ] Tables created: `job_dependencies`, `workflow_jobs`.
- [ ] OmegaConf resolver `${jobs:...}` registered with `JobRefContext` ContextVar scoping; unauthorized read raises clearly.
- [ ] `load_merged_omegaconf` returns unresolved `DictConfig`; round-trip preserves `${jobs:...}` and `${stages:...}` literals.
- [ ] `JobStore` API matches spec's "JobStore API Surface" exactly (method names + signatures).
- [ ] CAS test: two concurrent `claim_for_submit` calls — exactly one wins.
- [ ] Cascade-skip test: blocking parent failure → child SKIPPED; `allow_failure=1` → child remains QUEUED.
- [ ] All existing tests still pass (`pytest tests/`).
- [ ] AGENT.md schema section updated.
- [ ] No producer/CLI changes in this PR.
