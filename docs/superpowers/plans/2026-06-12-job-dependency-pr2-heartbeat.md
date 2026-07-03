# PR2 — Heartbeat Scheduler + Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-06-12-job-dependency-mechanism-design.md` (sections "The heartbeat scheduler", "Atomic Claim & Recovery", "Heartbeat CLI surface", and "Workflow Timeout").

**Goal:** Add the global heartbeat scheduler as a new consumer that drains `QUEUED` jobs through `SUBMITTING → SUBMITTED → RUNNING → terminal`, plus the cross-platform service-management CLI (`devrun heartbeat …`). Producers still use the old paths in PR2; PR3 flips them.

**Architecture:** A single `devrun/heartbeat.py` module owns the loop. Per-tick phases: claim-sweep → workflow-deadline-expiry → cascade-skip → promote-ready-queued → poll-active. Service management lives behind a `HeartbeatService` abstraction with `linux.py` (systemd `--user`) and `darwin.py` (launchd `~/Library/LaunchAgents/`) backends. Heartbeat itself is platform-agnostic and only depends on `JobStore` APIs landed in PR1.

**Tech Stack:** Pure Python loop, `signal` for SIGTERM/SIGINT, `subprocess` for `systemctl`/`launchctl`, Jinja2 templates for unit files.

**Exit criteria:** `pytest tests/test_heartbeat_loop.py tests/test_heartbeat_cascade.py tests/test_heartbeat_promotion.py tests/test_heartbeat_claim.py tests/test_heartbeat_cancel.py tests/test_heartbeat_timeout.py tests/test_heartbeat_service.py -v` passes. `devrun heartbeat --help` shows the new subcommand group. Manually inserting a `QUEUED` row via `JobStore.enqueue` and running `devrun heartbeat` (foreground) drives it to `SUBMITTED` against a `LocalExecutor`. Full suite still green.

---

## File Structure

| File | Responsibility |
|---|---|
| `devrun/heartbeat.py` | `run_loop`, `tick`, the five phase functions, signal handling, lease/instance ID helpers |
| `devrun/services/__init__.py` | `get_service()` dispatch on `sys.platform`; `HeartbeatService` Protocol |
| `devrun/services/linux.py` | systemd `--user` backend: install/uninstall/start/stop/restart/is_active |
| `devrun/services/darwin.py` | launchd backend (LaunchAgent plist) |
| `devrun/templates/devrun-heartbeat.service.j2` | systemd unit template |
| `devrun/templates/com.devrun.heartbeat.plist.j2` | launchd plist template |
| `devrun/cli_heartbeat.py` | Typer subapp wired in as `app.add_typer(heartbeat_app, name="heartbeat")` |
| `tests/test_heartbeat_loop.py` | Tick phase isolation tests |
| `tests/test_heartbeat_cascade.py` | Cascade-skip across ticks |
| `tests/test_heartbeat_promotion.py` | Promotion happy path + REQUIRED/jobref failure paths |
| `tests/test_heartbeat_claim.py` | CAS under simulated concurrency + lease reclaim |
| `tests/test_heartbeat_cancel.py` | `CANCELING → CANCELLED` after `executor.cancel()` |
| `tests/test_heartbeat_timeout.py` | Workflow `deadline_at` expiry |
| `tests/test_heartbeat_service.py` | Platform dispatch w/ mocked `subprocess` |

---

## Task 1: Module skeleton + instance ID + tick stub

**Files:**
- Create: `devrun/heartbeat.py`
- Create: `tests/test_heartbeat_loop.py`

- [ ] **Step 1: Write failing test for `tick()` no-op on empty DB**

```python
from devrun.heartbeat import tick
from devrun.db.jobs import JobStore

def test_tick_empty_db_is_noop(tmp_path):
    db = JobStore(tmp_path / "jobs.db")
    tick(db, executor_router=None)  # no rows → no actions
```

- [ ] **Step 2: Run test** — Expected: FAIL (`tick` does not exist).

- [ ] **Step 3: Implement module skeleton**

```python
"""Global heartbeat scheduler. See spec for phase ordering."""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from devrun.db.jobs import JobStore

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def instance_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def tick(db: JobStore, executor_router) -> None:
    """Run one heartbeat tick. Each phase is a separate function for isolation testing."""
    _reclaim_stale_leases(db)
    _expire_workflow_deadlines(db)
    _cascade_skip_failed(db)
    _promote_ready_queued(db, executor_router)
    _poll_active_jobs(db, executor_router)


def _reclaim_stale_leases(db: JobStore) -> None:
    reclaimed = db.reclaim_expired_leases(now=_now())
    if reclaimed:
        logger.warning("Reclaimed %d stale leases: %s", len(reclaimed), reclaimed)


def _expire_workflow_deadlines(db: JobStore) -> None:
    for wf_id in db.fetch_expired_workflows(now=_now()):
        logger.warning("Workflow %s exceeded deadline; expiring", wf_id)
        db.expire_workflow(wf_id)


def _cascade_skip_failed(db: JobStore) -> None:
    skipped = db.cascade_skip_dependents()
    if skipped:
        logger.info("Cascade-skipped %d dependents: %s", len(skipped), skipped)


def _promote_ready_queued(db: JobStore, executor_router) -> None:
    pass  # implemented in Task 3


def _poll_active_jobs(db: JobStore, executor_router) -> None:
    pass  # implemented in Task 5


def run_loop(db_path: Path, interval: float = 10.0,
             tick_file: Path | None = None) -> None:
    """Foreground loop. Catches signals; exits cleanly after current tick."""
    from devrun.router import ExecutorRouter  # local import to avoid cycles
    db = JobStore(db_path)
    router = ExecutorRouter()
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_event.set())
    signal.signal(signal.SIGINT, lambda *_: _shutdown_event.set())
    logger.info("Heartbeat starting (interval=%ss, instance=%s)", interval, instance_id())
    while not _shutdown_event.is_set():
        try:
            tick(db, router)
        except Exception:
            logger.exception("Heartbeat tick failed; continuing")
        if tick_file is not None:
            tick_file.write_text(_now().isoformat())
        _shutdown_event.wait(interval)
    logger.info("Heartbeat shut down cleanly")
```

- [ ] **Step 4: Run test** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add devrun/heartbeat.py tests/test_heartbeat_loop.py
git commit -m "feat(heartbeat): add module skeleton with tick phase stubs"
```

## Task 2: Cascade-skip phase test

**Files:**
- Modify: `tests/test_heartbeat_cascade.py` (create).

The phase function already delegates to `JobStore.cascade_skip_dependents()` (tested in PR1). Add an integration test that runs `tick()` and observes the side effect across ticks.

- [ ] **Step 1: Write test** — populate parent `FAILED`, child `QUEUED` with `allow_failure=0`. Call `tick(db, None)`. Assert child is `SKIPPED`.

- [ ] **Step 2: Run test** — Expected: PASS (cascade logic already lives in JobStore from PR1).

- [ ] **Step 3: Add multi-hop test** — `A FAILED` → child `B QUEUED` → grandchild `C QUEUED`. After two ticks, all three are terminal (`A FAILED`, `B SKIPPED`, `C SKIPPED`).

```python
def test_cascade_multihop(jobstore):
    a = jobstore.enqueue(task_name="t", executor="local", params_template="", parameters={})
    b = jobstore.enqueue(task_name="t", executor="local", params_template="", parameters={})
    c = jobstore.enqueue(task_name="t", executor="local", params_template="", parameters={})
    jobstore.insert_dependency(child_job_id=b, parent_job_id=a, allow_failure=False)
    jobstore.insert_dependency(child_job_id=c, parent_job_id=b, allow_failure=False)
    jobstore.update_status(a, JobStatus.FAILED)
    for _ in range(2):
        tick(jobstore, executor_router=None)
    assert jobstore.get(b).status == JobStatus.SKIPPED
    assert jobstore.get(c).status == JobStatus.SKIPPED
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_heartbeat_cascade.py
git commit -m "test(heartbeat): cover cascade-skip across ticks"
```

## Task 3: Promotion phase — happy path

**Files:**
- Modify: `devrun/heartbeat.py`
- Create: `tests/test_heartbeat_promotion.py`

Implement the full promotion algorithm per spec phase 2:
1. `db.fetch_ready_queued()` → candidates.
2. For each: `db.claim_for_submit(...)` (CAS). Skip if lost.
3. Load `params_template` (YAML string) into OmegaConf.
4. Build `JobRefContext(allowed_parents=db.get_parent_parameters(job_id), calling_job_id=job_id)` and install via the ContextVar `set` / `reset` pattern.
5. Resolve via `OmegaConf.to_container(cfg, resolve=True)`. Catch resolver errors → `db.fail_promotion(skip_reason=...)`.
6. Scan resolved dict for any leftover `<REQUIRED:...>` string — fail if found.
7. `task = get_task_class(task_name)(); spec = task.prepare(resolved_params)`.
8. `record = executor.submit_with_retry(spec)` → grab `remote_job_id`, `log_path`.
9. `db.finalize_submit(job_id, remote_job_id, log_path, resolved_parameters=resolved_params)`.

Any exception in steps 3–8 → `db.fail_promotion(skip_reason=<exception text>)`.

- [ ] **Step 1: Write failing happy-path test**

```python
def test_promotion_happy_path(jobstore, mock_executor_router):
    jid = jobstore.enqueue(
        task_name="eval",
        executor="local",
        params_template="model: gpt-4\nseed: 1\n",
        parameters={"model": "gpt-4", "seed": 1},
    )
    mock_executor_router.executors["local"].next_submit_result = ("remote-123", "/tmp/log")
    tick(jobstore, mock_executor_router)
    rec = jobstore.get(jid)
    assert rec.status == JobStatus.SUBMITTED
    assert rec.remote_job_id == "remote-123"
    assert rec.log_path == "/tmp/log"
```

- [ ] **Step 2: Run test** — Expected: FAIL.

- [ ] **Step 3: Implement `_promote_ready_queued`**

```python
from devrun.jobref import JobRefContext, JOBREF_CONTEXT
from omegaconf import OmegaConf
from devrun.registry import get_task_class

_REQUIRED_RE = __import__("re").compile(r"^<REQUIRED(?::\s*.*?)?>$")

def _has_required_placeholder(obj) -> bool:
    if isinstance(obj, str):
        return bool(_REQUIRED_RE.match(obj))
    if isinstance(obj, dict):
        return any(_has_required_placeholder(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_required_placeholder(v) for v in obj)
    return False


def _promote_ready_queued(db, executor_router) -> None:
    inst = instance_id()
    lease = 20.0
    for cand in db.fetch_ready_queued(limit=100):
        if not db.claim_for_submit(job_id=cand.job_id, instance_id=inst,
                                   lease_seconds=lease):
            continue
        try:
            cfg = OmegaConf.create(cand.params_template or "{}")
            ctx = JobRefContext(
                allowed_parents=db.get_parent_parameters(cand.job_id),
                calling_job_id=cand.job_id,
            )
            token = JOBREF_CONTEXT.set(ctx)
            try:
                resolved = OmegaConf.to_container(cfg, resolve=True)
            finally:
                JOBREF_CONTEXT.reset(token)
            if _has_required_placeholder(resolved):
                db.fail_promotion(job_id=cand.job_id,
                                  skip_reason="unfilled <REQUIRED:...> placeholder")
                continue
            task = get_task_class(cand.task_name)()
            spec = task.prepare(resolved)
            executor = executor_router.get(cand.executor)
            submit_result = executor.submit_with_retry(spec)
            db.finalize_submit(
                job_id=cand.job_id,
                remote_job_id=submit_result.remote_job_id,
                log_path=spec.metadata.get("log_path"),
                resolved_parameters=resolved,
            )
            logger.info("Promoted job %s → SUBMITTED (remote=%s)",
                        cand.job_id, submit_result.remote_job_id)
        except Exception as exc:
            logger.exception("Promotion failed for %s", cand.job_id)
            db.fail_promotion(job_id=cand.job_id, skip_reason=str(exc))
```

(`SubmitResult` shape: existing executors return a `RemoteJob`-ish object; adapt to whatever the current `submit_with_retry` returns. Inspect `devrun/runner.py` first and match its access pattern.)

- [ ] **Step 4: Run test** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add devrun/heartbeat.py tests/test_heartbeat_promotion.py
git commit -m "feat(heartbeat): implement promotion phase with JobRefContext scoping"
```

## Task 4: Promotion — failure paths

**Files:**
- Modify: `tests/test_heartbeat_promotion.py`

- [ ] **Step 1: Add test — unfilled `<REQUIRED:...>`**

```python
def test_promotion_required_placeholder_fails(jobstore, mock_executor_router):
    jid = jobstore.enqueue(task_name="eval", executor="local",
                           params_template="model: <REQUIRED: model name>\n",
                           parameters={"model": "<REQUIRED: model name>"})
    tick(jobstore, mock_executor_router)
    rec = jobstore.get(jid)
    assert rec.status == JobStatus.FAILED
    assert "REQUIRED" in rec.skip_reason
```

- [ ] **Step 2: Add test — `${jobs:missing,x}` (no edge)**

```python
def test_promotion_unauthorized_jobref_fails(jobstore, mock_executor_router):
    jid = jobstore.enqueue(task_name="eval", executor="local",
                           params_template="model: ${jobs:nope,model}\n",
                           parameters={})
    tick(jobstore, mock_executor_router)
    rec = jobstore.get(jid)
    assert rec.status == JobStatus.FAILED
```

- [ ] **Step 3: Add test — executor.submit raises**

`mock_executor_router.executors["local"].submit_raises = RuntimeError("boom")` → assert FAILED with skip_reason containing "boom".

- [ ] **Step 4: Run tests** — Expected: PASS (Task 3 already handles failures generically).

- [ ] **Step 5: Commit**

```bash
git add tests/test_heartbeat_promotion.py
git commit -m "test(heartbeat): cover promotion failure paths (REQUIRED, jobref, executor)"
```

## Task 5: Poll-active phase + status mapping

**Files:**
- Modify: `devrun/heartbeat.py`
- Modify: `devrun/runner.py` — extract `_map_status` to module level so heartbeat can reuse without instantiating `TaskRunner`.

- [ ] **Step 1: Extract `_map_status`** — see `devrun/runner.py` for the current method. Move it to a module-level `map_executor_status(raw: str) -> JobStatus`. Update existing callers in `TaskRunner`.

- [ ] **Step 2: Run existing runner tests** — `pytest tests/test_runner.py -v`. Expected: PASS (mechanical refactor).

- [ ] **Step 3: Write failing test for poll phase**

```python
def test_poll_active_transitions_to_completed(jobstore, mock_executor_router):
    jid = jobstore.enqueue(task_name="eval", executor="local",
                           params_template="", parameters={})
    jobstore.update_status(jid, JobStatus.RUNNING, remote_job_id="r-1")
    mock_executor_router.executors["local"].status_returns = {"r-1": "completed"}
    tick(jobstore, mock_executor_router)
    assert jobstore.get(jid).status == JobStatus.COMPLETED
```

- [ ] **Step 4: Implement `_poll_active_jobs`**

```python
def _poll_active_jobs(db, executor_router) -> None:
    for rec in db.fetch_active_jobs():
        executor = executor_router.get(rec.executor)
        try:
            raw = executor.status(rec.remote_job_id)
        except Exception as exc:
            logger.warning("status() failed for %s: %s", rec.job_id, exc)
            continue
        mapped = map_executor_status(raw)
        if mapped != rec.status:
            db.update_status(rec.job_id, mapped)
```

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add devrun/heartbeat.py devrun/runner.py tests/test_heartbeat_promotion.py
git commit -m "feat(heartbeat): implement poll-active phase; extract map_executor_status"
```

## Task 6: Cancel phase (CANCELING → CANCELLED)

**Files:**
- Modify: `devrun/heartbeat.py`
- Create: `tests/test_heartbeat_cancel.py`

Cancel logic lives inside `_poll_active_jobs`: rows with `status=CANCELING` get `executor.cancel(remote_id)` called and then transition to `CANCELLED`.

- [ ] **Step 1: Write failing test**

```python
def test_canceling_transitions_to_cancelled(jobstore, mock_executor_router):
    jid = jobstore.enqueue(task_name="eval", executor="local", params_template="", parameters={})
    jobstore.update_status(jid, JobStatus.RUNNING, remote_job_id="r-1")
    jobstore.request_cancel(jid)
    assert jobstore.get(jid).status == JobStatus.CANCELING
    tick(jobstore, mock_executor_router)
    assert mock_executor_router.executors["local"].cancel_calls == ["r-1"]
    assert jobstore.get(jid).status == JobStatus.CANCELLED
```

- [ ] **Step 2: Extend `_poll_active_jobs`**

```python
def _poll_active_jobs(db, executor_router) -> None:
    for rec in db.fetch_active_jobs():
        executor = executor_router.get(rec.executor)
        if rec.status == JobStatus.CANCELING:
            try:
                executor.cancel(rec.remote_job_id)
            except Exception as exc:
                logger.warning("cancel() failed for %s: %s", rec.job_id, exc)
            db.update_status(rec.job_id, JobStatus.CANCELLED)
            continue
        try:
            raw = executor.status(rec.remote_job_id)
        except Exception as exc:
            logger.warning("status() failed for %s: %s", rec.job_id, exc)
            continue
        mapped = map_executor_status(raw)
        if mapped != rec.status:
            db.update_status(rec.job_id, mapped)
```

- [ ] **Step 3: Run tests** — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add devrun/heartbeat.py tests/test_heartbeat_cancel.py
git commit -m "feat(heartbeat): handle CANCELING → CANCELLED via executor.cancel"
```

## Task 7: Claim sweep test (lease expiry + orphan annotation)

**Files:**
- Create: `tests/test_heartbeat_claim.py`

These exercise the JobStore CAS primitives (PR1) through the heartbeat phase wrapper.

- [ ] **Step 1: Test — expired SUBMITTING with NULL `remote_job_id` is reclaimed**

```python
def test_expired_lease_reclaim_to_queued(jobstore):
    jid = jobstore.enqueue(task_name="t", executor="local", params_template="", parameters={})
    jobstore.claim_for_submit(job_id=jid, instance_id="A", lease_seconds=0.001)
    time.sleep(0.05)
    tick(jobstore, executor_router=None)  # reclaim phase runs first
    rec = jobstore.get(jid)
    assert rec.status == JobStatus.QUEUED
    assert "reclaimed" in (rec.skip_reason or "")
```

- [ ] **Step 2: Test — SUBMITTING with non-NULL `remote_job_id` is NOT reclaimed** (already submitted; just an annotation race).

- [ ] **Step 3: Run tests** — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_heartbeat_claim.py
git commit -m "test(heartbeat): cover stale-lease reclaim and orphan protection"
```

## Task 8: Workflow deadline expiry test

**Files:**
- Create: `tests/test_heartbeat_timeout.py`

- [ ] **Step 1: Test — past deadline expires non-terminal stages**

Populate a workflow with two stage jobs (one `QUEUED`, one `RUNNING`), set `workflows.deadline_at` to past. Call `tick`. Assert QUEUED→SKIPPED, RUNNING→CANCELING, workflow→TIMED_OUT.

- [ ] **Step 2: Run test** — Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_heartbeat_timeout.py
git commit -m "test(heartbeat): cover workflow deadline expiry"
```

## Task 9: `HeartbeatService` abstraction + dispatch

**Files:**
- Create: `devrun/services/__init__.py`
- Create: `devrun/services/linux.py`
- Create: `devrun/services/darwin.py`

- [ ] **Step 1: Define Protocol**

```python
# devrun/services/__init__.py
from __future__ import annotations
import sys
from typing import Protocol

class HeartbeatService(Protocol):
    def install(self, *, python_path: str, db_path: str) -> None: ...
    def uninstall(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def is_active(self) -> bool: ...


def get_service() -> HeartbeatService:
    if sys.platform == "darwin":
        from devrun.services.darwin import LaunchdService
        return LaunchdService()
    if sys.platform.startswith("linux"):
        from devrun.services.linux import SystemdUserService
        return SystemdUserService()
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
```

- [ ] **Step 2: Implement `SystemdUserService`** — shell out to `systemctl --user`:

```python
import subprocess
from pathlib import Path
from devrun.utils.templates import render_template

class SystemdUserService:
    UNIT_NAME = "devrun-heartbeat.service"
    UNIT_PATH = Path.home() / ".config/systemd/user" / UNIT_NAME

    def install(self, *, python_path, db_path):
        body = render_template("devrun-heartbeat.service.j2",
                               python_path=python_path, db_path=db_path)
        self.UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.UNIT_PATH.write_text(body)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", self.UNIT_NAME], check=True)

    def uninstall(self):
        subprocess.run(["systemctl", "--user", "disable", self.UNIT_NAME], check=False)
        self.UNIT_PATH.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

    def start(self):   subprocess.run(["systemctl","--user","start",self.UNIT_NAME], check=True)
    def stop(self):    subprocess.run(["systemctl","--user","stop",self.UNIT_NAME], check=True)
    def restart(self): subprocess.run(["systemctl","--user","restart",self.UNIT_NAME], check=True)
    def is_active(self) -> bool:
        r = subprocess.run(["systemctl","--user","is-active","--quiet",self.UNIT_NAME])
        return r.returncode == 0
```

- [ ] **Step 3: Implement `LaunchdService`** — analogous; plist at `~/Library/LaunchAgents/com.devrun.heartbeat.plist`; commands `launchctl load`/`unload`/`kickstart -k`/`print`.

- [ ] **Step 4: Templates**

```
# devrun/templates/devrun-heartbeat.service.j2
[Unit]
Description=devrun heartbeat scheduler

[Service]
Type=simple
ExecStart={{ python_path }} -m devrun.heartbeat --db {{ db_path }}
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=60
WorkingDirectory=%h

[Install]
WantedBy=default.target
```

```xml
<!-- devrun/templates/com.devrun.heartbeat.plist.j2 -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.devrun.heartbeat</string>
  <key>ProgramArguments</key>
  <array>
    <string>{{ python_path }}</string>
    <string>-m</string><string>devrun.heartbeat</string>
    <string>--db</string><string>{{ db_path }}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

- [ ] **Step 5: Commit**

```bash
git add devrun/services/ devrun/templates/devrun-heartbeat.service.j2 devrun/templates/com.devrun.heartbeat.plist.j2
git commit -m "feat(services): add cross-platform HeartbeatService (systemd/launchd)"
```

## Task 10: Service tests — mocked subprocess

**Files:**
- Create: `tests/test_heartbeat_service.py`

- [ ] **Step 1: Test platform dispatch**

```python
import sys, types
from unittest.mock import patch
from devrun.services import get_service

def test_get_service_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    from devrun.services.linux import SystemdUserService
    assert isinstance(get_service(), SystemdUserService)

def test_get_service_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    from devrun.services.darwin import LaunchdService
    assert isinstance(get_service(), LaunchdService)
```

- [ ] **Step 2: Test Linux install writes unit and calls systemctl**

```python
def test_systemd_install_writes_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []
    with patch("subprocess.run", side_effect=lambda *a, **kw: calls.append(a[0]) or types.SimpleNamespace(returncode=0)):
        from devrun.services.linux import SystemdUserService
        svc = SystemdUserService()
        svc.install(python_path="/usr/bin/python3", db_path="/x/jobs.db")
    unit = tmp_path / ".config/systemd/user/devrun-heartbeat.service"
    assert unit.exists()
    assert "ExecStart=/usr/bin/python3" in unit.read_text()
    assert ["systemctl","--user","daemon-reload"] in calls
```

- [ ] **Step 3: Mirror for `is_active`, `start`, `stop`, `uninstall`** (both backends).

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_heartbeat_service.py
git commit -m "test(services): cover platform dispatch + install/start/stop with mocked subprocess"
```

## Task 11: `devrun heartbeat` CLI subapp

**Files:**
- Create: `devrun/cli_heartbeat.py`
- Modify: `devrun/cli.py` — `app.add_typer(heartbeat_app, name="heartbeat")`.

Subcommands per spec section "Heartbeat CLI surface":
- `devrun heartbeat` (no subcommand): foreground loop. Refuses to start if `get_service().is_active()` is True.
- `devrun heartbeat start|stop|restart|install|uninstall`: delegate to service backend. `install` resolves `sys.executable` for `python_path` and the default DB path for `db_path`.
- `devrun heartbeat status`: print `is_active()`, plus DB counts by status (`SELECT status, COUNT(*) FROM jobs GROUP BY status`), plus the contents of `~/.devrun/heartbeat.tick` if it exists.

- [ ] **Step 1: Write the subapp**

```python
import sys
from pathlib import Path
import typer
from devrun.services import get_service
from devrun.heartbeat import run_loop
from devrun.db.jobs import JobStore, default_db_path

heartbeat_app = typer.Typer(no_args_is_help=False,
                            help="Heartbeat scheduler control")

@heartbeat_app.callback(invoke_without_command=True)
def foreground(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    svc = get_service()
    if svc.is_active():
        typer.echo("Heartbeat daemon is already running; refusing to start foreground.", err=True)
        raise typer.Exit(1)
    tick_file = Path.home() / ".devrun" / "heartbeat.tick"
    tick_file.parent.mkdir(parents=True, exist_ok=True)
    run_loop(default_db_path(), interval=10.0, tick_file=tick_file)

@heartbeat_app.command()
def install():
    get_service().install(python_path=sys.executable, db_path=str(default_db_path()))
    typer.echo("Installed heartbeat service.")

@heartbeat_app.command()
def uninstall(): get_service().uninstall(); typer.echo("Uninstalled.")
@heartbeat_app.command()
def start():     get_service().start();     typer.echo("Started.")
@heartbeat_app.command()
def stop():      get_service().stop();      typer.echo("Stopped.")
@heartbeat_app.command()
def restart():   get_service().restart();   typer.echo("Restarted.")

@heartbeat_app.command()
def status():
    svc = get_service()
    active = svc.is_active()
    db = JobStore(default_db_path())
    counts = db.status_counts()  # add this small helper to JobStore
    typer.echo(f"Service: {'active' if active else 'inactive'}")
    for s, n in counts.items():
        typer.echo(f"  {s}: {n}")
    tick = Path.home() / ".devrun" / "heartbeat.tick"
    if tick.exists():
        typer.echo(f"Last tick: {tick.read_text().strip()}")
```

- [ ] **Step 2: Add `JobStore.status_counts()` helper** — `SELECT status, COUNT(*) FROM jobs GROUP BY status` returning `dict[str, int]`. Add to `devrun/db/jobs.py`. Add a small test.

- [ ] **Step 3: Wire into `devrun/cli.py`**

```python
from devrun.cli_heartbeat import heartbeat_app
app.add_typer(heartbeat_app, name="heartbeat")
```

- [ ] **Step 4: Add `__main__` entry to `devrun/heartbeat.py`**

```python
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--interval", type=float, default=10.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    tick_file = Path.home() / ".devrun" / "heartbeat.tick"
    tick_file.parent.mkdir(parents=True, exist_ok=True)
    run_loop(args.db, interval=args.interval, tick_file=tick_file)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: CLI test** — `pytest tests/test_cli.py` plus a new test that `devrun heartbeat status` runs and exits 0 with an empty DB (mock `get_service().is_active() = False`).

- [ ] **Step 6: Commit**

```bash
git add devrun/cli_heartbeat.py devrun/cli.py devrun/heartbeat.py devrun/db/jobs.py tests/
git commit -m "feat(cli): add devrun heartbeat subcommand group"
```

## Task 12: AGENT.md update + full-suite smoke

**Files:**
- Modify: `AGENT.md`

- [ ] **Step 1: Run full suite** — `pytest tests/`. Expected: all green.

- [ ] **Step 2: Smoke test in foreground** — create a temp DB, `JobStore.enqueue(task_name="eval", executor="local", params_template="x: 1", parameters={"x": 1})`, run `python -m devrun.heartbeat --db /tmp/test_jobs.db --interval 1` for ~3s in another shell, assert the row transitions to SUBMITTED. Document this in the PR description (not a unit test).

- [ ] **Step 3: Update `AGENT.md`** — add a new "Heartbeat" subsection under "High-Level Modules":

```markdown
* **`devrun/heartbeat.py`:** Global scheduler. Loops `tick()` at a configurable interval. Each tick runs five phases in order: stale-lease reclaim, workflow-deadline expiry, cascade-skip of dependents, promotion of ready `QUEUED` jobs (claim CAS → resolve params with `JobRefContext` → executor submit → `finalize_submit`), and poll of active jobs. PR2 lands the scheduler; producers still use legacy paths until PR3.
* **`devrun/services/`:** Cross-platform service management. `get_service()` dispatches on `sys.platform` to `SystemdUserService` (Linux, `systemctl --user`) or `LaunchdService` (macOS, `launchctl` + LaunchAgent plist). Both implement `install/uninstall/start/stop/restart/is_active`.
* **`devrun/cli_heartbeat.py`:** Typer subapp wired as `devrun heartbeat`. Subcommands: foreground (default), `install`, `uninstall`, `start`, `stop`, `restart`, `status`.
```

- [ ] **Step 4: Commit**

```bash
git add AGENT.md
git commit -m "docs(agent): document heartbeat scheduler and service backends (PR2)"
```

---

## Self-review checklist (PR2)

- [ ] `tick()` runs five phases in the correct order (reclaim → expire → cascade → promote → poll).
- [ ] Promotion uses `JobRefContext` ContextVar (set/reset around resolve).
- [ ] Promotion failure paths exercised: `<REQUIRED:...>`, unauthorized `${jobs:...}`, executor exception.
- [ ] Cancel path: `CANCELING → CANCELLED` only after `executor.cancel()` (or after logging its failure).
- [ ] Workflow deadline test: QUEUED→SKIPPED, RUNNING→CANCELING, workflow→TIMED_OUT.
- [ ] Stale-lease reclaim test: SUBMITTING with NULL remote_job_id and past lease → QUEUED with `skip_reason` annotation.
- [ ] Service abstraction has no `subprocess` calls outside the backend modules.
- [ ] CLI tests cover dispatch and `status` rendering.
- [ ] Full existing test suite still green.
- [ ] AGENT.md modules section reflects new files.
- [ ] No producer/CLI behavior changes (PR3 territory).
