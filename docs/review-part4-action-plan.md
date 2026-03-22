# Devrun Code Review — Part 4: Consolidated Action Plan
> Priority-ordered implementation guide. Written 2026-03-22.
> Designed to be unambiguous for independent implementation.
>
> **STATUS: FULLY IMPLEMENTED** — All 24 items completed 2026-03-22.
> Commit: `0940ad8` on branch `fix/code-review-improvements`.
> See `CHANGELOG.md` for a prose summary of every fix.

---

## Priority 1 — SSH Executor (Blocks all remote SSH jobs)

### ✅ P1-A: Fix SSH log file naming (Bug 1 from Part 1)

**File:** `devrun/executors/ssh.py`

**Problem:** The log file is named using `$$` which expands to the local SSH client PID, not the remote process PID. `logs()` then tries to read a file named after the remote PID, which never matches.

**Implementation:**
1. Before building `remote_cmd`, generate a stable token: `import uuid; run_token = uuid.uuid4().hex[:12]`
2. Use `run_token` in the log path: `remote_log = f"/tmp/devrun_ssh_{run_token}.log"`
3. Pass `run_token` back as part of the job identifier. Change the return value from `remote_pid` to `f"{remote_pid}:{run_token}"` (colon-separated).
4. In `status()`, parse `job_id.split(":")[0]` to get the PID for `kill -0`.
5. In `logs()`, parse `job_id.split(":")[-1]` to get the run_token for `cat`.
6. In `cancel()`, parse `job_id.split(":")[0]` to get the PID for `kill`.

**Acceptance test:** After fix, `devrun logs <ssh_job_id>` must return the actual output of the remote command.

---

### ✅ P1-B: Shell-quote all interpolated values in SSH submit (Bugs 2, 3, 4)

**File:** `devrun/executors/ssh.py`, method `submit()`

**Problem:** Single-quote wrapping of `full_cmd` breaks on commands with single quotes. Env values and working_dir are unquoted.

**Implementation:**
```python
import shlex

# Fix env prefix (Bug 3)
env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in task_spec.env.items())

# Fix working_dir (Bug 4)
if task_spec.working_dir:
    full_cmd = f"cd {shlex.quote(task_spec.working_dir)} && {full_cmd}"

# Fix command quoting (Bug 2) — use printf %q or heredoc approach
# Replace single-quote wrapping with double-escaped heredoc:
remote_cmd = (
    f"nohup bash << 'DEVRUN_EOF' > {remote_log} 2>&1 & echo $!\n"
    f"{full_cmd}\n"
    f"DEVRUN_EOF"
)
```

Note: The heredoc approach (`<< 'DEVRUN_EOF'`) avoids all quoting issues because the command is passed verbatim to the remote bash. Verify that `run_ssh_command` passes the heredoc string correctly through `subprocess.run` (it does, since the whole thing is a single string passed to `ssh <target> <command>`).

---

### ✅ P1-C: Reduce SSH timeout for status/logs queries (Bug 7)

**File:** `devrun/utils/ssh.py`, `run_ssh_command()`

**Problem:** Default timeout is 300 seconds. A hanging status check blocks the CLI for 5 minutes.

**Implementation:** Add a separate `timeout` parameter at call sites:
- `SSHExecutor.status()` → call `run_ssh_command(..., timeout=30)`
- `SSHExecutor.logs()` → call `run_ssh_command(..., timeout=60)`
- `SSHExecutor.submit()` → keep `timeout=300` (long-running setup commands are OK here)

---

## Priority 2 — Slurm Executor (Blocks remote Slurm jobs)

### ✅ P2-A: Store and retrieve absolute log paths (Bug 8, 14)

**File:** `devrun/executors/slurm.py`, method `submit()`

**Problem:** `logs()` uses a relative `cat devrun_{job_id}.out` which fails unless called from the exact submission directory.

**Implementation:**
1. In `submit()`, after generating the sbatch script, compute the absolute output path:
   ```python
   log_dir = Path(task_spec.working_dir) if task_spec.working_dir else Path.cwd()
   log_path = str(log_dir / f"devrun_{slurm_job_id}.out")
   ```
2. Return from `submit()` as before (just the `slurm_job_id` string).
3. Store the log path via the runner: the runner must call `self._db.update_status(job_id, JobStatus.RUNNING, remote_job_id=slurm_job_id, log_path=log_path)`.
   - **This requires `submit()` to return both the job ID and the log path**, OR the runner passes the log path separately.
   - Simplest approach: change `submit()` return type to a dataclass or tuple, OR add a `log_path` attribute on `TaskSpec.metadata` that the executor reads and writes.
   - Recommended: Add `metadata["log_path"]` convention — `SlurmExecutor.submit()` sets `task_spec.metadata["log_path"]` before returning, and `TaskRunner._submit_single()` reads it and stores it:
     ```python
     # In runner.py._submit_single(), after executor.submit():
     log_path = task_spec.metadata.get("log_path")
     self._db.update_status(job_id, JobStatus.RUNNING, remote_job_id=remote_job_id, log_path=log_path)
     ```
4. In `SlurmExecutor.logs()`, retrieve log path from the DB:
   ```python
   # logs() receives job_id (the devrun local ID), not the slurm ID
   # The executor doesn't have DB access — so logs() must use slurm_job_id
   # The runner passes remote_job_id (slurm ID) as job_id to logs()
   # So: reconstruct path from slurm job_id and a known base dir, OR
   # store it in metadata and have the runner pass it somehow
   ```
   **Simpler alternative (recommended):** Change `BaseExecutor.logs(job_id)` signature to `logs(job_id, log_path=None)`. The runner calls `executor.logs(remote_id, log_path=record.log_path)`. `SlurmExecutor.logs()` uses `log_path` if provided, falls back to relative path construction.

---

### ✅ P2-B: Unique script filenames to prevent collision (Bug 9, 11)

**File:** `devrun/executors/slurm.py`, method `submit()`

**Problem:** Script filename collisions when submitting sweeps with the same job_name.

**Implementation:**
```python
import uuid
suffix = uuid.uuid4().hex[:8]
script_path = _SCRIPT_DIR / f"sbatch_{job_name}_{suffix}.sh"
# Remote path:
submit_path = self._upload_script(
    str(script_path),
    f"/tmp/devrun_sbatch_{job_name}_{suffix}.sh"
)
```

---

### ✅ P2-C: Expand `_map_status` for all SLURM states (Bug 12)

**File:** `devrun/runner.py`, method `_map_status()`

**Problem:** SLURM returns state strings not in the mapping; they silently become `UNKNOWN`.

**Implementation** — add to the `mapping` dict:
```python
mapping = {
    # ... existing entries ...
    "completing": JobStatus.RUNNING,
    "node_fail": JobStatus.FAILED,
    "out_of_memory": JobStatus.FAILED,
    "preempted": JobStatus.FAILED,
    "boot_fail": JobStatus.FAILED,
    "deadline": JobStatus.FAILED,
    "stopped": JobStatus.FAILED,
    "suspended": JobStatus.RUNNING,
    "requeued": JobStatus.PENDING,
    "resizing": JobStatus.RUNNING,
}
```

---

### ✅ P2-D: Fix `sacct` line matching (Bug 13)

**File:** `devrun/executors/slurm.py`, method `status()`

**Problem:** `sacct` returns step lines (`12345.batch`) before job line (`12345`); matching on `parts[0] == job_id` may miss it.

**Implementation:**
```python
for line in result.stdout.strip().splitlines():
    parts = line.split("|")
    if len(parts) >= 2 and parts[0] == job_id:  # exact match for bare job ID only
        return parts[1].lower()
# If only step lines were returned (shouldn't happen, but defensive):
return "unknown"
```

This is already correct for the happy path. The defensive fix is to add: search all lines for exact match first, then fall back to prefix match:
```python
lines = [l.split("|") for l in result.stdout.strip().splitlines() if "|" in l]
for parts in lines:
    if parts[0] == job_id and len(parts) >= 2:
        return parts[1].lower()
return "unknown"
```

---

### ✅ P2-E: Shell-quote env values in `generate_sbatch_script` (Bug 15)

**File:** `devrun/utils/slurm.py`

**Problem:** `export KEY=value` without quoting breaks for values with spaces.

**Implementation:**
```python
import shlex
for key, val in (env or {}).items():
    lines.append(f"export {key}={shlex.quote(str(val))}")
```

---

### ✅ P2-F: Use absolute paths for `--output/--error` in sbatch (Bug 10)

**File:** `devrun/utils/slurm.py`, `generate_sbatch_script()`

**Problem:** Relative output paths are unreliable when job is submitted remotely or from different CWDs.

**Implementation:** Add parameter `output_dir: str | None = None` and:
```python
if output_dir:
    lines.append(f"#SBATCH --output={output_dir}/devrun_%j.out")
    lines.append(f"#SBATCH --error={output_dir}/devrun_%j.err")
else:
    lines.append("#SBATCH --output=devrun_%j.out")
    lines.append("#SBATCH --error=devrun_%j.err")
```
`SlurmExecutor.submit()` passes `working_dir` as `output_dir`.

---

## Priority 3 — `swe_bench_eval` Task Hardening

### ✅ P3-A: Placeholder validation in `SWEBenchEvalTask.prepare()`

**File:** `devrun/tasks/swe_bench_eval.py`

**Implementation:** After extracting each required param, add:
```python
def _check_placeholder(name: str, value: str) -> None:
    if value and str(value).startswith("<") and str(value).endswith(">"):
        raise ValueError(
            f"params.{name} is still a template placeholder ({value!r}). "
            f"Set it with: devrun run swe_bench_eval params.{name}=/actual/path"
        )

_check_placeholder("dataset_name", dataset_name)
if working_dir:
    _check_placeholder("working_dir", working_dir)
```

---

### ✅ P3-B: Shell-quote all command arguments

**File:** `devrun/tasks/swe_bench_eval.py`

**Implementation:**
```python
import shlex

command_parts = [
    "python -m swebench.harness.run_evaluation",
    f"    --dataset_name {shlex.quote(str(dataset_name))}",
    f"    --split {shlex.quote(str(split))}",
    f"    --max_workers {int(max_workers)}",
    f"    --run_id {shlex.quote(str(run_id))}",
    f"    --predictions_path {shlex.quote(str(predictions_path))}",
]
if namespace:
    command_parts.append(f"    --namespace {shlex.quote(str(namespace))}")
```

---

### ✅ P3-C: Update `swe_bench_eval/slurm.yaml`

**File:** `devrun/configs/swe_bench_eval/slurm.yaml`

Add `max_workers: 32` and add a comment block reminding users to override the placeholders:

```yaml
task: swe_bench_eval
executor: slurm

params:
  # REQUIRED: override these on the command line or in a local config layer:
  #   devrun run swe_bench_eval/slurm \
  #     params.dataset_name=/path/to/SWE-bench_Verified \
  #     params.predictions_path=/path/to/predictions.jsonl \
  #     params.working_dir=/path/to/working/dir
  mem: 64G
  cpus_per_task: 32
  max_workers: 32
  walltime: "24:00:00"
  partition: debug
```

---

## Priority 4 — `swe_bench_agentic` Task Hardening

### ✅ P4-A: Remove hard filesystem checks for remote executors (Issue 8)

**File:** `devrun/tasks/swe_bench_agentic.py`

**Option A (minimal change):** Wrap checks in `try/except` and log warnings instead of raising:
```python
from pathlib import Path
if not Path(llm_config).is_file():
    import logging
    logging.getLogger("devrun.tasks.swe_bench_agentic").warning(
        "llm_config file not found locally: %s — assuming it exists on the remote host.",
        llm_config
    )
    # Do NOT raise — let the job fail on the remote side with a clear error

if not Path(dataset).exists():
    logging.getLogger("devrun.tasks.swe_bench_agentic").warning(
        "dataset path not found locally: %s — assuming it exists on the remote host.",
        dataset
    )
```

**Option B (better, requires interface change):** Pass `executor_entry` to `prepare()` and only check for `type == "local"`. See Part 2 §7 for the interface change.

---

### ✅ P4-B: Add `mkdir -p slurm_logs` to generated command (Issue 13)

**File:** `devrun/tasks/swe_bench_agentic.py`

In the block where `extra_sbatch` is built for array jobs, also prepend `mkdir -p slurm_logs` to `command_lines`:

```python
if array:
    array_str = str(array)
    if concurrency_limit:
        array_str += f"%{concurrency_limit}"
    extra_sbatch.append(f"--array {array_str}")
    extra_sbatch.append("--output=slurm_logs/slurm-%A_%a.out")
    extra_sbatch.append("--error=slurm_logs/slurm-%A_%a.err")
    # Ensure slurm_logs dir exists at job runtime
    command_lines.insert(0, "mkdir -p slurm_logs")
```

Do NOT hardcode `--oversubscribe` here. Remove it.

---

### ✅ P4-C: Fix trailing backslash construction (Issue 10)

**File:** `devrun/tasks/swe_bench_agentic.py`

Replace the current append-then-strip approach with a proper argument list join:

```python
python_args = [
    f"python {script}",
    f"    --dataset ${{DATASET}}",
    f"    --split {split}",
    f"    --max-iterations {max_iterations}",
    f"    --select {select_dir}/${{num}}.txt",
    f"    --workspace {workspace}",
    f'    --output-dir "${{OUTPUT_PATH}}"',
]
for flag in flags:
    python_args.append(f"    {flag}")

python_cmd = " \\\n".join(python_args)

# Then append after the setup lines:
command_lines.append(python_cmd)
command = "\n".join(command_lines)
```

This guarantees no trailing backslash.

---

### ✅ P4-D: Remove hardcoded `--oversubscribe` (Issue 12)

**File:** `devrun/tasks/swe_bench_agentic.py`, line 132

```python
# DELETE THIS LINE:
extra_sbatch.append("--oversubscribe")
```

Add it as an opt-in config param: `oversubscribe: false` in `default.yaml`, and in `prepare()`:
```python
if params.get("oversubscribe", False):
    extra_sbatch.append("--oversubscribe")
```

---

## Priority 5 — Test Suite Fixes

### ✅ P5-A: Fix `mock_job_store` fixture in `conftest.py`

Replace the broken connection-swap fixture with:
```python
@pytest.fixture
def job_store(tmp_path):
    """Real JobStore backed by a temp SQLite file. Use instead of mock_job_store."""
    store = JobStore(tmp_path / "test_jobs.db")
    yield store
    store.close()
```

Keep `mock_job_store` as an alias pointing to `job_store` for backward compat, or rename all usages.

---

### ✅ P5-B: Fix registry pollution from `test_registry.py`

In `test_registry.py`, change all tests that re-register known names to use unique names:
```python
# INSTEAD OF:
@register_executor("local")  # overwrites production class
class DuplicateExecutor(...): ...

# USE:
import uuid
test_name = f"local_warn_test_{uuid.uuid4().hex[:6]}"

@register_executor(test_name)
class DuplicateA(...): ...

@register_executor(test_name)
class DuplicateB(...): ...

assert any("Overwriting executor" in r.message for r in caplog.records)
```

Remove `@pytest.mark.skip` from `test_local_executor_registered` in `TestBuiltInRegistrations`.

---

### ✅ P5-C: Fix `test_insert_creates_timestamp` in `test_db.py`

1. Change `db/jobs.py:57` from `datetime.utcnow().isoformat()` to `datetime.now(timezone.utc).isoformat()`.
2. Change `models.py:86` default factory from `datetime.utcnow` to `lambda: datetime.now(timezone.utc)`.
3. Remove `@pytest.mark.skip` from the test and fix the comparison to use timezone-aware datetime objects.

---

### ✅ P5-D: Add `test_swe_bench_eval.py`

Create `tests/test_swe_bench_eval.py` with the test cases listed in Part 3 §11.1. Focus first on:
- `test_prepare_placeholder_validation()` — catches the most common misconfiguration
- `test_prepare_generates_correct_command()` — regression guard
- `test_prepare_resources_forwarded()` — verifies Slurm integration

---

### ✅ P5-E: Add `test_swe_bench_agentic.py`

Create `tests/test_swe_bench_agentic.py` with the test cases listed in Part 3 §11.2. Focus first on:
- `test_prepare_mkdir_slurm_logs()` — catches Issue 13
- `test_prepare_trailing_backslash_clean()` — catches Issue 10
- `test_prepare_array_flag_in_extra_sbatch()` — basic smoke test

---

### ✅ P5-F: Add SSH and Slurm executor behavioral tests

Create `tests/test_ssh_executor.py` and `tests/test_slurm_executor.py` using `unittest.mock.patch` on `run_ssh_command` and `subprocess.run` respectively. The critical test is `test_submit_log_path_stable` and `test_logs_retrieves_correct_file` — these would have caught Bug 1.

---

## Priority 6 — Systemic Code Quality

### ✅ P6-A: Wire up `submit_with_retry` in the runner

**File:** `devrun/runner.py`, method `_submit_single()`

Change:
```python
remote_job_id = executor.submit(task_spec)
```
to:
```python
remote_job_id = executor.submit_with_retry(task_spec, retries=3, retry_delay=5.0)
```

---

### ✅ P6-B: Replace all `datetime.utcnow()` with `datetime.now(timezone.utc)`

Files affected: `models.py:86`, `db/jobs.py:57`, `runner.py:117,176,241`.

---

### ✅ P6-C: Remove dead code `parse_sacct_status` from `utils/slurm.py`

The function `parse_sacct_status()` at line 90 is never called. `SlurmExecutor.status()` has its own inline sacct parsing. Either:
- Delete `parse_sacct_status()` from `utils/slurm.py`, or
- Refactor `SlurmExecutor.status()` to use it (then add tests for it).

---

## Implementation Order for a Lesser Model

> ✅ All 24 steps completed 2026-03-22 in commit `0940ad8`.

Execute in this exact sequence to avoid merge conflicts and dependency issues:

```
✅ 1.  P5-C   Fix datetime.utcnow() → affects models.py and db/jobs.py, unblocks P5-C test fix
✅ 2.  P2-C   Expand _map_status() in runner.py — standalone, no dependencies
✅ 3.  P1-C   SSH timeout reduction — 2-line change, no dependencies
✅ 4.  P2-E   shlex.quote in generate_sbatch_script — standalone
✅ 5.  P1-B   shlex.quote in ssh.py submit() — standalone
✅ 6.  P2-D   sacct line matching fix — standalone
✅ 7.  P1-A   SSH log file naming fix — most complex SSH change; do after P1-B
✅ 8.  P2-B   Unique script filenames in slurm.py — standalone
✅ 9.  P2-F   Absolute paths in generate_sbatch_script — depends on P2-E done
✅ 10. P2-A  Store/retrieve log path (requires P2-F done, touches runner.py and slurm.py)
✅ 11. P3-A  Placeholder validation in swe_bench_eval — standalone task change
✅ 12. P3-B  shlex.quote in swe_bench_eval command — standalone
✅ 13. P3-C  Update slurm.yaml for swe_bench_eval — config only
✅ 14. P4-A  Remove hard filesystem checks in swe_bench_agentic
✅ 15. P4-C  Fix trailing backslash in swe_bench_agentic
✅ 16. P4-B  Add mkdir -p slurm_logs
✅ 17. P4-D  Remove --oversubscribe hardcode
✅ 18. P5-A  Fix mock_job_store fixture
✅ 19. P5-B  Fix registry pollution
✅ 20. P6-A  Wire submit_with_retry in runner
✅ 21. P5-D  Write test_swe_bench_eval.py
✅ 22. P5-E  Write test_swe_bench_agentic.py
✅ 23. P5-F  Write test_ssh_executor.py and test_slurm_executor.py
✅ 24. P6-C  Remove dead parse_sacct_status or refactor
```

---

## Files Changed Summary

| File | Changes |
|---|---|
| `devrun/executors/ssh.py` | P1-A (log naming), P1-B (quoting), P1-C (timeout) |
| `devrun/executors/slurm.py` | P2-A (log path), P2-B (unique filenames) |
| `devrun/utils/slurm.py` | P2-D (sacct), P2-E (quoting), P2-F (absolute paths), P6-C (dead code) |
| `devrun/runner.py` | P2-C (_map_status), P2-A (log path storage), P6-A (retry), P6-B (utcnow) |
| `devrun/models.py` | P6-B (utcnow default factory) |
| `devrun/db/jobs.py` | P6-B (utcnow) |
| `devrun/tasks/swe_bench_eval.py` | P3-A (placeholder), P3-B (quoting) |
| `devrun/configs/swe_bench_eval/slurm.yaml` | P3-C (max_workers, comments) |
| `devrun/tasks/swe_bench_agentic.py` | P4-A (checks), P4-B (mkdir), P4-C (backslash), P4-D (oversubscribe) |
| `tests/conftest.py` | P5-A (fixture fix) |
| `tests/test_registry.py` | P5-B (pollution fix) |
| `tests/test_db.py` | P5-C (timestamp test), boilerplate reduction |
| `tests/test_swe_bench_eval.py` | P5-D (new file) |
| `tests/test_swe_bench_agentic.py` | P5-E (new file) |
| `tests/test_ssh_executor.py` | P5-F (new file) |
| `tests/test_slurm_executor.py` | P5-F (new file) |
