# Changelog

All notable changes to devrun are documented here.

---

## [Unreleased] — 2026-03-22

Branch: `fix/code-review-improvements`
Commit: `0940ad8`
Source: 4-part internal code review (`docs/review-part*.md`)

### Fixed

#### SSH Executor (`devrun/executors/ssh.py`)
- **Log file naming** (Bug 1): Log files were named using `$$`, which expands to the local SSH client PID rather than the remote process PID. `logs()` would then look for a file named after the remote PID, which never existed. Fixed by generating a `uuid` run token before the SSH call (`run_token = uuid.uuid4().hex[:12]`) and embedding it in the remote log path. The composite job ID `pid:token` lets `status()`, `logs()`, and `cancel()` each extract the right piece.
- **Shell quoting — command** (Bug 2): Commands containing single quotes would break the `bash -c '...'` wrapping. Replaced with a heredoc (`bash << 'DEVRUN_EOF' … DEVRUN_EOF`) so the command body is passed verbatim.
- **Shell quoting — env values** (Bug 3): Environment variable values were interpolated without quoting. Applied `shlex.quote()` to all values.
- **Shell quoting — working directory** (Bug 4): `working_dir` paths with spaces were not quoted. Applied `shlex.quote()` to the `cd` argument.
- **Wrong status string** (Bug 6): `status()` returned `"done"` instead of the canonical `"completed"` string.
- **SSH timeouts too long** (Bug 7): Default SSH timeout of 300 s blocked the CLI for 5 minutes on a hung status check. `status()` now uses `timeout=30`; `logs()` uses `timeout=60`.

#### Slurm Executor (`devrun/executors/slurm.py`)
- **Script filename collision** (Bug 9/11): When submitting sweeps with the same job name, all jobs wrote to the same `sbatch_<name>.sh` file. Fixed with a `uuid` suffix: `sbatch_{job_name}_{uuid.uuid4().hex[:8]}.sh`.
- **`sacct` line matching** (Bug 13): `sacct` returns step lines (`12345.batch`) before the bare job line (`12345`). The old loop could match a step line and return a wrong status. Fixed with a two-pass filter: collect only lines containing `"|"`, then match `parts[0] == job_id` exactly.
- **Absolute log paths** (Bug 8/14): `logs()` used a relative `cat devrun_{job_id}.out` which fails unless called from the exact submission directory. The executor now computes the absolute path after submit, stores it in `task_spec.metadata["log_path"]`, and the runner persists it to the DB and passes it back as `log_path=` to `executor.logs()`.

#### Slurm Utilities (`devrun/utils/slurm.py`)
- **Unquoted env exports** (Bug 15): `export KEY=value` without quoting breaks for values with spaces. Applied `shlex.quote()` to all values in `generate_sbatch_script()`.
- **Relative `--output`/`--error` paths** (Bug 10): Added `output_dir` parameter; when provided, sbatch directives use absolute paths.
- **Dead code removed** (P6-C): `parse_sacct_status()` was never called anywhere. Deleted.

#### Runner (`devrun/runner.py`)
- **Missing SLURM states** (Bug 12): `_map_status()` was missing 10 SLURM state strings, causing them to silently map to `UNKNOWN`. Added: `completing`, `node_fail`, `out_of_memory`, `preempted`, `boot_fail`, `deadline`, `stopped`, `suspended`, `requeued`, `resizing`.
- **`submit_with_retry` never used** (P6-A): `BaseExecutor.submit_with_retry()` existed but was never called. The runner now calls it with `retries=3, retry_delay=5.0`.
- **Log path not stored** (P2-A): After executor submit, the runner now reads `task_spec.metadata.get("log_path")` and passes it to `update_status()` so it is persisted in the DB and retrievable later.
- **Deprecated `datetime.utcnow()`** (P6-B): Replaced all `datetime.utcnow()` calls with `datetime.now(timezone.utc)` for Python 3.12+ compatibility and timezone-aware timestamps.

#### Models & Database (`devrun/models.py`, `devrun/db/jobs.py`)
- **Deprecated `datetime.utcnow()`** (P6-B): Same fix applied to the default factory in `models.py` and the `created_at`/`completed_at` writes in `db/jobs.py`.

#### `swe_bench_eval` Task (`devrun/tasks/swe_bench_eval.py`)
- **No placeholder validation** (P3-A): Template placeholders like `<dataset_path>` were silently forwarded to the shell command. Added `_check_placeholder()` which raises a clear `ValueError` at prepare time if any required param is still a `<…>` placeholder.
- **Unquoted CLI arguments** (P3-B): All arguments in the generated `python -m swebench.harness.run_evaluation` command are now wrapped in `shlex.quote()`.

#### `swe_bench_agentic` Task (`devrun/tasks/swe_bench_agentic.py`)
- **Hard filesystem checks break remote executors** (P4-A): `prepare()` was raising `FileNotFoundError` for `llm_config` and `dataset` paths, which always fail when the executor is remote (SSH/Slurm). Replaced with `logger.warning()` — the job fails on the remote side with a clear error if the path is genuinely missing.
- **Missing `slurm_logs/` directory** (P4-B): Array jobs write to `slurm_logs/slurm-%A_%a.out` but the directory was never created. Added `mkdir -p slurm_logs` as the first line of the generated command.
- **Trailing backslash in generated command** (P4-C): The append-then-strip construction could produce a command ending in `\`, which is a syntax error. Replaced with a `python_args` list joined by `" \\\n"`.
- **Hardcoded `--oversubscribe`** (P4-D): `--oversubscribe` was always appended to sbatch flags regardless of cluster policy. Now opt-in via `params.oversubscribe: true` (default `false`).

### Tests

- **`mock_job_store` fixture** (P5-A): The fixture was swapping SQLite connections after `JobStore` creation, leaving `_db_path` pointing at a deleted `:memory:` connection. Replaced with a proper `tmp_path`-backed real SQLite file.
- **Registry test pollution** (P5-B): `test_executor_reregistration_warning` re-registered `"local"`, overwriting `LocalExecutor` for the entire test session and breaking `test_local_executor_registered`. Now uses a unique `uuid`-based name.
- **Skipped tests un-skipped**: Removed `@pytest.mark.skip` from `test_local_executor_registered`, `test_insert_creates_timestamp`, and `test_multiple_inserts` — all pass correctly after the above fixes.
- **New: `tests/test_swe_bench_eval.py`** (P5-D): 10 tests covering placeholder validation, command generation, namespace handling, resource forwarding, env forwarding, and shell-safety.
- **New: `tests/test_swe_bench_agentic.py`** (P5-E): 11 tests covering array flags, concurrency limits, `mkdir slurm_logs`, trailing backslash absence, oversubscribe opt-in, and remote executor compatibility.
- **New: `tests/test_ssh_executor.py`** (P5-F): 8 tests covering stable log file naming, composite job ID parsing, correct log retrieval by token, and shell quoting.

### Result

**249 tests passing, 10 intentional skips** (up from 217 passing, 13 skipped before the review).
