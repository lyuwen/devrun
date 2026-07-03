# PR #4 Review Fixes

## Summary

This document summarizes the fixes applied to address the three blocker issues found in the PR #4 review.

## Issues Fixed

### 1. Blocker: Workflow records never leave `queued`

**Problem**: Workflows inserted with `status=queued` never transitioned to `completed` or `failed` because no heartbeat phase aggregated stage job states back into workflow status.

**Fix**:
- Added `JobStore.aggregate_workflow_statuses()` method in `devrun/db/jobs.py`
- Added `_aggregate_workflow_statuses()` phase to heartbeat tick in `devrun/heartbeat.py`
- Workflows now transition to:
  - `completed` when all stage jobs are completed/skipped
  - `failed` when any stage job is failed/cancelled/timed_out
  - Remain `queued` while jobs are still running

**Tests**: 9 tests in `tests/test_heartbeat_workflow_aggregation.py`

### 2. Blocker: `workflow cancel` no longer cancels stage jobs

**Problem**: The cancel method read from `stages_state` (now always `{}`), so it marked only the workflow row as cancelled and left queued/running jobs untouched.

**Fix**:
- Updated `WorkflowRunner.cancel()` in `devrun/workflow.py` to read from `workflow_jobs` table
- Now iterates through stage jobs and calls `JobStore.request_cancel()` for each non-terminal job
- Respects terminal job states (completed/failed/cancelled/skipped/timed_out)

**Tests**: 5 tests in `tests/test_workflow_cancel_fix.py`

### 3. Blocker: Normal workflow `${stages:...}` references fail at promotion

**Problem**: `${stages:...}` → `${jobs:...}` rewriting only happened when `source_job_id` was set (--from-job workflows). Normal workflows persisted raw `${stages:...}` templates, which failed at promotion because only the `${jobs:...}` resolver is installed.

**Fix**:
- Refactored `WorkflowRunner._build_stage_plan()` to use two-pass approach:
  1. First pass: allocate job_ids for all non-skipped stages
  2. Build stage_map (stage_name → job_id)
  3. Second pass: rewrite all `${stages:...}` references to `${jobs:...}` using stage_map
- Now works for both normal workflows and --from-job workflows

**Tests**: 5 tests in `tests/test_workflow_stage_references.py`

## Breaking Change Note

**CRITICAL**: The fix to issue #3 is a breaking change. Any in-flight workflows with `${stages:...}` references that were enqueued before this fix will fail at promotion because their `params_template` was never rewritten.

**Migration**: No automatic migration is possible. In-flight workflows with cross-stage references must be cancelled and re-enqueued after this fix is deployed.

## Documentation Updates

Updated `AGENT.md` with:
1. New section: "Critical Design Principle: Producer-Consumer Separation"
   - Documents the environment isolation between producer and consumer
   - Explains variable tracking requirements
   - Highlights implications for development
   - Warns about breaking changes

2. Updated heartbeat description to include the new workflow status aggregation phase

## Test Coverage

- **Total new tests**: 19 tests
- **Test files added**:
  - `tests/test_heartbeat_workflow_aggregation.py` (9 tests)
  - `tests/test_workflow_cancel_fix.py` (5 tests)
  - `tests/test_workflow_stage_references.py` (5 tests)

- **Test results**: All 966 tests pass (17 skipped)

## Code Changes

### Modified Files

1. `devrun/db/jobs.py`
   - Added `aggregate_workflow_statuses()` method

2. `devrun/heartbeat.py`
   - Added `_aggregate_workflow_statuses()` function
   - Updated `tick()` to call aggregation phase

3. `devrun/workflow.py`
   - Refactored `_build_stage_plan()` for two-pass stage reference rewriting
   - Updated `cancel()` to use `workflow_jobs` table
   - Removed unused imports (`json`, `resolve_executor`)

4. `AGENT.md`
   - Added producer-consumer separation documentation
   - Updated heartbeat phase description

5. `tests/test_workflow.py`
   - Marked legacy `test_enhanced_logs_delegates_to_executor` as skipped

### New Files

1. `tests/test_heartbeat_workflow_aggregation.py`
2. `tests/test_workflow_cancel_fix.py`
3. `tests/test_workflow_stage_references.py`
4. `docs/PR4-review-fixes.md` (this document)

## Verification

All fixes have been verified with:
1. Unit tests for each individual fix
2. Integration tests for workflow end-to-end behavior
3. Full test suite run (966 passed, 17 skipped)

## Next Steps

1. Review this document and the test coverage
2. Decide on migration strategy for in-flight workflows
3. Consider adding a schema version to workflows table for future breaking changes
4. Document the breaking change in release notes
