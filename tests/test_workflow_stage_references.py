"""Tests for ${stages:...} reference rewriting in workflows."""

import pytest
import yaml
from devrun.db.jobs import JobStore, WorkflowStageRow
from devrun.models import WorkflowConfig, WorkflowStage
from devrun.workflow import WorkflowRunner


def test_stage_references_rewritten_in_normal_workflow(tmp_path):
    """${stages:...} references should be rewritten to ${jobs:...} in normal workflows."""
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    config = WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(
                name="stage1",
                task="eval",
                executor="local",
                params={"output_dir": "/data/stage1"},
            ),
            WorkflowStage(
                name="stage2",
                task="eval",
                executor="local",
                depends_on="stage1",
                params={
                    "input_dir": "${stages:stage1,output_dir}",
                    "model": "test-model",
                },
            ),
        ],
    )

    wf_id = runner.run(config, dry_run=False)

    # Get stage rows from the database
    db = JobStore(tmp_path / "test.db")
    stage_rows = db.get_workflow_stages(wf_id)

    # Find stage2's params_template
    stage2_row = next(r for r in stage_rows if r.stage_name == "stage2")
    assert stage2_row.params_template is not None

    # Parse the template
    template_dict = yaml.safe_load(stage2_row.params_template)

    # Should contain ${jobs:...} not ${stages:...}
    input_dir = template_dict.get("input_dir", "")
    assert "${jobs:" in input_dir
    assert "${stages:" not in input_dir
    assert ",output_dir}" in input_dir


def test_stage_references_multiple_stages(tmp_path):
    """Multiple ${stages:...} references should all be rewritten."""
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    config = WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(
                name="stage1",
                task="eval",
                executor="local",
                params={"output_dir": "/data/stage1", "model": "model1"},
            ),
            WorkflowStage(
                name="stage2",
                task="eval",
                executor="local",
                depends_on="stage1",
                params={
                    "input_dir": "${stages:stage1,output_dir}",
                    "model_name": "${stages:stage1,model}",
                    "dataset": "test-data",
                },
            ),
        ],
    )

    wf_id = runner.run(config, dry_run=False)

    db = JobStore(tmp_path / "test.db")
    stage_rows = db.get_workflow_stages(wf_id)
    stage2_row = next(r for r in stage_rows if r.stage_name == "stage2")

    template_dict = yaml.safe_load(stage2_row.params_template)

    # Both references should be rewritten
    input_dir = template_dict.get("input_dir", "")
    model_name = template_dict.get("model_name", "")

    assert "${jobs:" in input_dir
    assert "${stages:" not in input_dir

    assert "${jobs:" in model_name
    assert "${stages:" not in model_name


def test_stage_references_three_stage_chain(tmp_path):
    """${stages:...} references should work in multi-stage workflows."""
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    config = WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(
                name="stage1",
                task="eval",
                executor="local",
                params={"output_dir": "/data/stage1"},
            ),
            WorkflowStage(
                name="stage2",
                task="eval",
                executor="local",
                depends_on="stage1",
                params={"input_dir": "${stages:stage1,output_dir}"},
            ),
            WorkflowStage(
                name="stage3",
                task="eval",
                executor="local",
                depends_on="stage2",
                params={"input_dir": "${stages:stage2,input_dir}"},
            ),
        ],
    )

    wf_id = runner.run(config, dry_run=False)

    db = JobStore(tmp_path / "test.db")
    stage_rows = db.get_workflow_stages(wf_id)

    # All stages should have ${jobs:...} not ${stages:...}
    for row in stage_rows:
        if row.params_template:
            assert "${stages:" not in row.params_template


def test_stage_references_with_from_job(tmp_path):
    """${stages:...} references should be rewritten to source job in --from-job mode."""
    db = JobStore(tmp_path / "test.db")
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    # Create a source workflow first
    config1 = WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(
                name="stage1",
                task="eval",
                executor="local",
                params={"output_dir": "/data/stage1"},
            ),
            WorkflowStage(
                name="stage2",
                task="eval",
                executor="local",
                depends_on="stage1",
                params={"input_dir": "${stages:stage1,output_dir}"},
            ),
        ],
    )
    wf_id1 = runner.run(config1, dry_run=False)
    stage_rows1 = db.get_workflow_stages(wf_id1)
    stage1_job_id = next(r.job_id for r in stage_rows1 if r.stage_name == "stage1")

    # Mark stage1 as completed
    from devrun.models import JobStatus
    db.update_status(stage1_job_id, JobStatus.COMPLETED)

    # Now run with --from-job, skipping stage1
    wf_id2 = runner.run(
        config1,
        dry_run=False,
        from_job=stage1_job_id,
        start_after="stage1",
    )

    # Stage2 in the new workflow should reference the source job
    stage_rows2 = db.get_workflow_stages(wf_id2)
    stage2_row = next(r for r in stage_rows2 if r.stage_name == "stage2")

    template_dict = yaml.safe_load(stage2_row.params_template)
    input_dir = template_dict.get("input_dir", "")

    # Should contain ${jobs:<source_job_id>,...}
    assert "${jobs:" in input_dir
    assert stage1_job_id in input_dir
    assert "${stages:" not in input_dir


def test_stage_references_no_references(tmp_path):
    """Stages without ${stages:...} references should not be affected."""
    runner = WorkflowRunner(db_path=tmp_path / "test.db")

    config = WorkflowConfig(
        workflow="test_wf",
        stages=[
            WorkflowStage(
                name="stage1",
                task="eval",
                executor="local",
                params={"output_dir": "/data/stage1", "model": "test-model"},
            ),
        ],
    )

    wf_id = runner.run(config, dry_run=False)

    db = JobStore(tmp_path / "test.db")
    stage_rows = db.get_workflow_stages(wf_id)
    stage1_row = stage_rows[0]

    template_dict = yaml.safe_load(stage1_row.params_template)

    # Plain values should remain unchanged
    assert template_dict["output_dir"] == "/data/stage1"
    assert template_dict["model"] == "test-model"
    assert "${stages:" not in stage1_row.params_template
    assert "${jobs:" not in stage1_row.params_template
