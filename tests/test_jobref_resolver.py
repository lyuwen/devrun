"""Tests for ${jobs:...} OmegaConf resolver with JobRefContext scoping (PR1 Task 5)."""

import pytest
from omegaconf import OmegaConf

from devrun.jobref import (
    JobRefContext,
    clear_jobref_context,
    install_jobref_context,
)


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
    install_jobref_context(JobRefContext(
        allowed_parents={"jobA": {"val": "A"}},
        calling_job_id="job1"
    ))
    cfg1 = OmegaConf.create({"x": "${jobs:jobA,val}"})
    result1 = OmegaConf.to_container(cfg1, resolve=True)
    clear_jobref_context()

    install_jobref_context(JobRefContext(
        allowed_parents={"jobB": {"val": "B"}},
        calling_job_id="job2"
    ))
    cfg2 = OmegaConf.create({"x": "${jobs:jobB,val}"})
    result2 = OmegaConf.to_container(cfg2, resolve=True)
    clear_jobref_context()

    assert result1["x"] == "A"
    assert result2["x"] == "B"
