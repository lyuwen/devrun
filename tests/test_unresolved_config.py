"""Tests for load_merged_omegaconf (PR1 Task 6)."""

import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from devrun.runner import load_merged_omegaconf


def test_load_merged_omegaconf_preserves_interpolations():
    """Unresolved config keeps ${jobs:...} references as strings."""
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td) / "configs" / "test_task"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "default.yaml"
        config_file.write_text(
            "task: dummy\n"
            "executor: local\n"
            "params:\n"
            "  output_dir: ${jobs:abc123,output_dir}\n"
            "  model: gpt-4\n"
        )

        cfg = load_merged_omegaconf(
            str(config_file), config_dirs=[Path(td) / "configs"]
        )

        yaml_str = OmegaConf.to_yaml(cfg, resolve=False)
        assert "${jobs:abc123,output_dir}" in yaml_str


def test_load_merged_omegaconf_vs_load_merged_config():
    """load_merged_config still resolves eagerly; load_merged_omegaconf does not."""
    from devrun.runner import load_merged_config

    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td) / "configs" / "test_task"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "default.yaml"
        config_file.write_text(
            "task: dummy\n"
            "executor: local\n"
            "params:\n"
            "  literal: hello\n"
            "  ref: ${params.literal}\n"
        )

        unresolved_cfg = load_merged_omegaconf(
            str(config_file), config_dirs=[Path(td) / "configs"]
        )
        yaml_unresolved = OmegaConf.to_yaml(unresolved_cfg, resolve=False)
        assert "${params.literal}" in yaml_unresolved

        resolved_dict = load_merged_config(
            str(config_file), config_dirs=[Path(td) / "configs"]
        )
        assert resolved_dict["params"]["ref"] == "hello"
