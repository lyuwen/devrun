"""Comprehensive tests for devrun doctor — config health checker and deprecation scanner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from devrun.cli import app
from devrun.doctor import (
    DEPRECATION_RULES,
    ConfigScope,
    DeprecationRule,
    Diagnostic,
    DoctorReport,
    Severity,
    _apply_fixes,
    _check_cross_references,
    _check_deprecations,
    _check_executor_types,
    _check_placeholders_for_file,
    _del_nested,
    _find_placeholders,
    _get_nested,
    _set_nested,
    _validate_executor_entry,
    _validate_task_config,
)


# ============================================================================
# Helper function tests
# ============================================================================


class TestGetNested:
    """Tests for _get_nested helper."""

    def test_valid_path_returns_value(self):
        data = {"a": {"b": {"c": 42}}}
        assert _get_nested(data, "a.b.c") == 42

    def test_missing_path_returns_none(self):
        data = {"a": {"b": 1}}
        assert _get_nested(data, "a.x.y") is None

    def test_deep_nested_path(self):
        data = {"l1": {"l2": {"l3": {"l4": "deep"}}}}
        assert _get_nested(data, "l1.l2.l3.l4") == "deep"

    def test_top_level_key(self):
        data = {"key": "val"}
        assert _get_nested(data, "key") == "val"

    def test_non_dict_intermediate_returns_none(self):
        data = {"a": "string"}
        assert _get_nested(data, "a.b") is None

    def test_returns_list_value(self):
        data = {"cmds": {"setup": ["echo hi", "echo bye"]}}
        assert _get_nested(data, "cmds.setup") == ["echo hi", "echo bye"]


class TestSetNested:
    """Tests for _set_nested helper."""

    def test_creates_value_at_path(self):
        data: dict = {}
        _set_nested(data, "a.b.c", 99)
        assert data == {"a": {"b": {"c": 99}}}

    def test_creates_intermediate_dicts(self):
        data: dict = {}
        _set_nested(data, "x.y.z", "val")
        assert isinstance(data["x"], dict)
        assert isinstance(data["x"]["y"], dict)
        assert data["x"]["y"]["z"] == "val"

    def test_overwrites_existing_value(self):
        data = {"a": {"b": "old"}}
        _set_nested(data, "a.b", "new")
        assert data["a"]["b"] == "new"

    def test_top_level_set(self):
        data: dict = {}
        _set_nested(data, "key", 1)
        assert data == {"key": 1}

    def test_overwrites_non_dict_intermediate(self):
        data = {"a": "not-a-dict"}
        _set_nested(data, "a.b", "val")
        assert data["a"] == {"b": "val"}


class TestDelNested:
    """Tests for _del_nested helper."""

    def test_deletes_leaf_key(self):
        data = {"a": {"b": "val", "c": "keep"}}
        assert _del_nested(data, "a.b") is True
        assert "b" not in data["a"]
        assert data["a"]["c"] == "keep"

    def test_cleans_up_empty_parent_dicts(self):
        data = {"a": {"b": {"c": "gone"}}}
        assert _del_nested(data, "a.b.c") is True
        # b should be cleaned up since it's now empty, and then a too
        assert "a" not in data

    def test_returns_false_on_missing_path(self):
        data = {"a": 1}
        assert _del_nested(data, "x.y.z") is False

    def test_returns_false_on_missing_leaf(self):
        data = {"a": {"b": 1}}
        assert _del_nested(data, "a.c") is False

    def test_partial_cleanup(self):
        data = {"a": {"b": {"c": "gone"}, "d": "stay"}}
        _del_nested(data, "a.b.c")
        # a.b cleaned up (empty), but a stays because it still has d
        assert "b" not in data["a"]
        assert data["a"]["d"] == "stay"


class TestFindPlaceholders:
    """Tests for _find_placeholders helper."""

    def test_finds_placeholder_in_flat_dict(self):
        data = {"dataset": "<path_to_dataset>"}
        result = _find_placeholders(data)
        assert len(result) == 1
        assert result[0] == ("dataset", "<path_to_dataset>")

    def test_finds_placeholder_as_substring(self):
        data = {"image": "<registry>/foo"}
        result = _find_placeholders(data)
        assert len(result) == 1
        assert result[0] == ("image", "<registry>/foo")

    def test_returns_empty_for_normal_values(self):
        data = {"key": "normal", "count": 42, "flag": True}
        result = _find_placeholders(data)
        assert result == []

    def test_finds_placeholders_nested_in_lists(self):
        data = {"cmds": ["echo <name>", "normal"]}
        result = _find_placeholders(data)
        assert len(result) == 1
        assert result[0][0] == "cmds[0]"
        assert result[0][1] == "echo <name>"

    def test_deeply_nested_placeholder(self):
        data = {"a": {"b": {"c": "<placeholder>"}}}
        result = _find_placeholders(data)
        assert len(result) == 1
        assert result[0] == ("a.b.c", "<placeholder>")

    def test_multiple_placeholders(self):
        data = {"x": "<one>", "y": {"z": "<two>"}}
        result = _find_placeholders(data)
        assert len(result) == 2


# ============================================================================
# Deprecation detection tests
# ============================================================================


class TestCheckDeprecations:
    """Tests for _check_deprecations scanner."""

    def test_dep001_triggers_on_extra_setup_commands(self):
        data = {"extra": {"setup_commands": ["source /opt/conda/bin/activate"]}}
        diags = _check_deprecations(Path("test.yaml"), data, ConfigScope.EXECUTORS)
        assert len(diags) == 1
        assert diags[0].rule_id == "DEP001"
        assert diags[0].severity == Severity.WARNING
        assert "extra.setup_commands" in diags[0].message

    def test_clean_config_no_deprecation_warnings(self):
        data = {"type": "local", "python_env": {"setup_commands": ["echo ok"]}}
        diags = _check_deprecations(Path("test.yaml"), data, ConfigScope.EXECUTORS)
        assert len(diags) == 0

    def test_dep001_does_not_trigger_on_task_scope(self):
        """DEP001 is EXECUTORS scope — should NOT trigger when scanning TASK scope."""
        data = {"extra": {"setup_commands": ["source env"]}}
        diags = _check_deprecations(Path("test.yaml"), data, ConfigScope.TASK)
        assert len(diags) == 0

    def test_dep001_includes_removed_in_version(self):
        data = {"extra": {"setup_commands": ["cmd"]}}
        diags = _check_deprecations(Path("test.yaml"), data, ConfigScope.EXECUTORS)
        assert "0.3.0" in diags[0].message


# ============================================================================
# Structural validation tests
# ============================================================================


class TestValidateTaskConfig:
    """Tests for _validate_task_config."""

    def test_valid_config_passes(self):
        raw = {"task": "eval", "executor": "local", "params": {"model": "gpt"}}
        diags = _validate_task_config(Path("test.yaml"), raw)
        assert len(diags) == 0

    def test_invalid_config_returns_error(self):
        # Has required top-level keys but with invalid types
        raw = {"task": 123, "executor": 456}
        diags = _validate_task_config(Path("test.yaml"), raw)
        assert len(diags) == 1
        assert diags[0].severity == Severity.ERROR
        assert diags[0].rule_id == "SCHEMA"

    def test_partial_config_skips_validation(self):
        """Configs missing task/executor are partial overlays — no error."""
        raw = {"params": {"model": "gpt"}}
        diags = _validate_task_config(Path("test.yaml"), raw)
        assert len(diags) == 0

    def test_minimal_valid_config(self):
        raw = {"task": "eval", "executor": "local"}
        diags = _validate_task_config(Path("test.yaml"), raw)
        assert len(diags) == 0


class TestValidateExecutorEntry:
    """Tests for _validate_executor_entry."""

    def test_valid_entry_passes(self):
        raw = {"type": "local"}
        diags = _validate_executor_entry(Path("exec.yaml"), "my_exec", raw)
        assert len(diags) == 0

    def test_missing_type_returns_error(self):
        raw = {"host": "example.com"}
        diags = _validate_executor_entry(Path("exec.yaml"), "bad_exec", raw)
        assert len(diags) == 1
        assert diags[0].severity == Severity.ERROR
        assert diags[0].rule_id == "SCHEMA"
        assert "bad_exec" in diags[0].message

    def test_valid_ssh_entry(self):
        raw = {"type": "ssh", "host": "server.example.com", "user": "admin"}
        diags = _validate_executor_entry(Path("exec.yaml"), "ssh_exec", raw)
        assert len(diags) == 0


# ============================================================================
# Cross-reference validation tests
# ============================================================================


class TestCheckCrossReferences:
    """Tests for _check_cross_references."""

    @patch("devrun.doctor.list_tasks", return_value=["eval", "inference"])
    def test_unknown_task_produces_warning(self, _mock_tasks):
        raw = {"task": "nonexistent_task", "executor": "local"}
        executor_names = {"local", "slurm"}
        diags = _check_cross_references(Path("t.yaml"), raw, executor_names)
        task_diags = [d for d in diags if "nonexistent_task" in d.message]
        assert len(task_diags) == 1
        assert task_diags[0].severity == Severity.WARNING
        assert task_diags[0].rule_id == "XREF"

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    def test_unknown_executor_produces_warning(self, _mock_tasks):
        raw = {"task": "eval", "executor": "missing_exec"}
        executor_names = {"local"}
        diags = _check_cross_references(Path("t.yaml"), raw, executor_names)
        exec_diags = [d for d in diags if "missing_exec" in d.message]
        assert len(exec_diags) == 1
        assert exec_diags[0].severity == Severity.WARNING

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    def test_valid_references_no_diagnostics(self, _mock_tasks):
        raw = {"task": "eval", "executor": "local"}
        executor_names = {"local"}
        diags = _check_cross_references(Path("t.yaml"), raw, executor_names)
        assert len(diags) == 0


class TestCheckExecutorTypes:
    """Tests for _check_executor_types."""

    @patch("devrun.doctor.list_executors", return_value=["local", "ssh", "slurm"])
    def test_unknown_type_produces_error(self, _mock):
        raw = {"type": "kubernetes"}
        diags = _check_executor_types(Path("exec.yaml"), "test", raw)
        assert len(diags) == 1
        assert diags[0].severity == Severity.ERROR
        assert diags[0].rule_id == "TYPE"
        assert "kubernetes" in diags[0].message

    @patch("devrun.doctor.list_executors", return_value=["local", "ssh", "slurm"])
    def test_valid_type_passes(self, _mock):
        raw = {"type": "local"}
        diags = _check_executor_types(Path("exec.yaml"), "test", raw)
        assert len(diags) == 0


# ============================================================================
# Placeholder detection tests
# ============================================================================


class TestCheckPlaceholdersForFile:
    """Tests for _check_placeholders_for_file."""

    def test_placeholder_detected(self):
        data = {"dataset": "<placeholder>"}
        diags = _check_placeholders_for_file(Path("t.yaml"), data)
        assert len(diags) == 1
        assert diags[0].severity == Severity.INFO
        assert diags[0].rule_id == "PLACEHOLDER"
        assert "dataset" in diags[0].message

    def test_substring_placeholder_detected(self):
        data = {"image": "<registry>/image"}
        diags = _check_placeholders_for_file(Path("t.yaml"), data)
        assert len(diags) == 1
        assert "<registry>" in diags[0].message

    def test_normal_values_no_diagnostics(self):
        data = {"model": "gpt-4", "batch_size": 16}
        diags = _check_placeholders_for_file(Path("t.yaml"), data)
        assert len(diags) == 0


# ============================================================================
# DoctorReport dataclass tests
# ============================================================================


class TestDoctorReport:
    """Tests for DoctorReport properties."""

    def test_has_errors_true(self):
        report = DoctorReport(diagnostics=[
            Diagnostic(severity=Severity.ERROR, file_path="f", rule_id="R", message="m"),
        ])
        assert report.has_errors is True

    def test_has_errors_false(self):
        report = DoctorReport(diagnostics=[
            Diagnostic(severity=Severity.WARNING, file_path="f", rule_id="R", message="m"),
        ])
        assert report.has_errors is False

    def test_counts(self):
        report = DoctorReport(diagnostics=[
            Diagnostic(severity=Severity.ERROR, file_path="f", rule_id="R", message="m"),
            Diagnostic(severity=Severity.ERROR, file_path="f", rule_id="R", message="m"),
            Diagnostic(severity=Severity.WARNING, file_path="f", rule_id="R", message="m"),
            Diagnostic(severity=Severity.INFO, file_path="f", rule_id="R", message="m"),
        ])
        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.info_count == 1

    def test_empty_report(self):
        report = DoctorReport()
        assert report.has_errors is False
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.info_count == 0


# ============================================================================
# Auto-fix tests
# ============================================================================


class TestApplyFixes:
    """Tests for _apply_fixes auto-fix engine."""

    def _dep001_rules(self):
        return [r for r in DEPRECATION_RULES if r.rule_id == "DEP001"]

    def test_autofix_creates_backup(self, tmp_path):
        """Auto-fix should create a .bak backup before modifying the file."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "type: local\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - source /opt/conda/bin/activate\n"
        )
        _apply_fixes(yaml_file, self._dep001_rules())
        backup = yaml_file.with_suffix(".yaml.bak")
        assert backup.exists()

    def test_value_moved_from_old_to_new_path(self, tmp_path):
        """The deprecated value should appear at the new path after fix.

        NOTE: _apply_fixes operates on the document root, so old_path/new_path
        must exist at the top level of the YAML document for the fix to apply.
        """
        yaml_file = tmp_path / "executors.yaml"
        # Flat structure matching the dotpath the rules expect at doc root
        yaml_file.write_text(
            "type: local\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - source /opt/conda/bin/activate\n"
        )
        diags = _apply_fixes(yaml_file, self._dep001_rules())
        assert any(d.fix_applied for d in diags)

        with open(yaml_file) as fh:
            result = yaml.safe_load(fh)
        assert result["python_env"]["setup_commands"] == [
            "source /opt/conda/bin/activate"
        ]
        # Old path should be gone
        assert _get_nested(result, "extra.setup_commands") is None

    def test_list_values_merged_not_replaced(self, tmp_path):
        """If new_path already has a list, old values should be appended."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "type: local\n"
            "python_env:\n"
            "  setup_commands:\n"
            "    - existing_cmd\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - new_cmd\n"
        )
        _apply_fixes(yaml_file, self._dep001_rules())

        with open(yaml_file) as fh:
            result = yaml.safe_load(fh)
        cmds = result["python_env"]["setup_commands"]
        assert "existing_cmd" in cmds
        assert "new_cmd" in cmds
        assert len(cmds) == 2

    def test_existing_backup_skips_with_warning(self, tmp_path):
        """If .bak already exists, fix should be skipped."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "type: local\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - cmd\n"
        )
        backup = yaml_file.with_suffix(".yaml.bak")
        backup.write_text("old backup")

        diags = _apply_fixes(yaml_file, self._dep001_rules())
        assert len(diags) == 1
        assert diags[0].severity == Severity.WARNING
        assert "Backup already exists" in diags[0].message
        assert not any(d.fix_applied for d in diags)

    def test_empty_parent_cleaned_up(self, tmp_path):
        """After migrating extra.setup_commands, an empty 'extra: {}' should be removed."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "type: local\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - cmd\n"
        )
        _apply_fixes(yaml_file, self._dep001_rules())

        with open(yaml_file) as fh:
            result = yaml.safe_load(fh)
        # extra should be gone since setup_commands was its only child
        assert "extra" not in result

    def test_comments_preserved_by_ruamel(self, tmp_path):
        """ruamel.yaml round-trip should preserve YAML comments."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "# Top-level comment\n"
            "type: local  # inline comment\n"
            "extra:\n"
            "  setup_commands:\n"
            "    - cmd\n"
        )
        _apply_fixes(yaml_file, self._dep001_rules())

        content = yaml_file.read_text()
        assert "# Top-level comment" in content
        assert "# inline comment" in content

    def test_no_rules_no_changes(self, tmp_path):
        """With no triggered rules, file should be unchanged."""
        yaml_file = tmp_path / "executors.yaml"
        original = "type: local\n"
        yaml_file.write_text(original)

        diags = _apply_fixes(yaml_file, [])
        assert len(diags) == 0
        assert yaml_file.read_text() == original

    def test_nested_executor_fix_applies_with_entry_names(self, tmp_path):
        """_apply_fixes with entry_names correctly migrates deprecations
        nested under executor name keys in executors.yaml."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "my_exec:\n"
            "  type: local\n"
            "  extra:\n"
            "    setup_commands:\n"
            "      - cmd\n"
        )
        diags = _apply_fixes(yaml_file, self._dep001_rules(), entry_names=["my_exec"])
        assert any(d.fix_applied for d in diags)
        # Verify the value was moved
        content = yaml.safe_load(yaml_file.read_text())
        assert content["my_exec"]["python_env"]["setup_commands"] == ["cmd"]
        assert "extra" not in content["my_exec"]

    def test_nested_executor_fix_without_entry_names_skips(self, tmp_path):
        """Without entry_names, _apply_fixes searches at document root and skips
        deprecations nested under executor name keys."""
        yaml_file = tmp_path / "executors.yaml"
        yaml_file.write_text(
            "my_exec:\n"
            "  type: local\n"
            "  extra:\n"
            "    setup_commands:\n"
            "      - cmd\n"
        )
        diags = _apply_fixes(yaml_file, self._dep001_rules())
        # No fix_applied because the old_path isn't at document root
        assert not any(d.fix_applied for d in diags)


# ============================================================================
# CLI integration tests
# ============================================================================


class TestDoctorCLI:
    """Tests for the `devrun doctor` CLI command."""

    def _make_clean_config(self, tmp_path):
        """Create a minimal clean config setup in tmp_path."""
        task_file = tmp_path / "eval.yaml"
        task_file.write_text(yaml.dump({
            "task": "eval",
            "executor": "local",
            "params": {"model": "gpt"},
        }))
        exec_file = tmp_path / "executors.yaml"
        exec_file.write_text(yaml.dump({
            "local": {"type": "local"},
        }))
        return task_file, exec_file

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    @patch("devrun.doctor.list_executors", return_value=["local"])
    def test_doctor_exits_0_on_clean_config(self, _le, _lt, tmp_path):
        task_file, exec_file = self._make_clean_config(tmp_path)
        with (
            patch("devrun.doctor.get_config_dirs", return_value=[tmp_path]),
            patch("devrun.doctor._find_executors_file", return_value=exec_file),
        ):
            runner = CliRunner()
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "0 error(s)" in result.stdout

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    @patch("devrun.doctor.list_executors", return_value=["local"])
    def test_doctor_exits_1_on_errors(self, _le, _lt, tmp_path):
        """An executor with unknown type should cause exit code 1."""
        task_file = tmp_path / "eval.yaml"
        task_file.write_text(yaml.dump({
            "task": "eval",
            "executor": "local",
            "params": {},
        }))
        exec_file = tmp_path / "executors.yaml"
        exec_file.write_text(yaml.dump({
            "local": {"type": "unknown_type"},
        }))
        with (
            patch("devrun.doctor.get_config_dirs", return_value=[tmp_path]),
            patch("devrun.doctor._find_executors_file", return_value=exec_file),
        ):
            runner = CliRunner()
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    @patch("devrun.doctor.list_executors", return_value=["local"])
    def test_doctor_fix_triggers_fix_path(self, _le, _lt, tmp_path):
        exec_file = tmp_path / "executors.yaml"
        exec_file.write_text(
            "local:\n  type: local\n  extra:\n    setup_commands:\n      - cmd\n"
        )
        # No task configs (empty dir with only executors.yaml)
        with (
            patch("devrun.doctor.get_config_dirs", return_value=[tmp_path]),
            patch("devrun.doctor._find_executors_file", return_value=exec_file),
        ):
            runner = CliRunner()
            result = runner.invoke(app, ["doctor", "--fix"])
        # Should have run without crashing; backup should exist
        backup = exec_file.with_suffix(".yaml.bak")
        assert backup.exists()

    @patch("devrun.doctor.list_tasks", return_value=["eval"])
    @patch("devrun.doctor.list_executors", return_value=["local"])
    def test_doctor_verbose_shows_output(self, _le, _lt, tmp_path):
        task_file, exec_file = self._make_clean_config(tmp_path)
        with (
            patch("devrun.doctor.get_config_dirs", return_value=[tmp_path]),
            patch("devrun.doctor._find_executors_file", return_value=exec_file),
        ):
            runner = CliRunner()
            result = runner.invoke(app, ["doctor", "--verbose"])
        assert result.exit_code == 0
        assert "Summary" in result.stdout
