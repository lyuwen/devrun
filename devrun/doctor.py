"""devrun doctor — config health checker and deprecation scanner."""

from __future__ import annotations

import enum
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError

from devrun.models import ExecutorEntry, TaskConfig
from devrun.registry import list_executors, list_tasks
from devrun.router import _find_executors_file
from devrun.runner import get_config_dirs

logger = logging.getLogger("devrun.doctor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ConfigScope(enum.Enum):
    EXECUTORS = "executors"
    TASK = "task"
    BOTH = "both"


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class DeprecationRule:
    rule_id: str
    scope: ConfigScope
    description: str
    old_path: str
    new_path: str
    removed_in: str | None = None
    transform: Callable[[Any], Any] | None = None


@dataclass
class Diagnostic:
    severity: Severity
    file_path: str
    rule_id: str
    message: str
    fix_applied: bool = False


@dataclass
class DoctorReport:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self.diagnostics)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == Severity.INFO)


# ---------------------------------------------------------------------------
# Deprecation rules
# ---------------------------------------------------------------------------


DEPRECATION_RULES: list[DeprecationRule] = [
    DeprecationRule(
        rule_id="DEP001",
        scope=ConfigScope.EXECUTORS,
        description="'extra.setup_commands' is deprecated; use 'python_env.setup_commands'",
        old_path="extra.setup_commands",
        new_path="python_env.setup_commands",
        removed_in="0.3.0",
    ),
]


# ---------------------------------------------------------------------------
# Nested-dict helpers
# ---------------------------------------------------------------------------


def _get_nested(data: dict, dotpath: str) -> Any | None:
    """Traverse nested dicts by dot-delimited path. Return None if any key is missing."""
    keys = dotpath.split(".")
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(data: dict, dotpath: str, value: Any) -> None:
    """Set a value at a dot-delimited path, creating intermediate dicts as needed."""
    keys = dotpath.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _del_nested(data: dict, dotpath: str) -> bool:
    """Delete a key at a dot-delimited path. Clean up empty parent dicts. Return True if deleted."""
    keys = dotpath.split(".")
    # Walk down, keeping a stack of (parent_dict, key) pairs for cleanup
    stack: list[tuple[dict, str]] = []
    current: Any = data
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        stack.append((current, key))
        current = current[key]
    if not isinstance(current, dict) or keys[-1] not in current:
        return False
    del current[keys[-1]]
    # Clean up empty parent dicts
    for parent, key in reversed(stack):
        if isinstance(parent[key], dict) and not parent[key]:
            del parent[key]
    return True


def _find_placeholders(data: Any, path_prefix: str = "") -> list[tuple[str, str]]:
    """Recursively walk data, return list of (dotpath, value) for any string containing ``<...>``."""
    results: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, val in data.items():
            child_path = f"{path_prefix}.{key}" if path_prefix else key
            results.extend(_find_placeholders(val, child_path))
    elif isinstance(data, list):
        for i, val in enumerate(data):
            child_path = f"{path_prefix}[{i}]"
            results.extend(_find_placeholders(val, child_path))
    elif isinstance(data, str) and re.search(r"<[^>]+>", data):
        results.append((path_prefix, data))
    return results


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------


def _discover_config_files() -> tuple[list[Path], list[Path]]:
    """Return (task_config_paths, executor_config_paths)."""
    task_configs: list[Path] = []
    for config_dir in get_config_dirs():
        if not config_dir.is_dir():
            continue
        for yaml_file in config_dir.glob("**/*.yaml"):
            if yaml_file.name == "executors.yaml":
                continue
            task_configs.append(yaml_file)

    executor_configs: list[Path] = []
    try:
        executor_configs.append(_find_executors_file())
    except FileNotFoundError:
        logger.debug("No executors.yaml found; skipping executor validation.")

    return task_configs, executor_configs


# ---------------------------------------------------------------------------
# Validation scanners
# ---------------------------------------------------------------------------


def _validate_task_config(file_path: Path, raw: dict) -> list[Diagnostic]:
    """Parse with TaskConfig; return ERROR diagnostics on validation failure."""
    diagnostics: list[Diagnostic] = []
    try:
        TaskConfig(**raw)
    except ValidationError as exc:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                file_path=str(file_path),
                rule_id="SCHEMA",
                message=f"Task config validation error: {exc}",
            )
        )
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                file_path=str(file_path),
                rule_id="SCHEMA",
                message=f"Unexpected error validating task config: {exc}",
            )
        )
    return diagnostics


def _validate_executor_entry(file_path: Path, name: str, raw: dict) -> list[Diagnostic]:
    """Parse with ExecutorEntry; return ERROR diagnostics on validation failure."""
    diagnostics: list[Diagnostic] = []
    try:
        ExecutorEntry(**raw)
    except ValidationError as exc:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                file_path=str(file_path),
                rule_id="SCHEMA",
                message=f"Executor '{name}' validation error: {exc}",
            )
        )
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                file_path=str(file_path),
                rule_id="SCHEMA",
                message=f"Unexpected error validating executor '{name}': {exc}",
            )
        )
    return diagnostics


def _check_cross_references(
    file_path: Path, raw: dict, executor_names: set[str]
) -> list[Diagnostic]:
    """Check that referenced task and executor names actually exist."""
    diagnostics: list[Diagnostic] = []
    if "task" in raw:
        known_tasks = set(list_tasks())
        if raw["task"] not in known_tasks:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    file_path=str(file_path),
                    rule_id="XREF",
                    message=f"Task '{raw['task']}' is not a registered task plugin.",
                )
            )
    if "executor" in raw:
        if raw["executor"] not in executor_names:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    file_path=str(file_path),
                    rule_id="XREF",
                    message=f"Executor '{raw['executor']}' is not defined in executors.yaml.",
                )
            )
    return diagnostics


def _check_executor_types(
    file_path: Path, name: str, raw: dict
) -> list[Diagnostic]:
    """Check that the executor's ``type`` is a registered executor plugin."""
    diagnostics: list[Diagnostic] = []
    exec_type = raw.get("type")
    if exec_type is not None:
        known = set(list_executors())
        if exec_type not in known:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    file_path=str(file_path),
                    rule_id="TYPE",
                    message=f"Executor '{name}' has unknown type '{exec_type}'. Known types: {sorted(known)}",
                )
            )
    return diagnostics


def _check_deprecations(
    file_path: Path, data: dict, scope: ConfigScope
) -> list[Diagnostic]:
    """Check for deprecated config paths based on DEPRECATION_RULES."""
    diagnostics: list[Diagnostic] = []
    for rule in DEPRECATION_RULES:
        if rule.scope not in (scope, ConfigScope.BOTH):
            continue
        old_val = _get_nested(data, rule.old_path)
        if old_val is not None:
            msg = rule.description
            if rule.removed_in:
                msg += f" (will be removed in {rule.removed_in})"
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    file_path=str(file_path),
                    rule_id=rule.rule_id,
                    message=msg,
                )
            )
    return diagnostics


def _check_placeholders_for_file(
    file_path: Path, data: dict
) -> list[Diagnostic]:
    """Find unfilled ``<placeholder>`` values in the config."""
    diagnostics: list[Diagnostic] = []
    for dotpath, value in _find_placeholders(data):
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                file_path=str(file_path),
                rule_id="PLACEHOLDER",
                message=f"Unfilled placeholder at '{dotpath}': {value}",
            )
        )
    return diagnostics


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def _apply_fixes(
    file_path: Path,
    triggered_rules: list[DeprecationRule],
    *,
    entry_names: list[str] | None = None,
) -> list[Diagnostic]:
    """Apply deprecation auto-fixes using ruamel.yaml to preserve comments.

    When *entry_names* is provided (e.g. for ``executors.yaml``), the rules
    are applied within each named sub-dict rather than at the document root.
    """
    from ruamel.yaml import YAML

    diagnostics: list[Diagnostic] = []

    if not triggered_rules:
        return diagnostics

    backup_path = file_path.with_suffix(".yaml.bak")

    if backup_path.exists():
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                file_path=str(file_path),
                rule_id="FIX",
                message=f"Backup already exists at {backup_path}; skipping auto-fix.",
            )
        )
        return diagnostics

    yaml_rt = YAML(typ="rt")
    with open(file_path) as fh:
        doc = yaml_rt.load(fh)

    if doc is None:
        return diagnostics

    # Create backup before making changes
    shutil.copy2(file_path, backup_path)

    # Determine the sub-dicts to fix: per-entry for executors, or root for tasks
    targets: list[tuple[str, dict]] = []
    if entry_names:
        for name in entry_names:
            if name in doc and isinstance(doc[name], dict):
                targets.append((name, doc[name]))
    else:
        targets.append(("", doc))

    for target_name, target_data in targets:
        for rule in triggered_rules:
            old_val = _get_nested(target_data, rule.old_path)
            if old_val is None:
                continue

            if rule.transform is not None:
                old_val = rule.transform(old_val)

            existing = _get_nested(target_data, rule.new_path)
            if isinstance(existing, list) and isinstance(old_val, list):
                existing.extend(old_val)
            else:
                _set_nested(target_data, rule.new_path, old_val)

            _del_nested(target_data, rule.old_path)

            prefix = f"[{target_name}] " if target_name else ""
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    file_path=str(file_path),
                    rule_id=rule.rule_id,
                    message=f"{prefix}Migrated '{rule.old_path}' → '{rule.new_path}'",
                    fix_applied=True,
                )
            )

    with open(file_path, "w") as fh:
        yaml_rt.dump(doc, fh)

    return diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_doctor(*, fix: bool = False, verbose: bool = False) -> DoctorReport:
    """Run all diagnostic checks and optionally apply auto-fixes."""
    report = DoctorReport()
    task_configs, executor_configs = _discover_config_files()

    # Collect executor names for cross-reference checks
    executor_names: set[str] = set()

    # --- Executor configs ---------------------------------------------------
    for exec_path in executor_configs:
        try:
            with open(exec_path) as fh:
                raw_all: dict = yaml.safe_load(fh) or {}
        except Exception as exc:
            report.diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    file_path=str(exec_path),
                    rule_id="LOAD",
                    message=f"Failed to load YAML: {exc}",
                )
            )
            continue

        triggered_rules: list[DeprecationRule] = []
        affected_entries: list[str] = []

        for name, entry_data in raw_all.items():
            if not isinstance(entry_data, dict):
                continue
            executor_names.add(name)
            report.diagnostics.extend(_validate_executor_entry(exec_path, name, entry_data))
            report.diagnostics.extend(_check_executor_types(exec_path, name, entry_data))
            report.diagnostics.extend(
                _check_deprecations(exec_path, entry_data, ConfigScope.EXECUTORS)
            )
            report.diagnostics.extend(_check_placeholders_for_file(exec_path, entry_data))

            # Track triggered deprecation rules for auto-fix
            for rule in DEPRECATION_RULES:
                if rule.scope in (ConfigScope.EXECUTORS, ConfigScope.BOTH):
                    if _get_nested(entry_data, rule.old_path) is not None:
                        if rule not in triggered_rules:
                            triggered_rules.append(rule)
                        if name not in affected_entries:
                            affected_entries.append(name)

        if fix and triggered_rules:
            report.diagnostics.extend(
                _apply_fixes(exec_path, triggered_rules, entry_names=affected_entries)
            )

    # --- Task configs -------------------------------------------------------
    for task_path in task_configs:
        try:
            with open(task_path) as fh:
                raw: dict = yaml.safe_load(fh) or {}
        except Exception as exc:
            report.diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    file_path=str(task_path),
                    rule_id="LOAD",
                    message=f"Failed to load YAML: {exc}",
                )
            )
            continue

        report.diagnostics.extend(_validate_task_config(task_path, raw))
        report.diagnostics.extend(_check_cross_references(task_path, raw, executor_names))
        report.diagnostics.extend(
            _check_deprecations(task_path, raw, ConfigScope.TASK)
        )
        report.diagnostics.extend(_check_placeholders_for_file(task_path, raw))

        # Auto-fix for task-scoped deprecation rules
        triggered_task_rules: list[DeprecationRule] = []
        for rule in DEPRECATION_RULES:
            if rule.scope in (ConfigScope.TASK, ConfigScope.BOTH):
                if _get_nested(raw, rule.old_path) is not None:
                    triggered_task_rules.append(rule)

        if fix and triggered_task_rules:
            report.diagnostics.extend(_apply_fixes(task_path, triggered_task_rules))

    return report
