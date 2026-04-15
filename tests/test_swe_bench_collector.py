"""Runtime tests for the swe_bench_collect.py.j2 collector script.

These tests verify the *runtime behaviour* of the generated Python collector
script, complementing ``test_swe_bench_collect.py`` which covers the task's
``prepare()`` output (command structure and config injection).

Strategy: render the Jinja2 template with known values, ``exec()`` the result
into a namespace, and exercise each function against mock directory trees built
under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from devrun.utils.templates import render_template


# ============================================================================
# Helpers
# ============================================================================

# Default template variables used across tests.
_DEFAULTS: dict[str, Any] = {
    "output_dir": "/placeholder",  # overridden per-test
    "ds_dir": "__mnt__data__SWE-bench_Verified-test",
    "model_name_or_path": "test-model",
    "predictions_path": "/placeholder/predictions.jsonl",
    "histories_path": "/placeholder/collected_histories.jsonl",
    "max_workers": 2,
}


def _render_collector(**overrides: Any) -> str:
    """Render the collector template with test defaults + overrides."""
    ctx = {**_DEFAULTS, **overrides}
    return render_template("swe_bench_collect.py.j2", **ctx)


def _load_collector(source: str) -> ModuleType:
    """Exec rendered collector source into a pseudo-module for function access."""
    ns: dict[str, Any] = {}
    exec(compile(source, "<collector>", "exec"), ns)
    mod = ModuleType("collector")
    mod.__dict__.update(ns)
    return mod


def _make_output_jsonl(instance_id: str, git_patch: str) -> str:
    """Build a valid output.jsonl body."""
    return json.dumps({
        "instance_id": instance_id,
        "test_result": {"git_patch": git_patch},
    })


def _build_instance_dir(
    base: Path,
    shard: str,
    ds_dir: str,
    provider: str,
    instance: str,
    *,
    output_content: str | None = None,
    history_files: dict[str, Any] | None = None,
) -> Path:
    """Create a mock leaf directory matching the collector's expected layout.

    ``output_dir / shard / ds_dir / provider / instance /``
    """
    leaf = base / shard / ds_dir / provider / instance
    leaf.mkdir(parents=True, exist_ok=True)
    if output_content is not None:
        (leaf / "output.jsonl").write_text(output_content, encoding="utf-8")
    if history_files:
        for name, data in history_files.items():
            (leaf / name).write_text(json.dumps(data), encoding="utf-8")
    return leaf


# ============================================================================
# Fixtures
# ============================================================================

DS_DIR = _DEFAULTS["ds_dir"]


@pytest.fixture()
def collector(tmp_path):
    """Return a loaded collector module with paths pointing at tmp_path."""
    pred_path = str(tmp_path / "predictions.jsonl")
    hist_path = str(tmp_path / "collected_histories.jsonl")
    src = _render_collector(
        output_dir=str(tmp_path / "outputs"),
        predictions_path=pred_path,
        histories_path=hist_path,
    )
    return _load_collector(src)


@pytest.fixture()
def output_root(tmp_path):
    """The mock output directory root."""
    root = tmp_path / "outputs"
    root.mkdir()
    return root


# ============================================================================
# 1. discover_candidates()
# ============================================================================


class TestDiscoverCandidates:
    def test_finds_leaf_dirs(self, collector, output_root):
        _build_instance_dir(output_root, "000", DS_DIR, "openai", "instance_1")
        _build_instance_dir(output_root, "000", DS_DIR, "openai", "instance_2")
        _build_instance_dir(output_root, "001", DS_DIR, "openai", "instance_3")

        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        names = sorted(os.path.basename(c) for c in candidates)
        assert names == ["instance_1", "instance_2", "instance_3"]

    def test_ignores_files_at_shard_level(self, collector, output_root):
        """Files directly under output_root should not be treated as shard dirs."""
        (output_root / "stray_file.txt").write_text("noise")
        _build_instance_dir(output_root, "000", DS_DIR, "openai", "inst_ok")

        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        assert len(candidates) == 1

    def test_missing_ds_dir_level(self, collector, output_root):
        """Shard exists but has no matching ds_dir sub-directory."""
        (output_root / "000").mkdir()
        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        assert candidates == []

    def test_empty_output_dir(self, collector, output_root):
        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        assert candidates == []

    def test_nonexistent_output_dir(self, collector, tmp_path):
        candidates = collector.discover_candidates(
            str(tmp_path / "does_not_exist"), DS_DIR
        )
        assert candidates == []

    def test_multiple_providers(self, collector, output_root):
        """Multiple provider directories under the same shard/ds_dir."""
        _build_instance_dir(output_root, "000", DS_DIR, "openai", "inst_a")
        _build_instance_dir(output_root, "000", DS_DIR, "anthropic", "inst_b")

        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        names = sorted(os.path.basename(c) for c in candidates)
        assert names == ["inst_a", "inst_b"]

    def test_symlinks_not_followed(self, collector, output_root):
        """Symlinks at shard level should be skipped (follow_symlinks=False)."""
        real = _build_instance_dir(output_root, "real", DS_DIR, "p", "inst")
        (output_root / "linked").symlink_to(output_root / "real")

        candidates = collector.discover_candidates(str(output_root), DS_DIR)
        # Only the real shard's instance should be found
        assert len(candidates) == 1
        assert "real" in candidates[0]


# ============================================================================
# 2. _check_output()
# ============================================================================


class TestCheckOutput:
    def test_valid_file(self, collector, output_root):
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst",
            output_content=_make_output_jsonl("inst", "patch"),
        )
        result = collector._check_output(str(leaf))
        assert result is not None
        dir_path, output_path = result
        assert dir_path == str(leaf)
        assert output_path.endswith("output.jsonl")

    def test_missing_file(self, collector, output_root):
        leaf = _build_instance_dir(output_root, "000", DS_DIR, "p", "inst")
        result = collector._check_output(str(leaf))
        assert result is None

    def test_empty_file(self, collector, output_root):
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=""
        )
        result = collector._check_output(str(leaf))
        assert result is None

    def test_nonexistent_dir(self, collector, tmp_path):
        result = collector._check_output(str(tmp_path / "ghost"))
        assert result is None


# ============================================================================
# 3. _extract_prediction()
# ============================================================================


class TestExtractPrediction:
    def test_valid_extraction(self, collector, output_root):
        content = _make_output_jsonl("django__django-12345", "diff --git ...")
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, dir_path = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is not None
        assert pred["instance_id"] == "django__django-12345"
        assert pred["model_name_or_path"] == "test-model"
        assert pred["model_patch"] == "diff --git ..."
        assert dir_path == str(leaf)

    def test_missing_instance_id(self, collector, output_root):
        content = json.dumps({"test_result": {"git_patch": "p"}})
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None

    def test_missing_git_patch(self, collector, output_root):
        content = json.dumps({"instance_id": "id1", "test_result": {}})
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None

    def test_null_test_result(self, collector, output_root):
        content = json.dumps({"instance_id": "id1", "test_result": None})
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None

    def test_malformed_json(self, collector, output_root):
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst",
            output_content="not valid json{{{",
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None

    def test_file_read_error(self, collector, tmp_path):
        """Non-existent output path should return None gracefully."""
        fake_dir = str(tmp_path / "gone")
        pred, dir_path = collector._extract_prediction(
            (fake_dir, str(tmp_path / "gone" / "output.jsonl"))
        )
        assert pred is None
        assert dir_path == fake_dir

    def test_empty_instance_id(self, collector, output_root):
        """Empty string instance_id should be treated as missing."""
        content = json.dumps({
            "instance_id": "",
            "test_result": {"git_patch": "patch"},
        })
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None

    def test_empty_git_patch(self, collector, output_root):
        """Empty string git_patch should be treated as missing."""
        content = json.dumps({
            "instance_id": "id1",
            "test_result": {"git_patch": ""},
        })
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", output_content=content
        )
        pred, _ = collector._extract_prediction(
            (str(leaf), str(leaf / "output.jsonl"))
        )
        assert pred is None


# ============================================================================
# 4. _collect_histories()
# ============================================================================


class TestCollectHistories:
    def test_collects_history_files(self, collector, output_root):
        histories = {
            "run1.history.json": {"steps": [1, 2, 3]},
            "run2.history.json": {"steps": [4, 5]},
        }
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", history_files=histories
        )
        result = collector._collect_histories(str(leaf))
        assert len(result) == 2
        sources = sorted(h["source_file"] for h in result)
        assert any("run1.history.json" in s for s in sources)
        assert any("run2.history.json" in s for s in sources)
        for h in result:
            assert "history" in h
            assert isinstance(h["history"], dict)

    def test_no_history_files(self, collector, output_root):
        leaf = _build_instance_dir(output_root, "000", DS_DIR, "p", "inst")
        result = collector._collect_histories(str(leaf))
        assert result == []

    def test_malformed_history_json_skipped(self, collector, output_root):
        leaf = _build_instance_dir(output_root, "000", DS_DIR, "p", "inst")
        (leaf / "bad.history.json").write_text("{{not json}}", encoding="utf-8")
        (leaf / "good.history.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
        result = collector._collect_histories(str(leaf))
        assert len(result) == 1
        assert result[0]["history"] == {"ok": True}

    def test_non_history_json_ignored(self, collector, output_root):
        """Files that don't end with .history.json should be ignored."""
        leaf = _build_instance_dir(output_root, "000", DS_DIR, "p", "inst")
        (leaf / "output.jsonl").write_text("{}", encoding="utf-8")
        (leaf / "notes.json").write_text("{}", encoding="utf-8")
        (leaf / "real.history.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
        result = collector._collect_histories(str(leaf))
        assert len(result) == 1

    def test_nonexistent_dir(self, collector, tmp_path):
        result = collector._collect_histories(str(tmp_path / "nowhere"))
        assert result == []

    def test_source_file_is_absolute_path(self, collector, output_root):
        histories = {"trace.history.json": {"t": 1}}
        leaf = _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst", history_files=histories
        )
        result = collector._collect_histories(str(leaf))
        assert len(result) == 1
        assert os.path.isabs(result[0]["source_file"])


# ============================================================================
# 5. End-to-end: render script, run as subprocess, verify outputs
# ============================================================================


class TestEndToEnd:
    """Run the full collector script as a subprocess and verify file outputs."""

    def _run_collector(self, tmp_path: Path, output_root: Path) -> subprocess.CompletedProcess:
        pred_path = str(tmp_path / "predictions.jsonl")
        hist_path = str(tmp_path / "collected_histories.jsonl")
        src = _render_collector(
            output_dir=str(output_root),
            predictions_path=pred_path,
            histories_path=hist_path,
        )
        script = tmp_path / "collector.py"
        script.write_text(src, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_basic_collection(self, tmp_path):
        output_root = tmp_path / "outputs"
        output_root.mkdir()
        _build_instance_dir(
            output_root, "000", DS_DIR, "openai", "django__django-1",
            output_content=_make_output_jsonl("django__django-1", "patch1"),
            history_files={"a.history.json": {"steps": [1]}},
        )
        _build_instance_dir(
            output_root, "001", DS_DIR, "openai", "flask__flask-2",
            output_content=_make_output_jsonl("flask__flask-2", "patch2"),
        )

        result = self._run_collector(tmp_path, output_root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Verify predictions.jsonl
        pred_path = tmp_path / "predictions.jsonl"
        assert pred_path.exists()
        preds = [json.loads(line) for line in pred_path.read_text().strip().splitlines()]
        assert len(preds) == 2
        ids = sorted(p["instance_id"] for p in preds)
        assert ids == ["django__django-1", "flask__flask-2"]
        for p in preds:
            assert p["model_name_or_path"] == "test-model"
            assert p["model_patch"] in ("patch1", "patch2")

        # Verify collected_histories.jsonl
        hist_path = tmp_path / "collected_histories.jsonl"
        assert hist_path.exists()
        histories = [json.loads(line) for line in hist_path.read_text().strip().splitlines()]
        assert len(histories) == 1
        assert histories[0]["history"] == {"steps": [1]}

    def test_mixed_valid_and_invalid(self, tmp_path):
        """Instances with missing fields should be skipped, valid ones collected."""
        output_root = tmp_path / "outputs"
        output_root.mkdir()
        # Valid
        _build_instance_dir(
            output_root, "000", DS_DIR, "p", "good",
            output_content=_make_output_jsonl("good-id", "good-patch"),
        )
        # Missing git_patch
        _build_instance_dir(
            output_root, "000", DS_DIR, "p", "bad_patch",
            output_content=json.dumps({"instance_id": "bad", "test_result": {}}),
        )
        # Empty output.jsonl
        _build_instance_dir(
            output_root, "000", DS_DIR, "p", "empty_out",
            output_content="",
        )
        # Malformed JSON
        _build_instance_dir(
            output_root, "000", DS_DIR, "p", "bad_json",
            output_content="not json!",
        )

        result = self._run_collector(tmp_path, output_root)
        assert result.returncode == 0

        preds = [
            json.loads(l) for l in
            (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
        ]
        assert len(preds) == 1
        assert preds[0]["instance_id"] == "good-id"

    def test_stdout_summary(self, tmp_path):
        """Script should print collection summary to stdout."""
        output_root = tmp_path / "outputs"
        output_root.mkdir()
        _build_instance_dir(
            output_root, "000", DS_DIR, "p", "inst",
            output_content=_make_output_jsonl("id1", "p"),
            history_files={"h.history.json": {"x": 1}},
        )

        result = self._run_collector(tmp_path, output_root)
        assert "Collected 1 predictions" in result.stdout
        assert "Collected 1 history files" in result.stdout

    def test_many_instances(self, tmp_path):
        """Verify parallel processing works with many candidates."""
        output_root = tmp_path / "outputs"
        output_root.mkdir()
        n = 50
        for i in range(n):
            shard = f"{i // 10:03d}"
            _build_instance_dir(
                output_root, shard, DS_DIR, "p", f"inst_{i:04d}",
                output_content=_make_output_jsonl(f"id_{i}", f"patch_{i}"),
            )

        result = self._run_collector(tmp_path, output_root)
        assert result.returncode == 0

        preds = [
            json.loads(l) for l in
            (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
        ]
        assert len(preds) == n


# ============================================================================
# 6. Empty directory: no candidates
# ============================================================================


class TestEmptyDirectory:
    def test_empty_output_produces_empty_files(self, tmp_path):
        """An empty output directory should produce empty output files."""
        output_root = tmp_path / "outputs"
        output_root.mkdir()

        pred_path = str(tmp_path / "predictions.jsonl")
        hist_path = str(tmp_path / "collected_histories.jsonl")
        src = _render_collector(
            output_dir=str(output_root),
            predictions_path=pred_path,
            histories_path=hist_path,
        )
        script = tmp_path / "collector.py"
        script.write_text(src, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

        assert Path(pred_path).exists()
        assert Path(pred_path).read_text() == ""
        assert Path(hist_path).exists()
        assert Path(hist_path).read_text() == ""
        assert "Collected 0 predictions" in result.stdout

    def test_empty_with_no_matching_ds_dir(self, tmp_path):
        """Shard dirs exist but none contain the expected ds_dir."""
        output_root = tmp_path / "outputs"
        (output_root / "000" / "wrong_ds_dir").mkdir(parents=True)
        (output_root / "001" / "also_wrong").mkdir(parents=True)

        pred_path = str(tmp_path / "predictions.jsonl")
        hist_path = str(tmp_path / "collected_histories.jsonl")
        src = _render_collector(
            output_dir=str(output_root),
            predictions_path=pred_path,
            histories_path=hist_path,
        )
        script = tmp_path / "collector.py"
        script.write_text(src, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert Path(pred_path).read_text() == ""


# ============================================================================
# 7. Template rendering integrity
# ============================================================================


class TestTemplateRendering:
    def test_special_chars_in_model_name(self, tmp_path):
        """Model name with quotes/special chars should render safely."""
        src = _render_collector(
            output_dir=str(tmp_path),
            model_name_or_path='model "with" quotes & $pecial',
            predictions_path=str(tmp_path / "p.jsonl"),
            histories_path=str(tmp_path / "h.jsonl"),
        )
        ns: dict[str, Any] = {}
        exec(compile(src, "<test>", "exec"), ns)
        assert ns["MODEL_NAME_OR_PATH"] == 'model "with" quotes & $pecial'

    def test_path_with_spaces(self, tmp_path):
        """Paths containing spaces should render correctly."""
        spaced = tmp_path / "path with spaces"
        spaced.mkdir()
        src = _render_collector(
            output_dir=str(spaced),
            predictions_path=str(spaced / "pred.jsonl"),
            histories_path=str(spaced / "hist.jsonl"),
        )
        ns: dict[str, Any] = {}
        exec(compile(src, "<test>", "exec"), ns)
        assert ns["OUTPUT_DIR"] == str(spaced)

    def test_max_workers_integer(self, tmp_path):
        """max_workers should be rendered as a bare integer, not a string."""
        src = _render_collector(
            output_dir=str(tmp_path),
            predictions_path=str(tmp_path / "p.jsonl"),
            histories_path=str(tmp_path / "h.jsonl"),
            max_workers=32,
        )
        ns: dict[str, Any] = {}
        exec(compile(src, "<test>", "exec"), ns)
        assert ns["MAX_WORKERS"] == 32
        assert isinstance(ns["MAX_WORKERS"], int)

    def test_strict_undefined_raises(self):
        """Missing template variables should raise, not silently produce ''."""
        with pytest.raises(Exception):
            # Deliberately omit required variable
            render_template(
                "swe_bench_collect.py.j2",
                output_dir="/x",
                ds_dir="d",
                # model_name_or_path missing
                predictions_path="p",
                histories_path="h",
                max_workers=1,
            )
