"""Unit tests for devrun.presets — PresetStore + OmegaConf resolver."""
from __future__ import annotations

import pytest
import yaml
from omegaconf import OmegaConf

from devrun.presets import PresetStore, _validate_name


# ============================================================================
# _validate_name
# ============================================================================


class TestValidateName:
    def test_valid_alphanumeric(self):
        _validate_name("myPreset123")  # should not raise

    def test_valid_with_underscore_and_dash(self):
        _validate_name("my_preset-1")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError, match="Invalid preset name"):
            _validate_name("has space")

    def test_invalid_dot(self):
        with pytest.raises(ValueError, match="Invalid preset name"):
            _validate_name("dotted.name")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Invalid preset name"):
            _validate_name("")


# ============================================================================
# PresetStore basics
# ============================================================================


class TestPresetStoreBasics:
    def test_set_and_get_roundtrip(self, tmp_presets):
        tmp_presets.set("model", "gpt4", {"temperature": 0.7})
        assert tmp_presets.get("model", "gpt4") == {"temperature": 0.7}

    def test_get_missing_field_raises_key_error(self, tmp_presets):
        with pytest.raises(KeyError, match="nosuchfield.nosuchname"):
            tmp_presets.get("nosuchfield", "nosuchname")

    def test_get_missing_name_raises_key_error(self, tmp_presets):
        tmp_presets.set("model", "exists", "value")
        with pytest.raises(KeyError, match="model.missing"):
            tmp_presets.get("model", "missing")

    def test_set_overwrites_existing(self, tmp_presets):
        tmp_presets.set("model", "gpt4", "old")
        tmp_presets.set("model", "gpt4", "new")
        assert tmp_presets.get("model", "gpt4") == "new"

    def test_delete_removes_preset(self, tmp_presets):
        tmp_presets.set("model", "ephemeral", "value")
        tmp_presets.delete("model", "ephemeral")
        with pytest.raises(KeyError):
            tmp_presets.get("model", "ephemeral")

    def test_delete_missing_raises_key_error(self, tmp_presets):
        with pytest.raises(KeyError, match="ghost.missing"):
            tmp_presets.delete("ghost", "missing")

    def test_delete_cleans_empty_field(self, tmp_presets):
        tmp_presets.set("cleanup", "only", "val")
        tmp_presets.delete("cleanup", "only")
        # The field dict should be gone entirely
        raw = yaml.safe_load(tmp_presets._path.read_text()) or {}
        assert "cleanup" not in raw

    def test_delete_keeps_sibling_names(self, tmp_presets):
        tmp_presets.set("field", "a", 1)
        tmp_presets.set("field", "b", 2)
        tmp_presets.delete("field", "a")
        assert tmp_presets.get("field", "b") == 2


# ============================================================================
# list_presets
# ============================================================================


class TestListPresets:
    def test_list_empty(self, tmp_presets):
        assert tmp_presets.list_presets() == {}

    def test_list_all_sorted(self, tmp_presets):
        tmp_presets.set("model", "zulu", "z")
        tmp_presets.set("model", "alpha", "a")
        tmp_presets.set("executor", "beta", "b")
        result = tmp_presets.list_presets()
        assert result == {
            "executor": ["beta"],
            "model": ["alpha", "zulu"],
        }

    def test_list_filtered_by_field(self, tmp_presets):
        tmp_presets.set("model", "gpt4", "m1")
        tmp_presets.set("executor", "local", "e1")
        result = tmp_presets.list_presets(field="model")
        assert result == {"model": ["gpt4"]}

    def test_list_filtered_missing_field(self, tmp_presets):
        assert tmp_presets.list_presets(field="nonexistent") == {}


# ============================================================================
# On-disk format
# ============================================================================


class TestOnDiskFormat:
    def test_values_stored_as_is(self, tmp_presets):
        tmp_presets.set("model", "gpt4", {"temperature": 0.7, "max_tokens": 100})
        raw = yaml.safe_load(tmp_presets._path.read_text())
        assert raw["model"]["gpt4"] == {"temperature": 0.7, "max_tokens": 100}

    def test_parent_dirs_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "presets.yaml"
        store = PresetStore(path=deep)
        store.set("f", "n", "v")
        assert deep.exists()

    def test_load_empty_file(self, tmp_presets):
        """An empty YAML file should behave like an empty store."""
        tmp_presets._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_presets._path.write_text("")
        assert tmp_presets.list_presets() == {}

    def test_load_corrupt_yaml(self, tmp_presets):
        """Non-dict YAML (e.g. a bare string) should behave like an empty store."""
        tmp_presets._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_presets._path.write_text("just a string\n")
        assert tmp_presets.list_presets() == {}


# ============================================================================
# Default path
# ============================================================================


class TestDefaultPath:
    def test_default_path_under_home(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
        store = PresetStore()
        assert store._path == fake_home / ".devrun" / "presets.yaml"


# ============================================================================
# Name validation through set()
# ============================================================================


class TestSetValidation:
    def test_set_rejects_invalid_field(self, tmp_presets):
        with pytest.raises(ValueError, match="Invalid preset name"):
            tmp_presets.set("bad field!", "name", "val")

    def test_set_rejects_invalid_name(self, tmp_presets):
        with pytest.raises(ValueError, match="Invalid preset name"):
            tmp_presets.set("field", "bad name!", "val")


# ============================================================================
# Various value types
# ============================================================================


class TestValueTypes:
    def test_string_value(self, tmp_presets):
        tmp_presets.set("f", "n", "hello")
        assert tmp_presets.get("f", "n") == "hello"

    def test_int_value(self, tmp_presets):
        tmp_presets.set("f", "n", 42)
        assert tmp_presets.get("f", "n") == 42

    def test_float_value(self, tmp_presets):
        tmp_presets.set("f", "n", 3.14)
        assert tmp_presets.get("f", "n") == pytest.approx(3.14)

    def test_list_value(self, tmp_presets):
        tmp_presets.set("f", "n", [1, 2, 3])
        assert tmp_presets.get("f", "n") == [1, 2, 3]

    def test_dict_value(self, tmp_presets):
        val = {"batch_size": 16, "lr": 0.001}
        tmp_presets.set("params", "training", val)
        assert tmp_presets.get("params", "training") == val

    def test_nested_dict_value(self, tmp_presets):
        val = {"model": {"name": "gpt4", "args": {"temp": 0.5}}}
        tmp_presets.set("configs", "deep", val)
        assert tmp_presets.get("configs", "deep") == val

    def test_bool_value(self, tmp_presets):
        tmp_presets.set("flags", "debug", True)
        assert tmp_presets.get("flags", "debug") is True

    def test_none_value(self, tmp_presets):
        tmp_presets.set("f", "n", None)
        assert tmp_presets.get("f", "n") is None


# ============================================================================
# OmegaConf resolver
# ============================================================================


class TestOmegaConfResolver:
    def test_resolver_resolves_stored_preset(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore()
        store.set("model", "gpt4", "gpt-4-turbo")

        from devrun.presets import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"model_name": "${preset:model,gpt4}"})
        assert OmegaConf.to_container(cfg, resolve=True)["model_name"] == "gpt-4-turbo"

    def test_resolver_missing_preset_raises(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home2"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        from omegaconf.errors import InterpolationResolutionError

        from devrun.presets import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"val": "${preset:nosuch,missing}"})
        with pytest.raises(InterpolationResolutionError, match="nosuch.missing"):
            OmegaConf.to_container(cfg, resolve=True)

    def test_resolver_works_in_nested_config(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home3"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore()
        store.set("executor", "local", "local-gpu")

        from devrun.presets import register_resolver
        register_resolver()

        cfg = OmegaConf.create({
            "job": {"executor": "${preset:executor,local}", "workers": 4},
        })
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["job"]["executor"] == "local-gpu"
        assert resolved["job"]["workers"] == 4


# ============================================================================
# Multiple fields isolation
# ============================================================================


class TestFieldIsolation:
    def test_same_name_different_fields(self, tmp_presets):
        tmp_presets.set("model", "default", "gpt-4")
        tmp_presets.set("executor", "default", "local")
        assert tmp_presets.get("model", "default") == "gpt-4"
        assert tmp_presets.get("executor", "default") == "local"

    def test_delete_in_one_field_doesnt_affect_other(self, tmp_presets):
        tmp_presets.set("model", "shared", "m")
        tmp_presets.set("executor", "shared", "e")
        tmp_presets.delete("model", "shared")
        assert tmp_presets.get("executor", "shared") == "e"


# ============================================================================
# OmegaConf resolver — dict return type
# ============================================================================


class TestResolverDictReturn:
    def test_resolver_returns_dict(self, tmp_path, monkeypatch):
        """Verify OmegaConf resolver can return a dict (not just strings)."""
        fake_home = tmp_path / "oc_dict"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore()
        store.set("litellm_extra_body", "anthropic_thinking", {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "max"},
        })

        from devrun.presets import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"litellm_extra_body": "${preset:litellm_extra_body,anthropic_thinking}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["litellm_extra_body"] == {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "max"},
        }

    def test_resolver_in_llm_config_context(self, tmp_path, monkeypatch):
        """Verify preset works inside a realistic llm_config structure."""
        fake_home = tmp_path / "oc_llm"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore()
        store.set("model", "anthropic_claude", "anthropic/claude-opus-4-6-thinking-hz")
        store.set("litellm_extra_body", "anthropic_thinking", {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "max"},
            "tool_choice": {"type": "auto"},
        })

        from devrun.presets import register_resolver
        register_resolver()

        cfg = OmegaConf.create({
            "llm_config": {
                "model": "${preset:model,anthropic_claude}",
                "litellm_extra_body": "${preset:litellm_extra_body,anthropic_thinking}",
                "log_completions": True,
            },
        })
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["llm_config"]["model"] == "anthropic/claude-opus-4-6-thinking-hz"
        assert resolved["llm_config"]["litellm_extra_body"]["thinking"]["type"] == "adaptive"
        assert resolved["llm_config"]["log_completions"] is True


# ============================================================================
# CLI integration tests
# ============================================================================


class TestCLI:
    def test_cli_set_string(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "set", "model", "gpt4", "openai/gpt-4o"])
        assert result.exit_code == 0
        assert "stored" in result.output.lower()

        # Verify it was actually stored
        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        assert store.get("model", "gpt4") == "openai/gpt-4o"

    def test_cli_set_json(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, [
            "presets", "set", "litellm_extra_body", "thinking",
            "--json", '{"thinking": {"type": "adaptive"}}',
        ])
        assert result.exit_code == 0

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        assert store.get("litellm_extra_body", "thinking") == {"thinking": {"type": "adaptive"}}

    def test_cli_set_file(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        yaml_file = tmp_path / "input.yaml"
        yaml_file.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))

        runner, app = cli_runner
        result = runner.invoke(app, [
            "presets", "set", "configs", "myconfig",
            "--file", str(yaml_file),
        ])
        assert result.exit_code == 0

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        assert store.get("configs", "myconfig") == {"key": "value", "nested": {"a": 1}}

    def test_cli_set_no_value_errors(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "set", "model", "name"])
        assert result.exit_code != 0

    def test_cli_get(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        store.set("model", "gpt4", "openai/gpt-4o")

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "get", "model", "gpt4"])
        assert result.exit_code == 0
        assert "openai/gpt-4o" in result.output

    def test_cli_get_missing(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "get", "model", "nope"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_cli_list(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        store.set("model", "gpt4", "openai/gpt-4o")
        store.set("executor", "local", "local-gpu")

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "list"])
        assert result.exit_code == 0
        assert "model" in result.output
        assert "gpt4" in result.output
        assert "executor" in result.output
        assert "local" in result.output

    def test_cli_list_empty(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "list"])
        assert result.exit_code == 0
        assert "no presets" in result.output.lower()

    def test_cli_list_filtered(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        store.set("model", "gpt4", "m1")
        store.set("executor", "local", "e1")

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "list", "model"])
        assert result.exit_code == 0
        assert "gpt4" in result.output
        assert "local" not in result.output

    def test_cli_delete(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = PresetStore(path=fake_home / ".devrun" / "presets.yaml")
        store.set("model", "gpt4", "m1")

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "delete", "model", "gpt4"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

        with pytest.raises(KeyError):
            store.get("model", "gpt4")

    def test_cli_delete_missing(self, cli_runner, tmp_path, monkeypatch):
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        runner, app = cli_runner
        result = runner.invoke(app, ["presets", "delete", "model", "nope"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
