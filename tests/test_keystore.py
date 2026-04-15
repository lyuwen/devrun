"""Unit tests for devrun.keystore — KeyStore + OmegaConf resolver + CLI + runner integration."""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from typer.testing import CliRunner

from devrun.cli import app
from devrun.keystore import KeyStore, _validate_name


# ============================================================================
# _validate_name
# ============================================================================


class TestValidateName:
    def test_valid_alphanumeric(self):
        _validate_name("myKey123")  # should not raise

    def test_valid_with_underscore_and_dash(self):
        _validate_name("my_key-1")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError, match="Invalid key name"):
            _validate_name("has space")

    def test_invalid_dot(self):
        with pytest.raises(ValueError, match="Invalid key name"):
            _validate_name("dotted.name")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Invalid key name"):
            _validate_name("")


# ============================================================================
# KeyStore basics
# ============================================================================


class TestKeyStoreBasics:
    def test_set_and_get_roundtrip(self, tmp_keystore):
        tmp_keystore.set("api_key", "sk-secret123")
        assert tmp_keystore.get("api_key") == "sk-secret123"

    def test_get_missing_raises_key_error(self, tmp_keystore):
        with pytest.raises(KeyError):
            tmp_keystore.get("nonexistent")

    def test_set_overwrites_existing(self, tmp_keystore):
        tmp_keystore.set("tok", "old")
        tmp_keystore.set("tok", "new")
        assert tmp_keystore.get("tok") == "new"

    def test_delete_removes_key(self, tmp_keystore):
        tmp_keystore.set("ephemeral", "value")
        tmp_keystore.delete("ephemeral")
        with pytest.raises(KeyError):
            tmp_keystore.get("ephemeral")

    def test_delete_missing_raises_key_error(self, tmp_keystore):
        with pytest.raises(KeyError):
            tmp_keystore.delete("ghost")

    def test_list_keys_empty(self, tmp_keystore):
        assert tmp_keystore.list_keys() == []

    def test_list_keys_sorted(self, tmp_keystore):
        tmp_keystore.set("zulu", "z")
        tmp_keystore.set("alpha", "a")
        tmp_keystore.set("mike", "m")
        assert tmp_keystore.list_keys() == ["alpha", "mike", "zulu"]


# ============================================================================
# On-disk format and file permissions
# ============================================================================


class TestOnDiskFormat:
    def test_values_stored_as_base64(self, tmp_keystore):
        tmp_keystore.set("secret", "hello world")
        raw = yaml.safe_load(tmp_keystore._path.read_text())
        assert raw["secret"] == base64.b64encode(b"hello world").decode()

    def test_file_permissions_0600(self, tmp_keystore):
        tmp_keystore.set("perm_test", "val")
        mode = tmp_keystore._path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_parent_dirs_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "keys.yaml"
        store = KeyStore(path=deep)
        store.set("k", "v")
        assert deep.exists()

    def test_load_empty_file(self, tmp_keystore):
        """An empty YAML file should behave like an empty store."""
        tmp_keystore._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_keystore._path.write_text("")
        assert tmp_keystore.list_keys() == []

    def test_load_corrupt_yaml(self, tmp_keystore):
        """Non-dict YAML (e.g. a bare string) should behave like an empty store."""
        tmp_keystore._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_keystore._path.write_text("just a string\n")
        assert tmp_keystore.list_keys() == []


# ============================================================================
# Default path
# ============================================================================


class TestDefaultPath:
    def test_default_path_under_home(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))
        store = KeyStore()
        assert store._path == fake_home / ".devrun" / "keys.yaml"


# ============================================================================
# Name validation through set()
# ============================================================================


class TestSetValidation:
    def test_set_rejects_invalid_name(self, tmp_keystore):
        with pytest.raises(ValueError, match="Invalid key name"):
            tmp_keystore.set("bad name!", "val")


# ============================================================================
# Unicode and special values
# ============================================================================


class TestSpecialValues:
    def test_unicode_value(self, tmp_keystore):
        tmp_keystore.set("emoji", "hello-world")
        assert tmp_keystore.get("emoji") == "hello-world"

    def test_empty_value(self, tmp_keystore):
        tmp_keystore.set("blank", "")
        assert tmp_keystore.get("blank") == ""

    def test_multiline_value(self, tmp_keystore):
        val = "line1\nline2\nline3"
        tmp_keystore.set("multi", val)
        assert tmp_keystore.get("multi") == val


# ============================================================================
# OmegaConf resolver
# ============================================================================


class TestOmegaConfResolver:
    def test_resolver_resolves_stored_key(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = KeyStore()
        store.set("my_token", "resolved-secret")

        # Re-register so the resolver picks up the patched home
        from devrun.keystore import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"token": "${key:my_token}"})
        assert OmegaConf.to_container(cfg, resolve=True)["token"] == "resolved-secret"

    def test_resolver_missing_key_raises(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home2"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        from omegaconf.errors import InterpolationResolutionError

        from devrun.keystore import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"val": "${key:no_such_key}"})
        with pytest.raises(InterpolationResolutionError, match="no_such_key"):
            OmegaConf.to_container(cfg, resolve=True)

    def test_resolver_works_in_nested_config(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "oc_home3"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        store = KeyStore()
        store.set("db_pass", "p@ssw0rd")

        from devrun.keystore import register_resolver
        register_resolver()

        cfg = OmegaConf.create({"database": {"password": "${key:db_pass}", "host": "localhost"}})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["database"]["password"] == "p@ssw0rd"
        assert resolved["database"]["host"] == "localhost"


# ============================================================================
# Keys CLI sub-app
# ============================================================================


class TestKeysCLI:
    """Tests for `devrun keys set/get/list/delete` CLI commands."""

    @pytest.fixture(autouse=True)
    def _isolate_home(self, tmp_path, monkeypatch):
        """Redirect Path.home() so CLI KeyStore() doesn't touch real home."""
        fake_home = tmp_path / "cli_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

    def test_set_get_roundtrip(self):
        runner = CliRunner()
        result = runner.invoke(app, ["keys", "set", "tok", "my-secret"])
        assert result.exit_code == 0
        assert "stored" in result.stdout.lower()

        result = runner.invoke(app, ["keys", "get", "tok"])
        assert result.exit_code == 0
        assert "my-secret" in result.stdout

    def test_list_empty(self):
        runner = CliRunner()
        result = runner.invoke(app, ["keys", "list"])
        assert result.exit_code == 0
        assert "no keys" in result.stdout.lower()

    def test_list_with_keys(self):
        runner = CliRunner()
        runner.invoke(app, ["keys", "set", "beta", "b"])
        runner.invoke(app, ["keys", "set", "alpha", "a"])
        result = runner.invoke(app, ["keys", "list"])
        assert result.exit_code == 0
        assert "alpha" in result.stdout
        assert "beta" in result.stdout

    def test_delete(self):
        runner = CliRunner()
        runner.invoke(app, ["keys", "set", "gone", "val"])
        result = runner.invoke(app, ["keys", "delete", "gone"])
        assert result.exit_code == 0
        assert "deleted" in result.stdout.lower()

        result = runner.invoke(app, ["keys", "get", "gone"])
        assert result.exit_code == 1

    def test_get_nonexistent_exit_code(self):
        runner = CliRunner()
        result = runner.invoke(app, ["keys", "get", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_delete_nonexistent_exit_code(self):
        runner = CliRunner()
        result = runner.invoke(app, ["keys", "delete", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_set_invalid_name_exit_code(self):
        runner = CliRunner()
        result = runner.invoke(app, ["keys", "set", "bad name", "val"])
        assert result.exit_code == 1


# ============================================================================
# Runner key integration
# ============================================================================


class TestRunnerKeyIntegration:
    """Test that ${key:name} resolves through TaskRunner._load_config."""

    def test_load_config_resolves_key_placeholder(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "runner_home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        # Store a secret
        store = KeyStore()
        store.set("api_token", "sk-runner-test")

        # Re-register so resolver uses patched home
        from devrun.keystore import register_resolver
        register_resolver()

        # Create a minimal task config YAML that uses ${key:api_token}
        config_dir = tmp_path / "configs" / "dummy_task"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "default.yaml"
        config_file.write_text(yaml.dump({
            "task": "eval",
            "executor": "local",
            "params": {"token": "${key:api_token}", "dataset": "test"},
        }))

        from devrun.runner import TaskRunner
        runner = TaskRunner()
        # Point _load_config at our temp config via direct file path
        cfg = runner._load_config(str(config_file))
        assert cfg.params["token"] == "sk-runner-test"
        assert cfg.params["dataset"] == "test"
