"""Unit tests for SWE-bench shared utilities."""
from __future__ import annotations

import pytest
from devrun.utils.swebench import derive_ds_dir


class TestDeriveDsDir:
    def test_absolute_path_with_split(self):
        result = derive_ds_dir("/mnt/huawei/users/lfu/datasets/SWE-bench_Verified", "test")
        assert result == "__mnt__huawei__users__lfu__datasets__SWE-bench_Verified-test"

    def test_leading_slash_becomes_double_underscore(self):
        result = derive_ds_dir("/data/foo", "dev")
        assert result.startswith("__data")

    def test_relative_path(self):
        result = derive_ds_dir("datasets/SWE-bench", "test")
        assert result == "datasets__SWE-bench-test"

    def test_trailing_slash_stripped(self):
        result = derive_ds_dir("/data/foo/", "test")
        assert result == "__data__foo__-test"

    def test_split_appended_with_dash(self):
        result = derive_ds_dir("/data", "validation")
        assert result.endswith("-validation")

    def test_empty_split(self):
        result = derive_ds_dir("/data/foo", "")
        assert result == "__data__foo-"
