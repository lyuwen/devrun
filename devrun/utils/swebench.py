"""Shared SWE-bench utilities used by multiple task plugins."""
from __future__ import annotations


def derive_ds_dir(dataset: str, split: str) -> str:
    """Derive the DS_DIR directory name from a dataset path and split.

    The OpenHands inference runner creates output directories using this
    naming convention. Both the agentic inference task and the collect
    task must use the same derivation to ensure path consistency.

    Example::

        derive_ds_dir("/mnt/huawei/users/lfu/datasets/SWE-bench_Verified", "test")
        # → "__mnt__huawei__users__lfu__datasets__SWE-bench_Verified-test"

    Note: A leading ``/`` becomes a leading ``__``.
    """
    return dataset.replace("/", "__") + "-" + split
