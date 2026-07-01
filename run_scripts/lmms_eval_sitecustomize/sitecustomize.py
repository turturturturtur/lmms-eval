"""Runtime guards for lmms-eval vLLM workers.

The DLC image can expose a system-level flash_attn build compiled against a
different torch/c10 ABI than the lmms-eval venv. vLLM probes flash_attn with
importlib.util.find_spec before importing rotary helpers. Hiding that package
lets vLLM fall back to its non-flash-attn rotary path while still using
FLASHINFER for attention.
"""

from __future__ import annotations

import importlib.util
import builtins
import os


_REAL_FIND_SPEC = importlib.util.find_spec
_REAL_IMPORT = builtins.__import__
_PATCHING_DATASETS = False


def _patch_datasets_list_feature() -> None:
    global _PATCHING_DATASETS
    if _PATCHING_DATASETS:
        return
    _PATCHING_DATASETS = True
    try:
        from datasets import Sequence
        from datasets.features import features

        features._FEATURE_TYPES.setdefault("List", Sequence)
    except Exception:
        pass
    finally:
        _PATCHING_DATASETS = False


def _innovator_find_spec(name: str, *args, **kwargs):
    if os.environ.get("INNOVATOR_LMMS_HIDE_FLASH_ATTN") == "1":
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
    return _REAL_FIND_SPEC(name, *args, **kwargs)


def _innovator_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _REAL_IMPORT(name, globals, locals, fromlist, level)
    if name == "datasets" or name.startswith("datasets."):
        _patch_datasets_list_feature()
    return module


importlib.util.find_spec = _innovator_find_spec
builtins.__import__ = _innovator_import
