import importlib
from pathlib import Path
import sys
import types

from lmms_eval.api.reasoning import strip_reasoning_tags


def test_strip_reasoning_tags_removes_paired_block():
    text = "<think>\nreasoning\n</think>\n\nYes"
    cleaned = strip_reasoning_tags(text, [["<think>", "</think>"]])
    assert cleaned == "Yes"


def test_strip_reasoning_tags_handles_prompt_prefilled_opening_tag():
    text = "reasoning from completion only\n</think>\n\nNo"
    cleaned = strip_reasoning_tags(text, [["<think>", "</think>"]])
    assert cleaned == "No"


def test_task_reasoning_utils_import_without_openai_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)

    fake_math_verify = types.ModuleType("math_verify")
    fake_math_verify.parse = lambda value: value
    fake_math_verify.verify = lambda _gold, _pred: False
    monkeypatch.setitem(sys.modules, "math_verify", fake_math_verify)

    class RaisingOpenAI:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("OpenAI client must not be created during import")

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = RaisingOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    module_path = Path(__file__).resolve().parents[2] / "lmms_eval/tasks/_task_utils/reasoning_utils.py"
    spec = importlib.util.spec_from_file_location("reasoning_utils_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._CLIENT is None
