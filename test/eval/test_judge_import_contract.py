"""Regression tests for judge-free task import in generation stage."""

import importlib
import sys
import types


TARGET_MODULES = (
    "lmms_eval.tasks.mmmu.utils",
    "lmms_eval.tasks.mathverse.mathverse_evals",
    "lmms_eval.tasks.mathverse.reasoning.utils",
    "lmms_eval.tasks.mathvista.mathvista_evals",
    "lmms_eval.tasks.mathvista.utils",
    "lmms_eval.tasks.mathvista.utils_qwen3",
)


def test_generation_task_import_does_not_construct_judge_clients(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_API_URL", "JUDGE_API_KEY", "JUDGE_BASE_URL", "API_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LMMS_EVAL_JUDGE_STAGE", "generation")

    import lmms_eval.llm_judge as llm_judge

    def fail_get_server(*_args, **_kwargs):
        raise AssertionError("judge server must not be constructed during task import")

    monkeypatch.setattr(llm_judge, "get_server", fail_get_server)

    fake_openai = types.ModuleType("openai")

    class RaisingOpenAI:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("OpenAI client must not be constructed during task import")

    fake_openai.OpenAI = RaisingOpenAI
    fake_openai.AzureOpenAI = RaisingOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    for module_name in TARGET_MODULES:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
