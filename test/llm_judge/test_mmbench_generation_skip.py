"""A generation-only skip flag must not replace a completed judge score with zero."""
import pytest
from types import SimpleNamespace

@pytest.mark.parametrize('language', ['en', 'cn', 'ru', 'ko', 'cc'])
def test_standalone_mmbench_ignores_generation_skip(language, monkeypatch, tmp_path):
    import importlib
    module = importlib.import_module(f'lmms_eval.tasks.mmbench.{language}_utils')
    monkeypatch.setenv('SKIP_MMBENCH_DEV_JUDGE', '1')
    monkeypatch.setattr(module, '_refresh_mmbench_evaluator', lambda: None)
    monkeypatch.setattr(module, 'mmbench_evaluator', SimpleNamespace(eval_result=lambda *a, **kw: (.75, {}, {})))
    monkeypatch.setattr(module, 'generate_submission_file', lambda *args: str(tmp_path/'result.json'))
    prefix = 'mmbench_cn_cc' if language == 'cc' else 'mmbench'
    assert getattr(module, prefix + '_aggregate_dev_results_eval_standalone')([{'prediction': 'A'}]) == 75.0
    assert getattr(module, prefix + '_aggregate_dev_results_eval')([], SimpleNamespace()) == 0.0
