import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lmms_eval import evaluator
from lmms_eval.api.instance import Instance
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker


@pytest.mark.parametrize("raw,filtered", [([['A']], ['A']), ([['reasoning: A']], ['A']), ([['?']], [''])])
def test_sample_save_always_keeps_raw_output(tmp_path, raw, filtered):
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.model_name_sanitized = "fixture"
    tracker.date_id = "run"
    sample = {"doc_id": 0, "doc": {}, "arguments": ["question"], "target": "A",
              "resps": raw, "filtered_resps": filtered, "exact_match": int(raw == [['A']])}
    original = copy.deepcopy(sample)
    tracker.save_results_samples("fixture", [sample])
    saved = json.loads((tmp_path / "fixture/run_samples_fixture.jsonl").read_text())
    assert saved["resps"] == raw[0]
    assert sample == original


def test_raw_output_survives_filter_failure(tmp_path, monkeypatch):
    instance = Instance(request_type="generate_until", arguments=("question",), idx=0,
                        metadata={"task": "fixture", "doc_id": 0, "repeats": 1})
    task = MagicMock()
    task.instances = task._instances = [instance]
    task.dump_config.return_value = {}
    task.apply_filters.side_effect = RuntimeError("broken parser")
    monkeypatch.setattr(evaluator, "get_task_list", lambda _: [SimpleNamespace(task=task, task_name="fixture")])
    monkeypatch.setattr(evaluator, "get_sample_size", lambda *args: 1)
    for key in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = MagicMock()
    model.generate_until.return_value = ["<think>raw reasoning</think>A"]
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.model_name_sanitized = "fixture"
    with pytest.raises(RuntimeError, match="broken parser"):
        evaluator.evaluate(lm=model, task_dict={}, evaluation_tracker=tracker, datetime_str="run")
    saved = json.loads((tmp_path / "fixture/run_raw_fixture_rank0.jsonl").read_text())
    assert saved["resps"] == ["<think>raw reasoning</think>A"]
    assert saved["doc_id"] == 0
