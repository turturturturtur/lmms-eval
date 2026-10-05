from types import SimpleNamespace
from unittest.mock import Mock

from lmms_eval.llm_judge.standalone import JudgeRunner


def test_mmmu_nested_metrics_are_reused_when_original_doc_was_dropped():
    metric = {'id': 'validation_Art_1', 'question_type': 'multiple-choice',
              'answer': 'A', 'parsed_pred': ['A'], 'subdomain': 'Art'}
    sample = {'doc_id': 1, 'input': 'A question', 'target': 'A',
              'filtered_resps': ['A'], 'mmmu_acc': metric}
    runner = JudgeRunner(judge_mode='rule')
    process = Mock(side_effect=KeyError('question_type'))
    result = runner._judge_sample(sample, SimpleNamespace(), process)
    assert result['judge_mode'] == 'rule_existing'
    assert result['metrics']['mmmu_acc'] == metric
    process.assert_not_called()


def test_mmmu_answer_can_be_recovered_for_explicit_judging():
    assert JudgeRunner._extract_answer_from_metrics({'mmmu_acc': {'answer': 'B'}}) == 'B'
