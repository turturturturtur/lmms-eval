import json
import pytest
from tools.reaggregate_mmbench_samples import score_file, main


def write_samples(path, predictions=('A', 'A')):
    rows=[]
    for index,pred in zip((1,1000001),predictions):
        rows.append({'gpt_eval_score':dict(index=index,question='Q',answer='A',prediction=pred,
            hint='',A='a',B='b',C='nan',D='nan',E='nan',category='test')})
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_cyclic_score_requires_all_variants_correct(tmp_path,monkeypatch):
    from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
    monkeypatch.setattr(MMBench_Evaluator,'extract_answer_from_item',lambda *a:pytest.fail('Unexpected API judge'))
    path=tmp_path/'samples.jsonl';write_samples(path,('A','B'))
    assert score_file(path)['accuracy_percent']==0
    write_samples(path)
    assert score_file(path)['accuracy_percent']==100


def test_comparison_rejects_different_sample_sets_without_output(tmp_path):
    a,b,out=[tmp_path/name for name in ['a.jsonl','b.jsonl','out.json']]
    write_samples(a);write_samples(b)
    b.write_text(b.read_text().replace('"Q"','"Different question"'))
    with pytest.raises(ValueError,match='Refusing comparison'):
        main(['--output',str(out),str(a),str(b)])
    assert not out.exists()
