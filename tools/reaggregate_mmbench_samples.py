"""Compare saved MMBench runs with one explicit offline scoring protocol.

No inference or API calls are made. The score uses the repository's static
answer extractor and requires every cyclic option variant to be correct.
It must not be labeled as a newly run LLM-judge score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

IDENTITY_FIELDS = ('index', 'question', 'hint', 'answer', 'A', 'B', 'C', 'D', 'E')


def score_file(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty sample file: {path}')
    records = [row.get('metrics', {}).get('gpt_eval_score', row.get('gpt_eval_score')) for row in rows]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError(f'Missing gpt_eval_score records in {path}')
    indices = [r['index'] for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError(f'Duplicate sample indices in {path}')
    if any(i % 1000000 not in indices for i in indices):
        raise ValueError(f'Missing base cyclic variant in {path}')
    identity = [{k:r.get(k) for k in IDENTITY_FIELDS} for r in sorted(records,key=lambda r:r['index'])]
    sample_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    evaluator = MMBench_Evaluator(sys_prompt='', API_KEY='', API_URL='', model_version='offline')
    score, categories, l2 = evaluator.eval_result(records, eval_method='static')
    result = dict(source=str(Path(path).resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
        sample_set_sha256=sample_hash, sample_count=len(records),
        base_questions=sum(i < 1000000 for i in indices), accuracy_percent=100*float(score),
        category_accuracy=categories, l2_category_accuracy=l2,
        protocol='mmbench-en-dev/static-extractor/cyclic-all-correct/v1')
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('samples',type=Path,nargs='+')
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    results=[score_file(p) for p in args.samples]
    if len({r['sample_set_sha256'] for r in results}) != 1:
        raise ValueError('Refusing comparison: sample identities/questions/options differ between runs.')
    root=Path(__file__).resolve().parents[1]
    code=['lmms_eval/tasks/mmbench/mmbench_evals.py','lmms_eval/tasks/_task_utils/mcq_extract.py']
    output=dict(schema='innovator-tune.mmbench-comparison/v1',results=results,
                code_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in code})
    text=json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with args.output.open('x') as f:f.write(text)

if __name__=='__main__':main()
