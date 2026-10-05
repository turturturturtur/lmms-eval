"""Re-score saved SMILES samples without inference or modifying original results.

Usage: python -m tools.reaggregate_smol_samples --output corrected.json samples*.jsonl
The published task score retains the existing conditional-on-valid-predictions
Morgan fingerprint protocol. Invalid/no-answer counts are always reported too.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from lmms_eval.tasks.SmolInstruct_v2.metrics import calculate_smiles_metrics

TASKS = {'forward_synthesis', 'retrosynthesis', 'molecule_generation', 'name_conversion-i2s'}


def score_file(path: Path) -> dict:
    content = path.read_bytes()
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not records:
        raise ValueError(f'Empty sample file: {path}')
    samples = [r['smol_metrics'] for r in records]
    tasks = {s['task'] for s in samples}
    if len(tasks) != 1 or not tasks <= TASKS:
        raise ValueError(f'Expected exactly one SMILES task in {path}: {tasks}')
    task = next(iter(tasks))
    ids = [r['doc_id'] for r in records]
    if len({str(i) for i in ids}) != len(ids):
        raise ValueError(f'Duplicate sample IDs: {path}')
    metrics = calculate_smiles_metrics(
        [[s['pred']] for s in samples], [[s['gold']] for s in samples],
        metrics=('exact_match', 'fingerprint', 'multiple_match') if task == 'retrosynthesis'
        else ('exact_match', 'fingerprint'))
    # JSON's default NaN encoding is not permitted for a completed evaluation.
    json.dumps(metrics, allow_nan=False)
    identity = sorted([(str(r['doc_id']), r.get('input'), s['gold'])
                       for r, s in zip(records, samples)])
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(content).hexdigest(),
                task=task, sample_count=len(samples),
                sample_set_sha256=hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                    sort_keys=True).encode()).hexdigest(), metrics=metrics,
                protocol='smolinstruct-v2/morgan-r2/valid-predictions/empty-is-zero/v1')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('samples', type=Path, nargs='+')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    source_dir = Path(__file__).resolve().parents[1] / 'lmms_eval/tasks/SmolInstruct_v2'
    results = dict(schema='innovator-tune.smol-reaggregation/v1',
        rdkit=importlib.metadata.version('rdkit'),
        code_sha256={name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                     for name in ('metrics.py', 'smiles_canonicalization.py', 'utils.py')},
        results=[score_file(path) for path in args.samples])
    encoded = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    main()
