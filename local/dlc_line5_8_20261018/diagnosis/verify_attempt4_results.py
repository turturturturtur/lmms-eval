"""Final completion criteria: cloud success, full sample coverage, raw responses and native metrics."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

work=Path(__file__).resolve().parents[1]
manifest=json.loads((work/'attempt4_internvl_processor/manifest.json').read_text())
expected={b['benchmark']:{m['task']:m['samples'] for m in b['members']}
    for b in json.loads((work/'preflight.json').read_text())['benchmarks']}
source=work/'attempt4_internvl_processor/source'
assert hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest()==manifest['source_manifest_sha256']
for name,digest in json.loads((source/'source_manifest.json').read_text()).items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name


def verify(row):
    cloud=json.loads((Path(row['directory'])/'cloud_status.json').read_text())
    summary={'row':row['row'],'job_id':row['job_id'],'status':cloud['Status'],'complete':False}
    if cloud['Status']!='Succeeded':
        return summary
    config=json.loads(Path(row['runtime_config']).read_text()); e=config['eval']
    assert config['model']['enable_thinking'] is False and e['limit']==-1
    logdir=Path(row['runtime_config']).parent
    assert json.loads((logdir/'gpu_gate_report.json').read_text())['passed']
    root=Path(e['output_path'])/e['timestamp']
    with (root/'task_status.tsv').open() as f:
        statuses=list(csv.DictReader(f,delimiter='\t'))
    assert len(statuses)==len(row['tasks']),statuses
    assert all(s['status']=='success' for s in statuses),statuses
    tasks={}
    for benchmark in row['tasks']:
        for task,count in expected[benchmark].items():
            samples=list(root.rglob(f'*_samples_{task}.jsonl'))
            assert len(samples)==1,(task,samples)
            seen=set(); repeat=1
            with samples[0].open() as f:
                for line in f:
                    item=json.loads(line); doc=item['doc_id']
                    assert doc not in seen,(task,doc,'duplicate'); seen.add(doc)
                    responses=item['resps']
                    assert isinstance(responses,list) and len(responses)==repeat,(task,doc,'response count')
                    assert all(isinstance(s,str) and '[LMMS_EVAL_REQUEST_FAILED' not in s and '[PREPROC_FAILED]' not in s for s in responses)
            assert seen==set(range(count)),(task,len(seen),count)
            aggregates=[]
            for path in root.rglob('*_results.json'):
                content=json.loads(path.read_text())
                if task in content['results']:
                    aggregates.append((path,content))
            assert len(aggregates)==1,(task,len(aggregates))
            result_path,result=aggregates[0]
            assert result['n-samples'][task]=={'original':count,'effective':count}
            metric_name = 'mcp_accuracy,none' if task.startswith('superchem_') else 'smol_metrics,none'
            metrics={metric_name: result['results'][task][metric_name]}
            assert metrics and all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics.values()),(task,metrics)
            assert all(0<=v<=1 for v in metrics.values()),(task,metrics)
            receipts=list((logdir/'responses'/task).glob('*.json'))
            assert len(receipts)==count*repeat,(task,len(receipts),count*repeat)
            identities=set(); finishes=Counter(); adapted=Counter()
            for p in receipts:
                receipt=json.loads(p.read_text()); identity=receipt['identity']; response=receipt['response']
                identities.add(identity['request_index'])
                finishes[response['choices'][0].get('finish_reason')]+=1
                if '_lmms_context_recovery' in response:
                    adapted[identity['doc_id']]+=1
            assert len(identities)==count*repeat
            assert {json.loads(p.read_text())['identity']['doc_id'] for p in receipts} == set(range(count))
            tasks[task]={'samples':count,'responses':count*repeat,'metrics':metrics,
                'finish_reasons':dict(finishes),'adapted_doc_trials':dict(adapted),
                'sample_file':str(samples[0]),'result_file':str(result_path)}
    summary.update(complete=True,tasks=tasks,result_root=str(root))
    return summary


rows=[verify(row) for row in manifest['jobs']]
report={'complete':all(r['complete'] for r in rows),'jobs':rows}
(work/'attempt4_internvl_processor/result_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
