import csv,json,collections,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path('/mnt/oss/tianleniu/Innovator-Tune/eval_recovery/portfix_step90_20261021')
report={'checked_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'root':str(root),'branches':{}}
for branch in ['base20_2gpu_a4','supplement_1gpu_a4','math_native_3gpu','math_reasoning_3gpu']:
    data={'results':[],'sample_checks':[],'task_statuses':[]}
    for p in sorted((root/branch/'results').rglob('*_results.json')):
        if 'smoke' in p.parts:continue
        j=json.loads(p.read_text());data['results'].append({'path':str(p),'metrics':j['results'],'n_samples':j.get('n-samples',{}),'judge':j.get('judge')})
    for p in sorted((root/branch/'results').rglob('task_status.tsv')):
        if 'smoke' not in p.parts:data['task_statuses'].extend(list(csv.DictReader(p.open(),delimiter='\t')))
    for p in sorted((root/branch/'results').rglob('*.jsonl')):
        if 'smoke' in p.parts or ('_samples_' not in p.name and '_raw_' not in p.name):continue
        n=0;ids=set();failed=empty=bad=0;finishes=collections.Counter()
        with p.open() as f:
            for line in f:
                if not line.strip():continue
                try:r=json.loads(line)
                except ValueError:bad+=1;continue
                n+=1;ids.add(str(r.get('doc_id',n)))
                resp=r.get('resps',r.get('response',r.get('text')))
                text=json.dumps(resp,ensure_ascii=False)
                if any(tag in text for tag in ['[LMMS_EVAL_REQUEST_FAILED','[PREPROC_FAILED]','[LMMS_EVAL_EMPTY_RESPONSE']):failed+=1
                def flatten(x):
                    if isinstance(x,list):return ''.join(flatten(v) for v in x)
                    return '' if x is None else str(x)
                if not flatten(resp).strip():empty+=1
                for x in r.get('finish_reasons',[]):finishes.update([str(x)])
        task=p.name.split('_samples_',1)[-1].removesuffix('.jsonl') if '_samples_' in p.name else p.name
        data['sample_checks'].append({'task':task,'path':str(p),'rows':n,'unique_doc_ids':len(ids),'bad_json':bad,'request_failure_markers':failed,'empty_resps':empty,'finish_reasons':dict(finishes)})
    report['branches'][branch]=data
    print(branch,'RESULT_FILES',len(data['results']),'SAMPLE_FILES',len(data['sample_checks']),'SAMPLE_ROWS',sum(x['rows'] for x in data['sample_checks']),'ERROR_MARKERS',sum(x['request_failure_markers'] for x in data['sample_checks']),'EMPTY_RESPS',sum(x['empty_resps'] for x in data['sample_checks']))
    for s in data['sample_checks']:
        if s['request_failure_markers'] or s['bad_json'] or s['empty_resps'] or '_raw_' in s['path']:print('DETAIL',s)
Path('/tmp/rl_eval_audit_step90_samples_20261023.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
