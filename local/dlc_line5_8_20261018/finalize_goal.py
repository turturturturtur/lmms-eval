"""Publish a completion receipt only after every requested full evaluation passes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def main():
    manifest = json.loads((BASE / 'active_manifest.json').read_text())
    first = json.loads((BASE / 'attempt3_webui_defaults/result_verification.json').read_text())
    retry = json.loads((BASE / 'attempt6_internvl_2gpu/result_verification.json').read_text())
    verified = {r['row']:r for r in first['jobs'] if r['row'] in {5,6}}
    verified.update({r['row']:r for r in retry['jobs']})
    assert set(verified) == {5,6,7}
    defaults = json.loads((BASE / 'attempt3_webui_defaults/webui_eval_defaults.json').read_text())
    records = []
    for row in manifest['jobs']:
        proof = verified[row['row']]
        assert proof['complete'] and proof['job_id'] == row['job_id']
        cmd = [str(ROOT / 'bin/dlc'),'get','job',row['job_id'],'--show_detail']
        cloud = json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
        assert cloud['Status'] == 'Succeeded', (row['row'],cloud['Status'])
        cfg = json.loads(Path(row['runtime_config']).read_text())
        for key in ['tp','max_model_len','gpu_memory_utilization','max_num_seqs','base_port','enable_thinking']:
            assert cfg['model'][key] == defaults['model'][key], key
        for key in ['concurrency','gen_kwargs','limit','debug','vllm_request_timeout_seconds','task_timeout_seconds']:
            assert cfg['eval'][key] == defaults['eval'][key], key
        row['status'] = 'Succeeded'
        records.append({**proof,'model':row['model'],'requested_benchmarks':row['tasks'],
            'cells':cfg['provenance']['cells'],'gpu_count':int(cloud['JobSpecs'][0]['ResourceConfig']['GPU']),
            'finished_at':cloud.get('GmtFinishTime'),'runtime_config':row['runtime_config']})
    sample_count = sum(t['samples'] for row in records for t in row['tasks'].values())
    assert sample_count == 35061, sample_count
    assert sum(len(row['tasks']) for row in records) == 18
    report = {'complete':True,'verified_at':datetime.now(timezone.utc).isoformat(),
        'requested_benchmark_cells':5,'leaf_tasks':18,'total_samples':sample_count,
        'jobs':records,'judge_required':False,'judge_model_if_required':'Qwen3.5-9B',
        'generation_settings_match_webui_defaults':True,
        'compatibility_notes':[
            'Gemma4 uses TRITON_ATTN to support its multimodal attention implementation.',
            'InternVL uses a worker-local fix for image kwargs leaking into the video processor.',
            'InternVL context-overflow samples lower image patch count while preserving all images and text.',
            'InternVL ran on two GPUs because free resources were fragmented; TP=2 and concurrency=32 were unchanged.'
        ],
        'source_sheet_revision':1193,'sheet_written':False,
        'sheet_note':'Scope from saved successful read; live Feishu user authorization unavailable.',
        'row8':'All benchmark cells populated in the saved source sheet; no evaluation requested.'}
    (BASE / 'completion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (BASE / 'active_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'complete':True,'jobs':len(records),'leaf_tasks':18,'total_samples':sample_count,
        'scores':{str(r['row']):{t:v['metrics'] for t,v in r['tasks'].items()} for r in records}},ensure_ascii=False))


if __name__ == '__main__':
    main()
