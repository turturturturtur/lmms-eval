"""Read cloud state and shared evaluation artifacts; never mutate remote jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

work=Path(__file__).resolve().parents[1]
manifest_path=work/'attempt2/manifest.json'
binary='/mnt/cpfsA/Innovator-Tune/bin/dlc'


def inspect(row):
    if not row.get('job_id'):
        return {'row':row['row'],'status':row['status']}
    p=subprocess.run([binary,'get','job',row['job_id'],'--show_detail'],capture_output=True,text=True,check=True)
    cloud=json.loads(p.stdout)
    logdir=Path(row['runtime_config']).parent
    (Path(row['directory'])/'cloud_status.json').write_text(json.dumps(cloud,indent=2)+'\n')
    summary={'row':row['row'],'job_id':row['job_id'],'status':cloud['Status'],
        'gate_passed':(logdir/'gpu_gate_report.json').exists(),
        'gate_responses':len(list((logdir/'gpu_gate_responses').glob('*/*.json'))),
        'responses':{p.name:len(list(p.glob('*.json'))) for p in (logdir/'responses').glob('*') if p.is_dir()}}
    log=logdir/'worker.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-5000)); tail=f.read().decode(errors='replace')
        summary['worker_tail']=tail.replace('\r','\n').splitlines()[-4:]
    for p in logdir.glob('lmms_eval_rank*.log'):
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2500)); tail=f.read().decode(errors='replace')
        progress=re.findall(r'\d+/\d+ \[[^\]\n]+\]',tail)
        summary.setdefault('task_progress',{})[p.name]=progress[-1] if progress else tail.replace('\r','\n').splitlines()[-1:]
    return summary


manifest=json.loads(manifest_path.read_text())
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=list(pool.map(inspect,manifest['jobs']))
report={'checked_at':datetime.now(timezone.utc).isoformat(),'jobs':rows}
(work/'attempt2/latest_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
