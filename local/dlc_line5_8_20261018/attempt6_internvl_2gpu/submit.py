"""Submit the verified batch once through the same CreateJob helper as WebUI."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune')
REPO = ROOT/'lmms-eval'
WORK = REPO/'local/dlc_line5_8_20261018/attempt6_internvl_2gpu'
MANIFEST = WORK/'manifest.json'


def save(data):
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(MANIFEST)


def main():
    report = json.loads((WORK/'preflight.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert report['passed'] and report['source_manifest_sha256']==manifest['source_manifest_sha256']
    source = WORK/'source'
    for name,digest in json.loads((source/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    for row in manifest['jobs']:
        if row.get('job_id'):
            continue
        assert row['status']=='Prepared', 'Uncertain prior submission must be reconciled first'
        directory = Path(row['directory'])
        d = json.loads((directory/'config_dlc.json').read_text())['dlc']
        assert d['workers']==1 and d['worker_gpu']==2 and d['priority']==6
        query = subprocess.run([str(ROOT/'bin/dlc'),'get','job','-n',row['job_name'],
            '-w','247143','--resource_id','quota1cauvwqouvn'],text=True,capture_output=True,check=True)
        (directory/'duplicate_check.txt').write_text(query.stdout)
        assert not re.findall(r'\b(dlc[a-z0-9]+)\b',query.stdout), 'Matching cloud job already exists'
        subprocess.run(['bash','-n',str(directory/'worker.sh')],check=True)
        args=[sys.executable,str(REPO/'run_scripts/dlc_create_job.py'),'submit','pytorchjob',
              '--name',row['job_name'],'--command',row['command']]
        for key in ['workers','worker_cpu','worker_gpu','worker_memory','worker_shared_memory',
                    'worker_image','priority','job_max_running_time_minutes','data_source_uris',
                    'resource_id','workspace_id','vpc_id','switch_id','security_group_id',
                    'extended_cidrs','region','endpoint']:
            args += ['--'+key,str(d[key])]
        preview = subprocess.run(args+['--preview'],text=True,capture_output=True,check=True)
        request = json.loads(preview.stdout)
        assert request['ResourceId']=='quota1cauvwqouvn' and request['WorkspaceId']=='247143'
        assert request['DataSources'][0]['AccessPointId']=='ap-1sqxfm3pf4vkkcqec'
        (directory/'submission_preview.json').write_text(preview.stdout)
        row.update(status='Submitting', submitted_at=datetime.now(timezone.utc).isoformat())
        save(manifest)
        result = subprocess.run(args,text=True,capture_output=True)
        receipt = result.stdout+result.stderr
        (directory/'submit.log').write_text(receipt)
        Path(row['runtime_config']).with_name('eval_submit.log').write_text(receipt)
        if result.returncode:
            row['status']='SubmissionNeedsReview'; save(manifest)
            raise RuntimeError(receipt)
        job_id, = re.findall(r'\b(dlc[a-z0-9]+)\b',result.stdout)
        row.update(job_id=job_id,status='Submitted')
        save(manifest)
        detail = subprocess.run([str(ROOT/'bin/dlc'),'get','job',job_id,'--show_detail'],
                                text=True,capture_output=True,check=True)
        cloud = json.loads(detail.stdout)
        assert cloud['JobId']==job_id and cloud['DisplayName']==row['job_name']
        assert cloud['ResourceId']=='quota1cauvwqouvn' and cloud['Priority']==6
        assert cloud['UserCommand']==row['command']
        resources=cloud['JobSpecs'][0]['ResourceConfig']
        assert int(resources['GPU'])==2 and int(resources['CPU'])==32
        (directory/'cloud_status.json').write_text(json.dumps(cloud,indent=2)+'\n')
        row['status']=cloud['Status']; save(manifest)
        print(json.dumps({'row':row['row'],'model':row['model'],'tasks':row['tasks'],
                          'job_id':job_id,'status':row['status']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
