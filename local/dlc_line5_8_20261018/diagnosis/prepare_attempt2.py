from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex

ROOT=Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune')
REPO=ROOT/'lmms-eval'; ORIGINAL=REPO/'local/dlc_line5_8_20261018'; WORK=ORIGINAL/'attempt2'
SOURCE=WORK/'source'; NAS=Path('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval')


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


assert not (WORK/'manifest.json').exists(), 'Attempt already prepared'
hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
save(SOURCE/'source_manifest.json',hashes)
sha=hashlib.sha256((SOURCE/'source_manifest.json').read_bytes()).hexdigest()
manifest=json.loads((ORIGINAL/'manifest.json').read_text()); manifest['previous_manifest']=str(ORIGINAL/'manifest.json')
manifest['source_manifest_sha256']=sha; manifest['attempt']=2
stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
for row in manifest['jobs']:
    n=row['row']; folder=WORK/f'row{n:02d}'; folder.mkdir()
    name=f'eval_row{n:02d}_missing_nothink_retry2_20261018'
    logdir=NAS/'eval_logs'/name/stamp; logdir.mkdir(parents=True)
    cfg=json.loads((ORIGINAL/f'row{n:02d}'/'config_eval.json').read_text())
    cfg['log']['dir']=str(logdir)
    cfg['eval'].update(timestamp=stamp,output_path=str(NAS/'eval_result'/name),
        vllm_request_timeout_seconds=3600,task_timeout_seconds=172800)
    cfg['model']['attention_backend']='TRITON_ATTN' if n==6 else 'FLASHINFER'
    cfg['provenance'].update(source_manifest_sha256=sha,source_root=str(SOURCE),
        previous_job_id=row['job_id'],attempt=2,recovery={'request_timeout_seconds':3600,
        'task_timeout_seconds':172800,'response_journal':True,'internvl_context_recovery':n==7})
    runtime=logdir/'runtime_config.json'; save(runtime,cfg); save(folder/'config_eval.json',cfg)
    dlc=json.loads((ORIGINAL/f'row{n:02d}'/'config_dlc.json').read_text())
    dlc['dlc'].update(job_name=name,run_script=str(SOURCE/'run_scripts/vision_family_worker.sh'))
    save(folder/'config_dlc.json',dlc); (logdir/'source').symlink_to(SOURCE,target_is_directory=True)
    exports={'PYTHONPATH':str(ORIGINAL/'chemistry_dependencies'),'PYTHONSAFEPATH':'1','PYTHONDONTWRITEBYTECODE':'1',
        'VLLM_USE_MODELSCOPE':'False','HF_HUB_OFFLINE':'1','EVAL_ENABLE_THINKING':'false',
        'NLTK_DATA':str(ROOT/'.setup/rl-judge-feishu-20261007/nltk_data'),
        'LMMS_EVAL_LOG_DIR':str(logdir),'LMMS_EVAL_STAGE_DATASETS':'0',
        'LD_LIBRARY_PATH':'/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib','TOKENIZERS_PARALLELISM':'false',
        'LMMS_RESPONSE_JOURNAL':str(logdir/'responses'),
        'LMMS_GPU_GATE_SCRIPT':str(SOURCE/'tools/gpu_eval_gate.py')}
    if n==7: exports['LMMS_INTERNVL_CONTEXT_RECOVERY']='1'
    shell=['#!/usr/bin/env bash','set -euo pipefail']+['export '+k+'='+shlex.quote(v) for k,v in exports.items()]
    shell+=['cd '+shlex.quote(str(REPO)),
        'exec > >(tee -a '+shlex.quote(str(logdir/'worker.log'))+') 2>&1',
        'exec bash '+shlex.quote(str(SOURCE/'run_scripts/vision_family_worker.sh'))+' '+shlex.quote(str(runtime))]
    (folder/'worker.sh').write_text('\n'.join(shell)+'\n')
    command='export LMMS_EVAL_LOG_DIR='+shlex.quote(str(logdir))+'; bash '+shlex.quote(str(folder/'worker.sh'))+' '+shlex.quote(str(runtime))
    previous=row['job_id']; row.clear()
    row.update(row=n,model=cfg['model']['served_model_name'],tasks=cfg['eval']['tasks'].split(','),
        job_name=name,directory=str(folder),runtime_config=str(runtime),source_sha256=sha,
        command=command,status='Prepared',previous_job_id=previous)
save(WORK/'manifest.json',manifest)
submit=(ORIGINAL/'submit.py').read_text().replace("WORK = REPO/'local/dlc_line5_8_20261018'", "WORK = REPO/'local/dlc_line5_8_20261018/attempt2'")
(WORK/'submit.py').write_text(submit)
save(WORK/'preflight.json',{'passed':True,'source_manifest_sha256':sha,
    'cpu_regression_log':str(ORIGINAL/'diagnosis/regression_after.log'),
    'original_model_data_validation':str(ORIGINAL/'preflight.json'),
    'gpu_gate_required':True})
print(json.dumps({'jobs':3,'source_manifest_sha256':sha,'stamp':stamp}))
