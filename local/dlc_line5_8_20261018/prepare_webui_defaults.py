"""Prepare a new batch from current WebUI defaults and current task definitions."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune')
REPO = ROOT / 'lmms-eval'
BASE = REPO / 'local/dlc_line5_8_20261018'
WORK = BASE / 'attempt3_webui_defaults'
SOURCE = WORK / 'source'
NAS = Path('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval')
sys.path.insert(0, str(REPO / 'tools'))
from create_runtime_snapshot import create_snapshot


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not (WORK / 'manifest.json').exists(), 'Already prepared'
    create_snapshot(REPO, SOURCE)
    shutil.copy2(BASE / 'attempt2/source/run_scripts/vision_family_worker.sh', SOURCE / 'run_scripts/vision_family_worker.sh')
    shutil.copy2(BASE / 'gpu_gate_webui.py', SOURCE / 'tools/gpu_gate_webui.py')
    # Transport-only recovery: preserve the current backend's request construction,
    # sampling, stop handling, reasoning metadata and all benchmark implementations.
    shutil.copy2(BASE / 'attempt2/source/lmms_eval/models/model_utils/request_recovery.py',
                 SOURCE / 'lmms_eval/models/model_utils/request_recovery.py')
    backend = SOURCE / 'lmms_eval/models/chat/vllm_backend.py'
    code = backend.read_text()
    code = code.replace('from lmms_eval.protocol import ChatMessages',
        'from lmms_eval.protocol import ChatMessages\nfrom lmms_eval.models.model_utils.request_recovery import post_completion, journal_path, write_receipt')
    old = '''        response = self.session.post(
            f"{url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()'''
    assert code.count(old) == 1
    code = code.replace(old, '''        return post_completion(self.session, url, payload, self.headers, self.timeout,
            recover_internvl=os.environ.get('LMMS_INTERNVL_CONTEXT_RECOVERY') == '1')''')
    old = '                    result = self._make_request(payload, url)'
    assert code.count(old) == 1
    code = code.replace(old, '''                    result = self._make_request(payload, url)
                    if os.environ.get('LMMS_RESPONSE_JOURNAL'):
                        req = reordered_requests[local_index]
                        identity = {'task': req.args[4], 'split': req.args[5],
                                    'doc_id': req.args[3], 'request_index': local_index}
                        receipt = journal_path(os.environ['LMMS_RESPONSE_JOURNAL'], payload, identity)
                        write_receipt(receipt, identity, result, time.time() - api_start)''')
    backend.write_text(code)
    hashes = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
    save(SOURCE / 'source_manifest.json', hashes)
    source_sha = hashlib.sha256((SOURCE / 'source_manifest.json').read_bytes()).hexdigest()
    defaults = json.loads((REPO / 'run_scripts/config_eval.json').read_text().replace('<USER>', 'tianleniu'))
    save(WORK / 'webui_eval_defaults.json', defaults)
    prior = json.loads((BASE / 'attempt2/manifest.json').read_text())
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    jobs = []
    for old in prior['jobs']:
        n = old['row']
        folder = WORK / f'row{n:02d}'
        folder.mkdir()
        name = f'eval_row{n:02d}_missing_webui_defaults_20261018'
        logdir = NAS / 'eval_logs' / name / stamp
        logdir.mkdir(parents=True)
        cfg = copy.deepcopy(defaults)
        previous = json.loads(Path(old['runtime_config']).read_text())
        cfg['model'] = copy.deepcopy(previous['model'])
        for key in ['tp','max_model_len','gpu_memory_utilization','max_num_seqs','base_port','enable_thinking']:
            cfg['model'][key] = defaults['model'][key]
        cfg['eval'].update(tasks=','.join(old['tasks']),timestamp=stamp,output_path=str(NAS / 'eval_result' / name),batch_size=1)
        cfg['log']['dir'] = str(logdir)
        cfg['env'].update(hf_token='',openai_api_key='',openai_api_url='')
        cfg['dlc'] = {'submit': False}
        cfg['provenance'] = {
            'sheet_revision': prior['sheet_revision'], 'sheet_row': n,
            'sheet_source': 'saved successful user read at 2026-10-18 01:51 CST; live read currently unauthorized',
            'cells': previous['provenance']['cells'], 'source_root': str(SOURCE),
            'source_manifest_sha256': source_sha, 'benchmark_source': str(REPO / 'lmms_eval/tasks'),
            'previous_job_id': old['job_id'], 'webui_defaults': str(WORK / 'webui_eval_defaults.json'),
            'judge_model_if_required': 'Qwen3.5-9B', 'judge_required': False,
            'compatibility': {'attention_backend': cfg['model']['attention_backend'],
                              'internvl_context_recovery': n == 7,
                              'note': 'Only a context-overflow HTTP 400 reduces image patch count; all images and text retained. Sampling remains at WebUI defaults.'},
        }
        runtime = logdir / 'runtime_config.json'
        save(runtime, cfg)
        save(folder / 'config_eval.json', cfg)
        dlc = json.loads((REPO / 'run_scripts/config_dlc.json').read_text().replace('<USER>','tianleniu'))
        dlc['dlc'].update(job_name=name,run_script=str(SOURCE / 'run_scripts/vision_family_worker.sh'))
        save(folder / 'config_dlc.json', dlc)
        (logdir / 'source').symlink_to(SOURCE, target_is_directory=True)
        exports = {'PYTHONPATH':str(BASE / 'chemistry_dependencies'),'PYTHONSAFEPATH':'1',
            'PYTHONDONTWRITEBYTECODE':'1','VLLM_USE_MODELSCOPE':'False','HF_HUB_OFFLINE':'1',
            'EVAL_ENABLE_THINKING':'false','TOKENIZERS_PARALLELISM':'false',
            'NLTK_DATA':str(ROOT / '.setup/rl-judge-feishu-20261007/nltk_data'),
            'LMMS_EVAL_LOG_DIR':str(logdir),'LMMS_EVAL_STAGE_DATASETS':'0',
            'LD_LIBRARY_PATH':'/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib',
            'LMMS_RESPONSE_JOURNAL':str(logdir / 'responses'),
            'LMMS_GPU_GATE_SCRIPT':str(SOURCE / 'tools/gpu_gate_webui.py')}
        if n == 7:
            exports['LMMS_INTERNVL_CONTEXT_RECOVERY'] = '1'
        shell = ['#!/usr/bin/env bash','set -euo pipefail']
        shell += ['export '+k+'='+shlex.quote(v) for k,v in exports.items()]
        shell += ['cd '+shlex.quote(str(REPO)),
                  'exec > >(tee -a '+shlex.quote(str(logdir / 'worker.log'))+') 2>&1',
                  'exec bash '+shlex.quote(str(SOURCE / 'run_scripts/vision_family_worker.sh'))+' '+shlex.quote(str(runtime))]
        (folder / 'worker.sh').write_text('\n'.join(shell)+'\n')
        command = 'export LMMS_EVAL_LOG_DIR='+shlex.quote(str(logdir))+'; bash '+shlex.quote(str(folder / 'worker.sh'))+' '+shlex.quote(str(runtime))
        jobs.append({'row':n,'model':old['model'],'tasks':old['tasks'],'job_name':name,
            'directory':str(folder),'runtime_config':str(runtime),'source_sha256':source_sha,
            'command':command,'status':'Prepared','previous_job_id':old['job_id']})
    save(WORK / 'manifest.json', {'source_url':prior['source_url'],'sheet_revision':prior['sheet_revision'],
        'jobs':jobs,'skipped_rows':prior['skipped_rows'],'source_manifest_sha256':source_sha,
        'attempt':3,'benchmark_definition':'current WebUI checkout; no task overrides'})
    code = (BASE / 'attempt2/submit.py').read_text().replace("WORK = REPO/'local/dlc_line5_8_20261018/attempt2'", "WORK = REPO/'local/dlc_line5_8_20261018/attempt3_webui_defaults'")
    (WORK / 'submit.py').write_text(code)
    print(json.dumps({'jobs':len(jobs),'directory':str(WORK),'source_manifest_sha256':source_sha}))


if __name__ == '__main__':
    main()
