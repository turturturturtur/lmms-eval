"""Prepare the five empty benchmark cells, preserving prior evaluation semantics."""
import copy
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

ROOT = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune')
REPO = ROOT / 'lmms-eval'
WORK = REPO / 'local/dlc_line5_8_20261018'
PRIOR = ROOT / '.setup/mmmu-pro-vision-all-20261007'
TASK_SOURCE = ROOT / '.setup/rl-judge-feishu-20261007/supplement_source_v2'
NAS = Path('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval')
sys.path.insert(0, str(REPO / 'tools'))
from create_runtime_snapshot import create_snapshot


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    data = json.loads((WORK / 'sheet_rows.json').read_text())['data']
    assert data['revision'] == 1193 and not data['has_more']
    records = csv.reader(io.StringIO(re.sub(r'^\[row=\d+\] ', '', data['annotated_csv'], flags=re.M)))
    rows = dict(zip(data['row_indices'], records))
    missing = {}
    for n in range(5, 9):
        missing[n] = [{'cell': data['col_indices'][j] + str(n), 'task': rows[3][j]}
                      for j in range(8, 45) if rows[3][j] and not rows[3][j].startswith('AVG.')
                      and rows[n][j].strip() in ('', '-')]
    assert {n: [v['task'] for v in items] for n, items in missing.items()} == {
        5: ['superchem_en', 'superchem_cn'], 6: ['SmolInstruct_v2'],
        7: ['superchem_en', 'superchem_cn'], 8: []}

    source = WORK / 'source'
    old_manifest = json.loads((TASK_SOURCE / 'source_manifest.json').read_text())
    mismatches = [name for name, digest in old_manifest.items()
                  if hashlib.sha256((TASK_SOURCE / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    create_snapshot(TASK_SOURCE, source)
    worker = source / 'run_scripts/vision_family_worker.sh'
    shutil.copy2(PRIOR / 'generic_runtime/run_scripts/vision_family_worker.sh', worker)
    # Runtime infrastructure comes from the currently installed WebUI checkout.
    for name in ['eval_common.sh', 'activate_runtime.sh']:
        shutil.copy2(REPO / 'run_scripts' / name, source / 'run_scripts' / name)
    for name in ['runtime_preflight.py', 'storage_policy.py']:
        shutil.copy2(REPO / 'lmms_eval' / name, source / 'lmms_eval' / name)
    for name in ['monitor_worker_resources.py', 'run_with_backend_watch.py']:
        shutil.copy2(REPO / 'tools' / name, source / 'tools' / name)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
    save(source / 'source_manifest.json', hashes)
    source_sha = hashlib.sha256((source / 'source_manifest.json').read_bytes()).hexdigest()

    deps = WORK / 'chemistry_dependencies'
    deps.mkdir()
    site = ROOT.parent / 'conda/envs/lmm/lib/python3.11/site-packages'
    for name in ['rdkit', 'rdkit.libs', 'rdkit-2025.9.1.dist-info', 'rdchiral', 'rdchiral-1.1.0.dist-info']:
        assert (site / name).exists()
        (deps / name).symlink_to(site / name, target_is_directory=True)

    old_view = Path(json.loads((PRIOR / 'row07/eval.json').read_text())['model']['path'])
    view = NAS / 'eval_model_views/internvl35_row07_nothink_20261018'
    view.mkdir(exist_ok=False)
    view_proof = []
    for p in old_view.iterdir():
        if p.is_dir():
            continue
        if p.is_symlink():
            target = Path(str(p.readlink()).replace('/mnt/nasB/', '/mnt/nasA/', 1))
            assert target.is_file()
            (view / p.name).symlink_to(target)
            view_proof.append({'name': p.name, 'target': str(target), 'size': target.stat().st_size})
        else:
            shutil.copy2(p, view / p.name)
    save(WORK / 'internvl_view_receipt.json', {'original': str(old_view), 'view': str(view), 'weights': view_proof})

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    batch = []
    for n in [5, 6, 7]:
        label = f'row{n:02}'
        folder = WORK / label
        folder.mkdir()
        cfg = json.loads((REPO / 'run_scripts/config_eval.json').read_text().replace('<USER>', 'tianleniu'))
        model = copy.deepcopy(json.loads((PRIOR / label / 'eval.json').read_text())['model'])
        model.update(tp=2, max_model_len=40960, gpu_memory_utilization=0.88, max_num_seqs=192,
                     enable_thinking=False, served_model_name={5:'Qwen3-VL-8B-Instruct',6:'gemma-4-12B-it',7:'InternVL3_5-8B'}[n])
        if n == 5:
            shutil.copy2(PRIOR / label / 'chat_template.jinja', folder / 'chat_template.jinja')
            model['chat_template'] = str(folder / 'chat_template.jinja')
        if n == 7:
            model['path'] = str(view)
            model['chat_template'] = str(view / 'chat_template.jinja')
        model['source_path'] = model['path']
        model['backend'] = 'vllm'
        cfg['model'] = model
        tasks = [v['task'] for v in missing[n]]
        name = f'eval_{label}_missing_nothink_20261018'
        logdir = NAS / 'eval_logs' / name / stamp
        logdir.mkdir(parents=True)
        cfg['log']['dir'] = str(logdir)
        cfg['eval'].update(tasks=','.join(tasks), timestamp=stamp, limit=-1, gen_kwargs='',
                           concurrency=32, debug=False, output_path=str(NAS / 'eval_result' / name))
        cfg['env'].update(hf_token='', openai_api_key='', openai_api_url='')
        cfg['dlc'] = {'submit':False}
        cfg['provenance'] = {'sheet_revision':1193, 'sheet_row':n, 'cells':missing[n],
                             'source_manifest_sha256':source_sha, 'source_root':str(source),
                             'benchmark_source':str(TASK_SOURCE), 'thinking':False}
        runtime = logdir / 'runtime_config.json'
        save(runtime, cfg)
        save(folder / 'config_eval.json', cfg)
        dlc = json.loads((REPO / 'run_scripts/config_dlc.json').read_text().replace('<USER>', 'tianleniu'))
        dlc['dlc'].update(job_name=name, run_script=str(worker))
        save(folder / 'config_dlc.json', dlc)
        (logdir / 'source').symlink_to(source, target_is_directory=True)
        exports = {
            'PYTHONPATH':str(deps), 'PYTHONSAFEPATH':'1', 'PYTHONDONTWRITEBYTECODE':'1',
            'VLLM_USE_MODELSCOPE':'False', 'HF_HUB_OFFLINE':'1',
            'NLTK_DATA':str(ROOT / '.setup/rl-judge-feishu-20261007/nltk_data'),
            'LMMS_EVAL_LOG_DIR':str(logdir), 'LMMS_EVAL_STAGE_DATASETS':'0',
            'LD_LIBRARY_PATH':str(ROOT.parent / 'conda/envs/lmm/lib'),
            'EVAL_ENABLE_THINKING':'false', 'TOKENIZERS_PARALLELISM':'false',
        }
        shell = ['#!/usr/bin/env bash', 'set -euo pipefail']
        shell += ['export ' + k + '=' + shlex.quote(v) for k,v in exports.items()]
        shell += ['cd ' + shlex.quote(str(REPO)),
                  'exec > >(tee -a ' + shlex.quote(str(logdir / 'worker.log')) + ') 2>&1',
                  'exec bash ' + shlex.quote(str(worker)) + ' ' + shlex.quote(str(runtime))]
        (folder / 'worker.sh').write_text('\n'.join(shell)+'\n')
        # Keep the runtime path directly visible to WebUI's UserCommand discovery.
        command = 'bash ' + shlex.quote(str(folder / 'worker.sh')) + ' ' + shlex.quote(str(runtime))
        batch.append({'row':n, 'model':rows[n][0], 'tasks':tasks, 'job_name':name,
                      'directory':str(folder), 'runtime_config':str(runtime),
                      'source_sha256':source_sha, 'command':command, 'status':'Prepared'})
    save(WORK / 'manifest.json', {'source_url':'https://scnfed7y9q7p.feishu.cn/wiki/QXfpwaKEyi7H1hkhFwscFL0bnHb?sheet=RAG5Fk',
          'sheet_revision':1193, 'jobs':batch, 'skipped_rows':{'8':'All benchmark cells already populated'},
          'source_manifest_sha256':source_sha, 'benchmark_source_manifest_sha256':hashlib.sha256((TASK_SOURCE/'source_manifest.json').read_bytes()).hexdigest()})
    print(json.dumps({'jobs':len(batch),'benchmark_cells':5,'source_manifest_sha256':source_sha}))


if __name__ == '__main__':
    main()
