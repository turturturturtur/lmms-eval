"""Retry only InternVL with an isolated fix for installed vLLM processor kwargs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune')
REPO = ROOT / 'lmms-eval'
BASE = REPO / 'local/dlc_line5_8_20261018'
OLD = BASE / 'attempt3_webui_defaults'
WORK = BASE / 'attempt4_internvl_processor'
SOURCE = WORK / 'source'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


assert not SOURCE.exists()
shutil.copytree(OLD / 'source', SOURCE)
site = SOURCE / 'run_scripts/lmms_eval_sitecustomize/sitecustomize.py'
code = site.read_text().replace('import os\n','import os\nimport sys\n')
anchor = 'def _innovator_import(name, globals=None, locals=None, fromlist=(), level=0):'
compat = '''def _patch_internvl_video_processor() -> None:
    # vLLM 0.26 passes shared image kwargs to the video constructor even for
    # image-only inputs. Keep image overrides intact while excluding those
    # image-only options from video initialization. Scoped to this worker.
    if os.environ.get("LMMS_INTERNVL_IMAGE_KWARGS_COMPAT") != "1":
        return
    module = sys.modules.get("vllm.transformers_utils.processors.internvl")
    cls = getattr(module, "InternVLVideoProcessor", None)
    if cls is None or getattr(cls, "_lmms_image_kwargs_compat", False):
        return
    original = cls.__init__

    def init(self, image_size, **kwargs):
        for key in ("min_dynamic_patch", "max_dynamic_patch", "dynamic_image_size", "use_thumbnail"):
            kwargs.pop(key, None)
        return original(self, image_size=image_size, **kwargs)

    cls.__init__ = init
    cls._lmms_image_kwargs_compat = True


'''
assert code.count(anchor) == 1
code = code.replace(anchor,compat+anchor)
code = code.replace('    return module\n',
    '    if name == "vllm.transformers_utils.processors.internvl":\n        _patch_internvl_video_processor()\n    return module\n')
site.write_text(code)
hashes = {str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
save(SOURCE / 'source_manifest.json',hashes)
sha = hashlib.sha256((SOURCE / 'source_manifest.json').read_bytes()).hexdigest()
previous = json.loads((OLD / 'manifest.json').read_text())
row = next(r for r in previous['jobs'] if r['row']==7)
previous_id = row.pop('job_id')
folder = WORK / 'row07'
folder.mkdir()
cfg = json.loads(Path(row['runtime_config']).read_text())
stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
name = 'eval_row07_webui_defaults_processorfix_'+datetime.now().strftime('%Y%m%d_%H%M%S')
oldlog = cfg['log']['dir']
logdir = Path('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs') / name / stamp
logdir.mkdir(parents=True)
runtime = logdir / 'runtime_config.json'
cfg['log']['dir'] = str(logdir)
cfg['eval'].update(timestamp=stamp,output_path=str(Path('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_result') / name))
cfg['provenance'].update(source_root=str(SOURCE),source_manifest_sha256=sha,previous_job_id=previous_id,attempt=4)
cfg['provenance']['compatibility']['internvl_video_image_kwargs_fix'] = True
save(runtime,cfg)
save(folder/'config_eval.json',cfg)
dlc = json.loads((OLD / 'row07/config_dlc.json').read_text())
dlc['dlc'].update(job_name=name,run_script=str(SOURCE / 'run_scripts/vision_family_worker.sh'))
save(folder/'config_dlc.json',dlc)
(logdir/'source').symlink_to(SOURCE,target_is_directory=True)
worker = (OLD/'row07/worker.sh').read_text().replace(str(OLD),str(WORK)).replace(oldlog,str(logdir))
worker = worker.replace('set -euo pipefail\n','set -euo pipefail\nexport LMMS_INTERNVL_IMAGE_KWARGS_COMPAT=1\n')
(folder/'worker.sh').write_text(worker)
row.update(directory=str(folder),runtime_config=str(runtime),source_sha256=sha,
    job_name=name,command=row['command'].replace(str(OLD),str(WORK)).replace(oldlog,str(logdir)),
    status='Prepared',previous_job_id=previous_id)
row.pop('submitted_at',None)
save(WORK/'manifest.json',{'jobs':[row],'source_url':previous['source_url'],'sheet_revision':previous['sheet_revision'],
    'source_manifest_sha256':sha,'attempt':4,'previous_manifest':str(OLD/'manifest.json')})
(WORK/'submit.py').write_text((OLD/'submit.py').read_text().replace(str(OLD.relative_to(REPO)),str(WORK.relative_to(REPO))))
print(json.dumps({'prepared':str(WORK),'source_manifest_sha256':sha,'row':7}))
