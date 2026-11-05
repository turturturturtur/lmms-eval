import os
from pathlib import Path
import subprocess

w = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018')
venv = w.parents[1] / '.venv'
env = os.environ.copy()
env.update(PYTHONPATH=str(w/'source/run_scripts/lmms_eval_sitecustomize')+':'+str(w/'source')+':'+str(w/'chemistry_dependencies'),
    PATH=str(venv/'bin')+':'+env['PATH'], PYTHONSAFEPATH='1', PYTHONDONTWRITEBYTECODE='1',
    VLLM_USE_MODELSCOPE='False', INNOVATOR_LMMS_HIDE_FLASH_ATTN='1',
    HF_HOME='/mnt/nasA/tianleniu/evaluation_cache/huggingface',
    HF_DATASETS_CACHE='/mnt/nasA/tianleniu/evaluation_cache/lmms_eval',
    LMMS_EVAL_DATASETS_CACHE='/mnt/nasA/tianleniu/evaluation_cache/lmms_eval',
    HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    LD_LIBRARY_PATH='/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib',
    NLTK_DATA='/mnt/cpfsA/user/tianleniu/Innovator-Tune/.setup/rl-judge-feishu-20261007/nltk_data')
with (w/'preflight.log').open('w') as f:
    p = subprocess.run([str(venv/'bin/python'),str(w/'preflight.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
print('preflight exit',p.returncode)
print((w/'preflight.log').read_text()[-9000:])
raise SystemExit(p.returncode)
