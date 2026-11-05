"""Run a diagnostic against the exact frozen runtime, without GPU assumptions."""
import os
from pathlib import Path
import subprocess
import sys

work = Path(__file__).resolve().parents[1]
repo = work.parents[1]
source = Path(os.environ.get('DIAG_SOURCE', str(work/'source')))
env = os.environ.copy()
env.update(PYTHONPATH=f'{source}/run_scripts/lmms_eval_sitecustomize:{source}:{work}/chemistry_dependencies',
    PATH=f'{repo}/.venv/bin:'+env['PATH'], PYTHONSAFEPATH='1', PYTHONDONTWRITEBYTECODE='1',
    VLLM_USE_MODELSCOPE='False', INNOVATOR_LMMS_HIDE_FLASH_ATTN='1',
    HF_HOME='/mnt/nasA/tianleniu/evaluation_cache/huggingface',
    HF_DATASETS_CACHE='/mnt/nasA/tianleniu/evaluation_cache/lmms_eval',
    LMMS_EVAL_DATASETS_CACHE='/mnt/nasA/tianleniu/evaluation_cache/lmms_eval',
    HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    LD_LIBRARY_PATH='/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib',
    NLTK_DATA='/mnt/cpfsA/user/tianleniu/Innovator-Tune/.setup/rl-judge-feishu-20261007/nltk_data')
raise SystemExit(subprocess.call([str(repo/'.venv/bin/python'), *sys.argv[1:]], env=env))
