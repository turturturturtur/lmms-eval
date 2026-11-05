#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/chemistry_dependencies
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_MODELSCOPE=False
export HF_HUB_OFFLINE=1
export NLTK_DATA=/mnt/cpfsA/user/tianleniu/Innovator-Tune/.setup/rl-judge-feishu-20261007/nltk_data
export LMMS_EVAL_LOG_DIR=/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_20261018/2026-10-18_01-55-36
export LMMS_EVAL_STAGE_DATASETS=0
export LD_LIBRARY_PATH=/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib
export EVAL_ENABLE_THINKING=false
export TOKENIZERS_PARALLELISM=false
cd /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval
exec > >(tee -a /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_20261018/2026-10-18_01-55-36/worker.log) 2>&1
exec bash /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/source/run_scripts/vision_family_worker.sh /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_20261018/2026-10-18_01-55-36/runtime_config.json
