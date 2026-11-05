#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/chemistry_dependencies
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_MODELSCOPE=False
export HF_HUB_OFFLINE=1
export EVAL_ENABLE_THINKING=false
export NLTK_DATA=/mnt/cpfsA/user/tianleniu/Innovator-Tune/.setup/rl-judge-feishu-20261007/nltk_data
export LMMS_EVAL_LOG_DIR=/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_retry2_20261018/2026-10-18_08-42-50
export LMMS_EVAL_STAGE_DATASETS=0
export LD_LIBRARY_PATH=/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib
export TOKENIZERS_PARALLELISM=false
export LMMS_RESPONSE_JOURNAL=/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_retry2_20261018/2026-10-18_08-42-50/responses
export LMMS_GPU_GATE_SCRIPT=/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/attempt2/source/tools/gpu_eval_gate.py
export LMMS_INTERNVL_CONTEXT_RECOVERY=1
cd /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval
exec > >(tee -a /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_retry2_20261018/2026-10-18_08-42-50/worker.log) 2>&1
exec bash /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/attempt2/source/run_scripts/vision_family_worker.sh /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_missing_nothink_retry2_20261018/2026-10-18_08-42-50/runtime_config.json
