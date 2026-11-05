#!/usr/bin/env bash
set -euo pipefail
export LMMS_EVAL_MAIN_GPU_NUM=4
export LMMS_INTERNVL_IMAGE_KWARGS_COMPAT=1
export PYTHONPATH=/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/chemistry_dependencies
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_MODELSCOPE=False
export HF_HUB_OFFLINE=1
export EVAL_ENABLE_THINKING=false
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA=/mnt/cpfsA/user/tianleniu/Innovator-Tune/.setup/rl-judge-feishu-20261007/nltk_data
export LMMS_EVAL_LOG_DIR=/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_webui_4gpu_processorfix_20261018_232015/2026-10-18_23-20-15
export LMMS_EVAL_STAGE_DATASETS=0
export LD_LIBRARY_PATH=/mnt/cpfsA/user/tianleniu/conda/envs/lmm/lib
export LMMS_RESPONSE_JOURNAL=/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_webui_4gpu_processorfix_20261018_232015/2026-10-18_23-20-15/responses
export LMMS_GPU_GATE_SCRIPT=/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/attempt5_internvl_4gpu/source/tools/gpu_gate_webui.py
export LMMS_INTERNVL_CONTEXT_RECOVERY=1
cd /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval
exec > >(tee -a /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_webui_4gpu_processorfix_20261018_232015/2026-10-18_23-20-15/worker.log) 2>&1
exec bash /mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018/attempt5_internvl_4gpu/source/run_scripts/vision_family_worker.sh /mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_logs/eval_row07_webui_4gpu_processorfix_20261018_232015/2026-10-18_23-20-15/runtime_config.json
