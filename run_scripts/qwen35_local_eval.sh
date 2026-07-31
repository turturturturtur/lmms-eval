#!/bin/bash
# Run a local Qwen3.5 lmms-eval job through the same strict processor
# compatibility resolver used by the DLC submitter.
#
# Usage:
#   bash run_scripts/qwen35_local_eval.sh MODEL TASKS OUTPUT [LIMIT] [THINKING]
#
# THINKING must be "on" or "off" and defaults to "off".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_SOURCE="${1:-}"
TASKS="${2:-}"
OUTPUT_PATH_INPUT="${3:-}"
LIMIT="${4:--1}"
THINKING_MODE="${5:-off}"

if [[ -z "${MODEL_SOURCE}" || -z "${TASKS}" || -z "${OUTPUT_PATH_INPUT}" ]]; then
    sed -n '1,9p' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
if [[ ! -d "${MODEL_SOURCE}" ]]; then
    echo "[ERROR] MODEL must be an existing local directory: ${MODEL_SOURCE}" >&2
    exit 2
fi
if [[ "${MODEL_SOURCE}" == *","* || "${MODEL_SOURCE}" == *"="* ]]; then
    echo "[ERROR] MODEL path must not contain ',' or '=' because model_args uses those delimiters: ${MODEL_SOURCE}" >&2
    exit 2
fi
if ! [[ "${LIMIT}" =~ ^-?[0-9]+$ ]] || (( LIMIT < -1 )); then
    echo "[ERROR] LIMIT must be -1 or a non-negative integer, got: ${LIMIT}" >&2
    exit 2
fi

case "$(printf '%s' "${THINKING_MODE}" | tr '[:upper:]' '[:lower:]')" in
    on)
        ENABLE_THINKING=True
        ;;
    off)
        ENABLE_THINKING=False
        ;;
    *)
        echo "[ERROR] THINKING must be on or off, got: ${THINKING_MODE}" >&2
        exit 2
        ;;
esac

VENV_PATH="${LMMS_EVAL_VENV_PATH:-${LMMS_EVAL_ROOT}/.venv}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "[ERROR] lmms-eval Python is missing or not executable: ${VENV_PATH}/bin/python" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "[ERROR] jq is required by qwen35_local_eval.sh" >&2
    exit 2
fi

CONFIG_TEMPLATE="${SCRIPT_DIR}/config_eval.json"
if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
    echo "[ERROR] Eval config template not found: ${CONFIG_TEMPLATE}" >&2
    exit 2
fi
MODEL_TP="$(jq -er '.model.tp | if type == "number" and floor == . and . > 0 then . else error("model.tp must be a positive integer") end' "${CONFIG_TEMPLATE}")"
MODEL_MAX_LEN="$(jq -er '.model.max_model_len | if type == "number" and floor == . and . > 0 then . else error("model.max_model_len must be a positive integer") end' "${CONFIG_TEMPLATE}")"
MODEL_GPU_MEM_UTIL="$(jq -er '.model.gpu_memory_utilization | if type == "number" and . > 0 and . < 1 then . else error("model.gpu_memory_utilization must be in (0,1)") end' "${CONFIG_TEMPLATE}")"
MODEL_MAX_NUM_SEQS="$(jq -er '.model.max_num_seqs | if type == "number" and floor == . and . > 0 then . else error("model.max_num_seqs must be a positive integer") end' "${CONFIG_TEMPLATE}")"

OUTPUT_PATH="$("${VENV_PATH}/bin/python" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OUTPUT_PATH_INPUT}")"
mkdir -p "${OUTPUT_PATH}"
MODEL_VIEW_ROOT="${OUTPUT_PATH}/model_views"

MODEL_ARGS="model=${MODEL_SOURCE},tensor_parallel_size=${MODEL_TP},gpu_memory_utilization=${MODEL_GPU_MEM_UTIL},max_model_len=${MODEL_MAX_LEN},max_num_seqs=${MODEL_MAX_NUM_SEQS},trust_remote_code=True,enable_thinking=${ENABLE_THINKING}"
COMMAND=(
    "${VENV_PATH}/bin/python"
    -m lmms_eval
    --model vllm
    --model_args "${MODEL_ARGS}"
    --model_processor_compat required
    --model_view_root "${MODEL_VIEW_ROOT}"
    --tasks "${TASKS}"
    --batch_size 1
    --limit "${LIMIT}"
    --log_samples
    --output_path "${OUTPUT_PATH}"
)

echo "[INFO] Qwen3.5 local eval source path: $(readlink -f "${MODEL_SOURCE}")"
echo "[INFO] Qwen3.5 local eval view root: ${MODEL_VIEW_ROOT}"
echo "[INFO] Qwen3.5 local eval tasks: ${TASKS}"
echo "[INFO] Qwen3.5 local eval output: ${OUTPUT_PATH}"
echo "[INFO] Qwen3.5 local eval thinking: ${ENABLE_THINKING}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

export PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${COMMAND[@]}"
