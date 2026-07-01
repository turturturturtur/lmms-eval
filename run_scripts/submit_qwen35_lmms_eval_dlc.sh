#!/bin/bash
# Minimal local DLC submit script for Qwen3.5 lmms-eval.
#
# Usage:
#   bash lmms-eval/run_scripts/submit_qwen35_lmms_eval_dlc.sh \
#     /mnt/cpfs/<USER>/models/Qwen3.5-9B \
#     EMVista \
#     /mnt/cpfs/<USER>/lmms-eval/eval_result/qwen35_emvista \
#     -1 \
#     off
#
# Positional arguments:
#   1. model path
#   2. lmms-eval tasks, comma separated
#   3. output path
#   4. limit, default -1
#   5. thinking mode: off|on, default off

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    sed -n '3,24p' "${BASH_SOURCE[0]}" >&2
}

MODEL_PATH="${1:-}"
TASKS="${2:-}"
OUTPUT_PATH="${3:-}"
LIMIT="${4:--1}"
THINKING_MODE="${5:-off}"

if [[ -z "${MODEL_PATH}" || -z "${TASKS}" || -z "${OUTPUT_PATH}" ]]; then
    usage
    exit 2
fi

if [[ ! "${LIMIT}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] limit must be an integer, got: ${LIMIT}" >&2
    exit 2
fi

THINKING_MODE="$(printf '%s' "${THINKING_MODE}" | tr '[:upper:]' '[:lower:]')"
case "${THINKING_MODE}" in
    on|true|1|yes)
        ENABLE_THINKING=true
        ;;
    off|false|0|no)
        ENABLE_THINKING=false
        ;;
    *)
        echo "[ERROR] thinking mode must be off or on, got: ${THINKING_MODE}" >&2
        exit 2
        ;;
esac

if ! command -v jq &>/dev/null; then
    echo "[ERROR] jq is required by submit_qwen35_lmms_eval_dlc.sh" >&2
    exit 1
fi

USER_NAME="${LMMS_EVAL_USER:-}"
if [[ -z "${USER_NAME}" ]]; then
    if [[ "${PROJECT_ROOT}" =~ ^/mnt/cpfsB/([^/]+)/Innovator-Tune$ ]]; then
        USER_NAME="${BASH_REMATCH[1]}"
    elif [[ "${PROJECT_ROOT}" =~ ^/mnt/cpfs/([^/]+)/Innovator-Tune$ ]]; then
        USER_NAME="${BASH_REMATCH[1]}"
    else
        echo "[ERROR] Cannot infer user from PROJECT_ROOT=${PROJECT_ROOT}; set LMMS_EVAL_USER." >&2
        exit 2
    fi
fi

DLC_TEMPLATE="${LMMS_EVAL_DLC_CONFIG:-${SCRIPT_DIR}/config_dlc.json}"
EVAL_TEMPLATE="${LMMS_EVAL_EVAL_CONFIG:-${SCRIPT_DIR}/config_eval.json}"
if [[ ! -f "${DLC_TEMPLATE}" ]]; then
    echo "[ERROR] DLC template not found: ${DLC_TEMPLATE}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_TEMPLATE}" ]]; then
    echo "[ERROR] Eval template not found: ${EVAL_TEMPLATE}" >&2
    exit 1
fi

CONFIG_DIR="${LMMS_EVAL_SUBMIT_CONFIG_DIR:-$(mktemp -d /tmp/qwen35_lmms_eval_dlc.XXXXXX)}"
mkdir -p "${CONFIG_DIR}"
DLC_CONFIG="${CONFIG_DIR}/config_dlc.json"
EVAL_CONFIG="${CONFIG_DIR}/config_eval.json"

replace_user_filter='
  def replace_user:
    if type == "object" then
      with_entries(.value |= replace_user)
    elif type == "array" then
      map(replace_user)
    elif type == "string" then
      gsub("<USER>"; $user)
    else
      .
    end;
  replace_user
'

jq \
    --arg user "${USER_NAME}" \
    --arg run_script "${SCRIPT_DIR}/qwen35_worker.sh" \
    "${replace_user_filter} | .dlc.run_script = \$run_script" \
    "${DLC_TEMPLATE}" > "${DLC_CONFIG}"

jq \
    --arg user "${USER_NAME}" \
    --arg model_path "${MODEL_PATH}" \
    --arg tasks "${TASKS}" \
    --arg output_path "${OUTPUT_PATH}" \
    --arg gen_kwargs "${GEN_KWARGS:-}" \
    --argjson limit "${LIMIT}" \
    --argjson enable_thinking "${ENABLE_THINKING}" \
    "${replace_user_filter}
     | .model.path = \$model_path
     | .model.reasoning_parser = \"qwen3\"
     | .model.enable_thinking = \$enable_thinking
     | .model.is_qwen3_vl = false
     | .eval.tasks = \$tasks
     | .eval.output_path = \$output_path
     | .eval.limit = \$limit
     | .eval.gen_kwargs = \$gen_kwargs" \
    "${EVAL_TEMPLATE}" > "${EVAL_CONFIG}"

echo "[INFO] Qwen3.5 lmms-eval DLC config dir: ${CONFIG_DIR}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] tasks=${TASKS}"
echo "[INFO] output_path=${OUTPUT_PATH}"
echo "[INFO] enable_thinking=${ENABLE_THINKING}"

bash "${SCRIPT_DIR}/qwen35_submit.sh" "${DLC_CONFIG}" "${EVAL_CONFIG}"
