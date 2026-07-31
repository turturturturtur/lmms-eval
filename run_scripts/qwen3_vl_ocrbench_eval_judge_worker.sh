#!/bin/bash
# qwen3_vl_ocrbench_eval_judge_worker.sh
#
# Run OCRBench normal lmms-eval first, then run standalone LLM-as-judge on the
# generated samples in the same DLC job.
#
# Usage:
#   bash run_scripts/qwen3_vl_ocrbench_eval_judge_worker.sh config_eval.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config_eval.json}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "[WARN] jq not found, attempting to install..."
    apt-get update -qq && apt-get install -y -qq jq || { echo "[ERROR] Failed to install jq."; exit 1; }
fi

cfg() { jq -r "$1" "${CONFIG}"; }

TASKS="$(cfg '.eval.tasks // ""')"
if [[ ",${TASKS}," != *",ocrbench,"* && "${TASKS}" != "ocrbench" && "${TASKS}" != *",ocrbench" && "${TASKS}" != "ocrbench,"* ]]; then
    echo "[ERROR] OCRBench eval+judge worker requires eval.tasks to include ocrbench, got: ${TASKS}" >&2
    exit 2
fi

echo "[INFO] Starting OCRBench normal evaluation."
export LMMS_EVAL_STAGE_DATASETS=0
bash "${SCRIPT_DIR}/qwen3_vl_worker.sh" "${CONFIG}"

OUTPUT_PATH_BASE="$(cfg '.eval.output_path')"
TIMESTAMP="$(cfg '.eval.timestamp // ""')"
if [[ -z "${TIMESTAMP}" || "${TIMESTAMP}" == "null" ]]; then
    TIMESTAMP="$(find "${OUTPUT_PATH_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}')"
fi
if [[ -z "${TIMESTAMP}" ]]; then
    echo "[ERROR] Could not resolve eval timestamp under ${OUTPUT_PATH_BASE}" >&2
    exit 3
fi

EVAL_RUN_ROOT="${OUTPUT_PATH_BASE}/${TIMESTAMP}"
OCRBENCH_TASK_DIR="${EVAL_RUN_ROOT}/tasks/ocrbench"
if [[ ! -d "${OCRBENCH_TASK_DIR}" ]]; then
    echo "[ERROR] OCRBench task output directory not found: ${OCRBENCH_TASK_DIR}" >&2
    exit 4
fi

OCRBENCH_SAMPLE_FILE="$(find "${OCRBENCH_TASK_DIR}" -type f -name '*samples_ocrbench*.jsonl' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${OCRBENCH_SAMPLE_FILE}" || ! -f "${OCRBENCH_SAMPLE_FILE}" ]]; then
    echo "[ERROR] Could not find OCRBench sample JSONL under ${OCRBENCH_TASK_DIR}" >&2
    find "${OCRBENCH_TASK_DIR}" -maxdepth 5 -type f -name '*.jsonl' -print >&2 || true
    exit 5
fi

VENV_PATH="$(cfg '.env.venv_path')"
if [[ -z "${VENV_PATH}" || "${VENV_PATH}" == "null" || ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found: ${VENV_PATH}" >&2
    exit 6
fi

HF_HOME_CFG="$(cfg '.env.hf_home')"
LMMS_EVAL_DATASETS_CACHE_CFG="$(cfg '.env.lmms_eval_datasets_cache // ""')"
export HF_HOME="${HF_HOME_CFG}"
if [[ -n "${LMMS_EVAL_DATASETS_CACHE_CFG}" && "${LMMS_EVAL_DATASETS_CACHE_CFG}" != "null" ]]; then
    export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE_CFG}"
    export HF_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE}"
else
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export LMMS_EVAL_DATASETS_CACHE="${HF_DATASETS_CACHE}"
fi
if [[ "${LMMS_EVAL_DATASETS_CACHE}" != "/mnt/cpfsB/evaluation_cache/lmms_eval" || ! -d "/mnt/cpfsB/evaluation_cache/lmms_eval" ]]; then
    echo "[ERROR] OCRBench worker requires mounted benchmark cache /mnt/cpfsB/evaluation_cache/lmms_eval" >&2
    exit 2
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_COLOR=1
export FORCE_COLOR=0
export LOGURU_NO_COLOR=1
export TOKENIZERS_PARALLELISM=false

source_secret_env() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        return
    fi
    local saved_flags=""
    [[ "$-" == *e* ]] && saved_flags="${saved_flags}e"
    [[ "$-" == *u* ]] && saved_flags="${saved_flags}u"
    set +eu
    set -a
    # shellcheck disable=SC1090
    source "${file}"
    set +a
    [[ "${saved_flags}" == *e* ]] && set -e
    [[ "${saved_flags}" == *u* ]] && set -u
}

source_secret_env "${EPIC_ROUTER_ENV_FILE:-}"
source_secret_env "${OMNIV2_JUDGE_ENV_FILE:-}"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_PARALLEL="${JUDGE_PARALLEL:-1}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-180}"
JUDGE_NUM_RETRIES="${JUDGE_NUM_RETRIES:-5}"
JUDGE_RETRY_DELAY="${JUDGE_RETRY_DELAY:-5}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_URL:-${EVAL_OPENAI_API_URL:-${RELAY_BASE_URL:-}}}}}}"
if [[ "${JUDGE_BASE_URL}" == */chat/completions ]]; then
    JUDGE_BASE_URL="${JUDGE_BASE_URL%/chat/completions}"
fi

if [[ -z "${JUDGE_API_KEY}" ]]; then
    echo "[ERROR] JUDGE_API_KEY/OPENAI_API_KEY is not set for OCRBench LLM-as-judge." >&2
    exit 7
fi
if [[ -z "${JUDGE_BASE_URL}" ]]; then
    echo "[ERROR] JUDGE_BASE_URL/OPENAI_API_BASE/OPENAI_API_URL is not set for OCRBench LLM-as-judge." >&2
    exit 8
fi

export JUDGE_MODEL
export JUDGE_API_KEY
export JUDGE_BASE_URL
export JUDGE_MAX_CONCURRENT="${JUDGE_PARALLEL}"
export JUDGE_TIMEOUT
export JUDGE_NUM_RETRIES
export JUDGE_RETRY_DELAY
export JUDGE_MAX_TOKENS
export JUDGE_DISABLE_THINKING="${JUDGE_DISABLE_THINKING:-1}"
export AUTOMIX_DISABLE_THINKING="${AUTOMIX_DISABLE_THINKING:-1}"
export OPENAI_API_KEY="${JUDGE_API_KEY}"
export OPENAI_API_BASE="${JUDGE_BASE_URL}"
export OPENAI_API_URL="${JUDGE_BASE_URL}"
export API_TYPE=openai
export JUDGE_API_TYPE=openai
export PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

JUDGE_OUTPUT_DIR="${EVAL_RUN_ROOT}/judge/ocrbench"
JUDGE_LOG_DIR="${LMMS_EVAL_LOG_DIR:-${EVAL_RUN_ROOT}/logs}"
mkdir -p "${JUDGE_OUTPUT_DIR}" "${JUDGE_LOG_DIR}"
JUDGE_LOG="${JUDGE_LOG_DIR}/ocrbench_llm_judge.log"

echo "[INFO] Starting OCRBench LLM-as-judge."
echo "[INFO] OCRBench samples: ${OCRBENCH_SAMPLE_FILE}"
echo "[INFO] Judge output    : ${JUDGE_OUTPUT_DIR}"
echo "[INFO] Judge model     : ${JUDGE_MODEL}"
echo "[INFO] Judge parallel  : ${JUDGE_PARALLEL}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

set +e
python -m lmms_eval judge \
    --input_result "${OCRBENCH_SAMPLE_FILE}" \
    --task ocrbench \
    --judge-model "${JUDGE_MODEL}" \
    --judge-api-key "${JUDGE_API_KEY}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --parallel "${JUDGE_PARALLEL}" \
    --mode judge \
    --output-dir "${JUDGE_OUTPUT_DIR}" \
    2>&1 | tee "${JUDGE_LOG}"
judge_rc=${PIPESTATUS[0]}
set -e

if [[ "${judge_rc}" != "0" ]]; then
    echo "[ERROR] OCRBench LLM-as-judge failed with exit code ${judge_rc}. Log: ${JUDGE_LOG}" >&2
    exit "${judge_rc}"
fi

JUDGED_SAMPLE_FILE="$(find "${JUDGE_OUTPUT_DIR}" -type f -name '*samples_ocrbench*.jsonl' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${JUDGED_SAMPLE_FILE}" || ! -f "${JUDGED_SAMPLE_FILE}" ]]; then
    echo "[ERROR] OCRBench judge completed but no judged sample JSONL was written under ${JUDGE_OUTPUT_DIR}" >&2
    exit 9
fi

echo "[INFO] OCRBench normal eval and LLM-as-judge completed successfully."
echo "[INFO] Normal samples: ${OCRBENCH_SAMPLE_FILE}"
echo "[INFO] Judged samples: ${JUDGED_SAMPLE_FILE}"
