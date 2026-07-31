#!/bin/bash
# qwen35_worker.sh
# Qwen3.5 lmms-eval worker: launch persistent vLLM backends and run benchmarks
# one by one. This is a standalone Qwen3.5 path with its own submitter/worker.
#
# Usage:
#   bash run_scripts/qwen35_worker.sh [config.json] [optional_model_path] [optional_judge_config.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNOVATOR_SITECUSTOMIZE="${INNOVATOR_LMMS_SITECUSTOMIZE:-${SCRIPT_DIR}/lmms_eval_sitecustomize}"

source "${SCRIPT_DIR}/eval_common.sh"

CONFIG="${1:-${SCRIPT_DIR}/config_eval.json}"
CMD_MODEL_PATH="${2:-}"
JUDGE_CONFIG="${3:-}"

load_config "${CONFIG}" "${CMD_MODEL_PATH}"
export PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

validate_benchmark_cache_mount() {
    if [[ ! -d "/mnt/cpfsB" ]]; then
        echo "[ERROR] CPFSB mount is missing: /mnt/cpfsB" >&2
        return 2
    fi
    if [[ ! -d "${LMMS_EVAL_BENCHMARK_CACHE}" ]]; then
        echo "[ERROR] Benchmark cache is missing from CPFSB: ${LMMS_EVAL_BENCHMARK_CACHE}" >&2
        return 2
    fi
    if [[ "${LMMS_EVAL_DATASETS_CACHE:-}" != "${LMMS_EVAL_BENCHMARK_CACHE}" ]]; then
        echo "[ERROR] Eval must use benchmark cache ${LMMS_EVAL_BENCHMARK_CACHE}, got: ${LMMS_EVAL_DATASETS_CACHE:-<unset>}" >&2
        return 2
    fi
    if [[ "${HF_DATASETS_CACHE:-}" != "${LMMS_EVAL_BENCHMARK_CACHE}" ]]; then
        echo "[ERROR] HF_DATASETS_CACHE must match benchmark cache ${LMMS_EVAL_BENCHMARK_CACHE}, got: ${HF_DATASETS_CACHE:-<unset>}" >&2
        return 2
    fi
}

validate_benchmark_cache_mount
MODEL_BACKEND="$(cfg '.model.backend // "vllm"')"
MODEL_BACKEND="$(printf '%s' "${MODEL_BACKEND}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MODEL_BACKEND}" != "vllm" && "${MODEL_BACKEND}" != "openai" ]]; then
    echo "[ERROR] model.backend must be vllm or openai, got: ${MODEL_BACKEND}" >&2
    exit 2
fi
if [[ -n "${JUDGE_CONFIG}" ]]; then
    if [[ ! -f "${JUDGE_CONFIG}" ]]; then
        echo "[ERROR] Inline judge config not found: ${JUDGE_CONFIG}" >&2
        exit 2
    fi
    INLINE_JUDGE_BACKEND="$(jq -r '.judge.backend // ""' "${JUDGE_CONFIG}")"
    if [[ "${INLINE_JUDGE_BACKEND}" != "vllm" ]]; then
        echo "[ERROR] qwen35_worker.sh only accepts an inline judge config with judge.backend=vllm, got: ${INLINE_JUDGE_BACKEND:-<empty>}" >&2
        exit 2
    fi
fi
BATCH_SIZE="$(cfg_int '.eval.batch_size // 1')"
if (( BATCH_SIZE < 1 )); then
    echo "[ERROR] eval.batch_size must be positive, got: ${BATCH_SIZE}" >&2
    exit 2
fi
SYSTEM_INSTRUCTION=$(cfg '.eval.system_instruction // ""')
CONFIG_REASONING_PARSER="$(cfg '.model.reasoning_parser // "qwen3"')"
MODEL_REASONING_PARSER="${EVAL_REASONING_PARSER:-${CONFIG_REASONING_PARSER}}"
CONFIG_ENABLE_THINKING="$(cfg '.model.enable_thinking // false')"
MODEL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-${CONFIG_ENABLE_THINKING}}"

normalize_bool_arg() {
    local value="$1"
    local name="$2"
    local allow_empty="${3:-0}"
    local lowered
    lowered="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
    case "${lowered}" in
        1|true|yes|on)
            printf 'True'
            ;;
        0|false|no|off)
            printf 'False'
            ;;
        ""|null)
            if [[ "${allow_empty}" == "1" ]]; then
                printf ''
            else
                echo "[ERROR] ${name} must be a boolean, got empty" >&2
                exit 2
            fi
            ;;
        *)
            echo "[ERROR] ${name} must be a boolean, got: ${value}" >&2
            exit 2
            ;;
    esac
}

if [[ -z "${MODEL_REASONING_PARSER}" || "${MODEL_REASONING_PARSER}" == "null" ]]; then
    MODEL_REASONING_PARSER="qwen3"
fi
if [[ -n "${EVAL_IS_QWEN3_VL:-}" ]]; then
    EVAL_IS_QWEN3_VL_NORMALIZED="$(normalize_bool_arg "${EVAL_IS_QWEN3_VL}" "EVAL_IS_QWEN3_VL")"
    if [[ "${EVAL_IS_QWEN3_VL_NORMALIZED}" != "False" ]]; then
        echo "[ERROR] qwen35_worker.sh is Qwen3.5-only; EVAL_IS_QWEN3_VL must not be true." >&2
        exit 2
    fi
fi
CONFIG_IS_QWEN3_VL="$(cfg '.model.is_qwen3_vl // false')"
CONFIG_IS_QWEN3_VL="$(normalize_bool_arg "${CONFIG_IS_QWEN3_VL}" "model.is_qwen3_vl")"
if [[ "${CONFIG_IS_QWEN3_VL}" != "False" ]]; then
    echo "[ERROR] qwen35_worker.sh is Qwen3.5-only; config model.is_qwen3_vl must be false." >&2
    exit 2
fi
MODEL_IS_QWEN3_VL="False"
MODEL_ENABLE_THINKING="$(normalize_bool_arg "${MODEL_ENABLE_THINKING}" "model.enable_thinking" 1)"
CONFIG_ENFORCE_EAGER="$(cfg '.model.enforce_eager // false')"
MODEL_ENFORCE_EAGER="$(normalize_bool_arg "${CONFIG_ENFORCE_EAGER}" "model.enforce_eager")"

innovator_vllm_pythonpath() {
    if [[ -d "${INNOVATOR_SITECUSTOMIZE}" ]]; then
        if [[ -n "${PYTHONPATH:-}" ]]; then
            echo "${INNOVATOR_SITECUSTOMIZE}:${PYTHONPATH}"
        else
            echo "${INNOVATOR_SITECUSTOMIZE}"
        fi
    else
        echo "${PYTHONPATH:-}"
    fi
}

prepend_pythonpath_bins() {
    if [[ -z "${PYTHONPATH:-}" ]]; then
        return
    fi

    local entry
    local bin_entries=()
    IFS=':' read -ra _PYTHONPATH_BIN_ENTRIES <<< "${PYTHONPATH}"
    for entry in "${_PYTHONPATH_BIN_ENTRIES[@]}"; do
        if [[ -n "${entry}" && -d "${entry}/bin" ]]; then
            bin_entries+=("${entry}/bin")
        fi
    done
    if [[ "${#bin_entries[@]}" -gt 0 ]]; then
        local joined_bins
        joined_bins="$(IFS=:; printf '%s' "${bin_entries[*]}")"
        export PATH="${joined_bins}${PATH:+:${PATH}}"
        echo "[INFO][Machine ${MACHINE_RANK}] PYTHONPATH bin path prepended: ${joined_bins}"
    fi
}

setup_native_libs() {
    local candidates=()
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidates+=("${CONDA_PREFIX}/lib")
    fi
    if [[ -n "${VENV_PATH:-}" && -x "${VENV_PATH}/bin/python" ]]; then
        local resolved_python
        resolved_python="$(readlink -f "${VENV_PATH}/bin/python" 2>/dev/null || true)"
        if [[ -n "${resolved_python}" ]]; then
            candidates+=("$(dirname "$(dirname "${resolved_python}")")/lib")
        fi
    fi
    if [[ -n "${LMMS_EVAL_EXTRA_LD_LIBRARY_DIRS:-}" ]]; then
        local extra_lib_dirs=()
        IFS=':' read -ra extra_lib_dirs <<< "${LMMS_EVAL_EXTRA_LD_LIBRARY_DIRS}"
        candidates+=("${extra_lib_dirs[@]}")
    fi

    local lib_dir
    for lib_dir in "${candidates[@]}"; do
        if [[ -f "${lib_dir}/libstdc++.so.6" ]]; then
            export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
            echo "[INFO][Machine ${MACHINE_RANK}] Native lib path prepended: ${lib_dir}"
            return
        fi
    done

    echo "[WARN][Machine ${MACHINE_RANK}] No conda libstdc++.so.6 found; using system libstdc++."
}

check_runtime_deps() {
    local _vllm_pythonpath
    _vllm_pythonpath="$(innovator_vllm_pythonpath)"
    VENV_PATH="${VENV_PATH}" INNOVATOR_LMMS_HIDE_FLASH_ATTN=1 PYTHONPATH="${_vllm_pythonpath}" "${VENV_PATH}/bin/python" - <<'PY'
import importlib.metadata as md
import importlib.util
import os
import shutil
import sys
from packaging.version import Version

errors = []


def dist_version(name: str):
    try:
        return Version(md.version(name))
    except md.PackageNotFoundError:
        errors.append(f"missing package: {name}")
        return None


hub = dist_version("huggingface-hub")
dist_version("vllm")
if hub is not None and not (Version("0.34.0") <= hub < Version("1.0.0")):
    errors.append(f"huggingface-hub must be >=0.34.0,<1.0, got {hub}")

pydantic = dist_version("pydantic")
if pydantic is not None and pydantic < Version("2.12.0"):
    errors.append(f"pydantic must be >=2.12.0 for vLLM, got {pydantic}")

for name in (
    "pytablewriter",
    "nvidia-ml-py",
    "sacrebleu",
    "evaluate",
    "rouge-score",
    "rouge",
    "jiwer",
    "tenacity",
    "python-dotenv",
    "openpyxl",
):
    dist_version(name)

instrumentator = dist_version("prometheus-fastapi-instrumentator")
if instrumentator is not None and instrumentator < Version("8.0.2"):
    print(
        "[WARN] "
        "prometheus-fastapi-instrumentator must be >=8.0.2 for "
        f"FastAPI/Starlette compatibility, got {instrumentator}",
        file=sys.stderr,
    )

if shutil.which("ninja") is None:
    errors.append("missing executable: ninja")

if importlib.util.find_spec("pynvml") is None:
    errors.append("missing import: pynvml")

if importlib.util.find_spec("flashinfer") is None:
    errors.append("missing import: flashinfer")
else:
    try:
        __import__("flashinfer")
    except Exception as exc:
        errors.append(f"flashinfer import failed: {exc!r}")

if importlib.util.find_spec("flash_attn") is not None:
    errors.append("Innovator lmms-eval guard did not hide system flash_attn")

if errors:
    print("[ERROR] lmms-eval venv dependency check failed:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    print(
        f"Fix with: {os.environ.get('VENV_PATH', '${VENV_PATH}')}/bin/python -m pip install "
        "'huggingface-hub>=0.34,<1' 'pydantic>=2.12,<3' pytablewriter nvidia-ml-py ninja "
        "'prometheus-fastapi-instrumentator>=8.0.2' sacrebleu evaluate rouge-score rouge jiwer tenacity python-dotenv openpyxl",
        file=sys.stderr,
    )
    raise SystemExit(1)

print("[INFO] lmms-eval venv dependency check passed.")
PY
}

validate_qwen35_model_compat() {
    [[ "${MODEL_BACKEND}" == "vllm" ]] || return 0

    local processor_compat
    processor_compat="$(cfg '.model.processor_compat // "required"')"
    case "${processor_compat}" in
        auto|required) ;;
        off)
            echo "[ERROR][Machine ${MACHINE_RANK}] qwen35_worker.sh does not allow model.processor_compat=off for a local Qwen3.5 vLLM model." >&2
            return 2
            ;;
        *)
            echo "[ERROR][Machine ${MACHINE_RANK}] model.processor_compat must be auto, required, or off; got: ${processor_compat}" >&2
            return 2
            ;;
    esac

    local configured_resolved
    configured_resolved="$(cfg '.model.resolved_path // ""')"
    if [[ -n "${configured_resolved}" && "${configured_resolved}" != "null" ]]; then
        if [[ "$(readlink -f "${configured_resolved}")" != "$(readlink -f "${MODEL}")" ]]; then
            echo "[ERROR][Machine ${MACHINE_RANK}] model.path must equal model.resolved_path in worker runtime config: path=${MODEL}, resolved_path=${configured_resolved}" >&2
            return 2
        fi
    fi

    local result_path="${LOG_DIR}/model_worker_preflight.json"
    local stderr_path="${LOG_DIR}/model_worker_preflight.stderr.log"
    local rc
    set +e
    INNOVATOR_LMMS_HIDE_FLASH_ATTN=1 \
        PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${VENV_PATH}/bin/python" \
        "${LMMS_EVAL_ROOT}/lmms_eval/models/model_utils/qwen35_model_compat.py" \
        check \
        --model "${MODEL}" \
        > "${result_path}" \
        2> "${stderr_path}"
    rc=$?
    set -e
    if [[ -s "${stderr_path}" ]]; then
        sed "s/^/[MODEL_PREFLIGHT][Machine ${MACHINE_RANK}] /" "${stderr_path}" >&2
    fi
    if (( rc != 0 )); then
        echo "[ERROR][Machine ${MACHINE_RANK}] Qwen3.5 model check failed before vLLM launch (exit_code=${rc})." >&2
        echo "[ERROR][Machine ${MACHINE_RANK}] model.path=${MODEL}" >&2
        echo "[ERROR][Machine ${MACHINE_RANK}] Re-run qwen35_submit.sh or qwen35_local_eval.sh to prepare a verified compatibility view." >&2
        return "${rc}"
    fi
    if ! jq -e --arg model "$(readlink -f "${MODEL}")" '
        (.resolved_path == $model)
        and (.processor_class | type == "string" and length > 0)
        and (.transformers_version | type == "string" and length > 0)
    ' "${result_path}" >/dev/null; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Invalid Qwen3.5 worker preflight result: ${result_path}" >&2
        return 2
    fi
    echo "[INFO][Machine ${MACHINE_RANK}] Qwen3.5 model preflight passed: model=${MODEL} processor=$(jq -r '.processor_class' "${result_path}") transformers=$(jq -r '.transformers_version' "${result_path}")"
}

resolve_qwen35_stop_token_ids() {
    local _vllm_pythonpath
    _vllm_pythonpath="$(innovator_vllm_pythonpath)"
    MODEL_STOP_TOKEN_IDS_JSON="$(
        INNOVATOR_LMMS_HIDE_FLASH_ATTN=1 \
        PYTHONPATH="${_vllm_pythonpath}" \
        "${VENV_PATH}/bin/python" - "${MODEL}" <<'PY'
import json
import os
import sys

from transformers import AutoTokenizer


model = sys.argv[1]
stop_token = "<|im_end|>"
tokenizer = AutoTokenizer.from_pretrained(
    model,
    trust_remote_code=True,
    local_files_only=os.environ.get("TRANSFORMERS_OFFLINE") == "1",
)
token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
if len(token_ids) != 1:
    raise ValueError(
        f"Qwen3.5 stop token {stop_token!r} must encode to exactly one token, got {token_ids!r}"
    )
token_id = token_ids[0]
if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
    raise ValueError(f"Qwen3.5 stop token ID must be a non-negative integer, got {token_id!r}")
decoded_token = tokenizer.convert_ids_to_tokens(token_id)
if decoded_token != stop_token:
    raise ValueError(
        f"Qwen3.5 stop token round-trip mismatch: expected {stop_token!r}, got {decoded_token!r}"
    )
print(json.dumps([token_id], separators=(",", ":")))
PY
    )"
    if ! jq -e 'type == "array" and length == 1 and all(.[]; type == "number" and floor == . and . >= 0)' \
        <<< "${MODEL_STOP_TOKEN_IDS_JSON}" >/dev/null; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Invalid Qwen3.5 stop token ID payload: ${MODEL_STOP_TOKEN_IDS_JSON}" >&2
        return 2
    fi
    echo "[INFO][Machine ${MACHINE_RANK}] Qwen3.5 model stop token IDs: ${MODEL_STOP_TOKEN_IDS_JSON}"
}

check_api_runtime_deps() {
    VENV_PATH="${VENV_PATH}" "${VENV_PATH}/bin/python" - <<'PY'
import importlib.metadata as md
import sys

errors = []
for name in (
    "openai",
    "pytablewriter",
    "sacrebleu",
    "evaluate",
    "rouge-score",
    "rouge",
    "jiwer",
    "tenacity",
    "python-dotenv",
    "openpyxl",
):
    try:
        md.version(name)
    except md.PackageNotFoundError:
        errors.append(f"missing package: {name}")

if errors:
    print("[ERROR] lmms-eval API dependency check failed:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)

print("[INFO] lmms-eval API dependency check passed.")
PY
}

compute_api_resources() {
    LOCAL_GPU_NUM=0
    NPROC_PER_NODE=1
    NUM_MACHINES=${WORLD_SIZE}
    MACHINE_RANK=${RANK}
    MAIN_GPU_NUM=0
    NUM_BACKENDS=0
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

task_slug() {
    local task="$1"
    local slug
    slug="$(printf '%s' "${task}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//')"
    if [[ -z "${slug}" ]]; then
        slug="task"
    fi
    printf '%s' "${slug}"
}

build_vllm_backend_model_args() {
    if [[ -z "${MODEL_STOP_TOKEN_IDS_JSON:-}" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Qwen3.5 stop token IDs were not resolved" >&2
        return 2
    fi
    local args
    args="base_url=${BACKEND_URLS},model=${MODEL_NAME},api_key=EMPTY,timeout=${VLLM_REQUEST_TIMEOUT_SECONDS},num_concurrent=${CONCURRENCY},adaptive_max_concurrency=${CONCURRENCY},max_new_tokens=${MAX_NEW_TOKENS},max_pixels=${MAX_PIXELS},min_pixels=78400,is_qwen3_vl=${MODEL_IS_QWEN3_VL},shuffle_requests=True,stop_token_ids=${MODEL_STOP_TOKEN_IDS_JSON}"
    if [[ -n "${EVAL_MODEL_STOP_STRINGS_JSON:-}" ]]; then
        if ! jq -e 'type == "array" and length > 0 and all(.[]; type == "string" and length > 0)' \
            <<< "${EVAL_MODEL_STOP_STRINGS_JSON}" >/dev/null; then
            echo "[ERROR][Machine ${MACHINE_RANK}] EVAL_MODEL_STOP_STRINGS_JSON must be a non-empty JSON array of non-empty strings" >&2
            return 2
        fi
        args="${args},stop_strings=${EVAL_MODEL_STOP_STRINGS_JSON}"
    fi
    if [[ -n "${MODEL_ENABLE_THINKING}" ]]; then
        args="${args},enable_thinking=${MODEL_ENABLE_THINKING}"
    fi
    printf '%s' "${args}"
}

build_openai_model_args() {
    if [[ -z "${OPENAI_API_URL:-}" || "${OPENAI_API_URL}" == "null" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] env.openai_api_url is required for API evaluation." >&2
        exit 2
    fi
    if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "null" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] env.openai_api_key is required for API evaluation." >&2
        exit 2
    fi

    local args
    args="model=${MODEL},num_concurrent=${CONCURRENCY},adaptive_max_concurrency=${CONCURRENCY},max_new_tokens=${MAX_NEW_TOKENS},max_pixels=${MAX_PIXELS},min_pixels=78400,is_qwen3_vl=${MODEL_IS_QWEN3_VL},httpx_trust_env=False"
    printf '%s' "${args}"
}

split_lmms_tasks() {
    TASK_ARRAY=()
    local raw_task task
    IFS=',' read -ra _RAW_TASK_ARRAY <<< "${TASKS}"
    for raw_task in "${_RAW_TASK_ARRAY[@]}"; do
        task="$(trim_whitespace "${raw_task}")"
        if [[ -n "${task}" ]]; then
            TASK_ARRAY+=("${task}")
        fi
    done

    if [[ "${#TASK_ARRAY[@]}" -eq 0 ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] No lmms-eval tasks parsed from TASKS=${TASKS}" >&2
        exit 2
    fi
}

write_task_manifest_row() {
    local task="$1"
    local status="$2"
    local started_at="$3"
    local ended_at="$4"
    local rc="$5"
    local task_output_path="$6"
    local status_reason="$7"
    local log_path="$8"

    if [[ "${MACHINE_RANK}" != "0" ]]; then
        return
    fi

    local manifest="${OUTPUT_PATH}/task_status.tsv"
    local expected_header="task	status	started_at	ended_at	exit_code	output_path	status_reason	log_path"
    if [[ ! -f "${manifest}" ]]; then
        printf '%s\n' "${expected_header}" > "${manifest}"
    else
        local actual_header
        actual_header="$(head -n 1 "${manifest}")"
        if [[ "${actual_header}" != "${expected_header}" ]]; then
            echo "[ERROR][Machine ${MACHINE_RANK}] Unexpected task manifest header in ${manifest}: ${actual_header}" >&2
            return 2
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task}" "${status}" "${started_at}" "${ended_at}" "${rc}" "${task_output_path}" "${status_reason}" "${log_path}" \
        >> "${manifest}"
}

launch_vllm_backends() {
    if ! command -v setsid >/dev/null 2>&1; then
        echo "[ERROR][Machine ${MACHINE_RANK}] setsid is required for owned vLLM process-group cleanup." >&2
        return 2
    fi
    BACKEND_URLS=""
    PIDS=()
    BACKEND_LOGS=()
    for (( i=0; i<NUM_BACKENDS; i++ )); do
        PORT=$(( MODEL_BASE_PORT + i ))
        START_GPU=$(( i * MODEL_TP ))
        GPUS=""
        for (( g=START_GPU; g<START_GPU+MODEL_TP; g++ )); do
            GPUS="${GPUS}${g},"
        done
        GPUS=${GPUS%,}

        MODEL_LOG="${LOG_DIR}/vllm_model_rank${RANK}_port${PORT}.log"
        local reasoning_args=()
        if [[ -n "${MODEL_REASONING_PARSER}" && "${MODEL_REASONING_PARSER}" != "null" ]]; then
            reasoning_args=(--reasoning-parser "${MODEL_REASONING_PARSER}")
        fi
        local eager_args=()
        if [[ "${MODEL_ENFORCE_EAGER}" == "True" ]]; then
            eager_args=(--enforce-eager)
        fi

        echo "[INFO][Machine ${MACHINE_RANK}] Starting model vLLM  GPUs=${GPUS}  port=${PORT}  backend=FLASHINFER  reasoning_parser=${MODEL_REASONING_PARSER:-none}  is_qwen3_vl=${MODEL_IS_QWEN3_VL}  enable_thinking=${MODEL_ENABLE_THINKING:-unset}  enforce_eager=${MODEL_ENFORCE_EAGER}..."

        INNOVATOR_LMMS_HIDE_FLASH_ATTN=1 \
        PYTHONPATH="$(innovator_vllm_pythonpath)" \
        CUDA_VISIBLE_DEVICES=${GPUS} setsid "${VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server \
            --model                  "${MODEL}" \
            --served-model-name      "${MODEL_NAME}" \
            --tensor-parallel-size   "${MODEL_TP}" \
            --max-model-len          "${MODEL_MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${MODEL_GPU_MEM_UTIL}" \
            --max-num-seqs           "${MODEL_MAX_NUM_SEQS}" \
            --port                   "${PORT}" \
            --attention-backend      FLASHINFER \
            --mm-encoder-tp-mode data \
            --trust-remote-code \
            --enable-prefix-caching \
            "${eager_args[@]}" \
            "${reasoning_args[@]}" \
            > "${MODEL_LOG}" 2>&1 &
        PIDS+=("$!")
        BACKEND_LOGS+=("${MODEL_LOG}")
        BACKEND_URLS="${BACKEND_URLS}http://localhost:${PORT}/v1;"
    done
    BACKEND_URLS=${BACKEND_URLS%;}
}

run_lmms_eval_task() {
    local task="$1"
    local task_index="$2"
    local task_total="$3"
    export SKIP_MMBENCH_DEV_JUDGE=1

    local _MACHINE_RANK=${MACHINE_RANK}
    local _RANK=${RANK}
    local slug
    slug="$(task_slug "${task}")"
    local task_output_path="${OUTPUT_PATH}/tasks/${slug}"
    local eval_log="${LOG_DIR}/lmms_eval_rank${_RANK}_${slug}.log"
    local eval_master_port=$(( MASTER_PORT + task_index - 1 ))
    local started_at
    started_at="$(date -Is)"

    mkdir -p "${task_output_path}"
    if [[ "${_MACHINE_RANK}" == "0" ]]; then
        jq --arg task "${task}" '.eval.tasks = $task' "${CONFIG}" > "${task_output_path}/config.json"
    fi

    echo "[INFO][Machine ${_MACHINE_RANK}] Launching lmms-eval task ${task_index}/${task_total}: ${task}  output=${task_output_path}  log=${eval_log}"

    local system_args=()
    if [[ -n "${SYSTEM_INSTRUCTION}" && "${SYSTEM_INSTRUCTION}" != "null" ]]; then
        system_args=(--system_instruction "${SYSTEM_INSTRUCTION}")
    fi
    local model_backend_name
    local model_args
    local launcher=()
    local batch_size_arg="1"
    if [[ "${MODEL_BACKEND}" == "openai" ]]; then
        model_backend_name="openai"
        model_args="$(build_openai_model_args)"
        launcher=("${VENV_PATH}/bin/python")
        batch_size_arg="${BATCH_SIZE}"
    else
        model_backend_name="vllm_backend"
        model_args="$(build_vllm_backend_model_args)"
        launcher=(
            "${VENV_PATH}/bin/python" -m torch.distributed.run
            --nnodes="${NUM_MACHINES}"
            --node_rank="${_MACHINE_RANK}"
            --nproc_per_node="${NPROC_PER_NODE}"
            --master_addr="${MASTER_ADDR}"
            --master_port="${eval_master_port}"
        )
        batch_size_arg="1"
    fi
    local gen_args=()
    if [[ -n "${GEN_KWARGS}" && "${GEN_KWARGS}" != "null" ]]; then
        gen_args=(--gen_kwargs "${GEN_KWARGS}")
    fi

    set +e
    timeout \
        --signal=TERM \
        --kill-after="${TASK_TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TASK_TIMEOUT_SECONDS}s" \
        "${launcher[@]}" \
        -m lmms_eval \
        --model       "${model_backend_name}" \
        --model_args  "${model_args}" \
        "${gen_args[@]}" \
        --tasks       "${task}" \
        --batch_size  "${batch_size_arg}" \
        --output_path "${task_output_path}" \
        --verbosity   "${VERBOSITY}" \
        --log_samples \
        "${system_args[@]}" \
        --limit "${LIMIT}" \
        > "${eval_log}" 2>&1
    local rc=$?
    set -e

    local ended_at
    ended_at="$(date -Is)"
    local classification
    classification="$(classify_lmms_eval_task_status "${rc}" "${TASK_TIMEOUT_SECONDS}")"
    local task_status="${classification%%$'\t'*}"
    local status_reason="${classification#*$'\t'}"
    if [[ "${task_status}" == "success" && "${_MACHINE_RANK}" == "0" ]]; then
        local validation_error
        if ! validation_error="$(validate_lmms_eval_task_outputs "${task_output_path}" 2>&1)"; then
            task_status="failed"
            status_reason="missing_eval_outputs: ${validation_error}"
            rc=1
        fi
    fi

    if [[ "${task_status}" == "success" ]]; then
        echo "[INFO][Machine ${_MACHINE_RANK}] lmms-eval task completed: ${task}"
        write_task_manifest_row "${task}" "${task_status}" "${started_at}" "${ended_at}" "${rc}" "${task_output_path}" "${status_reason}" "${eval_log}"
    elif [[ "${task_status}" == "timeout" ]]; then
        echo "[ERROR][Machine ${_MACHINE_RANK}] lmms-eval task timed out: ${task}  timeout=${TASK_TIMEOUT_SECONDS}s  kill_after=${TASK_TIMEOUT_KILL_AFTER_SECONDS}s  exit_code=${rc}  log=${eval_log}" >&2
        clean_failed_task_output "${task}" "${task_output_path}"
        write_task_manifest_row "${task}" "${task_status}" "${started_at}" "${ended_at}" "${rc}" "${task_output_path}" "${status_reason}" "${eval_log}"
    elif [[ "${task_status}" == "failed" ]]; then
        echo "[ERROR][Machine ${_MACHINE_RANK}] lmms-eval task failed: ${task}  exit_code=${rc}  log=${eval_log}" >&2
        clean_failed_task_output "${task}" "${task_output_path}"
        write_task_manifest_row "${task}" "${task_status}" "${started_at}" "${ended_at}" "${rc}" "${task_output_path}" "${status_reason}" "${eval_log}"
    else
        echo "[ERROR][Machine ${_MACHINE_RANK}] Unknown classified task status for ${task}: ${task_status}" >&2
        return 2
    fi

    return "${rc}"
}

run_lmms_eval() {
    split_lmms_tasks

    mkdir -p "${OUTPUT_PATH}/tasks"
    if [[ "${MACHINE_RANK}" == "0" ]]; then
        cp "${CONFIG}" "${OUTPUT_PATH}/config.json"
    fi

    echo "[INFO][Machine ${MACHINE_RANK}] lmms-eval ${MODEL_BACKEND} mode: ${#TASK_ARRAY[@]} task(s), output=${OUTPUT_PATH}"

    local failed_tasks=()
    local index=0
    local task
    for task in "${TASK_ARRAY[@]}"; do
        index=$(( index + 1 ))
        if ! run_lmms_eval_task "${task}" "${index}" "${#TASK_ARRAY[@]}"; then
            failed_tasks+=("${task}")
        fi
    done

    if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Evaluation finished with failed task(s): ${failed_tasks[*]}" >&2
        return 1
    fi

    echo "[INFO][Machine ${MACHINE_RANK}] Evaluation completed successfully for all tasks."
}

wait_for_eval_gpu_release() {
    local active_pids=""
    local attempt
    echo "[INFO][Machine ${MACHINE_RANK}] Waiting for all eval GPU processes to exit before local judge..."
    for attempt in {1..60}; do
        active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u | tr '\n' ' ')"
        if [[ -z "${active_pids}" ]]; then
            echo "[INFO][Machine ${MACHINE_RANK}] Eval GPU processes released; all 8 GPUs are available for local judge."
            return 0
        fi
        sleep 2
    done
    echo "[ERROR][Machine ${MACHINE_RANK}] Eval GPU processes did not exit before local judge: ${active_pids}" >&2
    return 1
}

run_inline_local_judge() {
    if [[ -z "${JUDGE_CONFIG}" ]]; then
        return 0
    fi
    if (( WORLD_SIZE != 1 || RANK != 0 )); then
        echo "[ERROR] Inline local judge requires WORLD_SIZE=1 and RANK=0, got WORLD_SIZE=${WORLD_SIZE} RANK=${RANK}" >&2
        return 2
    fi
    local gpu_count
    gpu_count="$(nvidia-smi -L | wc -l)"
    if (( gpu_count != 8 )); then
        echo "[ERROR] Inline local judge requires exactly 8 visible GPUs, got: ${gpu_count}" >&2
        return 2
    fi

    if [[ "${MODEL_BACKEND}" == "vllm" ]]; then
        echo "[INFO][Machine ${MACHINE_RANK}] Eval succeeded; stopping eval vLLM backends before inline local judge."
        cleanup_vllm
    fi
    wait_for_eval_gpu_release

    echo "[INFO][Machine ${MACHINE_RANK}] Starting inline local vLLM judge in the current DLC worker."
    bash "${SCRIPT_DIR}/run_judge.sh" "${JUDGE_CONFIG}"
    echo "[INFO][Machine ${MACHINE_RANK}] Inline local vLLM judge completed successfully."
}

if [[ "${MODEL_BACKEND}" == "openai" ]]; then
    compute_api_resources
    setup_logging
    ensure_venv
    ensure_timeout_command
    prepend_pythonpath_bins
    setup_native_libs
    check_api_runtime_deps

    stage_datasets &
    DATASET_STAGE_PID=$!
    if [[ -n "${DATASET_STAGE_PID:-}" ]]; then
        wait "${DATASET_STAGE_PID}"
    fi

    run_lmms_eval
else
    compute_resources
    setup_logging
    ensure_venv
    ensure_timeout_command
    prepend_pythonpath_bins
    setup_native_libs
    check_runtime_deps
    validate_qwen35_model_compat
    resolve_qwen35_stop_token_ids
    setup_cleanup_trap

    launch_vllm_backends

    stage_datasets &
    DATASET_STAGE_PID=$!

    if ! wait_for_backends; then
        if [[ -n "${DATASET_STAGE_PID:-}" ]] && kill -0 "${DATASET_STAGE_PID}" 2>/dev/null; then
            kill -TERM "${DATASET_STAGE_PID}" 2>/dev/null || true
        fi
        [[ -n "${DATASET_STAGE_PID:-}" ]] && wait "${DATASET_STAGE_PID}" 2>/dev/null || true
        exit 1
    fi

    if [[ -n "${DATASET_STAGE_PID:-}" ]]; then
        wait "${DATASET_STAGE_PID}"
    fi

    run_lmms_eval
fi

run_inline_local_judge
