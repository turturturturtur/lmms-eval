#!/bin/bash
# qwen3_vl_worker.sh
# Qwen3.5/Qwen3-VL lmms-eval worker: launch persistent vLLM backends and run
# benchmarks one by one. This mirrors the AutoMix Qwen3.5 eval worker behavior
# while keeping all paths inside this lmms-eval checkout.
#
# Usage:
#   bash run_scripts/qwen3_vl_worker.sh [config.json] [optional_model_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNOVATOR_SITECUSTOMIZE="${INNOVATOR_LMMS_SITECUSTOMIZE:-${SCRIPT_DIR}/lmms_eval_sitecustomize}"

source "${SCRIPT_DIR}/eval_common.sh"

CONFIG="${1:-${SCRIPT_DIR}/config_eval.json}"
CMD_MODEL_PATH="${2:-}"

load_config "${CONFIG}" "${CMD_MODEL_PATH}"
export PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SYSTEM_INSTRUCTION=$(cfg '.eval.system_instruction // ""')
CONFIG_REASONING_PARSER="$(cfg '.model.reasoning_parser // ""')"
MODEL_REASONING_PARSER="${EVAL_REASONING_PARSER:-${CONFIG_REASONING_PARSER}}"
CONFIG_IS_QWEN3_VL="$(cfg '.model.is_qwen3_vl // true')"
MODEL_IS_QWEN3_VL="${EVAL_IS_QWEN3_VL:-${CONFIG_IS_QWEN3_VL}}"
CONFIG_ENABLE_THINKING="$(cfg '.model.enable_thinking // ""')"
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

MODEL_IS_QWEN3_VL="$(normalize_bool_arg "${MODEL_IS_QWEN3_VL}" "model.is_qwen3_vl")"
MODEL_ENABLE_THINKING="$(normalize_bool_arg "${MODEL_ENABLE_THINKING}" "model.enable_thinking" 1)"

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
    local args
    args="base_url=${BACKEND_URLS},model=${MODEL_NAME},api_key=EMPTY,timeout=${VLLM_REQUEST_TIMEOUT_SECONDS},num_concurrent=${CONCURRENCY},adaptive_max_concurrency=${CONCURRENCY},max_new_tokens=${MAX_NEW_TOKENS},max_pixels=${MAX_PIXELS},min_pixels=78400,is_qwen3_vl=${MODEL_IS_QWEN3_VL},shuffle_requests=True"
    if [[ -n "${MODEL_ENABLE_THINKING}" ]]; then
        args="${args},enable_thinking=${MODEL_ENABLE_THINKING}"
    fi
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
    BACKEND_URLS=""
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

        echo "[INFO][Machine ${MACHINE_RANK}] Starting model vLLM  GPUs=${GPUS}  port=${PORT}  backend=FLASHINFER  reasoning_parser=${MODEL_REASONING_PARSER:-none}  is_qwen3_vl=${MODEL_IS_QWEN3_VL}  enable_thinking=${MODEL_ENABLE_THINKING:-unset}..."

        INNOVATOR_LMMS_HIDE_FLASH_ATTN=1 \
        PYTHONPATH="$(innovator_vllm_pythonpath)" \
        CUDA_VISIBLE_DEVICES=${GPUS} "${VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server \
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
            "${reasoning_args[@]}" \
            > "${MODEL_LOG}" 2>&1 &
        PIDS+=($!)
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
    local model_args
    model_args="$(build_vllm_backend_model_args)"

    set +e
    timeout \
        --signal=TERM \
        --kill-after="${TASK_TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TASK_TIMEOUT_SECONDS}s" \
        "${VENV_PATH}/bin/torchrun" \
        --nnodes="${NUM_MACHINES}" \
        --node_rank="${_MACHINE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${eval_master_port}" \
        -m lmms_eval \
        --model       vllm_backend \
        --model_args  "${model_args}" \
        --gen_kwargs  "${GEN_KWARGS}" \
        --tasks       "${task}" \
        --batch_size  1 \
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

    echo "[INFO][Machine ${MACHINE_RANK}] Persistent vLLM lmms-eval mode: ${#TASK_ARRAY[@]} task(s), output=${OUTPUT_PATH}"

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

compute_resources
setup_logging
ensure_venv
ensure_timeout_command
prepend_pythonpath_bins
setup_native_libs
check_runtime_deps
setup_cleanup_trap

launch_vllm_backends

stage_datasets &
DATASET_STAGE_PID=$!

wait_for_backends

if [[ -n "${DATASET_STAGE_PID:-}" ]]; then
    wait "${DATASET_STAGE_PID}" 2>/dev/null || true
fi

run_lmms_eval
