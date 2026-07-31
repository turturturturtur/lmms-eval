#!/bin/bash
# eval_common.sh
# Shared utilities for lmms-eval + vLLM worker scripts.
# Usage: source "$(dirname "$0")/eval_common.sh"

set -euo pipefail

# Pre-populated benchmark cache mounted from CPFSB.  DLC workers must consume
# this cache directly; copying it to another shared path only adds startup I/O.
LMMS_EVAL_BENCHMARK_CACHE="/mnt/cpfsB/evaluation_cache/lmms_eval"

# ── Guard: must be sourced ────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[ERROR] eval_common.sh should be sourced, not executed directly."
    exit 1
fi

# ── JSON helpers ──────────────────────────────────────────────────────────────
cfg()     { jq -r "$1"       "${CONFIG}"; }
cfg_bool() { jq -r "$1 // false" "${CONFIG}"; }
cfg_int() { jq -r "$1 // 0" "${CONFIG}"; }

cfg_required_positive_int() {
    local jq_expr="$1"
    local name="$2"
    local value

    if ! value="$(jq -er "${jq_expr}" "${CONFIG}")"; then
        echo "[ERROR] Missing required positive integer config: ${name}" >&2
        exit 2
    fi
    if [[ -z "${value}" || "${value}" == "null" ]]; then
        echo "[ERROR] Missing required positive integer config: ${name}" >&2
        exit 2
    fi
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
    if (( value <= 0 )); then
        echo "[ERROR] ${name} must be > 0, got: ${value}" >&2
        exit 2
    fi
    printf '%s' "${value}"
}

ensure_timeout_command() {
    if ! command -v timeout &>/dev/null; then
        echo "[ERROR] GNU timeout command is required for per-task lmms-eval hard timeouts." >&2
        exit 2
    fi
}

classify_lmms_eval_task_status() {
    local rc="$1"
    local timeout_seconds="$2"

    if ! [[ "${rc}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] task exit code must be a non-negative integer, got: ${rc}" >&2
        return 2
    fi
    if ! [[ "${timeout_seconds}" =~ ^[0-9]+$ ]] || (( timeout_seconds <= 0 )); then
        echo "[ERROR] task timeout seconds must be a positive integer, got: ${timeout_seconds}" >&2
        return 2
    fi

    case "${rc}" in
        0)
            printf 'success\tcompleted'
            ;;
        124|137|143)
            printf 'timeout\ttimeout_after_%ss' "${timeout_seconds}"
            ;;
        *)
            printf 'failed\texit_code_%s' "${rc}"
            ;;
    esac
}

clean_failed_task_output() {
    local task="$1"
    local task_output_path="$2"

    if [[ "${MACHINE_RANK}" != "0" ]]; then
        return
    fi
    if [[ "${task_output_path}" == "${OUTPUT_PATH}/tasks" || "${task_output_path}" == "${OUTPUT_PATH}/tasks/" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Refusing to clean task root path: ${task_output_path}" >&2
        return 2
    fi
    case "${task_output_path}" in
        "${OUTPUT_PATH}/tasks/"*)
            ;;
        *)
            echo "[ERROR][Machine ${MACHINE_RANK}] Refusing to clean unexpected task output path: ${task_output_path}" >&2
            return 2
            ;;
    esac

    rm -rf -- "${task_output_path}"
    mkdir -p "${task_output_path}"
    jq --arg task "${task}" '.eval.tasks = $task' "${CONFIG}" > "${task_output_path}/config.json"
}

validate_lmms_eval_task_outputs() {
    local task_output_path="$1"
    local result_count sample_count

    result_count="$(find "${task_output_path}" -type f -name '*results*.json' | wc -l)"
    sample_count="$(find "${task_output_path}" -type f -name '*samples_*.jsonl' | wc -l)"
    if (( result_count == 0 )); then
        echo "missing results json under ${task_output_path}" >&2
        return 1
    fi
    if (( sample_count == 0 )); then
        echo "missing samples jsonl under ${task_output_path}" >&2
        return 1
    fi
}

# ── parse gen_kwargs ──────────────────────────────────────────────────────────
parse_gen_kwarg() {
    local key=$1
    local default=$2
    local value=""
    local item
    IFS=',' read -ra _GEN_KWARG_ITEMS <<< "${GEN_KWARGS}"
    for item in "${_GEN_KWARG_ITEMS[@]}"; do
        if [[ "${item}" == "${key}="* ]]; then
            value="${item#*=}"
            break
        fi
    done
    echo "${value:-$default}"
}

# ── load configuration and derive variables ───────────────────────────────────
load_config() {
    CONFIG="${1:-$(dirname "$0")/config_eval.json}"
    CMD_MODEL_PATH="${2:-}"

    [[ ! -f "${CONFIG}" ]] && { echo "[ERROR] Config not found: ${CONFIG}"; exit 1; }
    if ! command -v jq &>/dev/null; then
        echo "[WARN] jq not found, attempting to install..."
        apt-get update -qq && apt-get install -y -qq jq || { echo "[ERROR] Failed to install jq."; exit 1; }
    fi

    # environment
    API_TYPE=$(cfg '.env.api_type // ""')
    [[ -n "${API_TYPE}" && "${API_TYPE}" != "null" ]] && export API_TYPE="${API_TYPE}"

    OPENAI_API_KEY=$(cfg '.env.openai_api_key // ""')
    [[ -n "${OPENAI_API_KEY}" && "${OPENAI_API_KEY}" != "null" ]] && export OPENAI_API_KEY="${OPENAI_API_KEY}"

    OPENAI_API_URL=$(cfg '.env.openai_api_url // ""')
    [[ -n "${OPENAI_API_URL}" && "${OPENAI_API_URL}" != "null" ]] && export OPENAI_API_URL="${OPENAI_API_URL}"

    JUDGE_API_KEY=$(cfg '.env.judge_api_key // ""')
    [[ -n "${JUDGE_API_KEY}" && "${JUDGE_API_KEY}" != "null" ]] && export JUDGE_API_KEY="${JUDGE_API_KEY}"

    JUDGE_BASE_URL=$(cfg '.env.judge_base_url // ""')
    [[ -n "${JUDGE_BASE_URL}" && "${JUDGE_BASE_URL}" != "null" ]] && export JUDGE_BASE_URL="${JUDGE_BASE_URL}"

    export HF_HOME=$(cfg '.env.hf_home')
    export HF_TOKEN=$(cfg '.env.hf_token')
    HF_DATASETS_CACHE_CFG=$(cfg '.env.hf_datasets_cache // ""')
    if [[ -n "${HF_DATASETS_CACHE_CFG}" && "${HF_DATASETS_CACHE_CFG}" != "null" ]]; then
        export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_CFG}"
    else
        export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    fi

    LMMS_EVAL_DATASETS_CACHE=$(cfg '.env.lmms_eval_datasets_cache // ""')
    if [[ -n "${LMMS_EVAL_DATASETS_CACHE}" && "${LMMS_EVAL_DATASETS_CACHE}" != "null" ]]; then
        export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE}"
        # Task utilities such as SuperChem use HF_DATASETS_CACHE directly;
        # keep it aligned with lmms-eval's explicit cache override.
        if [[ -n "${HF_DATASETS_CACHE_CFG}" && "${HF_DATASETS_CACHE_CFG}" != "null" && "${HF_DATASETS_CACHE_CFG}" != "${LMMS_EVAL_DATASETS_CACHE}" ]]; then
            echo "[ERROR] env.hf_datasets_cache must equal env.lmms_eval_datasets_cache, got: ${HF_DATASETS_CACHE_CFG} vs ${LMMS_EVAL_DATASETS_CACHE}" >&2
            exit 2
        fi
        export HF_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE}"
    fi

    HF_DATASETS_OFFLINE=$(cfg_bool '.env.hf_datasets_offline')
    TRANSFORMERS_OFFLINE=$(cfg_bool '.env.transformers_offline')
    [[ "${HF_DATASETS_OFFLINE}" == "true" ]] && export HF_DATASETS_OFFLINE=1 || unset HF_DATASETS_OFFLINE
    [[ "${TRANSFORMERS_OFFLINE}" == "true" ]] && export TRANSFORMERS_OFFLINE=1 || unset TRANSFORMERS_OFFLINE

    export NO_COLOR=1
    export FORCE_COLOR=0
    export LOGURU_NO_COLOR=1

    VENV_PATH=$(cfg '.env.venv_path')

    # logs
    LOG_BASE=$(cfg '.log.dir')

    # distributed
    MASTER_ADDR="${MASTER_ADDR:-$(cfg '.distributed.master_addr')}"
    MASTER_PORT="${MASTER_PORT:-$(cfg_int '.distributed.master_port')}"
    WORLD_SIZE="${WORLD_SIZE:-$(cfg_int '.distributed.world_size')}"
    RANK="${RANK:-$(cfg_int '.distributed.rank')}"

    # model
    MODEL_FROM_JSON=$(cfg '.model.path')
    MODEL="${CMD_MODEL_PATH:-$MODEL_FROM_JSON}"
    MODEL_TP=$(cfg_int '.model.tp')
    MODEL_MAX_MODEL_LEN=$(cfg_int '.model.max_model_len')
    MODEL_GPU_MEM_UTIL=$(cfg '.model.gpu_memory_utilization')
    MODEL_MAX_NUM_SEQS=$(cfg_int '.model.max_num_seqs')
    MODEL_BASE_PORT=$(cfg_int '.model.base_port')
    MODEL_NAME=$(cfg '.model.served_model_name // ""')
    if [[ -z "${MODEL_NAME}" || "${MODEL_NAME}" == "null" ]]; then
        MODEL_NAME=$(basename "${MODEL}")
    fi
    MODEL_STARTUP_TIMEOUT_SECONDS=$(cfg_required_positive_int '.model.startup_timeout_seconds // 1800' 'model.startup_timeout_seconds')

    # eval
    TASKS=$(cfg '.eval.tasks')
    OUTPUT_PATH_BASE=$(cfg '.eval.output_path')
    # 优先使用 config 里由 submitter 写入的统一时间戳；本地运行时不存在则自行生成
    TIMESTAMP=$(cfg '.eval.timestamp // ""')
    [[ -z "${TIMESTAMP}" || "${TIMESTAMP}" == "null" ]] && TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    OUTPUT_PATH="${OUTPUT_PATH_BASE}/${TIMESTAMP}"
    CONCURRENCY=$(cfg_int '.eval.concurrency // 128')
    LIMIT=$(cfg_int '.eval.limit // -1')

    DEBUG=$(cfg '.eval.debug // false')
    [[ "${DEBUG}" == "null" || -z "${DEBUG}" ]] && DEBUG="false"
    if [[ "${DEBUG}" == "true" ]]; then
        VERBOSITY="DEBUG"
    else
        VERBOSITY=$(cfg '.eval.verbosity')
        [[ "${VERBOSITY}" == "null" || -z "${VERBOSITY}" ]] && VERBOSITY="INFO"
    fi

    GEN_KWARGS=$(cfg '.eval.gen_kwargs // ""')
    MAX_NEW_TOKENS=$(parse_gen_kwarg "max_new_tokens" "32768")
    MAX_PIXELS=$(parse_gen_kwarg "max_pixels" "4014080")
    VLLM_REQUEST_TIMEOUT_SECONDS=$(cfg_required_positive_int '.eval.vllm_request_timeout_seconds // 300' 'eval.vllm_request_timeout_seconds')
    TASK_TIMEOUT_SECONDS=$(cfg_required_positive_int '.eval.task_timeout_seconds' 'eval.task_timeout_seconds')
    TASK_TIMEOUT_KILL_AFTER_SECONDS=$(cfg_required_positive_int '.eval.task_timeout_kill_after_seconds' 'eval.task_timeout_kill_after_seconds')
}

# ── validate that virtual environment exists ──────────────────────────────────
ensure_venv() {
    if [[ -z "${VENV_PATH}" || "${VENV_PATH}" == "null" ]]; then
        VENV_PATH="$(dirname "$0")/../.venv"
    fi
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "[ERROR] Virtual environment not found: ${VENV_PATH}"
        exit 1
    fi
    echo "[INFO][Machine ${MACHINE_RANK}] Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
}

# ── optionally stage pre-cached datasets ──────────────────────────────────────
stage_datasets() {
    # 仅在 DLC 提交场景下由 submitter 显式开启；本地调试默认跳过 staging
    if [[ "${LMMS_EVAL_STAGE_DATASETS:-}" != "1" ]]; then
        return
    fi

    local src="${LMMS_EVAL_BENCHMARK_CACHE}"
    local dst="${LMMS_EVAL_DATASETS_CACHE:-}"
    if [[ ! -d "${src}" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] Benchmark cache is not mounted: ${src}" >&2
        return 2
    fi
    if [[ -z "${dst}" || "${dst}" == "null" ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] LMMS_EVAL_DATASETS_CACHE is required when staging is enabled." >&2
        return 2
    fi

    local src_real dst_real
    src_real="$(readlink -f -- "${src}")"
    dst_real="$(readlink -f -- "${dst}" 2>/dev/null || true)"
    if [[ "${src_real}" == "${dst_real}" ]]; then
        echo "[INFO][Machine ${MACHINE_RANK}] Dataset staging skipped; using existing benchmark cache ${src_real}."
        return 0
    fi

    echo "[INFO][Machine ${MACHINE_RANK}] Staging datasets from ${src} to ${dst} ..."
    mkdir -p -- "${dst}"
    cp -r -- "${src}"/* "${dst}/"
}

# ── compute GPU / machine role ────────────────────────────────────────────────
compute_resources() {
    LOCAL_GPU_NUM=$(nvidia-smi -L | wc -l)
    MAIN_GPU_NUM=${LOCAL_GPU_NUM}
    if [[ -n "${LMMS_EVAL_MAIN_GPU_NUM:-}" ]]; then
        if ! [[ "${LMMS_EVAL_MAIN_GPU_NUM}" =~ ^[0-9]+$ ]] || (( LMMS_EVAL_MAIN_GPU_NUM <= 0 )); then
            echo "[ERROR] LMMS_EVAL_MAIN_GPU_NUM must be a positive integer, got: ${LMMS_EVAL_MAIN_GPU_NUM}" >&2
            exit 2
        fi
        if (( LMMS_EVAL_MAIN_GPU_NUM > LOCAL_GPU_NUM )); then
            echo "[ERROR] LMMS_EVAL_MAIN_GPU_NUM(${LMMS_EVAL_MAIN_GPU_NUM}) > local GPUs(${LOCAL_GPU_NUM})" >&2
            exit 2
        fi
        MAIN_GPU_NUM=${LMMS_EVAL_MAIN_GPU_NUM}
    fi
    NPROC_PER_NODE=${MAIN_GPU_NUM}
    if [[ "${WORLD_SIZE}" -le "${NPROC_PER_NODE}" ]]; then
        # DLC semantic: WORLD_SIZE = num_machines, RANK = machine_rank
        NUM_MACHINES=${WORLD_SIZE}
        MACHINE_RANK=${RANK}
    else
        # Traditional accelerate semantic: WORLD_SIZE = total processes
        NUM_MACHINES=$(( (WORLD_SIZE + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
        MACHINE_RANK=$(( RANK / NPROC_PER_NODE ))
    fi
    NUM_BACKENDS=$(( LOCAL_GPU_NUM / MODEL_TP ))
    if [[ -n "${LMMS_EVAL_NUM_BACKENDS:-}" ]]; then
        if ! [[ "${LMMS_EVAL_NUM_BACKENDS}" =~ ^[0-9]+$ ]] || (( LMMS_EVAL_NUM_BACKENDS <= 0 )); then
            echo "[ERROR] LMMS_EVAL_NUM_BACKENDS must be a positive integer, got: ${LMMS_EVAL_NUM_BACKENDS}" >&2
            exit 2
        fi
        if (( LMMS_EVAL_NUM_BACKENDS * MODEL_TP > LOCAL_GPU_NUM )); then
            echo "[ERROR] LMMS_EVAL_NUM_BACKENDS(${LMMS_EVAL_NUM_BACKENDS}) * MODEL_TP(${MODEL_TP}) > local GPUs(${LOCAL_GPU_NUM})" >&2
            exit 2
        fi
        NUM_BACKENDS=${LMMS_EVAL_NUM_BACKENDS}
    fi

    if (( MODEL_TP > LOCAL_GPU_NUM )); then
        echo "[ERROR] MODEL_TP(${MODEL_TP}) > local GPUs(${LOCAL_GPU_NUM})"
        exit 1
    fi
    if (( NUM_BACKENDS == 0 )); then
        echo "[ERROR] NUM_BACKENDS is 0, check model.tp config"
        exit 1
    fi
}

# ── setup logging directory ───────────────────────────────────────────────────
setup_logging() {
    if [[ -n "${LMMS_EVAL_LOG_DIR:-}" ]]; then
        LOG_DIR="${LMMS_EVAL_LOG_DIR}"
    else
        LOG_DIR="${LOG_BASE}/$(date +%Y-%m-%d_%H-%M-%S)"
    fi
    mkdir -p "${LOG_DIR}"

    echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Config          : ${CONFIG}"
    echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Rank            : ${RANK}/${WORLD_SIZE}  master=${MASTER_ADDR}:${MASTER_PORT}"
    echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Local GPUs      : ${LOCAL_GPU_NUM}  main=${MAIN_GPU_NUM} (TP=${MODEL_TP}, backends=${NUM_BACKENDS})"
    echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Log dir         : ${LOG_DIR}"
    if [[ "${DEBUG}" == "true" ]]; then
        echo "[WARN][Machine ${MACHINE_RANK}/${NUM_MACHINES}] DEBUG mode    : ENABLED (vLLM backends will NOT be killed on exit)"
    fi
}

# ── process cleanup ───────────────────────────────────────────────────────────
PIDS=()
BACKEND_LOGS=()
cleanup_vllm() {
    trap - EXIT INT TERM
    if [[ "${DEBUG}" == "true" ]]; then
        echo "[INFO][Machine ${MACHINE_RANK}] DEBUG mode enabled, skipping vLLM cleanup."
        echo "[INFO][Machine ${MACHINE_RANK}] PIDs to keep running: ${PIDS[*]}"
        echo "[INFO][Machine ${MACHINE_RANK}] To manually stop: kill ${PIDS[*]}"
        return
    fi
    [[ ${#PIDS[@]} -eq 0 ]] && return
    echo "[INFO][Machine ${MACHINE_RANK}] Stopping owned vLLM process groups..."
    local pid
    for pid in "${PIDS[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        elif kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    local grace_seconds="${LMMS_EVAL_CLEANUP_GRACE_SECONDS:-10}"
    local deadline=$((SECONDS + grace_seconds))
    local still_running=1
    while (( still_running == 1 && SECONDS < deadline )); do
        still_running=0
        for pid in "${PIDS[@]}"; do
            if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
                still_running=1
                break
            fi
        done
        (( still_running == 1 )) && sleep 1
    done
    for pid in "${PIDS[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null; then
            echo "[WARN][Machine ${MACHINE_RANK}] Force-killing owned vLLM process group ${pid}" >&2
            kill -KILL -- "-${pid}" 2>/dev/null || true
        elif kill -0 "${pid}" 2>/dev/null; then
            echo "[WARN][Machine ${MACHINE_RANK}] Force-killing owned vLLM PID ${pid} before process-group initialization" >&2
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    done
    PIDS=()
    BACKEND_LOGS=()
    echo "[INFO][Machine ${MACHINE_RANK}] Done."
}
setup_cleanup_trap() {
    if ! [[ "${LMMS_EVAL_CLEANUP_GRACE_SECONDS:-10}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] LMMS_EVAL_CLEANUP_GRACE_SECONDS must be a non-negative integer, got: ${LMMS_EVAL_CLEANUP_GRACE_SECONDS}" >&2
        return 2
    fi
    trap cleanup_vllm EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

# ── launch vLLM backends ──────────────────────────────────────────────────────
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
        echo "[INFO][Machine ${MACHINE_RANK}] Starting model vLLM  GPUs=${GPUS}  port=${PORT}..."

        CUDA_VISIBLE_DEVICES=${GPUS} setsid "${VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server \
            --model                  "${MODEL}" \
            --served-model-name      "${MODEL_NAME}" \
            --tensor-parallel-size   "${MODEL_TP}" \
            --max-model-len          "${MODEL_MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${MODEL_GPU_MEM_UTIL}" \
            --max-num-seqs           "${MODEL_MAX_NUM_SEQS}" \
            --port                   "${PORT}" \
            --mm-encoder-tp-mode data \
            --trust-remote-code \
            --enable-prefix-caching \
            > "${MODEL_LOG}" 2>&1 &
        PIDS+=("$!")
        BACKEND_LOGS+=("${MODEL_LOG}")
        BACKEND_URLS="${BACKEND_URLS}http://localhost:${PORT}/v1;"
    done
    BACKEND_URLS=${BACKEND_URLS%;}
}

# ── wait for backends to be ready ─────────────────────────────────────────────
wait_for_backends() {
    local url_array=()
    IFS=';' read -ra url_array <<< "${BACKEND_URLS}"
    if (( ${#url_array[@]} == 0 )); then
        echo "[ERROR][Machine ${MACHINE_RANK}] No backend URLs were registered." >&2
        return 1
    fi
    if (( ${#url_array[@]} != ${#PIDS[@]} )); then
        echo "[ERROR][Machine ${MACHINE_RANK}] Backend URL/PID count mismatch: urls=${#url_array[@]} pids=${#PIDS[@]}" >&2
        return 1
    fi
    if (( ${#url_array[@]} != ${#BACKEND_LOGS[@]} )); then
        echo "[ERROR][Machine ${MACHINE_RANK}] Backend URL/log count mismatch: urls=${#url_array[@]} logs=${#BACKEND_LOGS[@]}" >&2
        return 1
    fi

    local poll_seconds="${LMMS_EVAL_BACKEND_POLL_SECONDS:-2}"
    if ! [[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR][Machine ${MACHINE_RANK}] LMMS_EVAL_BACKEND_POLL_SECONDS must be a positive integer, got: ${poll_seconds}" >&2
        return 2
    fi
    local deadline=$((SECONDS + MODEL_STARTUP_TIMEOUT_SECONDS))
    local ready=()
    local index
    for (( index=0; index<${#url_array[@]}; index++ )); do
        ready+=(0)
    done

    echo "[INFO][Machine ${MACHINE_RANK}] Waiting for all backends to be ready (timeout ${MODEL_STARTUP_TIMEOUT_SECONDS}s, expected_model=${MODEL_NAME})..."
    while true; do
        local pending=0
        for (( index=0; index<${#url_array[@]}; index++ )); do
            local pid="${PIDS[index]}"
            local url="${url_array[index]}"
            local log_path="${BACKEND_LOGS[index]}"
            if ! kill -0 "${pid}" 2>/dev/null; then
                local exit_code=0
                wait "${pid}" || exit_code=$?
                echo "[ERROR][Machine ${MACHINE_RANK}] Backend ${url} PID ${pid} exited before readiness (exit_code=${exit_code})." >&2
                echo "[ERROR][Machine ${MACHINE_RANK}] Last 120 lines from ${log_path}:" >&2
                tail -n 120 "${log_path}" >&2 2>/dev/null || true
                return 1
            fi
            [[ "${ready[index]}" == "1" ]] && continue
            pending=$((pending + 1))

            local response_file
            response_file="$(mktemp "${TMPDIR:-/tmp}/lmms_eval_models_XXXXXX.json")"
            local http_code
            http_code="$(curl -sS --connect-timeout 2 --max-time 5 \
                -o "${response_file}" -w "%{http_code}" "${url}/models" 2>/dev/null || true)"
            if [[ "${http_code}" == "200" ]]; then
                local observed_models
                if ! observed_models="$(jq -cer '[.data[]?.id] | if length > 0 and all(.[]; type == "string" and length > 0) then . else error("missing model ids") end' "${response_file}" 2>/dev/null)"; then
                    echo "[ERROR][Machine ${MACHINE_RANK}] Backend ${url} returned HTTP 200 with an invalid /v1/models payload: $(tr '\n' ' ' < "${response_file}")" >&2
                    rm -f "${response_file}"
                    return 1
                fi
                if ! jq -e --arg expected "${MODEL_NAME}" 'index($expected) != null' <<< "${observed_models}" >/dev/null; then
                    echo "[ERROR][Machine ${MACHINE_RANK}] Backend model identity mismatch at ${url}: expected=${MODEL_NAME}, observed=${observed_models}" >&2
                    echo "[ERROR][Machine ${MACHINE_RANK}] Last 120 lines from ${log_path}:" >&2
                    tail -n 120 "${log_path}" >&2 2>/dev/null || true
                    rm -f "${response_file}"
                    return 1
                fi
                ready[index]=1
                pending=$((pending - 1))
                echo "[INFO][Machine ${MACHINE_RANK}] Ready: ${url} model=${MODEL_NAME} pid=${pid}"
            fi
            rm -f "${response_file}"
        done

        if (( pending == 0 )); then
            for (( index=0; index<${#PIDS[@]}; index++ )); do
                if ! kill -0 "${PIDS[index]}" 2>/dev/null; then
                    local final_exit_code=0
                    wait "${PIDS[index]}" || final_exit_code=$?
                    echo "[ERROR][Machine ${MACHINE_RANK}] Backend ${url_array[index]} PID ${PIDS[index]} exited before readiness returned (exit_code=${final_exit_code})." >&2
                    tail -n 120 "${BACKEND_LOGS[index]}" >&2 2>/dev/null || true
                    return 1
                fi
            done
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "[ERROR][Machine ${MACHINE_RANK}] Timed out after ${MODEL_STARTUP_TIMEOUT_SECONDS}s waiting for ${pending} backend(s)." >&2
            for (( index=0; index<${#url_array[@]}; index++ )); do
                if [[ "${ready[index]}" != "1" ]]; then
                    echo "[ERROR][Machine ${MACHINE_RANK}] Not ready: ${url_array[index]} pid=${PIDS[index]} log=${BACKEND_LOGS[index]}" >&2
                    tail -n 120 "${BACKEND_LOGS[index]}" >&2 2>/dev/null || true
                fi
            done
            return 1
        fi
        sleep "${poll_seconds}"
    done
}

# ── run lmms-eval ─────────────────────────────────────────────────────────────
run_lmms_eval() {
    export SKIP_MMBENCH_DEV_JUDGE=1

    mkdir -p "${OUTPUT_PATH}"
    cp "${CONFIG}" "${OUTPUT_PATH}/config.json"
    local _MACHINE_RANK=${MACHINE_RANK}
    local _RANK=${RANK}
    EVAL_LOG="${LOG_DIR}/lmms_eval_rank${_RANK}.log"
    echo "[INFO][Machine ${_MACHINE_RANK}] Launching lmms-eval  tasks=${TASKS}  output=${OUTPUT_PATH}  log= ${EVAL_LOG}"

    # Use torchrun directly instead of accelerate launch.
    # DLC PyTorchJob sets WORLD_SIZE/RANK as node-level info, but accelerate launch
    # gets confused by these env vars and only spawns a single process.
    # torchrun handles multi-node/multi-process correctly and lmms-eval auto-detects
    # torch.distributed.is_initialized() to set distributed_executor_backend=torchrun.
    "${VENV_PATH}/bin/torchrun" \
        --nnodes="${NUM_MACHINES}" \
        --node_rank="${_MACHINE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m lmms_eval \
        --model       vllm_backend \
        --model_args  "base_url=${BACKEND_URLS},model=${MODEL_NAME},api_key=EMPTY,timeout=${VLLM_REQUEST_TIMEOUT_SECONDS},num_concurrent=${CONCURRENCY},adaptive_max_concurrency=${CONCURRENCY},max_new_tokens=${MAX_NEW_TOKENS},max_pixels=${MAX_PIXELS},min_pixels=78400,is_qwen3_vl=True,shuffle_requests=True" \
        --gen_kwargs  "${GEN_KWARGS}" \
        --tasks       "${TASKS}" \
        --batch_size  1 \
        --output_path "${OUTPUT_PATH}" \
        --verbosity   "${VERBOSITY}" \
        --log_samples \
        --limit "${LIMIT}" \
        > "${EVAL_LOG}" 2>&1

    echo "[INFO][Machine ${_MACHINE_RANK}] Evaluation completed successfully."
}
