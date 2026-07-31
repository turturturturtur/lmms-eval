#!/bin/bash
# tools/start_vllm_judge.sh
#
# 启动本地 vLLM 作为 judge 后端，并等待其就绪。
#
# 用法：
#   bash tools/start_vllm_judge.sh \
#       --model-path <path> \
#       --served-model-name <name> \
#       --tp <int> \
#       --max-model-len <int> \
#       --gpu-memory-utilization <float> \
#       --max-num-seqs <int> \
#       --port <int> \
#       --log <log_file>
#
# 成功启动后输出机器可读元数据：VLLM_PID=<pid> 与 VLLM_OWNED=<0|1>

set -euo pipefail

# ── 解析参数 ─────────────────────────────────────────────────────────────────
MODEL_PATH=""
SERVED_MODEL_NAME=""
TP=1
MAX_MODEL_LEN=32768
GPU_MEM_UTIL="0.8"
MAX_NUM_SEQS=512
PORT=8002
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"; shift 2 ;;
        --tp)
            TP="$2"; shift 2 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-memory-utilization)
            GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --log)
            LOG_FILE="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${MODEL_PATH}" ]] && { echo "[ERROR] --model-path is required" >&2; exit 1; }
[[ -z "${LOG_FILE}" ]] && { echo "[ERROR] --log is required" >&2; exit 1; }
[[ ! -d "${MODEL_PATH}" ]] && { echo "[ERROR] Model directory not found: ${MODEL_PATH}" >&2; exit 1; }
for pair in "tp:${TP}" "max-model-len:${MAX_MODEL_LEN}" "max-num-seqs:${MAX_NUM_SEQS}" "port:${PORT}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "[ERROR] --${name} must be a positive integer, got: ${value}" >&2
        exit 1
    fi
done
if ! [[ "${GPU_MEM_UTIL}" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || [[ "${GPU_MEM_UTIL}" == "0" ]]; then
    echo "[ERROR] --gpu-memory-utilization must be in (0, 1], got: ${GPU_MEM_UTIL}" >&2
    exit 1
fi

VISIBLE_GPU_TOKENS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra VISIBLE_GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_COUNT="$(nvidia-smi -L | wc -l)"
    for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
        VISIBLE_GPU_TOKENS+=("${gpu}")
    done
fi
if (( ${#VISIBLE_GPU_TOKENS[@]} < TP )); then
    echo "[ERROR] Not enough visible GPUs for TP=${TP}: ${#VISIBLE_GPU_TOKENS[@]} available" >&2
    exit 1
fi
SELECTED_GPU_TOKENS=("${VISIBLE_GPU_TOKENS[@]:0:TP}")
JUDGE_CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${SELECTED_GPU_TOKENS[*]}")"

# 确保日志目录存在
mkdir -p "$(dirname "${LOG_FILE}")"

# 默认 served model name 与模型路径 basename 一致
if [[ -z "${SERVED_MODEL_NAME}" ]]; then
    SERVED_MODEL_NAME=$(basename "${MODEL_PATH}")
fi

JUDGE_BASE_URL="http://localhost:${PORT}/v1"

# ── 检查是否已有可用的 vLLM 在跑 ──────────────────────────────────────────────
check_existing_vllm() {
    local url="$1"
    local expected_model="$2"
    local http_status
    http_status=$(curl -sS --connect-timeout 2 --max-time 5 \
        -o /dev/null -w "%{http_code}" "${url}/models" 2>/dev/null || echo "000")
    [[ "${http_status}" != "200" ]] && return 1

    # 用 python 检查返回的模型列表中是否包含 expected_model
    local matched
    matched=$(curl -sS --connect-timeout 2 --max-time 5 "${url}/models" 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m.get('id','') for m in data.get('data',[])]
    expected = sys.argv[1]
    print('true' if expected in models else 'false')
except Exception:
    print('false')
" "${expected_model}" 2>/dev/null)

    [[ "${matched}" == "true" ]]
}

if check_existing_vllm "${JUDGE_BASE_URL}" "${SERVED_MODEL_NAME}"; then
    echo "[INFO] Found existing vLLM on port ${PORT} with model ${SERVED_MODEL_NAME}, reusing it." >&2
    # 尝试找到已有进程的 PID
    EXISTING_PID=$(lsof -ti :"${PORT}" 2>/dev/null | head -n1 || echo "")
    echo "VLLM_PID=${EXISTING_PID}"
    echo "VLLM_OWNED=0"
    exit 0
fi

# ── 启动新的 vLLM ────────────────────────────────────────────────────────────
echo "[INFO] Starting vLLM judge backend..." >&2
echo "[INFO] Model: ${MODEL_PATH}" >&2
echo "[INFO] Served model name: ${SERVED_MODEL_NAME}" >&2
echo "[INFO] TP: ${TP}, Port: ${PORT}" >&2
echo "[INFO] CUDA_VISIBLE_DEVICES: ${JUDGE_CUDA_VISIBLE_DEVICES}" >&2
echo "[INFO] Log file: ${LOG_FILE}" >&2

# 使用独立 session/process group，保证 cleanup 能清理 vLLM 的 EngineCore 子进程。
if ! command -v setsid >/dev/null 2>&1; then
    echo "[ERROR] setsid is required for owned vLLM process-group cleanup." >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES}" setsid python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --port "${PORT}" \
    --attention-backend FLASHINFER \
    --mm-encoder-tp-mode data \
    --enforce-eager \
    --enable-prefix-caching \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
cleanup_failed_start() {
    trap - EXIT INT TERM
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill -TERM "${VLLM_PID}" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-${VLLM_PID}" 2>/dev/null || kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup_failed_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 等待 vLLM 就绪
echo "[INFO] Waiting for vLLM to be ready (timeout: 10min)..." >&2
check_http() {
    curl -sS --connect-timeout 2 --max-time 5 \
        -o /dev/null -w "%{http_code}" "$1/models" 2>/dev/null
}
retries=0
while [[ "$(check_http "${JUDGE_BASE_URL}")" != "200" ]]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM judge backend exited before becoming ready. Tail of ${LOG_FILE}:" >&2
        tail -n 80 "${LOG_FILE}" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    retries=$((retries + 1))
    if (( retries >= 120 )); then
        echo "[ERROR] Timeout waiting for vLLM" >&2
        exit 1
    fi
    echo "[INFO] Waiting... (${retries}/120)" >&2
done
if ! check_existing_vllm "${JUDGE_BASE_URL}" "${SERVED_MODEL_NAME}"; then
    echo "[ERROR] vLLM judge backend model identity mismatch after HTTP readiness: expected=${SERVED_MODEL_NAME}" >&2
    tail -n 80 "${LOG_FILE}" 2>/dev/null || true
    exit 1
fi

trap - EXIT INT TERM
echo "[INFO] vLLM judge backend ready at ${JUDGE_BASE_URL}" >&2
echo "VLLM_PID=${VLLM_PID}"
echo "VLLM_OWNED=1"
