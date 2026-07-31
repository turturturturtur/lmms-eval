#!/bin/bash
# run_judge.sh
#
# 用法：bash run_judge.sh [config.json]
#
# 功能：
#   1. 支持两种 judge 后端：
#      - api: 使用 OpenAI 兼容 API（如 yunwu.ai, DashScope 等）
#      - vllm: 本地启动 vLLM 作为 judge 后端
#   2. 根据配置文件自动选择后端并执行 judge
#   3. 支持批量 judge 多个 task

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ══════════════════════════════════════════════════════════════════════════════
# §0  读取 JSON 配置
# ══════════════════════════════════════════════════════════════════════════════
CONFIG="${1:-${SCRIPT_DIR}/config_judge.json}"
# 如果默认配置文件不存在，尝试使用 config_math.json 作为备选
if [[ ! -f "${CONFIG}" && "${CONFIG}" == *"config_judge.json" ]]; then
    if [[ -f "${SCRIPT_DIR}/config_math.json" ]]; then
        echo "[INFO] config_judge.json not found, trying config_math.json"
        CONFIG="${SCRIPT_DIR}/config_math.json"
    fi
fi
[[ ! -f "${CONFIG}" ]] && { echo "[ERROR] Config not found: ${CONFIG}"; exit 1; }
JSON_PYTHON="$(command -v python3 || command -v python || true)"
[[ -z "${JSON_PYTHON}" ]] && { echo "[ERROR] python3 or python is required to read JSON config."; exit 1; }

cfg() {
    "${JSON_PYTHON}" - "${CONFIG}" "$1" <<'PY'
import ast
import json
import sys

config_path, expr = sys.argv[1], sys.argv[2]
with open(config_path, encoding="utf-8") as f:
    data = json.load(f)


def parse_default(raw):
    raw = raw.strip()
    if raw == "empty":
        return ""
    if raw == "null":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw


path_expr, default = expr, None
has_default = False
if " // " in expr:
    path_expr, default_expr = expr.split(" // ", 1)
    default = parse_default(default_expr)
    has_default = True

if not path_expr.startswith("."):
    raise SystemExit(f"Unsupported config expression: {expr}")

value = data
missing = False
for part in [p for p in path_expr[1:].split(".") if p]:
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        missing = True
        break

if missing or value is None:
    value = default if has_default else None

if value is None:
    print("null")
elif isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}
cfg_int() {
    if [[ "$1" == *" // "* ]]; then
        cfg "$1"
    else
        cfg "$1 // 0"
    fi
}
cfg_bool() { 
    local expr="$1"
    if [[ "${expr}" != *" // "* ]]; then
        expr="${expr} // false"
    fi
    local val=$(cfg "${expr}")
    [[ "$val" == "true" ]] && echo "true" || echo "false"
}

# ── 环境 ──────────────────────────────────────────────────────────────────────
export HF_HOME=$(cfg '.env.hf_home // "/mnt/cpfsB/public_data/public_dataset/.cache/huggingface"')
# Priority: existing env var (e.g. from ~/.bashrc) > config file
export HF_TOKEN="${HF_TOKEN:-$(cfg '.env.hf_token // empty')}"
HF_DATASETS_CACHE_CFG=$(cfg '.env.hf_datasets_cache // empty')
if [[ -n "${HF_DATASETS_CACHE_CFG}" && "${HF_DATASETS_CACHE_CFG}" != "null" ]]; then
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_CFG}"
else
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
fi
# unset HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=1

# Disable ANSI colors in logs (for clean log files)
export NO_COLOR=1
export FORCE_COLOR=0
export LOGURU_NO_COLOR=1

# 虚拟环境路径
VENV_PATH=$(cfg '.env.venv_path // "/mnt/cpfsB/<USER>/Innovator-Tune/lmms-eval/.venv"')

# 数据集缓存与离线模式
LMMS_EVAL_DATASETS_CACHE=$(cfg '.env.lmms_eval_datasets_cache // empty')
if [[ -n "${LMMS_EVAL_DATASETS_CACHE}" && "${LMMS_EVAL_DATASETS_CACHE}" != "null" ]]; then
    export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE}"
    # Judge task utilities also use HF_DATASETS_CACHE directly; keep both
    # variables on the same pre-populated benchmark cache.
    if [[ -n "${HF_DATASETS_CACHE_CFG}" && "${HF_DATASETS_CACHE_CFG}" != "null" && "${HF_DATASETS_CACHE_CFG}" != "${LMMS_EVAL_DATASETS_CACHE}" ]]; then
        echo "[ERROR] env.hf_datasets_cache must equal env.lmms_eval_datasets_cache, got: ${HF_DATASETS_CACHE_CFG} vs ${LMMS_EVAL_DATASETS_CACHE}" >&2
        exit 2
    fi
    export HF_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE}"
fi

validate_benchmark_cache_mount() {
    if [[ ! -d "/mnt/cpfsB" || ! -d "/mnt/cpfsB/evaluation_cache/lmms_eval" ]]; then
        echo "[ERROR] CPFSB benchmark cache is not mounted: /mnt/cpfsB/evaluation_cache/lmms_eval" >&2
        return 2
    fi
    if [[ "${LMMS_EVAL_DATASETS_CACHE:-}" != "/mnt/cpfsB/evaluation_cache/lmms_eval" ]]; then
        echo "[ERROR] Judge must use benchmark cache /mnt/cpfsB/evaluation_cache/lmms_eval, got: ${LMMS_EVAL_DATASETS_CACHE:-<unset>}" >&2
        return 2
    fi
    if [[ "${HF_DATASETS_CACHE:-}" != "/mnt/cpfsB/evaluation_cache/lmms_eval" ]]; then
        echo "[ERROR] HF_DATASETS_CACHE must match benchmark cache /mnt/cpfsB/evaluation_cache/lmms_eval, got: ${HF_DATASETS_CACHE:-<unset>}" >&2
        return 2
    fi
}

HF_DATASETS_OFFLINE=$(cfg_bool '.env.hf_datasets_offline')
[[ "${HF_DATASETS_OFFLINE}" == "true" ]] && export HF_DATASETS_OFFLINE=1 || unset HF_DATASETS_OFFLINE

TRANSFORMERS_OFFLINE=$(cfg_bool '.env.transformers_offline')
[[ "${TRANSFORMERS_OFFLINE}" == "true" ]] && export TRANSFORMERS_OFFLINE=1 || unset TRANSFORMERS_OFFLINE

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_BASE=$(cfg '.log.dir // "/mnt/cpfsB/<USER>/vllm_logs"')
LOG_DIR="${LOG_BASE}/judge_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOG_DIR}"

# ── Judge 配置 ────────────────────────────────────────────────────────────────
JUDGE_BACKEND=$(cfg '.judge.backend // "vllm"')  # api 或 vllm
JUDGE_PARALLEL=$(cfg_int '.judge.parallel // 32')

# ── API 后端配置 (当 backend=api 时使用) ───────────────────────────────────────
API_KEY=$(cfg '.judge.api.key // empty')
API_BASE_URL=$(cfg '.judge.api.base_url // empty')

# ── vLLM 后端配置 (当 backend=vllm 时使用) ─────────────────────────────────────
VLLM_MODEL_PATH=$(cfg '.judge.vllm.model_path // "/mnt/cpfsB/tianleniu/Innovator-Tune/models/Qwen3.5-9B"')
VLLM_PROCESSOR_COMPAT=$(cfg '.judge.vllm.processor_compat // "auto"')
VLLM_TP=$(cfg_int '.judge.vllm.tp // 8')
VLLM_MAX_MODEL_LEN=$(cfg_int '.judge.vllm.max_model_len // 40960')
VLLM_GPU_MEM_UTIL=$(cfg '.judge.vllm.gpu_memory_utilization // "0.88"')
VLLM_MAX_NUM_SEQS=$(cfg_int '.judge.vllm.max_num_seqs // 192')
VLLM_PORT=$(cfg_int '.judge.vllm.port // 8002')
JUDGE_MODEL=$(cfg '.judge.model // empty')
if [[ -z "${JUDGE_MODEL}" || "${JUDGE_MODEL}" == "null" ]]; then
    JUDGE_MODEL=$(basename "${VLLM_MODEL_PATH}")
fi

case "${JUDGE_BACKEND}" in
    api|vllm) ;;
    *) echo "[ERROR] Unknown judge backend: ${JUDGE_BACKEND}. Use 'api' or 'vllm'" >&2; exit 2 ;;
esac
if ! [[ "${JUDGE_PARALLEL}" =~ ^[0-9]+$ ]] || (( JUDGE_PARALLEL < 1 )); then
    echo "[ERROR] judge.parallel must be a positive integer, got: ${JUDGE_PARALLEL}" >&2
    exit 2
fi
if [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    if [[ ! -d "${VLLM_MODEL_PATH}" ]]; then
        echo "[ERROR] Local judge model directory not found: ${VLLM_MODEL_PATH}" >&2
        exit 2
    fi
    case "${VLLM_PROCESSOR_COMPAT}" in
        off|auto|required) ;;
        *)
            echo "[ERROR] judge.vllm.processor_compat must be off, auto, or required; got: ${VLLM_PROCESSOR_COMPAT}" >&2
            exit 2
            ;;
    esac
    for pair in "tp:${VLLM_TP}" "max_model_len:${VLLM_MAX_MODEL_LEN}" "max_num_seqs:${VLLM_MAX_NUM_SEQS}" "port:${VLLM_PORT}"; do
        name="${pair%%:*}"
        value="${pair#*:}"
        if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
            echo "[ERROR] judge.vllm.${name} must be a positive integer, got: ${value}" >&2
            exit 2
        fi
    done
fi

# ── 评测配置 ──────────────────────────────────────────────────────────────────
# 优先从 input_result_path 读取，如果没有则尝试从旧的 output_path 读取
INPUT_RESULT_PATH=$(cfg '.eval.input_result_path // empty')
if [[ -z "${INPUT_RESULT_PATH}" || "${INPUT_RESULT_PATH}" == "null" ]]; then
    INPUT_RESULT_PATH=$(cfg '.eval.output_path // "/mnt/cpfsB/<USER>/eval_result"')
fi
TASKS=$(cfg '.eval.tasks // empty')
OUTPUT_PATH=$(cfg '.eval.output_path // "/mnt/cpfsB/<USER>/judge_results"')
VERBOSITY=$(cfg '.eval.verbosity // "INFO"')
SKIP_JUDGED=$(cfg_bool '.eval.skip_judged // false')

# ── Debug 模式 ────────────────────────────────────────────────────────────────
DEBUG=$(cfg_bool '.eval.debug // false')

# ══════════════════════════════════════════════════════════════════════════════
# §1  日志 & 摘要
# ══════════════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "lmms-eval Judge Script"
echo "=========================================="
echo "[INFO] Config          : ${CONFIG}"
echo "[INFO] Judge Backend   : ${JUDGE_BACKEND}"
echo "[INFO] Judge Model     : ${JUDGE_MODEL}"
echo "[INFO] Judge Parallel  : ${JUDGE_PARALLEL}"
echo "[INFO] Tasks           : ${TASKS}"
echo "[INFO] Input Path      : ${INPUT_RESULT_PATH}"
echo "[INFO] Output Path     : ${OUTPUT_PATH}"
echo "[INFO] Log dir         : ${LOG_DIR}"
if [[ "${DEBUG}" == "true" ]]; then
    echo "[WARN] DEBUG mode    : ENABLED"
fi
echo "=========================================="

# ══════════════════════════════════════════════════════════════════════════════
# §2  激活虚拟环境
# ══════════════════════════════════════════════════════════════════════════════
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found: ${VENV_PATH}"
    exit 1
fi

echo "[INFO] Activating virtual environment: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
INNOVATOR_SITECUSTOMIZE="${LMMS_EVAL_ROOT}/run_scripts/lmms_eval_sitecustomize"
export PYTHONPATH="${INNOVATOR_SITECUSTOMIZE}:${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export INNOVATOR_LMMS_HIDE_FLASH_ATTN=1

# 验证 lmms-eval 可用
if ! python -c "import lmms_eval" 2>/dev/null; then
    echo "[ERROR] lmms-eval not found in virtual environment"
    echo "[INFO] Please install: pip install -e /mnt/cpfsB/<USER>/lmms-eval"
    exit 1
fi

validate_qwen35_judge_model_compat() {
    [[ "${JUDGE_BACKEND}" == "vllm" ]] || return 0
    [[ "${VLLM_PROCESSOR_COMPAT}" != "off" ]] || return 0

    local result_path="${LOG_DIR}/judge_model_preflight.json"
    local stderr_path="${LOG_DIR}/judge_model_preflight.stderr.log"
    local rc
    set +e
    python - "${VLLM_MODEL_PATH}" "${VLLM_PROCESSOR_COMPAT}" >"${result_path}" 2>"${stderr_path}" <<'PY'
import json
import sys

from lmms_eval.models.model_utils.qwen35_model_compat import (
    SUPPORTED_MODEL_TYPES,
    ModelCompatibilityError,
    check_model,
    get_local_model_type,
)


model_path, mode = sys.argv[1:]
try:
    model_type = get_local_model_type(model_path)
    if model_type not in SUPPORTED_MODEL_TYPES:
        if mode == "required":
            raise ModelCompatibilityError(
                "required judge processor compatibility only supports Qwen3.5; "
                f"got model_type={model_type!r}: {model_path}"
            )
        payload = {
            "model_type": model_type,
            "processor_compat": mode,
            "resolved_path": model_path,
            "status": "not_applicable",
        }
    else:
        payload = {
            **check_model(model_path),
            "processor_compat": mode,
            "status": "verified",
        }
except ModelCompatibilityError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    raise SystemExit(2)

print(json.dumps(payload, indent=2, sort_keys=True))
PY
    rc=$?
    set -e
    if [[ -s "${stderr_path}" ]]; then
        sed 's/^/[JUDGE_MODEL_PREFLIGHT] /' "${stderr_path}" >&2
    fi
    if (( rc != 0 )); then
        echo "[ERROR] Qwen3.5 judge model check failed before vLLM launch (exit_code=${rc})." >&2
        echo "[ERROR] judge.vllm.model_path=${VLLM_MODEL_PATH}" >&2
        echo "[ERROR] Prepare the model through qwen35_submit.sh/qwen35_local_eval.sh and use its verified resolved_path." >&2
        return "${rc}"
    fi
    echo "[INFO] Judge model processor preflight passed: $(tr '\n' ' ' < "${result_path}")"
}

validate_qwen35_judge_model_compat
validate_benchmark_cache_mount

# ══════════════════════════════════════════════════════════════════════════════
# §3  进程管理（cleanup on exit / signal）
# ══════════════════════════════════════════════════════════════════════════════
VLLM_PID=""
VLLM_OWNED=0
cleanup() {
    trap - EXIT INT TERM
    
    if [[ "${DEBUG}" == "true" ]]; then
        echo "[INFO] DEBUG mode enabled, skipping cleanup."
        [[ -n "${VLLM_PID}" ]] && echo "[INFO] vLLM PID to keep: ${VLLM_PID}"
        return
    fi
    
    if [[ "${VLLM_OWNED}" == "1" && -n "${VLLM_PID}" ]]; then
        echo "[INFO] Stopping vLLM judge backend (PID: ${VLLM_PID})..."
        kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill -TERM "${VLLM_PID}" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-${VLLM_PID}" 2>/dev/null || kill -KILL "${VLLM_PID}" 2>/dev/null || true
        echo "[INFO] vLLM stopped."
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ══════════════════════════════════════════════════════════════════════════════
# §4  启动 Judge 后端
# ══════════════════════════════════════════════════════════════════════════════
JUDGE_BASE_URL=""
JUDGE_API_KEY=""

# 启动 vLLM 的独立脚本路径
START_VLLM_SCRIPT="${LMMS_EVAL_ROOT}/tools/start_vllm_judge.sh"
if [[ ! -f "${START_VLLM_SCRIPT}" ]]; then
    echo "[ERROR] Local vLLM launcher not found: ${START_VLLM_SCRIPT}" >&2
    exit 2
fi

start_vllm_backend() {
    JUDGE_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    JUDGE_API_KEY="EMPTY"
    
    # 先检查是否已有可用的 vLLM 在跑
    local http_status
    http_status=$(curl -sS --connect-timeout 2 --max-time 5 \
        -o /dev/null -w "%{http_code}" "${JUDGE_BASE_URL}/models" 2>/dev/null || echo "000")
    if [[ "${http_status}" == "200" ]]; then
        local matched
        matched=$(curl -sS --connect-timeout 2 --max-time 5 "${JUDGE_BASE_URL}/models" 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m.get('id','') for m in data.get('data',[])]
    expected = sys.argv[1]
    print('true' if expected in models else 'false')
except Exception:
    print('false')
" "${JUDGE_MODEL}" 2>/dev/null)
        if [[ "${matched}" == "true" ]]; then
            echo "[INFO] Found existing vLLM on port ${VLLM_PORT} with model ${JUDGE_MODEL}, reusing it."
            VLLM_PID=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null | head -n1 || echo "")
            VLLM_OWNED=0
            return
        fi
    fi
    
    # 没有可用服务，启动新的
    VLLM_LOG="${LOG_DIR}/vllm_judge_backend.log"
    
    local _output
    _output=$(bash "${START_VLLM_SCRIPT}" \
        --model-path "${VLLM_MODEL_PATH}" \
        --served-model-name "${JUDGE_MODEL}" \
        --tp "${VLLM_TP}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --port "${VLLM_PORT}" \
        --log "${VLLM_LOG}" \
    ) || { echo "[ERROR] Failed to start vLLM judge backend"; exit 1; }
    
    # 打印子脚本日志（过滤掉机器可读的进程元数据行）
    echo "${_output}" | grep -v -E '^VLLM_(PID|OWNED)='
    
    VLLM_PID=$(echo "${_output}" | grep '^VLLM_PID=' | cut -d= -f2)
    VLLM_OWNED=$(echo "${_output}" | grep '^VLLM_OWNED=' | cut -d= -f2)
    if [[ "${VLLM_OWNED}" != "1" || -z "${VLLM_PID}" ]]; then
        echo "[ERROR] Local vLLM launcher did not return an owned PID" >&2
        exit 2
    fi
}

load_api_keys() {
    # 如果配置中没提供，尝试从环境变量获取
    if [[ -z "${API_KEY}" || "${API_KEY}" == "null" ]]; then
        # 尝试加载 setup_api_keys.sh 中的变量
        # 注意：被 source 的脚本可能包含未绑定变量，因此需要临时关闭 set -eu
        _saved_flags=""
        if [[ "$-" == *e* ]]; then _saved_flags="${_saved_flags}e"; fi
        if [[ "$-" == *u* ]]; then _saved_flags="${_saved_flags}u"; fi
        
        # 首先尝试 Qwen3-VL/evaluation 目录下的 setup_api_keys.sh
        if [[ -f "/mnt/cpfsB/<USER>/Qwen3-VL/evaluation/setup_api_keys.sh" ]]; then
            echo "[INFO] Loading API keys from /mnt/cpfsB/<USER>/Qwen3-VL/evaluation/setup_api_keys.sh"
            set +eu
            set -a
            source "/mnt/cpfsB/<USER>/Qwen3-VL/evaluation/setup_api_keys.sh" 2>/dev/null || true
            set +a
            [[ "${_saved_flags}" == *e* ]] && set -e
            [[ "${_saved_flags}" == *u* ]] && set -u
        elif [[ -f "$(dirname "$0")/../Qwen3-VL/evaluation/setup_api_keys.sh" ]]; then
            echo "[INFO] Loading API keys from Qwen3-VL/evaluation/setup_api_keys.sh"
            set +eu
            set -a
            source "$(dirname "$0")/../Qwen3-VL/evaluation/setup_api_keys.sh" 2>/dev/null || true
            set +a
            [[ "${_saved_flags}" == *e* ]] && set -e
            [[ "${_saved_flags}" == *u* ]] && set -u
        elif [[ -f "$(dirname "$0")/setup_api_keys.sh" ]]; then
            echo "[INFO] Loading API keys from setup_api_keys.sh"
            set +eu
            set -a
            source "$(dirname "$0")/setup_api_keys.sh" 2>/dev/null || true
            set +a
            [[ "${_saved_flags}" == *e* ]] && set -e
            [[ "${_saved_flags}" == *u* ]] && set -u
        fi
        
        # 优先级：显式 judge 环境变量 > OpenAI 兼容变量 > 配置文件
        API_KEY="${JUDGE_API_KEY:-${OPENAI_COMPATIBLE_KEY:-${OPENAI_API_KEY:-${CHATGPT_DASHSCOPE_API_KEY:-}}}}"
        API_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_COMPATIBLE_URL:-${OPENAI_API_BASE:-${DASHSCOPE_API_BASE:-${API_BASE_URL}}}}}"
    fi
}

if [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    start_vllm_backend
    export OPENAI_API_KEY="${JUDGE_API_KEY}"
    export OPENAI_API_URL="${JUDGE_BASE_URL%/}/chat/completions"
    export OPENAI_API_BASE="${JUDGE_BASE_URL}"
    export OPENAI_BASE_URL="${JUDGE_BASE_URL}"
    export API_TYPE="openai"
    export JUDGE_API_TYPE="openai"
    export JUDGE_DISABLE_THINKING=1
    export MODEL_VERSION="${JUDGE_MODEL}"
    export JUDGE_MODEL_NAME="${JUDGE_MODEL}"
    export SCIVQR_OPEN_JUDGE_MODEL="${JUDGE_MODEL}"
    export SCIVQR_REASONING_JUDGE_MODEL="${JUDGE_MODEL}"
    
elif [[ "${JUDGE_BACKEND}" == "api" ]]; then
    # ── 使用 API 后端 ─────────────────────────────────────────────────────────
    echo "[INFO] Using API backend for judging"
    
    load_api_keys
    
    if [[ -z "${API_KEY}" ]]; then
        echo "[ERROR] API backend selected but no API key provided. Please set OPENAI_API_KEY or JUDGE_API_KEY environment variable, or configure judge.api.key in config."
        exit 1
    fi
    if [[ -z "${API_BASE_URL}" ]]; then
        echo "[ERROR] API backend selected but no base URL provided. Please set OPENAI_API_BASE or JUDGE_BASE_URL environment variable, or configure judge.api.base_url in config."
        exit 1
    fi
    
    _api_ready=false
    # 修正 base_url：去掉末尾的 /chat/completions
    _test_url="${API_BASE_URL%/chat/completions}"
    echo "[INFO] Testing API connectivity: ${_test_url}"
    http_status=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        "${_test_url}/models" 2>/dev/null || echo "000")
    if [[ "${http_status}" == "200" ]]; then
        _api_ready=true
        JUDGE_BASE_URL="${_test_url}"
        JUDGE_API_KEY="${API_KEY}"
        echo "[INFO] API backend ready at ${JUDGE_BASE_URL}"
    else
        echo "[WARN] API connectivity test failed (HTTP ${http_status})"
    fi
    
    if [[ "${_api_ready}" != "true" ]]; then
        echo "[ERROR] API backend unavailable; refusing to fall back to local vLLM for an API-judge run."
        echo "[ERROR] Please check JUDGE_BASE_URL/OPENAI_API_BASE and JUDGE_API_KEY/OPENAI_API_KEY."
        exit 1
    fi
    
    # Export for embedded task evaluators (e.g. mathvista) that read env vars at import time
    # OPENAI_API_URL must keep the original /chat/completions suffix because tasks like mmbench
    # use requests.post directly against the full endpoint URL.
    export OPENAI_API_KEY="${JUDGE_API_KEY}"
    export OPENAI_API_URL="${API_BASE_URL}"
    export OPENAI_API_BASE="${JUDGE_BASE_URL}"
    export API_TYPE="openai"
    
fi

# ══════════════════════════════════════════════════════════════════════════════
# §5  准备 Judge 参数
# ══════════════════════════════════════════════════════════════════════════════

# 构建任务列表
if [[ -z "${TASKS}" || "${TASKS}" == "null" ]]; then
    echo "[ERROR] No tasks specified in config. Please set eval.tasks"
    exit 1
fi

# 确定输入路径
if [[ -d "${INPUT_RESULT_PATH}" ]]; then
    INPUT_FLAG="${INPUT_RESULT_PATH}"
elif [[ -f "${INPUT_RESULT_PATH}" ]]; then
    INPUT_FLAG="${INPUT_RESULT_PATH}"
else
    echo "[ERROR] Input result path not found: ${INPUT_RESULT_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"

# ══════════════════════════════════════════════════════════════════════════════
# §6  执行 Judge
# ══════════════════════════════════════════════════════════════════════════════
JUDGE_LOG="${LOG_DIR}/judge.log"
echo "[INFO] Starting judge..."
echo "[INFO] Log file: ${JUDGE_LOG}"
echo ""

# 设置 judge 环境变量
export JUDGE_MODEL="${JUDGE_MODEL}"
export JUDGE_API_KEY="${JUDGE_API_KEY}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL}"
export JUDGE_MAX_CONCURRENT="${JUDGE_PARALLEL}"
export LMMS_EVAL_JUDGE_STAGE="post_eval"
export LOGURU_LEVEL="INFO"

# 执行 lmms-eval judge
python -m lmms_eval judge \
    --input_result "${INPUT_FLAG}" \
    --task "${TASKS}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-api-key "${JUDGE_API_KEY}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --parallel "${JUDGE_PARALLEL}" \
    --output-dir "${OUTPUT_PATH}" \
    2>&1 | tee "${JUDGE_LOG}"

JUDGE_EXIT_CODE=${PIPESTATUS[0]}

if [[ ${JUDGE_EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Judge completed successfully!"
    echo "=========================================="
    echo "[INFO] Output directory: ${OUTPUT_PATH}"
    echo "[INFO] Log directory: ${LOG_DIR}"
else
    echo ""
    echo "=========================================="
    echo "Judge failed with exit code: ${JUDGE_EXIT_CODE}"
    echo "=========================================="
    exit ${JUDGE_EXIT_CODE}
fi
