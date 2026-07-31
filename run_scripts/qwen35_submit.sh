#!/bin/bash
# qwen35_submit.sh
# Qwen3.5 submitter entrypoint: reads a DLC config + an eval config, then
# submits a DLC PyTorchJob that runs qwen35_worker.sh directly.
# Optionally accepts a judge config. API judge keeps the separate CPU-only DLC
# workflow; local vLLM judge runs inline in the same 8-GPU eval DLC worker.
#
# Usage:
#   bash run_scripts/qwen35_submit.sh <dlc_config.json> <eval_config.json> [judge_config.json] [-- <dlc auth flags>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DLC_RESOURCE_ID="quotaev2tl4w6aw0"
REQUIRED_NAS_MOUNT_URI="nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB"
REQUIRED_CPFSB_MOUNT_PATH="/mnt/cpfsB"
BENCHMARK_DATASETS_CACHE="/mnt/cpfsB/evaluation_cache/lmms_eval"

if (( $# < 2 )); then
    echo "[ERROR] Usage: bash $(basename "$0") <dlc_config.json> <eval_config.json> [judge_config.json] [-- <dlc auth flags>]"
    exit 1
fi

DLC_CONFIG="$1"
EVAL_CONFIG="$2"
shift 2

JUDGE_CONFIG=""
if (( $# > 0 )) && [[ "$1" != "--" && "$1" != -* ]]; then
    JUDGE_CONFIG="$1"
    shift
fi
if (( $# > 0 )) && [[ "$1" == "--" ]]; then
    shift
fi

DLC_AUTH_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --access_id|--access_key|-a|-k|--security_token|-S)
            if (( $# < 2 )); then
                echo "[ERROR] Missing value for DLC credential flag: $1" >&2
                exit 2
            fi
            DLC_AUTH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --access_id=*|--access_key=*|--security_token=*)
            DLC_AUTH_ARGS+=("$1")
            shift
            ;;
        --ignore_local_config|-I)
            DLC_AUTH_ARGS+=("$1")
            shift
            ;;
        *)
            echo "[ERROR] Unsupported extra DLC flag: $1" >&2
            exit 2
            ;;
    esac
done

for f in "${DLC_CONFIG}" "${EVAL_CONFIG}"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Config not found: $f"
        echo "[ERROR] Usage: bash $(basename "$0") <dlc_config.json> <eval_config.json> [judge_config.json] [-- <dlc auth flags>]"
        exit 1
    fi
done
if [[ -n "${JUDGE_CONFIG}" && ! -f "${JUDGE_CONFIG}" ]]; then
    echo "[ERROR] Judge config not found: ${JUDGE_CONFIG}"
    echo "[ERROR] Usage: bash $(basename "$0") <dlc_config.json> <eval_config.json> [judge_config.json] [-- <dlc auth flags>]"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "[WARN] jq not found, attempting to install..."
    apt-get update -qq && apt-get install -y -qq jq || { echo "[ERROR] Failed to install jq."; exit 1; }
fi

# ── helpers for reading the two configs ───────────────────────────────────────
dlc_cfg()     { jq -r "$1"       "${DLC_CONFIG}"; }
dlc_cfg_int() { jq -r "$1 // 0" "${DLC_CONFIG}"; }
dlc_judge_cfg() { jq -er "$1" "${DLC_CONFIG}"; }
dlc_judge_cfg_int() { jq -er "$1 | tonumber" "${DLC_CONFIG}"; }
eval_cfg()     { jq -r "$1"       "${EVAL_CONFIG}"; }
eval_cfg_int() { jq -r "$1 // 0" "${EVAL_CONFIG}"; }
judge_cfg()     { jq -r "$1"       "${JUDGE_CONFIG}"; }
judge_cfg_int() { jq -r "$1 // 0" "${JUDGE_CONFIG}"; }

JUDGE_BACKEND=""
if [[ -n "${JUDGE_CONFIG}" ]]; then
    JUDGE_BACKEND="$(judge_cfg '.judge.backend // ""')"
    case "${JUDGE_BACKEND}" in
        api|vllm) ;;
        *)
            echo "[ERROR] judge.backend must be api or vllm, got: ${JUDGE_BACKEND:-<empty>}" >&2
            exit 2
            ;;
    esac
fi

require_non_empty() {
    local value="$1"
    local field="$2"
    if [[ -z "${value}" || "${value}" == "null" ]]; then
        echo "[ERROR] Missing ${field} in ${DLC_CONFIG}" >&2
        exit 2
    fi
}

require_single_resource_id() {
    local value="$1"
    local field="$2"
    require_non_empty "${value}" "${field}"
    if [[ "${value}" != "${DEFAULT_DLC_RESOURCE_ID}" ]]; then
        echo "[ERROR] ${field} must be ${DEFAULT_DLC_RESOURCE_ID}, got: ${value}" >&2
        exit 8
    fi
}

require_required_nas_mount() {
    local value="$1"
    local field="$2"
    require_non_empty "${value}" "${field}"
    case ",${value}," in
        *",${REQUIRED_NAS_MOUNT_URI},"*) ;;
        *)
            echo "[ERROR] ${field} must include ${REQUIRED_NAS_MOUNT_URI}, got: ${value}" >&2
            exit 9
            ;;
    esac
}

require_required_cpfsb_mount() {
    local value="$1"
    local field="$2"
    require_non_empty "${value}" "${field}"

    local uri
    local uris=()
    IFS=',' read -ra uris <<< "${value}"
    for uri in "${uris[@]}"; do
        if [[ "${uri}" == cpfs://*::${REQUIRED_CPFSB_MOUNT_PATH} ]]; then
            return 0
        fi
    done

    echo "[ERROR] ${field} must include a CPFS URI mounted at ${REQUIRED_CPFSB_MOUNT_PATH}, got: ${value}" >&2
    exit 9
}

# ── resolve job name (needs model info from eval config) ──────────────────────
MODEL=$(eval_cfg '.model.source_path // .model.path')
MODEL_TP=$(eval_cfg_int '.model.tp')
LOG_BASE=$(eval_cfg '.log.dir')
MODEL_BACKEND=$(eval_cfg '.model.backend // "vllm"')
MODEL_BACKEND="$(printf '%s' "${MODEL_BACKEND}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MODEL_BACKEND}" != "vllm" && "${MODEL_BACKEND}" != "openai" ]]; then
    echo "[ERROR] model.backend must be vllm or openai, got: ${MODEL_BACKEND}" >&2
    exit 2
fi
if [[ -z "${MODEL}" || "${MODEL}" == "null" ]]; then
    echo "[ERROR] model.source_path/model.path must be non-empty in ${EVAL_CONFIG}" >&2
    exit 2
fi

JOB_NAME_FROM_CFG=$(dlc_cfg '.dlc.job_name // ""')
if [[ -n "${JOB_NAME_FROM_CFG}" && "${JOB_NAME_FROM_CFG}" != "null" ]]; then
    JOB_NAME="${JOB_NAME_FROM_CFG}"
else
    JOB_NAME="eval_$(basename "${MODEL}")_tp${MODEL_TP}_$(date +%m%d_%H%M%S)"
fi
if [[ "${JOB_NAME}" != eval_* ]]; then
    echo "[ERROR] DLC eval job name must start with eval_, got: ${JOB_NAME}" >&2
    exit 7
fi

# ── validate DLC binary ───────────────────────────────────────────────────────
DLC_BINARY=$(dlc_cfg '.dlc.binary')
if [[ -z "${DLC_BINARY}" || "${DLC_BINARY}" == "null" ]]; then
    echo "[ERROR] DLC binary not configured in ${DLC_CONFIG} (dlc.binary)"
    exit 1
fi
if [[ ! -x "${DLC_BINARY}" ]]; then
    echo "[ERROR] DLC binary not found or not executable: ${DLC_BINARY}"
    exit 1
fi

# 统一时间戳由 submitter 生成，保证所有 worker 目录一致
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FIXED_LOG_DIR="${LOG_BASE}/${JOB_NAME}/${TIMESTAMP}"
mkdir -p "${FIXED_LOG_DIR}"

# ── Qwen3.5 model processor preflight ────────────────────────────────────────
MODEL_SOURCE_INPUT="${MODEL}"
MODEL_SOURCE="${MODEL_SOURCE_INPUT}"
RESOLVED_MODEL="${MODEL}"
MODEL_PREFLIGHT_JSON="${FIXED_LOG_DIR}/model_preflight.json"
MODEL_PREFLIGHT_LOG="${FIXED_LOG_DIR}/model_preflight.stderr.log"
MODEL_VIEW_MANIFEST_PATH=""
MODEL_VIEW_ROOT=""
PROCESSOR_COMPAT="off"
HAS_MODEL_PREFLIGHT=false
SERVED_MODEL_NAME=$(eval_cfg '.model.served_model_name // ""')
if [[ -z "${SERVED_MODEL_NAME}" || "${SERVED_MODEL_NAME}" == "null" ]]; then
    SERVED_MODEL_NAME="$(basename "${MODEL_SOURCE}")"
fi
if ! [[ "${SERVED_MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] model.served_model_name must match ^[A-Za-z0-9._-]+$ so it is safe in lmms-eval model_args; got: ${SERVED_MODEL_NAME}" >&2
    exit 2
fi

if [[ "${MODEL_BACKEND}" == "vllm" ]]; then
    PROCESSOR_COMPAT=$(eval_cfg '.model.processor_compat // "required"')
    case "${PROCESSOR_COMPAT}" in
        auto|required) ;;
        off)
            echo "[ERROR] qwen35_submit.sh does not allow model.processor_compat=off for a local Qwen3.5 vLLM model; use auto or required." >&2
            exit 2
            ;;
        *)
            echo "[ERROR] model.processor_compat must be auto, required, or off; got: ${PROCESSOR_COMPAT}" >&2
            exit 2
            ;;
    esac

    EVAL_VENV_PATH=$(eval_cfg '.env.venv_path // ""')
    if [[ -z "${EVAL_VENV_PATH}" || "${EVAL_VENV_PATH}" == "null" ]]; then
        EVAL_VENV_PATH="${LMMS_EVAL_ROOT}/.venv"
    fi
    if [[ ! -x "${EVAL_VENV_PATH}/bin/python" ]]; then
        echo "[ERROR] Evaluation Python is missing or not executable: ${EVAL_VENV_PATH}/bin/python" >&2
        exit 2
    fi

    MODEL_VIEW_ROOT=$(eval_cfg '.model.view_root // ""')
    if [[ -z "${MODEL_VIEW_ROOT}" || "${MODEL_VIEW_ROOT}" == "null" ]]; then
        echo "[ERROR] model.view_root must be an explicit absolute shared path for Qwen3.5 vLLM evaluation." >&2
        exit 2
    fi
    if [[ "${MODEL_VIEW_ROOT}" != /* ]]; then
        echo "[ERROR] model.view_root must be an absolute shared path, got: ${MODEL_VIEW_ROOT}" >&2
        exit 2
    fi

    echo "[INFO] Qwen3.5 model preflight source: ${MODEL_SOURCE}"
    echo "[INFO] Qwen3.5 processor compatibility mode: ${PROCESSOR_COMPAT}"
    echo "[INFO] Qwen3.5 compatibility view root: ${MODEL_VIEW_ROOT}"
    set +e
    PYTHONPATH="${LMMS_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${EVAL_VENV_PATH}/bin/python" \
        "${LMMS_EVAL_ROOT}/lmms_eval/models/model_utils/qwen35_model_compat.py" \
        prepare \
        --source "${MODEL_SOURCE}" \
        --view-root "${MODEL_VIEW_ROOT}" \
        --run-id "${JOB_NAME}_${TIMESTAMP}" \
        > "${MODEL_PREFLIGHT_JSON}" \
        2> "${MODEL_PREFLIGHT_LOG}"
    MODEL_PREFLIGHT_RC=$?
    set -e
    if [[ -s "${MODEL_PREFLIGHT_LOG}" ]]; then
        sed 's/^/[MODEL_PREFLIGHT] /' "${MODEL_PREFLIGHT_LOG}" >&2
    fi
    if (( MODEL_PREFLIGHT_RC != 0 )); then
        echo "[ERROR] Qwen3.5 model preflight failed with exit code ${MODEL_PREFLIGHT_RC}." >&2
        echo "[ERROR] source_path=${MODEL_SOURCE}" >&2
        echo "[ERROR] preflight_log=${MODEL_PREFLIGHT_LOG}" >&2
        exit "${MODEL_PREFLIGHT_RC}"
    fi
    if ! jq -e '
        (.source_path | type == "string" and length > 0)
        and (.resolved_path | type == "string" and length > 0)
        and (.processor_class | type == "string" and length > 0)
        and (.source_manifest_sha256 | type == "string" and length == 64)
    ' "${MODEL_PREFLIGHT_JSON}" >/dev/null; then
        echo "[ERROR] Qwen3.5 model preflight returned an invalid result: ${MODEL_PREFLIGHT_JSON}" >&2
        exit 2
    fi
    RESOLVED_MODEL=$(jq -er '.resolved_path' "${MODEL_PREFLIGHT_JSON}")
    MODEL_SOURCE=$(jq -er '.source_path' "${MODEL_PREFLIGHT_JSON}")
    if [[ "${MODEL_SOURCE}" != /* || ! -d "${MODEL_SOURCE}" ]]; then
        echo "[ERROR] Qwen3.5 model preflight canonical source path is invalid: ${MODEL_SOURCE}" >&2
        exit 2
    fi
    if [[ ! -d "${RESOLVED_MODEL}" ]]; then
        echo "[ERROR] Qwen3.5 model preflight resolved path is not visible: ${RESOLVED_MODEL}" >&2
        exit 2
    fi
    if [[ -f "${RESOLVED_MODEL}/.qwen35_processor_compat_manifest.json" ]]; then
        MODEL_VIEW_MANIFEST_PATH="${RESOLVED_MODEL}/.qwen35_processor_compat_manifest.json"
    fi
    HAS_MODEL_PREFLIGHT=true
    echo "[INFO] Qwen3.5 model preflight passed."
    echo "[INFO] Qwen3.5 model resolved path: ${RESOLVED_MODEL}"
    echo "[INFO] Qwen3.5 processor class: $(jq -r '.processor_class' "${MODEL_PREFLIGHT_JSON}")"
else
    printf '{}\n' > "${MODEL_PREFLIGHT_JSON}"
    : > "${MODEL_PREFLIGHT_LOG}"
fi

EXTRA_LD_LIBRARY_DIRS=$(eval_cfg '.env.extra_ld_library_dirs // ""')
EXTRA_LD_EXPORT=""
if [[ -n "${EXTRA_LD_LIBRARY_DIRS}" && "${EXTRA_LD_LIBRARY_DIRS}" != "null" ]]; then
    if [[ "${EXTRA_LD_LIBRARY_DIRS}" == *"'"* ]]; then
        echo "[ERROR] env.extra_ld_library_dirs must not contain single quotes: ${EXTRA_LD_LIBRARY_DIRS}" >&2
        exit 2
    fi
    EXTRA_LD_EXPORT=" export LMMS_EVAL_EXTRA_LD_LIBRARY_DIRS='${EXTRA_LD_LIBRARY_DIRS}';"
fi

# ── generate runtime config for workers ───────────────────────────────────────
# Workers should never submit again, and cluster runs are always non-debug.
# 同时将统一时间戳写入 runtime config，worker 会用它作为 output_path。
# Qwen3.5 链路在这里固定模型家族参数，避免走旧的 VL 专用处理分支。
JUDGE_RUNTIME_API_KEY=""
JUDGE_RUNTIME_BASE_URL=""
if [[ "${JUDGE_BACKEND}" == "api" ]]; then
    JUDGE_RUNTIME_API_KEY="$(judge_cfg '.judge.api.key // ""')"
    JUDGE_RUNTIME_BASE_URL="$(judge_cfg '.judge.api.base_url // ""')"
fi
RUNTIME_CONFIG="${FIXED_LOG_DIR}/runtime_config.json"
LMMS_EVAL_COMMIT="$(git -C "${LMMS_EVAL_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
LMMS_EVAL_GIT_DIRTY=false
if [[ -n "$(git -C "${LMMS_EVAL_ROOT}" status --porcelain=v1 --untracked-files=normal 2>/dev/null || true)" ]]; then
    LMMS_EVAL_GIT_DIRTY=true
fi
LMMS_EVAL_TREE_STATE_SHA256="$(
    (
        cd "${LMMS_EVAL_ROOT}"
        git diff --binary HEAD -- 2>/dev/null || true
        while IFS= read -r -d '' untracked_file; do
            printf '\0untracked\0%s\0' "${untracked_file}"
            sha256sum -- "${untracked_file}"
        done < <(git ls-files --others --exclude-standard -z 2>/dev/null)
    ) | sha256sum | awk '{print $1}'
)"
jq \
  --arg ts "${TIMESTAMP}" \
  --arg judge_api_key "${JUDGE_RUNTIME_API_KEY}" \
  --arg judge_base_url "${JUDGE_RUNTIME_BASE_URL}" \
  --arg source_input_model "${MODEL_SOURCE_INPUT}" \
  --arg source_model "${MODEL_SOURCE}" \
  --arg resolved_model "${RESOLVED_MODEL}" \
  --arg served_model_name "${SERVED_MODEL_NAME}" \
  --arg processor_compat "${PROCESSOR_COMPAT}" \
  --arg model_view_root "${MODEL_VIEW_ROOT}" \
  --arg model_view_manifest_path "${MODEL_VIEW_MANIFEST_PATH}" \
  --arg lmms_eval_commit "${LMMS_EVAL_COMMIT}" \
  --arg lmms_eval_tree_state_sha256 "${LMMS_EVAL_TREE_STATE_SHA256}" \
  --arg benchmark_datasets_cache "${BENCHMARK_DATASETS_CACHE}" \
  --argjson lmms_eval_git_dirty "${LMMS_EVAL_GIT_DIRTY}" \
  --argjson has_model_preflight "${HAS_MODEL_PREFLIGHT}" \
  --slurpfile model_preflight "${MODEL_PREFLIGHT_JSON}" \
  '
  if (.model | type) != "object" then
    error("config.model must be an object")
  else
    .
  end
  | if (.env | type) != "object" then
      error("config.env must be an object")
    else
      .
    end
  | .dlc.submit = false
  | .eval.debug = false
  | .eval.timestamp = $ts
  | .env.lmms_eval_datasets_cache = $benchmark_datasets_cache
  | .env.hf_datasets_cache = $benchmark_datasets_cache
  | if ($judge_api_key | length) > 0 then
      .env.judge_api_key = $judge_api_key
    else
      .
    end
  | if ($judge_base_url | length) > 0 then
      .env.judge_base_url = $judge_base_url
    else
      .
    end
  | if (($judge_api_key | length) > 0) and (((.env.openai_api_key // "") | tostring | length) == 0) then
      .env.openai_api_key = $judge_api_key
    else
      .
    end
  | if (($judge_base_url | length) > 0) and (((.env.openai_api_url // "") | tostring | length) == 0) then
      .env.openai_api_url = $judge_base_url
    else
      .
    end
  | .model.reasoning_parser = "qwen3"
  | .model.is_qwen3_vl = false
  | .model.startup_timeout_seconds = (.model.startup_timeout_seconds // 1800)
  | if $has_model_preflight then
      .model.path = $resolved_model
      | .model.source_input_path = $source_input_model
      | .model.source_path = $source_model
      | .model.resolved_path = $resolved_model
      | .model.served_model_name = $served_model_name
      | .model.processor_compat = $processor_compat
      | .model.view_root = $model_view_root
      | .model.view_manifest_path = $model_view_manifest_path
      | .model.preflight = (
          $model_preflight[0]
          + {
              lmms_eval_commit: $lmms_eval_commit,
              lmms_eval_git_dirty: $lmms_eval_git_dirty,
              lmms_eval_tree_state_sha256: $lmms_eval_tree_state_sha256
            }
        )
    else
      .
    end
  | if (.model.enable_thinking == null) then
      .model.enable_thinking = false
    else
      .
    end
' "${EVAL_CONFIG}" > "${RUNTIME_CONFIG}"

# ── resolve absolute paths for worker script and runtime config ───────────────
WORKER_SCRIPT_FROM_CFG=$(dlc_cfg '.dlc.run_script // ""')
if [[ -n "${WORKER_SCRIPT_FROM_CFG}" && "${WORKER_SCRIPT_FROM_CFG}" != "null" ]]; then
    WORKER_SCRIPT="${WORKER_SCRIPT_FROM_CFG}"
else
    WORKER_SCRIPT="${SCRIPT_DIR}/qwen35_worker.sh"
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "[ERROR] Worker script not found: ${WORKER_SCRIPT}"
    echo "[ERROR] Configure dlc.run_script with the CPFS path visible inside the DLC container."
    exit 1
fi
if [[ "$(basename "${WORKER_SCRIPT}")" != "qwen35_worker.sh" ]]; then
    echo "[ERROR] Qwen3.5 submitter must run qwen35_worker.sh, got: ${WORKER_SCRIPT}" >&2
    exit 1
fi

# ── read DLC parameters ───────────────────────────────────────────────────────
WORKERS=$(dlc_cfg_int '.dlc.workers')
WORKER_GPU=$(dlc_cfg_int '.dlc.worker_gpu')
WORKER_CPU=$(dlc_cfg_int '.dlc.worker_cpu')
WORKER_MEMORY=$(dlc_cfg '.dlc.worker_memory')
WORKER_SHARED_MEMORY=$(dlc_cfg '.dlc.worker_shared_memory')
PRIORITY=$(dlc_cfg_int '.dlc.priority')
RUNNING_TIMEOUT=$(dlc_cfg_int '.dlc.running_timeout')
JOB_MAX_RUNNING_TIME_MINUTES=$(dlc_cfg_int '.dlc.job_max_running_time_minutes // 10080')
WORKER_IMAGE=$(dlc_cfg '.dlc.worker_image')
DATA_SOURCE_URIS=$(dlc_cfg '.dlc.data_source_uris')
RESOURCE_ID=$(dlc_cfg '.dlc.resource_id')
WORKSPACE_ID=$(dlc_cfg '.dlc.workspace_id')
VPC_ID=$(dlc_cfg '.dlc.vpc_id')
SWITCH_ID=$(dlc_cfg '.dlc.switch_id')
SECURITY_GROUP_ID=$(dlc_cfg '.dlc.security_group_id')
EXTENDED_CIDRS=$(dlc_cfg '.dlc.extended_cidrs')
REGION=$(dlc_cfg '.dlc.region // ""')
ENDPOINT=$(dlc_cfg '.dlc.endpoint // ""')

require_single_resource_id "${RESOURCE_ID}" "dlc.resource_id"
require_required_nas_mount "${DATA_SOURCE_URIS}" "dlc.data_source_uris"
require_required_cpfsb_mount "${DATA_SOURCE_URIS}" "dlc.data_source_uris"

if ! [[ "${PRIORITY}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] DLC priority must be an integer, got: ${PRIORITY}"
    exit 2
fi
if (( PRIORITY > 9 )); then
    echo "[ERROR] Refusing to submit Qwen3.5 eval DLC job with priority=${PRIORITY}; maximum allowed is 9."
    exit 6
fi
if [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    if (( WORKERS != 1 )); then
        echo "[ERROR] Inline local vLLM judge requires dlc.workers=1, got: ${WORKERS}" >&2
        exit 2
    fi
    if (( WORKER_GPU != 8 )); then
        echo "[ERROR] Inline local vLLM judge requires dlc.worker_gpu=8, got: ${WORKER_GPU}" >&2
        exit 2
    fi
fi

# ── build DLC command ─────────────────────────────────────────────────────────
quote_for_single_quotes() {
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

parse_job_id_from_text() {
    awk -F'|' '{for (i = 1; i <= NF; i++) {if ($i ~ /dlc[[:alnum:]]+/) {gsub(/[[:space:]]/, "", $i); print $i; exit}}}'
}

resolve_job_id_by_name() {
    local job_name="$1"
    local resource_id="$2"
    local workspace_id="$3"
    local job_id=""
    for _ in {1..24}; do
        job_id="$("${DLC_BINARY}" get job \
            -n "${job_name}" \
            -w "${workspace_id}" \
            --resource_id="${resource_id}" \
            "${OPTIONAL_DLC_ARGS[@]}" \
            | parse_job_id_from_text)"
        if [[ -n "${job_id}" ]]; then
            printf "%s" "${job_id}"
            return 0
        fi
        sleep 5
    done
    return 1
}

status_of_detail() {
    python - "$1" <<'PY'
import json
import sys

text = open(sys.argv[1], encoding="utf-8").read()
start = text.find("{")
if start < 0:
    raise ValueError(f"No JSON object found in {sys.argv[1]}")
print(json.loads(text[start:]).get("Status", ""))
PY
}

summarize_detail() {
    python - "$1" <<'PY'
import json
import sys

text = open(sys.argv[1], encoding="utf-8").read()
start = text.find("{")
if start < 0:
    raise ValueError(f"No JSON object found in {sys.argv[1]}")
data = json.loads(text[start:])
parts = []
for key in ("DisplayName", "JobId", "Status", "ReasonCode", "ReasonMessage", "Duration"):
    if key in data and data[key] not in (None, ""):
        parts.append(f"{key}={data[key]}")
print(" ".join(parts))
PY
}

wait_for_job_success() {
    local job_id="$1"
    local stage="$2"
    local timeout_seconds="$3"
    local resource_id="$4"
    local workspace_id="$5"
    local poll_interval="${DLC_POLL_INTERVAL_SECONDS:-60}"
    local detail_dir="${FIXED_LOG_DIR}/job_details"
    mkdir -p "${detail_dir}"

    echo "[INFO] Waiting for ${stage} DLC job ${job_id}; timeout=${timeout_seconds}s interval=${poll_interval}s"
    local deadline=$((SECONDS + timeout_seconds))
    local status=""
    local detail_file=""
    while (( SECONDS < deadline )); do
        detail_file="${detail_dir}/${stage}_${job_id}_$(date +%Y%m%d_%H%M%S).json"
        "${DLC_BINARY}" get job "${job_id}" \
            -w "${workspace_id}" \
            --resource_id="${resource_id}" \
            --show_detail \
            "${OPTIONAL_DLC_ARGS[@]}" \
            > "${detail_file}"
        status="$(status_of_detail "${detail_file}")"
        echo "[INFO] ${stage} job status: $(summarize_detail "${detail_file}")"
        if [[ "${status}" == "Succeeded" ]]; then
            echo "[INFO] ${stage} DLC job succeeded: ${job_id}"
            return 0
        fi
        if [[ "${status}" == "Failed" || "${status}" == "Stopped" ]]; then
            echo "[ERROR] ${stage} DLC job ended with status=${status}. Detail: ${detail_file}" >&2
            return 1
        fi
        sleep "${poll_interval}"
    done

    echo "[ERROR] Timeout while waiting for ${stage} DLC job ${job_id}. Last detail: ${detail_file}" >&2
    return 1
}

OPTIONAL_DLC_ARGS=()
if [[ -n "${REGION}" && "${REGION}" != "null" ]]; then
    OPTIONAL_DLC_ARGS+=(--region="${REGION}")
fi
if [[ -n "${ENDPOINT}" && "${ENDPOINT}" != "null" ]]; then
    OPTIONAL_DLC_ARGS+=(--endpoint="${ENDPOINT}")
fi
if (( ${#DLC_AUTH_ARGS[@]} > 0 )); then
    OPTIONAL_DLC_ARGS+=("${DLC_AUTH_ARGS[@]}")
fi

build_judge_submit_args() {
    local judge_runtime_config="$1"
    local judge_job_name="$2"
    local judge_log_dir="$3"
    local judge_script="$4"

    local judge_workers
    local judge_worker_gpu
    local judge_worker_cpu
    local judge_worker_memory
    local judge_worker_shared_memory
    local judge_priority
    local judge_job_max_running_time_minutes
    local judge_worker_image
    local judge_data_source_uris
    local judge_resource_id
    local judge_workspace_id
    local judge_vpc_id
    local judge_switch_id
    local judge_security_group_id
    local judge_extended_cidrs

    if ! jq -e '(.dlc.judge | type) == "object"' "${DLC_CONFIG}" >/dev/null; then
        echo "[ERROR] Judge DLC requires a complete dlc.judge CPU-only template in ${DLC_CONFIG}" >&2
        exit 8
    fi

    judge_workers=$(dlc_judge_cfg_int '.dlc.judge.workers')
    judge_worker_gpu=$(dlc_judge_cfg_int '.dlc.judge.worker_gpu')
    judge_worker_cpu=$(dlc_judge_cfg_int '.dlc.judge.worker_cpu')
    judge_worker_memory=$(dlc_judge_cfg '.dlc.judge.worker_memory')
    judge_worker_shared_memory=$(dlc_judge_cfg '.dlc.judge.worker_shared_memory')
    judge_priority=$(dlc_judge_cfg_int '.dlc.judge.priority')
    judge_job_max_running_time_minutes=$(dlc_judge_cfg_int '.dlc.judge.job_max_running_time_minutes')
    judge_worker_image=$(dlc_judge_cfg '.dlc.judge.worker_image')
    judge_data_source_uris=$(dlc_judge_cfg '.dlc.judge.data_source_uris')
    judge_resource_id=$(dlc_judge_cfg '.dlc.judge.resource_id')
    judge_workspace_id=$(dlc_judge_cfg '.dlc.judge.workspace_id')
    judge_vpc_id=$(dlc_judge_cfg '.dlc.judge.vpc_id')
    judge_switch_id=$(dlc_judge_cfg '.dlc.judge.switch_id')
    judge_security_group_id=$(dlc_judge_cfg '.dlc.judge.security_group_id')
    judge_extended_cidrs=$(dlc_judge_cfg '.dlc.judge.extended_cidrs')

    require_non_empty "${judge_worker_memory}" "dlc.judge.worker_memory"
    require_non_empty "${judge_worker_shared_memory}" "dlc.judge.worker_shared_memory"
    require_non_empty "${judge_worker_image}" "dlc.judge.worker_image"
    require_non_empty "${judge_data_source_uris}" "dlc.judge.data_source_uris"
    require_non_empty "${judge_resource_id}" "dlc.judge.resource_id"
    require_non_empty "${judge_workspace_id}" "dlc.judge.workspace_id"
    require_non_empty "${judge_vpc_id}" "dlc.judge.vpc_id"
    require_non_empty "${judge_switch_id}" "dlc.judge.switch_id"
    require_non_empty "${judge_security_group_id}" "dlc.judge.security_group_id"
    require_non_empty "${judge_extended_cidrs}" "dlc.judge.extended_cidrs"
    require_single_resource_id "${judge_resource_id}" "dlc.judge.resource_id"
    require_required_nas_mount "${judge_data_source_uris}" "dlc.judge.data_source_uris"
    require_required_cpfsb_mount "${judge_data_source_uris}" "dlc.judge.data_source_uris"

    if [[ "${judge_worker_gpu}" != "0" ]]; then
        echo "[ERROR] Judge DLC must be CPU-only; expected worker_gpu=0, got ${judge_worker_gpu}" >&2
        exit 8
    fi
    if (( judge_workers < 1 )); then
        echo "[ERROR] Judge DLC workers must be positive, got ${judge_workers}" >&2
        exit 8
    fi
    if (( judge_worker_cpu < 1 )); then
        echo "[ERROR] Judge DLC worker_cpu must be positive, got ${judge_worker_cpu}" >&2
        exit 8
    fi
    if (( judge_priority > 9 )); then
        echo "[ERROR] Refusing to submit judge DLC job with priority=${judge_priority}; maximum allowed is 9." >&2
        exit 6
    fi
    if (( judge_job_max_running_time_minutes < 1 )); then
        echo "[ERROR] Judge DLC job_max_running_time_minutes must be positive, got ${judge_job_max_running_time_minutes}" >&2
        exit 8
    fi

    local judge_inner_command
    local judge_command
    judge_inner_command="set -euo pipefail; cd ${LMMS_EVAL_ROOT}; export LMMS_EVAL_LOG_DIR=${judge_log_dir}; bash ${judge_script} ${judge_runtime_config}"
    judge_command="/bin/bash -c '$(quote_for_single_quotes "${judge_inner_command}")'"

    JUDGE_SUBMIT_ARGS=(
        submit pytorchjob
        --name="${judge_job_name}"
        --priority="${judge_priority}"
        --workers="${judge_workers}"
        --worker_cpu="${judge_worker_cpu}"
        --worker_gpu="${judge_worker_gpu}"
        --worker_memory="${judge_worker_memory}"
        --worker_shared_memory="${judge_worker_shared_memory}"
        --worker_image="${judge_worker_image}"
        --job_max_running_time_minutes="${judge_job_max_running_time_minutes}"
        --data_source_uris="${judge_data_source_uris}"
        --resource_id="${judge_resource_id}"
        --workspace_id="${judge_workspace_id}"
        --vpc_id="${judge_vpc_id}"
        --switch_id="${judge_switch_id}"
        --security_group_id="${judge_security_group_id}"
        --extended_cidrs="${judge_extended_cidrs}"
        "${OPTIONAL_DLC_ARGS[@]}"
        --command="${judge_command}"
    )
    JUDGE_RESOURCE_ID="${judge_resource_id}"
    JUDGE_WORKSPACE_ID="${judge_workspace_id}"
}

prepare_judge_runtime_config() {
    local eval_result_path="$1"
    local judge_runtime_config="$2"
    local judge_log_dir="$3"
    local judge_output_path="${eval_result_path}/judge"

    jq \
        --arg input_path "${eval_result_path}" \
        --arg output_path "${judge_output_path}" \
        --arg log_dir "${judge_log_dir}" \
        '.eval.input_result_path = $input_path
         | .eval.output_path = $output_path
         | .eval.debug = false
         | .log.dir = $log_dir
         | .env.lmms_eval_datasets_cache = "/mnt/cpfsB/evaluation_cache/lmms_eval"
         | .env.hf_datasets_cache = "/mnt/cpfsB/evaluation_cache/lmms_eval"' \
        "${JUDGE_CONFIG}" > "${judge_runtime_config}"
}

submit_and_resolve_job_id() {
    local job_name="$1"
    local stage="$2"
    local resource_id="$3"
    local workspace_id="$4"
    shift 4
    local submit_log="${FIXED_LOG_DIR}/${stage}_submit.log"
    local submit_output
    local rc

    set +e
    submit_output="$("${DLC_BINARY}" "$@" 2>&1)"
    rc=$?
    set -e
    printf "%s\n" "${submit_output}" | tee "${submit_log}" >&2
    if [[ "${rc}" != "0" ]]; then
        echo "[ERROR] DLC ${stage} submit failed with exit code ${rc}. Log: ${submit_log}" >&2
        exit "${rc}"
    fi

    local job_id
    job_id="$(printf "%s\n" "${submit_output}" | parse_job_id_from_text)"
    if [[ -z "${job_id}" ]]; then
        job_id="$(resolve_job_id_by_name "${job_name}" "${resource_id}" "${workspace_id}")" || {
            echo "[ERROR] Could not resolve ${stage} DLC JobId for ${job_name}" >&2
            exit 67
        }
    fi
    printf "%s" "${job_id}"
}

print_dry_run_command() {
    local label="$1"
    shift
    printf "[DRY_RUN][%s]" "${label}"
    local redact_next=0
    local arg
    for arg in "$@"; do
        if [[ "${redact_next}" == "1" ]]; then
            printf " %q" "********"
            redact_next=0
            continue
        fi
        case "${arg}" in
            --access_id|--access_key|-a|-k|--security_token|-S)
                printf " %q" "${arg}"
                redact_next=1
                ;;
            --access_id=*|--access_key=*|--security_token=*)
                printf " %q" "${arg%%=*}=********"
                ;;
            *)
                printf " %q" "${arg}"
                ;;
        esac
    done
    printf "\n"
}

HAS_JUDGE_CONFIG=0
if [[ -n "${JUDGE_CONFIG}" ]]; then
    HAS_JUDGE_CONFIG=1
fi

EVAL_OUTPUT_BASE=$(eval_cfg '.eval.output_path')
EVAL_RESULT_PATH="${EVAL_OUTPUT_BASE}/${TIMESTAMP}"
JUDGE_JOB_ID=""
JUDGE_JOB_NAME=""
JUDGE_RUNTIME_CONFIG=""
JUDGE_LOG_DIR=""
JUDGE_SCRIPT=""
JUDGE_RESOURCE_ID=""
JUDGE_WORKSPACE_ID=""
if [[ "${HAS_JUDGE_CONFIG}" == "1" ]]; then
    JUDGE_LOG_DIR="${FIXED_LOG_DIR}/judge_logs"
    JUDGE_RUNTIME_CONFIG="${FIXED_LOG_DIR}/judge_runtime_config.json"
    prepare_judge_runtime_config "${EVAL_RESULT_PATH}" "${JUDGE_RUNTIME_CONFIG}" "${JUDGE_LOG_DIR}"
    if [[ "${JUDGE_BACKEND}" == "api" ]]; then
        JUDGE_JOB_NAME_FROM_CFG=$(dlc_cfg '.dlc.judge_job_name // ""')
        if [[ -n "${JUDGE_JOB_NAME_FROM_CFG}" && "${JUDGE_JOB_NAME_FROM_CFG}" != "null" ]]; then
            JUDGE_JOB_NAME="${JUDGE_JOB_NAME_FROM_CFG}"
        else
            JUDGE_JOB_NAME="judge_${JOB_NAME}_${TIMESTAMP}"
        fi
        JUDGE_JOB_NAME="$(printf "%s" "${JUDGE_JOB_NAME}" | tr -c '[:alnum:]_-' '_' | cut -c1-120)"

        JUDGE_SCRIPT_FROM_CFG=$(dlc_cfg '.dlc.judge_run_script // ""')
        if [[ -n "${JUDGE_SCRIPT_FROM_CFG}" && "${JUDGE_SCRIPT_FROM_CFG}" != "null" ]]; then
            JUDGE_SCRIPT="${JUDGE_SCRIPT_FROM_CFG}"
        else
            JUDGE_SCRIPT="${SCRIPT_DIR}/run_judge.sh"
        fi
        if [[ ! -f "${JUDGE_SCRIPT}" ]]; then
            echo "[ERROR] Judge script not found: ${JUDGE_SCRIPT}" >&2
            exit 1
        fi
        build_judge_submit_args "${JUDGE_RUNTIME_CONFIG}" "${JUDGE_JOB_NAME}" "${JUDGE_LOG_DIR}" "${JUDGE_SCRIPT}"
    fi
fi

INLINE_JUDGE_ARG=""
if [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    INLINE_JUDGE_ARG=" \"\" ${JUDGE_RUNTIME_CONFIG}"
fi
INNER_COMMAND="set -euo pipefail; cd ${LMMS_EVAL_ROOT}; export LMMS_EVAL_LOG_DIR=${FIXED_LOG_DIR}; export LMMS_EVAL_STAGE_DATASETS=0;${EXTRA_LD_EXPORT} bash ${WORKER_SCRIPT} ${RUNTIME_CONFIG}${INLINE_JUDGE_ARG}"
COMMAND="/bin/bash -c '$(quote_for_single_quotes "${INNER_COMMAND}")'"
DLC_SUBMIT_ARGS=(
    submit pytorchjob
    --name="${JOB_NAME}"
    --priority="${PRIORITY}"
    --workers="${WORKERS}"
    --worker_cpu="${WORKER_CPU}"
    --worker_gpu="${WORKER_GPU}"
    --worker_memory="${WORKER_MEMORY}"
    --worker_shared_memory="${WORKER_SHARED_MEMORY}"
    --worker_image="${WORKER_IMAGE}"
    --job_max_running_time_minutes="${JOB_MAX_RUNNING_TIME_MINUTES}"
    --data_source_uris="${DATA_SOURCE_URIS}"
    --resource_id="${RESOURCE_ID}"
    --workspace_id="${WORKSPACE_ID}"
    --vpc_id="${VPC_ID}"
    --switch_id="${SWITCH_ID}"
    --security_group_id="${SECURITY_GROUP_ID}"
    --extended_cidrs="${EXTENDED_CIDRS}"
    "${OPTIONAL_DLC_ARGS[@]}"
    --command="${COMMAND}"
)

# ── submit ────────────────────────────────────────────────────────────────────
echo "[INFO] Safety override for cluster run: debug=false"
echo "[INFO] Submitting eval DLC job: ${JOB_NAME}"
echo "[INFO] Worker script: ${WORKER_SCRIPT}"
echo "[INFO] Job max running time: ${JOB_MAX_RUNNING_TIME_MINUTES} minutes"
if [[ "${JUDGE_BACKEND}" == "api" ]]; then
    echo "[INFO] API judge enabled; a CPU-only judge DLC will be submitted after eval succeeds."
    echo "[INFO] Judge script: ${JUDGE_SCRIPT}"
    echo "[INFO] Judge runtime config: ${JUDGE_RUNTIME_CONFIG}"
elif [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    echo "[INFO] Local vLLM judge enabled; eval and judge will run sequentially in the same 8-GPU DLC job."
    echo "[INFO] Inline judge runtime config: ${JUDGE_RUNTIME_CONFIG}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    print_dry_run_command "eval" "${DLC_BINARY}" "${DLC_SUBMIT_ARGS[@]}"
    if [[ "${JUDGE_BACKEND}" == "api" ]]; then
        print_dry_run_command "judge" "${DLC_BINARY}" "${JUDGE_SUBMIT_ARGS[@]}"
    fi
    exit 0
fi

EVAL_JOB_ID="$(submit_and_resolve_job_id "${JOB_NAME}" "eval" "${RESOURCE_ID}" "${WORKSPACE_ID}" "${DLC_SUBMIT_ARGS[@]}")"

echo "[INFO] Job submitted successfully."
echo "[INFO] Eval JobId: ${EVAL_JOB_ID}"
echo "[INFO] Expected log locations:"
echo "  - vLLM logs: ${FIXED_LOG_DIR}"
echo "  - eval log:  ${FIXED_LOG_DIR}"
echo "[INFO] Unified timestamp: ${TIMESTAMP}"

if [[ "${HAS_JUDGE_CONFIG}" != "1" ]]; then
    exit 0
fi

wait_for_job_success "${EVAL_JOB_ID}" "eval" "${RUNNING_TIMEOUT}" "${RESOURCE_ID}" "${WORKSPACE_ID}"

if [[ "${JUDGE_BACKEND}" == "vllm" ]]; then
    echo "[INFO] Eval + inline local vLLM judge workflow completed successfully in DLC job: ${EVAL_JOB_ID}"
    echo "[INFO] Eval result path: ${EVAL_RESULT_PATH}"
    echo "[INFO] Judge output path: ${EVAL_RESULT_PATH}/judge"
    exit 0
fi

echo "[INFO] Eval completed; submitting CPU-only judge DLC job: ${JUDGE_JOB_NAME}"
JUDGE_JOB_ID="$(submit_and_resolve_job_id "${JUDGE_JOB_NAME}" "judge" "${JUDGE_RESOURCE_ID}" "${JUDGE_WORKSPACE_ID}" "${JUDGE_SUBMIT_ARGS[@]}")"
echo "[INFO] Judge JobId: ${JUDGE_JOB_ID}"

JUDGE_RUNNING_TIMEOUT=$(dlc_judge_cfg_int '.dlc.judge.running_timeout')
wait_for_job_success "${JUDGE_JOB_ID}" "judge" "${JUDGE_RUNNING_TIMEOUT}" "${JUDGE_RESOURCE_ID}" "${JUDGE_WORKSPACE_ID}"

echo "[INFO] Eval + CPU judge DLC workflow completed successfully."
echo "[INFO] Eval result path: ${EVAL_RESULT_PATH}"
echo "[INFO] Judge output path: ${EVAL_RESULT_PATH}/judge"
