"""
LMMs-Eval Web UI Server - FastAPI backend with static file serving.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import mimetypes
import os
import platform
import re
import secrets
import signal
import shlex
import socket
import subprocess
import time
import uuid
from collections import Counter
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lmms_eval.tui.discovery import get_discovery_cache

app = FastAPI(title="LMMs-Eval Web UI", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "web" / "dist"
LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = LMMS_EVAL_ROOT / "lmms_eval" / "tasks"
RUN_SCRIPTS_DIR = LMMS_EVAL_ROOT / "run_scripts"
DLC_SUBMIT_SCRIPT = RUN_SCRIPTS_DIR / "qwen35_submit.sh"
DEFAULT_DLC_CONFIG_PATH = RUN_SCRIPTS_DIR / "config_dlc.json"
DEFAULT_EVAL_CONFIG_PATH = RUN_SCRIPTS_DIR / "config_eval.json"
DEFAULT_JUDGE_CONFIG_PATH = RUN_SCRIPTS_DIR / "config_judge.json"
DEFAULT_RUN_MODE = "dlc"
DEFAULT_DLC_WORKERS = 1
DEFAULT_DLC_WORKER_GPU = 8
DEFAULT_DLC_WORKSPACE_ID = "240810"
DEFAULT_DLC_RESOURCE_ID = "quotaev2tl4w6aw0"
REQUIRED_NAS_MOUNT_URI = "nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB"
DEFAULT_DLC_POOL_TOTAL_GPU = 256
DEFAULT_DLC_POOL_GPU_PER_NODE = 8
DEFAULT_DLC_POOL_CPU_PER_NODE = 124
DEFAULT_MODEL_TP = 2
DEFAULT_MAX_MODEL_LEN = 40960
DEFAULT_GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_MAX_NUM_SEQS = 192
DEFAULT_BASE_PORT = 8941
DEFAULT_CONCURRENCY = 32
DEFAULT_GEN_KWARGS = ""
DEFAULT_REASONING_PARSER = "qwen3"
DEFAULT_ENABLE_THINKING = False
DEFAULT_EVAL_INFERENCE_MODE = "ckpt"
DEFAULT_API_EVAL_TYPE = "openai"
DEFAULT_API_EVAL_URL = "http://gw-k6isjixc1ij25ms7q4.cn-shanghai.pai-eas.aliyuncs.com/api/predict/router_fs_eval/v1"
DEFAULT_API_EVAL_MODEL = "/mnt/data/jingyichai/sft/checkpoints/qwen36_kimi_e2b_3tool_6k_0629_4nodes/iter_0000206_hf"
DEFAULT_JUDGE_BACKEND = "vllm"
DEFAULT_JUDGE_API_URL = "http://8.130.30.251:8801/v1"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_LOCAL_JUDGE_MODEL_PATH = "/mnt/cpfsB/tianleniu/Innovator-Tune/models/Qwen3.5-9B"
DEFAULT_LOCAL_JUDGE_MODEL = "Qwen3.5-9B"
DEFAULT_LOCAL_JUDGE_TP = 8
DEFAULT_LOCAL_JUDGE_MAX_MODEL_LEN = 40960
DEFAULT_LOCAL_JUDGE_GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_LOCAL_JUDGE_MAX_NUM_SEQS = 192
DEFAULT_LOCAL_JUDGE_PORT = 8002
DEFAULT_LOCAL_JUDGE_PARALLEL = 32
DEFAULT_AUTH_SESSION_TTL_SECONDS = 15 * 24 * 60 * 60
AUTH_VALIDATION_TIMEOUT_SECONDS = 20
AUTH_IDENTITY_TIMEOUT_SECONDS = 20
USER_PLACEHOLDER = "<USER>"
USER_PLACEHOLDER_ALIASES = (USER_PLACEHOLDER, "<USERNAME>")
DEFAULT_DLC_BINARY = "/mnt/cpfsB/<USER>/dlc"
DEFAULT_DLC_PATH_TEMPLATE = DEFAULT_DLC_BINARY
QWEN35_WORKER_BASENAME = "qwen35_worker.sh"
DEFAULT_AUTH_FILE_PATH = LMMS_EVAL_ROOT / "local" / "webui_users.json"
AUTH_FILE_ENV = "LMMS_EVAL_WEBUI_AUTH_FILE"
AUTH_SESSION_TTL_ENV = "LMMS_EVAL_WEBUI_SESSION_TTL_SECONDS"
AUTH_ALLOWED_ORIGINS_ENV = "LMMS_EVAL_WEBUI_ALLOWED_ORIGINS"
AUTH_COOKIE_NAME = "lmms_eval_webui_session"
AUTH_ADMIN_ROLE = "admin"
AUTH_USER_ROLE = "user"
AUTH_VALID_ROLES = {AUTH_USER_ROLE, AUTH_ADMIN_ROLE}
DLC_SAMPLE_MEDIA_MAX_TOKENS = 4096
DLC_SAMPLE_MEDIA_TOKENS: dict[str, Path] = {}
AUTH_PROTECTED_PREFIXES = (
    "/auth/me",
    "/auth/logout",
    "/defaults",
    "/models",
    "/tasks",
    "/eval",
    "/dlc",
    "/logs",
)
EVAL_JOB_NAME_PREFIX = "eval_"
DEFAULT_EVAL_JOB_NAME = "eval_qwen35_9b_feishu20"
JUDGE_JOB_NAME_PREFIX = "judge_"
VIEW_LOG_JOB_NAME_PREFIXES = (EVAL_JOB_NAME_PREFIX, JUDGE_JOB_NAME_PREFIX)
MASKED_SECRET = "********"
DLC_ACCESS_ID_FLAG = "--access_id"
DLC_ACCESS_KEY_FLAG = "--access_key"
DLC_IGNORE_LOCAL_CONFIG_FLAG = "--ignore_local_config"
JUDGE_INPUT_RESULT_PLACEHOLDER = "${EVAL_RESULT_PATH}"
JUDGE_OUTPUT_PATH_PLACEHOLDER = "${EVAL_RESULT_PATH}/judge"
LLM_AS_JUDGE_EXACT_TASKS = {
    "molparse",
    "openrxn",
    "ocrbench",
    "simplevqa",
}
LLM_AS_JUDGE_TASK_PATTERNS = (
    r"^mmbench(?:_|-).*(?:_|-)dev$",
    r"^mmbench(?:_|-)cn(?:_|-)cc$",
    r"^mmmu.*qwen3.*official$",
    r"^mmmu_pro.*qwen3.*official$",
    r"^sfe(?:-|_).*$",
    r"^scivqr_(?:open|reasoning)$",
    r"^mathverse.*(?:reasoning|qwen3).*$",
    r"^mathvista.*(?:reasoning|qwen3).*$",
    r"^mathvision.*(?:reasoning|qwen3).*$",
    r"^wemath.*(?:reasoning|qwen3).*$",
)
DLC_REGION = "cn-wulanchabu"
DLC_ENDPOINT = "pai-dlc.cn-wulanchabu.aliyuncs.com"
DLC_JOBS_CACHE_TTL_SECONDS = 15
DLC_POOL_USAGE_CACHE_TTL_SECONDS = 20
DLC_STOP_TIMEOUT_SECONDS = 60
PAISTUDIO_USER_USAGE_CACHE_TTL_SECONDS = 60
AIWORKSPACE_MEMBER_CACHE_TTL_SECONDS = 300
DLC_POOL_ACTIVE_STATUSES = ("Running", "EnvPreparing", "Pending", "Queuing")
DLC_KILLABLE_STATUSES = {"running", "queuing", "queueing", "envpreparing"}
PAISTUDIO_PRODUCT = "paistudio"
AIWORKSPACE_PRODUCT = "aiworkspace"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
EVAL_JOB_NAME_PATTERN = re.compile(r"^eval_[A-Za-z0-9_-]+$")


def _allowed_cors_origins() -> list[str]:
    raw = os.getenv(AUTH_ALLOWED_ORIGINS_ENV, "").strip()
    if not raw:
        return []
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError(f"{AUTH_ALLOWED_ORIGINS_ENV}=* is not allowed with credentialed WebUI auth")
    return origins


# Same-origin WebUI requests do not need CORS. Set LMMS_EVAL_WEBUI_ALLOWED_ORIGINS
# to a comma-separated list only when running a separate frontend dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage
_jobs: dict[str, dict[str, Any]] = {}
_task_manager: Any | None = None
_dataset_cache: dict[tuple[str, str | None, str], Any] = {}
_dlc_jobs_cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
_dlc_pool_usage_cache: tuple[float, dict[str, Any]] | None = None
_paistudio_user_name_cache: tuple[float, dict[str, str]] | None = None
_aiworkspace_member_name_cache: tuple[float, dict[str, str]] | None = None
_auth_sessions: dict[str, dict[str, Any]] = {}


def _auth_file_path() -> Path:
    configured_path = os.getenv(AUTH_FILE_ENV, "").strip()
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_AUTH_FILE_PATH


def _auth_session_ttl_seconds() -> int:
    raw = os.getenv(AUTH_SESSION_TTL_ENV, str(DEFAULT_AUTH_SESSION_TTL_SECONDS)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"{AUTH_SESSION_TTL_ENV} must be an integer") from exc
    if value <= 0:
        raise HTTPException(status_code=500, detail=f"{AUTH_SESSION_TTL_ENV} must be positive")
    return value


def _auth_error(message: str, *, status_code: int = 500) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def _load_auth_admins() -> dict[str, dict[str, str]]:
    path = _auth_file_path()
    if not path.exists() or not path.is_file():
        raise _auth_error(f"WebUI auth admin file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise _auth_error(f"Failed to read WebUI auth admin file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _auth_error(f"WebUI auth admin file is invalid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise _auth_error("WebUI auth admin file must contain a JSON object")
    raw_admins = data.get("admins")
    if raw_admins is None:
        raw_admins = []
    if not isinstance(raw_admins, list):
        raise _auth_error("WebUI auth admin file admins field must be a list")

    admins: dict[str, dict[str, str]] = {}
    seen_access_key_ids: set[str] = set()

    for idx, raw_admin in enumerate(raw_admins):
        if not isinstance(raw_admin, dict):
            raise _auth_error(f"WebUI auth admin at index {idx} must be an object")
        if "secret_access_key" in raw_admin:
            raise _auth_error(f"WebUI auth admin at index {idx} must not contain secret_access_key")
        normalized: dict[str, str] = {}

        access_key_id = raw_admin.get("access_key_id")
        if not isinstance(access_key_id, str) or not access_key_id.strip():
            raise _auth_error(f"WebUI auth admin at index {idx} is missing non-empty access_key_id")
        normalized["access_key_id"] = access_key_id.strip()

        username = raw_admin.get("username", normalized["access_key_id"])
        if username is not None and not isinstance(username, str):
            raise _auth_error(f"WebUI auth admin at index {idx} has invalid username")
        normalized["username"] = (username or normalized["access_key_id"]).strip() or normalized["access_key_id"]

        display_name = raw_admin.get("display_name", normalized["username"])
        if display_name is not None and not isinstance(display_name, str):
            raise _auth_error(f"WebUI auth admin at index {idx} has invalid display_name")
        normalized["display_name"] = (display_name or normalized["username"]).strip() or normalized["username"]

        access_key_id = normalized["access_key_id"]
        if access_key_id in seen_access_key_ids:
            raise _auth_error(f"Duplicate WebUI auth admin access_key_id: {access_key_id}")
        seen_access_key_ids.add(access_key_id)
        normalized["role"] = AUTH_ADMIN_ROLE
        admins[access_key_id] = normalized

    return admins


def _validate_auth_credentials(access_key_id: str, secret_access_key: str) -> bool:
    dlc_binary = _resolve_dlc_binary()
    args = [
        dlc_binary,
        "get",
        "job",
        "--workspace_id",
        DEFAULT_DLC_WORKSPACE_ID,
        "--page_size",
        "1",
        "--page_num",
        "1",
        "--region",
        DLC_REGION,
        "--endpoint",
        DLC_ENDPOINT,
        "--access_id",
        access_key_id,
        "--access_key",
        secret_access_key,
        "--ignore_local_config",
    ]
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=AUTH_VALIDATION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _load_authenticated_aliyun_identity(access_key_id: str, secret_access_key: str) -> dict[str, str]:
    args = [
        "aliyun",
        "sts",
        "GetCallerIdentity",
        "--mode",
        "AK",
        "--access-key-id",
        access_key_id,
        "--access-key-secret",
        secret_access_key,
        "--region",
        DLC_REGION,
    ]
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=AUTH_IDENTITY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    def text_value(*keys: str) -> str:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    identity = {
        "aliyun_user_id": text_value("UserId", "UserID", "PrincipalId"),
        "aliyun_account_id": text_value("AccountId"),
        "aliyun_principal_id": text_value("PrincipalId"),
        "aliyun_arn": text_value("Arn"),
    }
    return {key: value for key, value in identity.items() if value}


def _build_authenticated_user(access_key_id: str, secret_access_key: str) -> dict[str, str] | None:
    normalized_access_key_id = access_key_id.strip()
    normalized_secret_access_key = secret_access_key.strip()
    if not normalized_access_key_id or not normalized_secret_access_key:
        raise _auth_error("Access Key ID and Secret Access Key are required", status_code=400)

    if not _validate_auth_credentials(normalized_access_key_id, normalized_secret_access_key):
        return None

    identity = _load_authenticated_aliyun_identity(normalized_access_key_id, normalized_secret_access_key)
    admins = _load_auth_admins()
    admin = admins.get(normalized_access_key_id)
    if admin is not None:
        user = copy.deepcopy(admin)
        user["secret_access_key"] = normalized_secret_access_key
        user.update(identity)
        return user

    return {
        "username": normalized_access_key_id,
        "display_name": normalized_access_key_id,
        "access_key_id": normalized_access_key_id,
        "secret_access_key": normalized_secret_access_key,
        "role": AUTH_USER_ROLE,
        **identity,
    }


def _public_auth_user(user: dict[str, Any]) -> dict[str, str]:
    return {
        "username": str(user["username"]),
        "display_name": str(user.get("display_name") or user["username"]),
        "role": str(user["role"]),
        "access_key_id": str(user["access_key_id"]),
    }


def _create_auth_session(user: dict[str, str]) -> tuple[str, dict[str, Any]]:
    ttl_seconds = _auth_session_ttl_seconds()
    now = time.time()
    session_id = secrets.token_urlsafe(32)
    session = {
        **copy.deepcopy(user),
        "created_at": now,
        "expires_at": now + ttl_seconds,
    }
    _auth_sessions[session_id] = session
    return session_id, session


def _get_auth_session(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    session = _auth_sessions.get(session_id)
    if session is None:
        return None
    expires_at = float(session.get("expires_at") or 0)
    if expires_at <= time.time():
        _auth_sessions.pop(session_id, None)
        return None
    return session


def _delete_auth_session(session_id: str | None) -> None:
    if session_id:
        _auth_sessions.pop(session_id, None)


def _is_protected_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in AUTH_PROTECTED_PREFIXES)


def _require_authenticated_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "auth_user", None)
    if isinstance(user, dict):
        return user
    session = _get_auth_session(request.cookies.get(AUTH_COOKIE_NAME))
    if session is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    request.state.auth_user = session
    return session


def _require_admin_user(request: Request) -> dict[str, Any]:
    user = _require_authenticated_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


@app.middleware("http")
async def _webui_auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    if _is_protected_path(request.url.path):
        session = _get_auth_session(request.cookies.get(AUTH_COOKIE_NAME))
        if session is None:
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})
        request.state.auth_user = session

    return await call_next(request)


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        from lmms_eval.tasks import TaskManager

        _task_manager = TaskManager(verbosity="ERROR")
    return _task_manager


def _get_task_dataset_spec(task_name: str) -> tuple[str, str | None, str, dict[str, Any]]:
    manager = _get_task_manager()
    info = manager.task_index.get(task_name)
    if info is None or info.get("type") != "task":
        raise HTTPException(status_code=404, detail="Task not found")

    yaml_path = info.get("yaml_path")
    if not yaml_path or yaml_path == -1:
        raise HTTPException(status_code=404, detail="Task config not found")

    from lmms_eval import utils as lmms_utils

    config = lmms_utils.load_yaml_config(yaml_path, mode="full")
    if not isinstance(config, dict):
        raise HTTPException(status_code=500, detail="Task config is invalid")

    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise HTTPException(status_code=404, detail="Task dataset is not configured")

    dataset_name = config.get("dataset_name")
    if dataset_name is not None and not isinstance(dataset_name, str):
        dataset_name = None

    split: str | None = None
    for key in ("test_split", "validation_split", "train_split", "split"):
        value = config.get(key)
        if isinstance(value, str) and value:
            split = value
            break
    if split is None:
        raise HTTPException(status_code=404, detail="Task split is not configured")

    dataset_kwargs = config.get("dataset_kwargs")
    if not isinstance(dataset_kwargs, dict):
        dataset_kwargs = {}

    return dataset_path, dataset_name, split, dataset_kwargs


def _get_dataset(dataset_path: str, dataset_name: str | None, split: str, dataset_kwargs: dict[str, Any]):
    cache_key = (dataset_path, dataset_name, split)
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    from datasets import load_dataset

    kwargs = dict(dataset_kwargs)
    if dataset_name:
        dataset = load_dataset(dataset_path, dataset_name, split=split, **kwargs)
    else:
        dataset = load_dataset(dataset_path, split=split, **kwargs)
    _dataset_cache[cache_key] = dataset
    return dataset


def _serialize_pil_image(image) -> tuple[bytes, str]:
    fmt = getattr(image, "format", None)
    pil_format = fmt.upper() if isinstance(fmt, str) and fmt else "PNG"
    mime = f"image/{pil_format.lower()}"
    if pil_format == "JPG":
        pil_format = "JPEG"
        mime = "image/jpeg"

    buffer = io.BytesIO()
    image.save(buffer, format=pil_format)
    return buffer.getvalue(), mime


def _extract_image_blob(value: Any) -> tuple[bytes, str] | None:
    if value is None:
        return None

    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return _serialize_pil_image(value)
    except ImportError:
        pass

    if isinstance(value, (bytes, bytearray)):
        return bytes(value), "image/png"

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            path_hint = value.get("path") if isinstance(value.get("path"), str) else None
            guessed, _ = mimetypes.guess_type(path_hint or "")
            media_type = guessed if guessed and guessed.startswith("image/") else "image/png"
            return bytes(raw_bytes), media_type

        for candidate_key in ("image", "img", "picture"):
            if candidate_key in value:
                nested = _extract_image_blob(value[candidate_key])
                if nested is not None:
                    return nested

        for nested_value in value.values():
            nested = _extract_image_blob(nested_value)
            if nested is not None:
                return nested

    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _extract_image_blob(item)
            if nested is not None:
                return nested

    return None


def _extract_video_path(value: Any, dataset_cache_dir: str | None) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

        resolved = resolve_media_reference(value, media_type="video", cache_dir=dataset_cache_dir)
        if isinstance(resolved, str) and Path(resolved).exists():
            return resolved
        return None

    if isinstance(value, dict):
        for key in ("video", "video_path", "path", "file", "clip_path"):
            candidate = value.get(key)
            path = _extract_video_path(candidate, dataset_cache_dir)
            if path is not None:
                return path

        for nested in value.values():
            path = _extract_video_path(nested, dataset_cache_dir)
            if path is not None:
                return path

    if isinstance(value, (list, tuple)):
        for item in value:
            path = _extract_video_path(item, dataset_cache_dir)
            if path is not None:
                return path

    return None


def _resolve_dataset_media(task_name: str, doc_id: int) -> tuple[str, bytes | str, str]:
    dataset_path, dataset_name, split, dataset_kwargs = _get_task_dataset_spec(task_name)
    dataset = _get_dataset(dataset_path, dataset_name, split, dataset_kwargs)

    try:
        record = dataset[doc_id]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Sample doc_id not found in dataset") from exc

    image_payload = _extract_image_blob(record)
    if image_payload is not None:
        image_bytes, media_type = image_payload
        return "bytes", image_bytes, media_type

    cache_dir = dataset_kwargs.get("cache_dir") if isinstance(dataset_kwargs.get("cache_dir"), str) else None
    video_path = _extract_video_path(record, cache_dir)
    if video_path is not None:
        guessed, _ = mimetypes.guess_type(video_path)
        media_type = guessed if guessed and guessed.startswith("video/") else "video/mp4"
        return "file", video_path, media_type

    raise HTTPException(status_code=404, detail="No image/video found in dataset sample")


def get_version() -> str:
    """Get lmms-eval version from package metadata."""
    try:
        return pkg_version("lmms_eval")
    except Exception:
        return "0.6.0"


def get_git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def get_repo_root() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    except Exception:
        return ""


def _detect_env_setup() -> str:
    """Auto-detect environment activation command.

    Builds: cd <repo_root> && source .venv/bin/activate
    Returns empty string if no venv found.
    """
    repo_root = get_repo_root()
    if repo_root:
        activate = Path(repo_root) / ".venv" / "bin" / "activate"
        if activate.exists():
            return f"cd {repo_root} && source .venv/bin/activate"
    return ""


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {path}")
    return data


def _split_tasks(tasks: str | list[str]) -> list[str]:
    if isinstance(tasks, str):
        return [task.strip() for task in tasks.split(",") if task.strip()]
    if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
        return [task.strip() for task in tasks if task.strip()]
    raise ValueError("tasks must be a comma-separated string or a list of strings")


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _default_dlc_config() -> dict[str, Any]:
    config = copy.deepcopy(_load_json_file(DEFAULT_DLC_CONFIG_PATH))
    dlc = config.get("dlc")
    if not isinstance(dlc, dict):
        raise RuntimeError(f"Missing dlc object in {DEFAULT_DLC_CONFIG_PATH}")
    dlc["resource_id"] = _require_default_config_resource_id(dlc.get("resource_id"), field="dlc.resource_id")
    dlc["data_source_uris"] = _require_default_config_nas_mount(
        dlc.get("data_source_uris"), field="dlc.data_source_uris"
    )
    judge = dlc.get("judge")
    if isinstance(judge, dict):
        judge["resource_id"] = _require_default_config_resource_id(judge.get("resource_id"), field="dlc.judge.resource_id")
        judge["data_source_uris"] = _require_default_config_nas_mount(
            judge.get("data_source_uris"), field="dlc.judge.data_source_uris"
        )
    dlc["submit"] = True
    dlc["job_name"] = DEFAULT_EVAL_JOB_NAME
    dlc["run_script"] = f"/mnt/cpfsB/{USER_PLACEHOLDER}/Innovator-Tune/lmms-eval/run_scripts/{QWEN35_WORKER_BASENAME}"
    dlc["workers"] = DEFAULT_DLC_WORKERS
    dlc["worker_gpu"] = DEFAULT_DLC_WORKER_GPU
    dlc["workspace_id"] = DEFAULT_DLC_WORKSPACE_ID
    return config


def _require_default_config_resource_id(resource_id: Any, *, field: str) -> str:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise RuntimeError(f"Missing {field} in {DEFAULT_DLC_CONFIG_PATH}")
    normalized = resource_id.strip()
    if normalized != DEFAULT_DLC_RESOURCE_ID:
        raise RuntimeError(f"{field} in {DEFAULT_DLC_CONFIG_PATH} must be {DEFAULT_DLC_RESOURCE_ID}, got: {normalized}")
    return normalized


def _require_default_config_nas_mount(data_source_uris: Any, *, field: str) -> str:
    if not isinstance(data_source_uris, str) or not data_source_uris.strip():
        raise RuntimeError(f"Missing {field} in {DEFAULT_DLC_CONFIG_PATH}")
    normalized = data_source_uris.strip()
    if REQUIRED_NAS_MOUNT_URI not in [item.strip() for item in normalized.split(",")]:
        raise RuntimeError(f"{field} in {DEFAULT_DLC_CONFIG_PATH} must include {REQUIRED_NAS_MOUNT_URI}")
    return normalized


def _default_eval_config() -> dict[str, Any]:
    config = copy.deepcopy(_load_json_file(DEFAULT_EVAL_CONFIG_PATH))
    for key in ("env", "log", "distributed", "model", "eval"):
        if not isinstance(config.get(key), dict):
            raise RuntimeError(f"Missing {key} object in {DEFAULT_EVAL_CONFIG_PATH}")
    return config


def _default_judge_config() -> dict[str, Any]:
    config = copy.deepcopy(_load_json_file(DEFAULT_JUDGE_CONFIG_PATH))
    for key in ("env", "log", "judge", "eval"):
        if not isinstance(config.get(key), dict):
            raise RuntimeError(f"Missing {key} object in {DEFAULT_JUDGE_CONFIG_PATH}")
    judge = config["judge"]
    if not isinstance(judge.get("api"), dict):
        raise RuntimeError(f"Missing judge.api object in {DEFAULT_JUDGE_CONFIG_PATH}")
    api = judge["api"]
    model = judge.get("model", DEFAULT_JUDGE_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(f"Missing judge.model in {DEFAULT_JUDGE_CONFIG_PATH}")
    base_url = api.get("base_url", DEFAULT_JUDGE_API_URL)
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError(f"Missing judge.api.base_url in {DEFAULT_JUDGE_CONFIG_PATH}")
    judge["model"] = model.strip()
    api["base_url"] = base_url.strip()
    return config


def _default_env_vars() -> str:
    env = _default_eval_config()["env"]
    return _dict_to_env_vars({str(key): value for key, value in env.items()})


def _default_user() -> str:
    configured_user = os.getenv("LMMS_EVAL_WEBUI_USER") or os.getenv("LMMS_EVAL_USER")
    if configured_user and configured_user.strip():
        return configured_user.strip()

    match = re.match(r"^/mnt/(?:cpfsB|cpfs)/([^/]+)/Innovator-Tune/lmms-eval$", str(LMMS_EVAL_ROOT))
    if match:
        return match.group(1)
    return ""


def _replace_user_placeholder(value: Any, user: str) -> Any:
    user = user.strip()
    if not user:
        return value
    if isinstance(value, str):
        for placeholder in USER_PLACEHOLDER_ALIASES:
            value = value.replace(placeholder, user)
        return value
    if isinstance(value, list):
        return [_replace_user_placeholder(item, user) for item in value]
    if isinstance(value, dict):
        return {key: _replace_user_placeholder(item, user) for key, item in value.items()}
    return value


def _contains_user_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return any(placeholder in value for placeholder in USER_PLACEHOLDER_ALIASES)
    if isinstance(value, list):
        return any(_contains_user_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_user_placeholder(item) for item in value.values())
    return False


def _task_requires_llm_judge(task: str) -> bool:
    normalized = task.strip().lower()
    if not normalized:
        return False
    if normalized in LLM_AS_JUDGE_EXACT_TASKS:
        return True
    return any(re.search(pattern, normalized) for pattern in LLM_AS_JUDGE_TASK_PATTERNS)


def _llm_as_judge_tasks(tasks: list[str]) -> list[str]:
    return [task for task in tasks if _task_requires_llm_judge(task)]


def _sync_judge_api_to_eval_env(
    config: dict[str, Any],
    request: EvalRequest | PreviewRequest | ExportYamlRequest,
) -> None:
    """Expose judge credentials to eval-time scorers that import OpenAI clients."""
    if _validate_judge_backend(request.judge_backend) != "api":
        return
    api_key = request.judge_api_key.strip()
    api_url = request.judge_api_url.strip()
    if not api_key:
        return

    env = config.setdefault("env", {})
    if not isinstance(env, dict):
        raise HTTPException(status_code=400, detail="eval config env must be an object")
    env["judge_api_key"] = api_key
    if api_url:
        env["judge_base_url"] = api_url
    if not str(env.get("openai_api_key") or "").strip():
        env["openai_api_key"] = api_key
    if api_url and not str(env.get("openai_api_url") or "").strip():
        env["openai_api_url"] = api_url
    if not str(env.get("api_type") or "").strip():
        env["api_type"] = DEFAULT_API_EVAL_TYPE


def get_system_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "repo_root": get_repo_root(),
    }


# --- Models ---


class AuthLoginRequest(BaseModel):
    access_key_id: str
    secret_access_key: str


class AuthUserResponse(BaseModel):
    username: str
    display_name: str
    role: str
    access_key_id: str
    expires_at: float


class ModelInfo(BaseModel):
    id: str
    name: str


class TaskInfo(BaseModel):
    id: str
    name: str
    group: bool = False
    requires_llm_judge: bool = False


class TaskCreateRequest(BaseModel):
    task_id: str
    yaml_content: str
    python_content: str
    overwrite: bool = False


class TaskCreateResponse(BaseModel):
    task_id: str
    task_dir: str
    yaml_path: str
    python_path: str
    discovered_task_count: int


class EvalRequest(BaseModel):
    user: str = ""
    job_name: str = DEFAULT_EVAL_JOB_NAME
    eval_inference_mode: str = DEFAULT_EVAL_INFERENCE_MODE
    model: str
    api_url: str = DEFAULT_API_EVAL_URL
    api_key: str = ""
    dlc_path: str = DEFAULT_DLC_PATH_TEMPLATE
    model_args: str = ""
    tasks: list[str]
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_api_url: str = ""
    judge_api_key: str = ""
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None
    env_setup: str = ""
    run_mode: str = DEFAULT_RUN_MODE
    dlc_config: dict[str, Any] = Field(default_factory=dict)
    model_tp: int = DEFAULT_MODEL_TP
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    base_port: int = DEFAULT_BASE_PORT
    concurrency: int = DEFAULT_CONCURRENCY
    gen_kwargs: str = DEFAULT_GEN_KWARGS
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    debug: bool = False


class EvalStartResponse(BaseModel):
    job_id: str
    command: str


class PreviewRequest(BaseModel):
    user: str = ""
    job_name: str = DEFAULT_EVAL_JOB_NAME
    eval_inference_mode: str = DEFAULT_EVAL_INFERENCE_MODE
    model: str
    api_url: str = DEFAULT_API_EVAL_URL
    api_key: str = ""
    dlc_path: str = DEFAULT_DLC_PATH_TEMPLATE
    model_args: str = ""
    tasks: list[str]
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_api_url: str = ""
    judge_api_key: str = ""
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None
    env_setup: str = ""
    run_mode: str = DEFAULT_RUN_MODE
    dlc_config: dict[str, Any] = Field(default_factory=dict)
    model_tp: int = DEFAULT_MODEL_TP
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    base_port: int = DEFAULT_BASE_PORT
    concurrency: int = DEFAULT_CONCURRENCY
    gen_kwargs: str = DEFAULT_GEN_KWARGS
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    debug: bool = False


class PreviewResponse(BaseModel):
    command: str


class ExportYamlRequest(BaseModel):
    user: str = ""
    job_name: str = DEFAULT_EVAL_JOB_NAME
    eval_inference_mode: str = DEFAULT_EVAL_INFERENCE_MODE
    model: str
    api_url: str = DEFAULT_API_EVAL_URL
    api_key: str = ""
    dlc_path: str = DEFAULT_DLC_PATH_TEMPLATE
    model_args: str = ""
    tasks: list[str]
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_api_url: str = ""
    judge_api_key: str = ""
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None
    env_setup: str = ""
    run_mode: str = DEFAULT_RUN_MODE
    dlc_config: dict[str, Any] = Field(default_factory=dict)
    model_tp: int = DEFAULT_MODEL_TP
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    base_port: int = DEFAULT_BASE_PORT
    concurrency: int = DEFAULT_CONCURRENCY
    gen_kwargs: str = DEFAULT_GEN_KWARGS
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    debug: bool = False


class ExportYamlResponse(BaseModel):
    yaml_content: str


class ImportYamlRequest(BaseModel):
    yaml_content: str


class ImportYamlResponse(BaseModel):
    user: str = ""
    job_name: str = DEFAULT_EVAL_JOB_NAME
    eval_inference_mode: str = DEFAULT_EVAL_INFERENCE_MODE
    model: str = ""
    api_url: str = DEFAULT_API_EVAL_URL
    api_key: str = ""
    dlc_path: str = DEFAULT_DLC_PATH_TEMPLATE
    model_args: str = ""
    tasks: list[str] = []
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_api_url: str = ""
    judge_api_key: str = ""
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = None
    output_path: str = "./logs/"
    log_samples: bool = False
    verbosity: str = "INFO"
    device: str | None = None
    run_mode: str = DEFAULT_RUN_MODE
    dlc_config: dict[str, Any] = Field(default_factory=dict)
    model_tp: int = DEFAULT_MODEL_TP
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    base_port: int = DEFAULT_BASE_PORT
    concurrency: int = DEFAULT_CONCURRENCY
    gen_kwargs: str = DEFAULT_GEN_KWARGS
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    debug: bool = False


class DefaultsResponse(BaseModel):
    user: str = ""
    job_name: str
    eval_inference_mode: str = DEFAULT_EVAL_INFERENCE_MODE
    model: str
    api_url: str = DEFAULT_API_EVAL_URL
    api_key: str = ""
    dlc_path: str
    model_args: str = ""
    tasks: list[str]
    judge_backend: str = DEFAULT_JUDGE_BACKEND
    judge_api_url: str = ""
    judge_api_key: str = ""
    env_vars: str
    batch_size: int
    limit: int | None
    output_path: str
    log_samples: bool
    verbosity: str
    device: str | None = None
    env_setup: str = ""
    run_mode: str
    dlc_config: dict[str, Any]
    model_tp: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    base_port: int
    concurrency: int
    gen_kwargs: str
    enable_thinking: bool
    debug: bool


class LogRunSummary(BaseModel):
    run_id: str
    model_name: str
    date: str
    tasks: list[str]
    metrics: dict[str, dict[str, Any]]
    total_evaluation_time_seconds: Any | None = None
    config: dict[str, Any]
    n_samples: dict[str, Any]


class LogSamplesResponse(BaseModel):
    samples: list[dict[str, Any]]
    total: int
    offset: int
    limit: int


class DlcJobSummary(BaseModel):
    job_id: str
    name: str
    status: str
    workspace_id: str = ""
    resource_id: str = ""
    job_type: str = ""
    priority: str = ""
    user_name: str = ""
    user_id: str = ""
    job_stage: str = ""
    lmms_tasks: list[str] = Field(default_factory=list)
    llm_judge_tasks: list[str] = Field(default_factory=list)
    requires_llm_judge: bool = False
    create_time: str = ""
    submitted_time: str = ""
    running_time: str = ""
    finish_time: str = ""
    duration_seconds: str = ""
    result_root: str | None = None
    has_results: bool = False
    can_kill: bool = False
    kill_disabled_reason: str = ""


class DlcJobsResponse(BaseModel):
    jobs: list[DlcJobSummary]
    total: int
    fetched_at: str
    source: str


class DlcPoolMetric(BaseModel):
    used: int
    total: int
    percent: float
    capacity_source: str


class DlcPoolJobUsage(BaseModel):
    job_id: str
    name: str
    status: str
    workspace_id: str = ""
    resource_id: str = ""
    gpu: int
    cpu: int
    pod_count: int


class DlcPoolUsageResponse(BaseModel):
    workspace_id: str
    resource_id: str
    resource_name: str = ""
    active_statuses: list[str]
    gpu: DlcPoolMetric
    cpu: DlcPoolMetric
    jobs: list[DlcPoolJobUsage]
    errors: list[str] = Field(default_factory=list)
    fetched_at: str
    source: str


class DlcJobDetailResponse(BaseModel):
    job: dict[str, Any]
    result_root: str | None = None
    runtime_config_path: str | None = None
    log_dir: str | None = None
    result_status: str


class DlcJobKillResponse(BaseModel):
    job_id: str
    status: str
    message: str


class DlcMetricRow(BaseModel):
    metric_id: str
    display_name: str
    lmms_tasks: str
    status: str
    value: Any | None = None
    value_text: str
    metric_name: str
    stderr: Any | None = None
    started_at: str = ""
    ended_at: str = ""
    wall_seconds: Any | None = None
    total_evaluation_time_seconds: Any | None = None
    n_samples: Any | None = None
    result_json: str | None = None
    sample_jsonls: list[str] = Field(default_factory=list)
    value_source: str = ""


class DlcMetricsResponse(BaseModel):
    job_id: str
    result_root: str | None
    metrics: list[DlcMetricRow]
    summary_files: list[str]
    message: str = ""


class ChoiceAnswerBucket(BaseModel):
    option: str
    count: int
    ratio: float


class ChoiceAnswerStats(BaseModel):
    is_multiple_choice: bool
    correct_answers: list[ChoiceAnswerBucket] = Field(default_factory=list)
    target_answers: list[ChoiceAnswerBucket] = Field(default_factory=list)
    total: int = 0
    filtered_total: int = 0
    wrong_total: int = 0
    unknown_correctness_total: int = 0
    correct_answer_total: int = 0
    target_answer_total: int = 0


class DlcSampleMedia(BaseModel):
    url: str
    label: str
    source: str
    media_type: str


class DlcMetricSamplesResponse(BaseModel):
    job_id: str
    metric_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total: int
    offset: int
    limit: int
    sample_files: list[str]
    answer_stats: ChoiceAnswerStats


def _resolve_logs_root(logs_path: str) -> Path:
    return Path(logs_path).expanduser().resolve()


def _ensure_path_within_base(base_path: Path, target_path: Path) -> None:
    try:
        target_path.relative_to(base_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes logs directory") from exc


def _resolve_run_results_path(logs_path: str, run_id: str) -> Path:
    logs_root = _resolve_logs_root(logs_path)
    if not logs_root.exists() or not logs_root.is_dir():
        raise HTTPException(status_code=404, detail="Logs path not found")

    decoded_run_id = Path(unquote(run_id))
    if decoded_run_id.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid run_id")

    run_path = (logs_root / decoded_run_id).resolve()
    _ensure_path_within_base(logs_root, run_path)
    return run_path


def _infer_cpfs_user() -> str:
    candidates = [Path.cwd(), LMMS_EVAL_ROOT]
    for candidate in candidates:
        parts = candidate.resolve().parts
        for idx, part in enumerate(parts[:-1]):
            if part in {"cpfs", "cpfsB"} and idx > 0 and parts[idx - 1] == "mnt":
                return parts[idx + 1]
    return os.environ.get("LMMS_EVAL_WEB_UI_USER", "")


def _replace_default_user(value: Any) -> Any:
    user = _infer_cpfs_user()
    if not user:
        return value
    return _replace_user_placeholder(value, user)


def _resolve_dlc_binary() -> str:
    configured = os.environ.get("LMMS_EVAL_DLC_BINARY", "").strip()
    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    try:
        dlc_config = _replace_default_user(_default_dlc_config())
        dlc = dlc_config.get("dlc", {})
        if isinstance(dlc, dict) and isinstance(dlc.get("binary"), str):
            candidates.append(str(dlc["binary"]))
    except Exception:
        pass

    candidates.append(DEFAULT_DLC_BINARY)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)

    raise HTTPException(
        status_code=500,
        detail="DLC binary not found. Set LMMS_EVAL_DLC_BINARY or fix run_scripts/config_dlc.json.",
    )


def _run_dlc_command(args: list[str], *, timeout: int = 30) -> str:
    binary = _resolve_dlc_binary()
    command = [binary, *args, "--region", DLC_REGION, "--endpoint", DLC_ENDPOINT]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"DLC command timed out: {' '.join(command)}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise HTTPException(status_code=502, detail=f"DLC command failed: {detail}")
    return completed.stdout


def _run_authenticated_dlc_command(args: list[str], auth_user: dict[str, Any], *, timeout: int = 30) -> str:
    binary = _resolve_dlc_binary()
    credential_args = _build_dlc_credential_args(auth_user, mask_secrets=False)
    redacted_credential_args = _build_dlc_credential_args(auth_user, mask_secrets=True)
    command = [binary, *args, *credential_args, "--region", DLC_REGION, "--endpoint", DLC_ENDPOINT]
    redacted_command = [
        binary,
        *args,
        *redacted_credential_args,
        "--region",
        DLC_REGION,
        "--endpoint",
        DLC_ENDPOINT,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"DLC command timed out: {' '.join(redacted_command)}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        access_key = str(auth_user.get("access_key_id") or "")
        secret_access_key = str(auth_user.get("secret_access_key") or "")
        if access_key:
            detail = detail.replace(access_key, MASKED_SECRET)
        if secret_access_key:
            detail = detail.replace(secret_access_key, MASKED_SECRET)
        raise HTTPException(status_code=502, detail=f"DLC command failed: {detail}")
    return completed.stdout


def _run_paistudio_command(args: list[str], *, timeout: int = 30) -> str:
    command = ["aliyun", PAISTUDIO_PRODUCT, *args, "--region", DLC_REGION]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="aliyun CLI not found; cannot query PAI quota") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"PaiStudio quota command timed out: {' '.join(command)}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise HTTPException(status_code=502, detail=f"PaiStudio quota command failed: {detail}")
    return completed.stdout


def _run_aiworkspace_command(args: list[str], *, timeout: int = 30) -> str:
    command = ["aliyun", AIWORKSPACE_PRODUCT, *args, "--region", DLC_REGION]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="aliyun CLI not found; cannot query PAI workspace members") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"AIWorkspace command timed out: {' '.join(command)}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise HTTPException(status_code=502, detail=f"AIWorkspace command failed: {detail}")
    return completed.stdout


def _parse_dlc_table(output: str) -> list[dict[str, str]]:
    headers: list[str] | None = None
    rows: list[dict[str, str]] = []

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        values = [part.strip() for part in stripped.strip("|").split("|")]
        if headers is None:
            headers = values
            continue
        if len(values) != len(headers):
            continue
        row = {headers[idx]: values[idx] for idx in range(len(headers))}
        if rows and "JobId" in row and not row.get("JobId") and any(row.values()):
            for key, value in row.items():
                if value:
                    rows[-1][key] = f"{rows[-1].get(key, '')}{value}"
            continue
        rows.append(row)

    return rows


def _extract_dlc_user_name(row: dict[str, str]) -> str:
    for key in ("UserName", "Username", "User", "OwnerName", "Owner", "Creator", "CreatedBy", "Submitter"):
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _normalize_dlc_job_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "job_id": row.get("JobId", ""),
        "name": row.get("Name", ""),
        "status": row.get("JobStatus", ""),
        "workspace_id": row.get("WorkspaceId", ""),
        "resource_id": row.get("ResourceId", ""),
        "resource_name": row.get("ResourceName", ""),
        "job_type": row.get("JobType", ""),
        "priority": row.get("Priority", ""),
        "user_name": _extract_dlc_user_name(row),
        "user_id": row.get("UserId", ""),
        "create_time": row.get("CreateTime", ""),
        "submitted_time": row.get("SubmittedTime", ""),
        "running_time": row.get("RunningTime", ""),
        "finish_time": row.get("FinishTime", ""),
        "duration_seconds": row.get("Duration(seconds)", ""),
    }


def _first_nonempty_text(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _normalize_dlc_status(status: Any) -> str:
    return str(status or "").strip().lower()


def _dlc_job_display_name(job: dict[str, Any]) -> str:
    return _first_nonempty_text(job, "DisplayName", "Name", "name", "display_name")


def _dlc_job_status(job: dict[str, Any]) -> str:
    return _first_nonempty_text(job, "Status", "JobStatus", "status", "job_status")


def _dlc_job_owner_user_id(job: dict[str, Any]) -> str:
    return _first_nonempty_text(job, "UserId", "user_id", "OwnerId", "owner_id", "CreatorId", "creator_id")


def _is_dlc_job_killable_status(status: Any) -> bool:
    return _normalize_dlc_status(status) in DLC_KILLABLE_STATUSES


def _dlc_job_kill_permission(job: dict[str, Any], auth_user: dict[str, Any]) -> tuple[bool, str]:
    name = _dlc_job_display_name(job)
    if not _is_view_log_job_name(name):
        return False, f"DLC job name must start with {_view_log_job_prefix_label()}"

    status = _dlc_job_status(job)
    if not _is_dlc_job_killable_status(status):
        return False, f"DLC job status is not killable: {status or 'unknown'}"

    if auth_user.get("role") == AUTH_ADMIN_ROLE:
        return True, ""

    current_user_id = str(auth_user.get("aliyun_user_id") or "").strip()
    if not current_user_id:
        return False, "Current Alibaba Cloud user id is unavailable; only admins can kill this job"

    owner_user_id = _dlc_job_owner_user_id(job)
    if not owner_user_id:
        return False, "DLC job owner user id is unavailable"

    if current_user_id != owner_user_id:
        return False, "Only the job owner or an admin can kill this DLC job"

    return True, ""


def _assert_dlc_job_kill_allowed(job: dict[str, Any], auth_user: dict[str, Any]) -> None:
    can_kill, reason = _dlc_job_kill_permission(job, auth_user)
    if can_kill:
        return
    if not _is_view_log_job_name(_dlc_job_display_name(job)):
        raise HTTPException(status_code=400, detail=reason)
    if not _is_dlc_job_killable_status(_dlc_job_status(job)):
        raise HTTPException(status_code=409, detail=reason)
    raise HTTPException(status_code=403, detail=reason)


def _with_dlc_job_kill_permission(job: dict[str, Any], auth_user: dict[str, Any]) -> dict[str, Any]:
    annotated = copy.deepcopy(job)
    can_kill, reason = _dlc_job_kill_permission(annotated, auth_user)
    annotated["can_kill"] = can_kill
    annotated["kill_disabled_reason"] = reason
    return annotated


def _is_eval_job_name(name: Any) -> bool:
    return isinstance(name, str) and name.startswith(EVAL_JOB_NAME_PREFIX)


def _is_judge_job_name(name: Any) -> bool:
    return isinstance(name, str) and name.startswith(JUDGE_JOB_NAME_PREFIX)


def _is_view_log_job_name(name: Any) -> bool:
    return _is_eval_job_name(name) or _is_judge_job_name(name)


def _view_log_job_prefix_label() -> str:
    return ",".join(VIEW_LOG_JOB_NAME_PREFIXES)


def _job_stage_from_name(name: Any) -> str:
    if _is_judge_job_name(name):
        return "judge"
    if _is_eval_job_name(name):
        return "eval"
    return ""


def _split_display_name_filters(display_name: str) -> list[str]:
    filters = [item.strip() for item in display_name.split(",") if item.strip()]
    return filters or list(VIEW_LOG_JOB_NAME_PREFIXES)


def _clear_dlc_runtime_caches() -> None:
    global _dlc_pool_usage_cache
    _dlc_jobs_cache.clear()
    _dlc_pool_usage_cache = None


def _default_dlc_resource_id() -> str:
    _default_dlc_config()
    return DEFAULT_DLC_RESOURCE_ID


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise HTTPException(status_code=500, detail=f"{name} must be a positive integer")
    return int(raw)


def _dlc_pool_capacity() -> tuple[int, int, str, str]:
    total_gpu = _positive_int_env("LMMS_EVAL_DLC_POOL_TOTAL_GPU")
    gpu_source = "env:LMMS_EVAL_DLC_POOL_TOTAL_GPU"
    if total_gpu is None:
        total_gpu = DEFAULT_DLC_POOL_TOTAL_GPU
        gpu_source = "default:256gpu"

    total_cpu = _positive_int_env("LMMS_EVAL_DLC_POOL_TOTAL_CPU")
    cpu_source = "env:LMMS_EVAL_DLC_POOL_TOTAL_CPU"
    if total_cpu is None:
        gpu_per_node = _positive_int_env("LMMS_EVAL_DLC_POOL_GPU_PER_NODE") or DEFAULT_DLC_POOL_GPU_PER_NODE
        cpu_per_node = _positive_int_env("LMMS_EVAL_DLC_POOL_CPU_PER_NODE") or DEFAULT_DLC_POOL_CPU_PER_NODE
        if total_gpu % gpu_per_node != 0:
            raise HTTPException(
                status_code=500,
                detail="LMMS_EVAL_DLC_POOL_TOTAL_GPU must be divisible by LMMS_EVAL_DLC_POOL_GPU_PER_NODE",
            )
        total_cpu = (total_gpu // gpu_per_node) * cpu_per_node
        cpu_source = f"default:{total_gpu // gpu_per_node}nodes_x_{cpu_per_node}cpu"
    return total_gpu, total_cpu, gpu_source, cpu_source


def _parse_resource_int(value: Any, *, field: str, job_id: str) -> int:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} for {job_id} must be non-negative")
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field} for {job_id} must be an integer, got {value!r}")


def _parse_quota_int(value: Any, *, field: str) -> int:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field} must be an integer, got {value!r}")


def _load_paistudio_json(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    output = _run_paistudio_command(args, timeout=timeout)
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="PaiStudio quota output is not valid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="PaiStudio quota output must be a JSON object")
    return data


def _load_aiworkspace_json(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    output = _run_aiworkspace_command(args, timeout=timeout)
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="AIWorkspace output is not valid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="AIWorkspace output must be a JSON object")
    return data


def _quota_total(quota_details: dict[str, Any], *, resource: str) -> int:
    total_quota = quota_details.get("AllocatableQuota") or quota_details.get("SchedulableQuota")
    if not isinstance(total_quota, dict):
        raise ValueError("QuotaDetails.AllocatableQuota missing")
    return _parse_quota_int(total_quota.get(resource), field=f"QuotaDetails.AllocatableQuota.{resource}")


def _quota_workload_rows(quota_id: str, *, page_size: int = 100, max_pages: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_number in range(1, max_pages + 1):
        data = _load_paistudio_json(
            [
                "ListQuotaWorkloads",
                "--QuotaId",
                quota_id,
                "--WorkspaceIds",
                DEFAULT_DLC_WORKSPACE_ID,
                "--ShowOwn",
                "true",
                "--PageSize",
                str(page_size),
                "--PageNumber",
                str(page_number),
            ],
            timeout=30,
        )
        workloads = data.get("Workloads")
        if not isinstance(workloads, list):
            raise HTTPException(status_code=502, detail="PaiStudio ListQuotaWorkloads missing Workloads")
        if not workloads:
            break
        for workload in workloads:
            if not isinstance(workload, dict):
                raise HTTPException(status_code=502, detail="PaiStudio workload item must be an object")
            rows.append(workload)
        if len(workloads) < page_size:
            break
    return rows


def _paistudio_user_name_map() -> dict[str, str]:
    global _paistudio_user_name_cache
    now = time.time()
    if _paistudio_user_name_cache and now - _paistudio_user_name_cache[0] < PAISTUDIO_USER_USAGE_CACHE_TTL_SECONDS:
        return copy.deepcopy(_paistudio_user_name_cache[1])

    quota_id = _default_dlc_resource_id()
    page_size = 100
    user_names: dict[str, str] = {}
    for page_number in range(1, 21):
        data = _load_paistudio_json(
            [
                "ListQuotaActiveUserUsages",
                "--QuotaId",
                quota_id,
                "--WorkspaceId",
                DEFAULT_DLC_WORKSPACE_ID,
                "--PageSize",
                str(page_size),
                "--PageNumber",
                str(page_number),
            ],
            timeout=30,
        )
        usages = data.get("QuotaUserUsages")
        if not isinstance(usages, list):
            raise HTTPException(status_code=502, detail="PaiStudio ListQuotaActiveUserUsages missing QuotaUserUsages")
        if not usages:
            break
        for usage in usages:
            if not isinstance(usage, dict):
                raise HTTPException(status_code=502, detail="PaiStudio user usage item must be an object")
            user_id = str(usage.get("UserId") or "").strip()
            username = str(usage.get("Username") or "").strip()
            if user_id and username:
                user_names[user_id] = username
        if len(usages) < page_size:
            break

    _paistudio_user_name_cache = (now, copy.deepcopy(user_names))
    return user_names


def _aiworkspace_member_name_map() -> dict[str, str]:
    global _aiworkspace_member_name_cache
    now = time.time()
    if _aiworkspace_member_name_cache and now - _aiworkspace_member_name_cache[0] < AIWORKSPACE_MEMBER_CACHE_TTL_SECONDS:
        return copy.deepcopy(_aiworkspace_member_name_cache[1])

    page_size = 100
    user_names: dict[str, str] = {}
    for page_number in range(1, 21):
        data = _load_aiworkspace_json(
            [
                "ListMembers",
                "--WorkspaceId",
                DEFAULT_DLC_WORKSPACE_ID,
                "--PageSize",
                str(page_size),
                "--PageNumber",
                str(page_number),
            ],
            timeout=30,
        )
        members = data.get("Members")
        if not isinstance(members, list):
            raise HTTPException(status_code=502, detail="AIWorkspace ListMembers missing Members")
        if not members:
            break
        for member in members:
            if not isinstance(member, dict):
                raise HTTPException(status_code=502, detail="AIWorkspace member item must be an object")
            user_id = str(member.get("UserId") or "").strip()
            username = str(member.get("MemberName") or member.get("DisplayName") or member.get("AccountName") or "").strip()
            if user_id and username:
                user_names[user_id] = username
        if len(members) < page_size:
            break

    _aiworkspace_member_name_cache = (now, copy.deepcopy(user_names))
    return user_names


def _workload_resource_int(workload: dict[str, Any], key: str, *, scheduled: bool = False) -> int:
    resource_key = "ScheduledResource" if scheduled else "Resource"
    resource = workload.get(resource_key)
    if not isinstance(resource, dict):
        return 0
    value = resource.get(key)
    if value in (None, ""):
        return 0
    return _parse_quota_int(value, field=f"Workload.{resource_key}.{key}")


def _build_dlc_pool_usage_from_quota() -> dict[str, Any]:
    quota_id = _default_dlc_resource_id()
    data = _load_paistudio_json(["GetQuota", "--QuotaId", quota_id], timeout=30)
    quota_details = data.get("QuotaDetails")
    if not isinstance(quota_details, dict):
        raise HTTPException(status_code=502, detail="PaiStudio GetQuota missing QuotaDetails")

    total_gpu = _quota_total(quota_details, resource="GPU")
    total_cpu = _quota_total(quota_details, resource="CPU")
    workloads = _quota_workload_rows(quota_id)
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    used_gpu = 0
    used_cpu = 0
    for workload in workloads:
        try:
            gpu = _workload_resource_int(workload, "GPU", scheduled=True)
            cpu = _workload_resource_int(workload, "CPU", scheduled=True)
        except Exception as exc:
            errors.append(f"{workload.get('WorkloadId') or workload.get('Name')}: {exc}")
            gpu = 0
            cpu = 0
        used_gpu += gpu
        used_cpu += cpu
        jobs.append(
            {
                "job_id": str(workload.get("WorkloadId") or workload.get("Name") or ""),
                "name": str(workload.get("WorkloadName") or workload.get("Name") or ""),
                "status": str(workload.get("WorkloadStatus") or workload.get("Phase") or workload.get("Status") or ""),
                "workspace_id": str(workload.get("WorkspaceId") or ""),
                "resource_id": str(workload.get("QuotaId") or quota_id),
                "gpu": gpu,
                "cpu": cpu,
                "pod_count": 1,
            }
        )

    def metric(used: int, total: int, source: str) -> dict[str, Any]:
        percent = round((used / total) * 100, 1) if total > 0 else 0.0
        return {
            "used": used,
            "total": total,
            "percent": percent,
            "capacity_source": source,
        }

    return {
        "workspace_id": DEFAULT_DLC_WORKSPACE_ID,
        "resource_id": quota_id,
        "resource_name": str(data.get("QuotaName") or ""),
        "active_statuses": [
            f"quota={quota_id}",
            f"workspace={DEFAULT_DLC_WORKSPACE_ID}",
            "workload_type=all",
            "usage=ScheduledResource",
        ],
        "gpu": metric(used_gpu, total_gpu, "paistudio:ListQuotaWorkloads.ScheduledResource/GetQuota:AllocatableQuota"),
        "cpu": metric(used_cpu, total_cpu, "paistudio:ListQuotaWorkloads.ScheduledResource/GetQuota:AllocatableQuota"),
        "jobs": jobs,
        "errors": errors,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": (
            f"aliyun paistudio GetQuota/ListQuotaWorkloads "
            f"quota={quota_id} workspace={DEFAULT_DLC_WORKSPACE_ID} workload_type=all region={DLC_REGION}"
        ),
    }


def _job_requested_resources(detail: dict[str, Any], *, job_id: str) -> tuple[int, int, int]:
    specs = detail.get("JobSpecs")
    if not isinstance(specs, list):
        raise ValueError(f"JobSpecs missing for {job_id}")

    total_gpu = 0
    total_cpu = 0
    total_pods = 0
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"JobSpecs item for {job_id} must be an object")
        pod_count = _parse_resource_int(spec.get("PodCount", 0), field="PodCount", job_id=job_id)
        resource_config = spec.get("ResourceConfig")
        if not isinstance(resource_config, dict):
            raise ValueError(f"ResourceConfig missing for {job_id}")
        gpu = _parse_resource_int(resource_config.get("GPU", 0), field="GPU", job_id=job_id)
        cpu = _parse_resource_int(resource_config.get("CPU", 0), field="CPU", job_id=job_id)
        total_gpu += pod_count * gpu
        total_cpu += pod_count * cpu
        total_pods += pod_count
    return total_gpu, total_cpu, total_pods


def _list_dlc_pool_job_rows(
    *,
    page_size: int,
    max_pages: int,
    statuses: tuple[str, ...],
    resource_id: str,
) -> list[dict[str, Any]]:
    page_size = max(1, min(page_size, 100))
    max_pages = max(1, min(max_pages, 20))
    rows: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()

    for status in statuses:
        for page_num in range(1, max_pages + 1):
            args = [
                "get",
                "job",
                "--workspace_id",
                DEFAULT_DLC_WORKSPACE_ID,
                "--resource_id",
                resource_id,
                "--status",
                status,
                "--page_size",
                str(page_size),
                "--page_num",
                str(page_num),
                "--show_detail",
            ]
            output = _run_dlc_command(args, timeout=30)
            page_rows = [_normalize_dlc_job_row(row) for row in _parse_dlc_table(output)]
            valid_page_rows = [row for row in page_rows if str(row.get("job_id") or "")]
            if not valid_page_rows:
                break
            for row in valid_page_rows:
                job_id = str(row.get("job_id") or "")
                if job_id in seen_job_ids:
                    continue
                seen_job_ids.add(job_id)
                rows.append(row)
            if len(valid_page_rows) < page_size:
                break
    return rows


def _build_dlc_pool_usage() -> dict[str, Any]:
    global _dlc_pool_usage_cache
    now = time.time()
    if _dlc_pool_usage_cache and now - _dlc_pool_usage_cache[0] < DLC_POOL_USAGE_CACHE_TTL_SECONDS:
        return copy.deepcopy(_dlc_pool_usage_cache[1])

    result = _build_dlc_pool_usage_from_quota()
    _dlc_pool_usage_cache = (now, copy.deepcopy(result))
    return result


def _build_dlc_pool_usage_from_active_jobs() -> dict[str, Any]:
    resource_id = _default_dlc_resource_id()
    total_gpu_capacity, total_cpu_capacity, gpu_capacity_source, cpu_capacity_source = _dlc_pool_capacity()
    rows = _list_dlc_pool_job_rows(
        page_size=100,
        max_pages=3,
        statuses=DLC_POOL_ACTIVE_STATUSES,
        resource_id=resource_id,
    )

    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    used_gpu = 0
    used_cpu = 0
    resource_name = ""
    for row in rows:
        job_id = str(row.get("job_id") or "")
        try:
            detail = _get_dlc_job_detail(job_id)
            gpu, cpu, pod_count = _job_requested_resources(detail, job_id=job_id)
        except Exception as exc:
            errors.append(f"{job_id}: {exc}")
            gpu = 0
            cpu = 0
            pod_count = 0
        used_gpu += gpu
        used_cpu += cpu
        if not resource_name:
            resource_name = str(row.get("resource_name") or "")
        jobs.append(
            {
                "job_id": job_id,
                "name": str(row.get("name") or ""),
                "status": str(row.get("status") or ""),
                "workspace_id": str(row.get("workspace_id") or ""),
                "resource_id": str(row.get("resource_id") or resource_id),
                "gpu": gpu,
                "cpu": cpu,
                "pod_count": pod_count,
            }
        )

    if gpu_capacity_source.startswith("default:") and used_gpu > total_gpu_capacity:
        total_gpu_capacity = used_gpu
        gpu_capacity_source = "lower-bound:active_jobs"
    if cpu_capacity_source.startswith("default:") and used_cpu > total_cpu_capacity:
        total_cpu_capacity = used_cpu
        cpu_capacity_source = "lower-bound:active_jobs"

    def metric(used: int, total: int, source: str) -> dict[str, Any]:
        percent = round((used / total) * 100, 1) if total > 0 else 0.0
        return {
            "used": used,
            "total": total,
            "percent": percent,
            "capacity_source": source,
        }

    result = {
        "workspace_id": DEFAULT_DLC_WORKSPACE_ID,
        "resource_id": resource_id,
        "resource_name": resource_name,
        "active_statuses": list(DLC_POOL_ACTIVE_STATUSES),
        "gpu": metric(used_gpu, total_gpu_capacity, gpu_capacity_source),
        "cpu": metric(used_cpu, total_cpu_capacity, cpu_capacity_source),
        "jobs": jobs,
        "errors": errors,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": f"{_resolve_dlc_binary()} workspace={DEFAULT_DLC_WORKSPACE_ID} resource={resource_id}",
    }
    _dlc_pool_usage_cache = (now, copy.deepcopy(result))
    return result


def _list_dlc_jobs_from_cli(
    *,
    page_size: int,
    max_pages: int,
    status: str,
    display_name: str,
) -> list[dict[str, Any]]:
    page_size = max(1, min(page_size, 100))
    max_pages = max(1, min(max_pages, 20))
    display_name_filters = _split_display_name_filters(display_name)
    cache_key = (page_size, max_pages, status, tuple(display_name_filters))
    cached = _dlc_jobs_cache.get(cache_key)
    now = time.time()
    if cached and now - cached[0] < DLC_JOBS_CACHE_TTL_SECONDS:
        return copy.deepcopy(cached[1])

    rows: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    for display_filter in display_name_filters:
        for page_num in range(1, max_pages + 1):
            args = [
                "get",
                "job",
                "--workspace_id",
                DEFAULT_DLC_WORKSPACE_ID,
                "--page_size",
                str(page_size),
                "--page_num",
                str(page_num),
                "--show_detail",
            ]
            if status:
                args.extend(["--status", status])
            if display_filter:
                args.extend(["--display_name", display_filter])
            output = _run_dlc_command(args, timeout=30)
            page_rows = [_normalize_dlc_job_row(row) for row in _parse_dlc_table(output)]
            if not page_rows:
                break
            for row in page_rows:
                job_id = str(row.get("job_id") or "")
                if not job_id or job_id in seen_job_ids:
                    continue
                if not _is_view_log_job_name(row.get("name")):
                    continue
                seen_job_ids.add(job_id)
                rows.append(row)
            if len(page_rows) < page_size:
                break

    user_names = _paistudio_user_name_map()
    user_names.update(_aiworkspace_member_name_map())
    for row in rows:
        user_id = str(row.get("user_id") or "")
        if user_id in user_names:
            row["user_name"] = user_names[user_id]
        row["job_stage"] = _job_stage_from_name(row.get("name"))
        row["lmms_tasks"] = []
        row["llm_judge_tasks"] = []
        row["requires_llm_judge"] = False
        row["result_root"] = None
        row["has_results"] = False
    _dlc_jobs_cache[cache_key] = (now, copy.deepcopy(rows))
    return rows


def _get_dlc_job_detail(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"dlc[a-z0-9]+", job_id):
        raise HTTPException(status_code=400, detail="Invalid DLC job id")
    output = _run_dlc_command(
        ["get", "job", job_id, "--workspace_id", DEFAULT_DLC_WORKSPACE_ID, "--show_detail"],
        timeout=30,
    )
    try:
        detail = json.loads(output)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="DLC detail output is not valid JSON") from exc
    if not isinstance(detail, dict):
        raise HTTPException(status_code=502, detail="DLC detail output must be a JSON object")
    return detail


def _extract_command_path(command: str, suffix: str) -> str | None:
    pattern = rf"(/[^\s;'\"|]+{re.escape(suffix)})"
    match = re.search(pattern, command)
    if not match:
        return None
    return match.group(1)


def _extract_cli_option_value(command: str, option: str) -> str | None:
    escaped = re.escape(option)
    match = re.search(rf"(?:^|\s){escaped}(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s;'\"]+)", command)
    if not match:
        return None
    return match.group(1).strip("'\"")


def _extract_log_dir(command: str) -> str | None:
    match = re.search(r"(?:export\s+)?LMMS_EVAL_LOG_DIR=([^;\s]+)", command)
    if not match:
        return None
    return match.group(1).strip("'\"")


def _runtime_config_path_from_detail(detail: dict[str, Any]) -> Path | None:
    command = detail.get("UserCommand")
    if not isinstance(command, str) or not command:
        return None
    command = str(_replace_default_user(command))
    runtime_path = _extract_command_path(command, "runtime_config.json")
    if not runtime_path:
        return None
    path = Path(runtime_path).expanduser()
    return path if path.is_file() else None


def _load_json_path(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _tasks_from_eval_config_data(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    eval_config = data.get("eval")
    if not isinstance(eval_config, dict):
        return []
    tasks = eval_config.get("tasks")
    if tasks is None:
        return []
    try:
        return _split_tasks(tasks)
    except ValueError:
        return []


def _tasks_from_config_path(path: Path) -> list[str]:
    return _tasks_from_eval_config_data(_load_json_path(path))


def _task_name_from_sample_jsonl(path: Path) -> str:
    stem = path.stem
    marker = "_samples_"
    if marker not in stem:
        return ""
    return stem.split(marker, 1)[1].strip()


def _tasks_from_result_root(result_root: Path | None) -> list[str]:
    if result_root is None or not result_root.exists() or not result_root.is_dir():
        return []

    tasks: list[str] = []
    for config_name in ("config.json", "runtime_config.json", "judge_runtime_config.json"):
        config_path = result_root / config_name
        if config_path.is_file():
            tasks.extend(_tasks_from_config_path(config_path))

    summary_jsonl = result_root / "summary.jsonl"
    if summary_jsonl.is_file():
        for item in _load_summary_jsonl(summary_jsonl):
            for key in ("lmms_tasks", "task", "display_name"):
                value = item.get(key)
                if isinstance(value, str):
                    tasks.extend(_split_tasks(value))

    for result_json in sorted(result_root.rglob("*_results.json")):
        data = _load_json_path(result_json)
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            tasks.extend(str(task) for task in data["results"].keys())

    for sample_jsonl in sorted(result_root.rglob("*_samples_*.jsonl")):
        task_name = _task_name_from_sample_jsonl(sample_jsonl)
        if task_name:
            tasks.append(task_name)

    return _dedupe_strings(tasks)


def _result_root_from_runtime_config(runtime_config_path: Path | None) -> Path | None:
    if runtime_config_path is None:
        return None
    data = _load_json_path(runtime_config_path)
    if not isinstance(data, dict):
        return None
    eval_config = data.get("eval")
    if not isinstance(eval_config, dict):
        return None
    output_path = eval_config.get("output_path")
    timestamp = eval_config.get("timestamp")
    if not isinstance(output_path, str) or not output_path:
        return None
    output_path = str(_replace_default_user(output_path))
    root = Path(output_path).expanduser()
    if isinstance(timestamp, str) and timestamp and timestamp != "null":
        root = root / timestamp
    return root if root.exists() and root.is_dir() else None


def _result_root_from_command_output_path(detail: dict[str, Any]) -> Path | None:
    command = detail.get("UserCommand")
    if not isinstance(command, str) or not command:
        return None
    command = str(_replace_default_user(command))
    output_path = _extract_cli_option_value(command, "--output_path")
    if not output_path:
        return None
    root = Path(output_path).expanduser()
    return root if root.exists() and root.is_dir() else None


def _safe_read_text(path: Path, *, max_bytes: int = 200000) -> str:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _load_summary_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append({str(key): _parse_json_maybe(value) for key, value in item.items()})
    except OSError:
        return []
    return rows


def _has_metric_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict)):
        return len(value) > 0
    return True


def _primary_metrics(result_data: dict[str, Any]) -> list[tuple[str, str, Any, Any | None]]:
    results = result_data.get("results")
    if not isinstance(results, dict):
        return []

    metrics: list[tuple[str, str, Any, Any | None]] = []
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, metric_value in task_metrics.items():
            if metric_name == "alias" or "stderr" in metric_name:
                continue
            if not _has_metric_value(metric_value):
                continue
            stderr_key = metric_name.replace(",none", "_stderr,none") if metric_name.endswith(",none") else f"{metric_name}_stderr"
            metrics.append((str(task_name), str(metric_name).replace(",none", ""), metric_value, task_metrics.get(stderr_key)))
            break
    return metrics


def _candidate_sample_jsonls(result_json: str | None, value_source: str = "", task_name: str = "") -> list[str]:
    paths: list[str] = []
    if value_source:
        for match in re.finditer(r"(/[^\s;:]+\.jsonl)", value_source):
            paths.append(match.group(1))
    if result_json:
        result_path = Path(result_json)
        if result_path.is_file():
            stem = result_path.stem
            if stem.endswith("_results"):
                prefix = stem.removesuffix("_results")
                sample_glob = f"{prefix}_samples_{task_name}.jsonl" if task_name else f"{prefix}_samples_*.jsonl"
                direct = sorted(result_path.parent.glob(sample_glob))
                paths.extend(str(path) for path in direct)
            if not task_name:
                paths.extend(str(path) for path in sorted(result_path.parent.glob("*_samples_*.jsonl")))

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _metric_value_text(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if not float(value).is_integer() else str(int(value))
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _result_json_metric_rows(
    *,
    idx_start: int,
    result_json: Path,
) -> list[DlcMetricRow]:
    result_data = _load_json_path(result_json)
    if not isinstance(result_data, dict):
        return []
    primary_metrics = _primary_metrics(result_data)
    if not primary_metrics:
        return []
    n_samples = result_data.get("n-samples")
    rows: list[DlcMetricRow] = []
    for offset, (task_name, metric_name, value, stderr) in enumerate(primary_metrics):
        task_n_samples = n_samples.get(task_name) if isinstance(n_samples, dict) else n_samples
        rows.append(
            DlcMetricRow(
                metric_id=str(idx_start + offset),
                display_name=task_name,
                lmms_tasks=task_name,
                status="success",
                value=value,
                value_text=_metric_value_text(value),
                metric_name=metric_name,
                stderr=stderr,
                total_evaluation_time_seconds=result_data.get("total_evaluation_time_seconds"),
                n_samples=task_n_samples,
                result_json=str(result_json),
                sample_jsonls=_candidate_sample_jsonls(str(result_json), task_name=task_name),
            )
        )
    return rows


def _sample_has_judge_metrics(item: dict[str, Any]) -> bool:
    if item.get("judge_mode"):
        return True
    metrics = item.get("metrics")
    if not isinstance(metrics, dict):
        return False
    judge_keys = {"llm_judge_score", "llm_judge_success", "llm_judge_failed", "llm_judge_skipped"}
    if any(key in metrics for key in judge_keys):
        return True
    for value in metrics.values():
        if isinstance(value, dict) and any(key in value for key in judge_keys):
            return True
    return False


def _append_numeric_metric(target: dict[str, list[float]], key: str, value: Any) -> None:
    if isinstance(value, bool):
        target.setdefault(key, []).append(float(value))
    elif isinstance(value, (int, float)):
        target.setdefault(key, []).append(float(value))


def _aggregate_judged_sample_jsonl(path: Path) -> tuple[dict[str, Any], int]:
    metric_values: dict[str, list[float]] = {}
    total = 0
    judged_rows = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                total += 1
                if not _sample_has_judge_metrics(item):
                    continue
                judged_rows += 1
                metrics = item.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                for key, value in metrics.items():
                    _append_numeric_metric(metric_values, str(key), value)
                    if isinstance(value, dict):
                        for nested_key in ("score", "hit", "correct", "accuracy", "llm_judge_score", "llm_judge_success", "llm_judge_failed"):
                            if nested_key in value:
                                _append_numeric_metric(metric_values, f"{key}.{nested_key}", value[nested_key])
    except OSError:
        return {}, 0

    if judged_rows == 0:
        return {}, total
    summary = {key: round(sum(values) / len(values), 4) for key, values in metric_values.items() if values}
    return summary, total


def _preferred_judged_sample_metric(summary: dict[str, Any]) -> str:
    for key in (
        "total_acc",
        "accuracy",
        "exact_match",
        "ocrbench_accuracy.score",
        "llm_judge_score",
        "llm_judge_success",
    ):
        if key in summary:
            return key
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            return key
    return ""


def _judged_sample_metric_rows(*, idx_start: int, result_root: Path) -> list[DlcMetricRow]:
    rows: list[DlcMetricRow] = []
    for sample_jsonl in sorted(result_root.rglob("*_samples_*.jsonl")):
        summary, n_samples = _aggregate_judged_sample_jsonl(sample_jsonl)
        metric_name = _preferred_judged_sample_metric(summary)
        if not metric_name:
            continue
        task_name = _task_name_from_sample_jsonl(sample_jsonl) or sample_jsonl.stem
        value = summary[metric_name]
        rows.append(
            DlcMetricRow(
                metric_id=str(idx_start + len(rows)),
                display_name=task_name,
                lmms_tasks=task_name,
                status="success",
                value=value,
                value_text=_metric_value_text(value),
                metric_name=metric_name,
                n_samples=n_samples,
                sample_jsonls=[str(sample_jsonl)],
                value_source="judged_samples",
            )
        )
    return rows


def _build_metric_rows(result_root: Path) -> tuple[list[DlcMetricRow], list[str]]:
    summary_files: list[str] = []
    rows: list[DlcMetricRow] = []
    resolved_root = result_root.resolve()

    summary_jsonl = result_root / "summary.jsonl"
    if summary_jsonl.is_file():
        summary_files.append(str(summary_jsonl))
        for item in _load_summary_jsonl(summary_jsonl):
            result_json_raw = item.get("result_json")
            result_json = Path(str(result_json_raw)) if isinstance(result_json_raw, str) and result_json_raw else None
            if result_json is None or not result_json.is_file():
                continue
            try:
                result_json.resolve().relative_to(resolved_root)
            except ValueError:
                continue
            result_rows = _result_json_metric_rows(idx_start=len(rows), result_json=result_json)
            for row in result_rows:
                row.started_at = str(item.get("started_at") or "")
                row.ended_at = str(item.get("ended_at") or "")
                row.wall_seconds = item.get("wall_seconds")
                row.total_evaluation_time_seconds = item.get("total_evaluation_time_seconds") or row.total_evaluation_time_seconds
                row.n_samples = item.get("n_samples") or row.n_samples
            rows.extend(result_rows)

    if not rows:
        result_jsons = sorted(result_root.rglob("*_results.json"))
        for result_json in result_jsons:
            rows.extend(_result_json_metric_rows(idx_start=len(rows), result_json=result_json))

    existing_sample_jsonls = {
        str(Path(sample_jsonl).resolve())
        for row in rows
        for sample_jsonl in row.sample_jsonls
    }
    for judged_row in _judged_sample_metric_rows(idx_start=len(rows), result_root=result_root):
        judged_paths = {str(Path(sample_jsonl).resolve()) for sample_jsonl in judged_row.sample_jsonls}
        if judged_paths & existing_sample_jsonls:
            continue
        judged_row.metric_id = str(len(rows))
        rows.append(judged_row)
        existing_sample_jsonls.update(judged_paths)

    summary_json = result_root / "summary.json"
    if summary_json.is_file():
        summary_files.append(str(summary_json))

    return rows, summary_files


def _stringify_sample_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _is_svg_image_header(header: bytes) -> bool:
    stripped = header.lstrip().lower()
    return stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in stripped)


def _image_file_media_type(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(str(path))
    if not guessed or not guessed.startswith("image/"):
        return None
    try:
        with path.open("rb") as f:
            header = f.read(512)
    except OSError:
        return None

    if guessed == "image/svg+xml":
        return guessed if _is_svg_image_header(header) else None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return guessed
    if header.startswith(b"\xff\xd8\xff"):
        return guessed
    if header.startswith((b"GIF87a", b"GIF89a")):
        return guessed
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return guessed
    if header.startswith(b"BM"):
        return guessed
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return guessed
    return None


def _iter_sample_strings(value: Any, depth: int = 0):
    if depth > 8:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_sample_strings(nested, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_sample_strings(nested, depth + 1)


def _data_image_media_type(value: str) -> str | None:
    match = re.match(r"^data:(image/[A-Za-z0-9.+-]+)[;,]", value, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _url_image_media_type(value: str) -> str | None:
    if not value.startswith(("http://", "https://")):
        return None
    without_fragment = value.split("#", 1)[0]
    without_query = without_fragment.split("?", 1)[0]
    guessed, _ = mimetypes.guess_type(without_query)
    return guessed if guessed and guessed.startswith("image/") else None


def _looks_like_local_image_reference(value: str) -> bool:
    if len(value) > 1024 or re.search(r"\s", value):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith("file://"):
        stripped = stripped.removeprefix("file://")
    guessed, _ = mimetypes.guess_type(unquote(stripped))
    return bool(guessed and guessed.startswith("image/"))


def _local_image_path(value: str, sample_dir: Path) -> tuple[Path, str] | None:
    raw_path = unquote(value.removeprefix("file://")) if value.startswith("file://") else value
    try:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = sample_dir / path
        path = path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    media_type = _image_file_media_type(path)
    if media_type is None:
        return None
    return path, media_type


def _register_dlc_sample_media_path(path: Path) -> str:
    if len(DLC_SAMPLE_MEDIA_TOKENS) >= DLC_SAMPLE_MEDIA_MAX_TOKENS:
        DLC_SAMPLE_MEDIA_TOKENS.clear()
    token = secrets.token_urlsafe(18)
    DLC_SAMPLE_MEDIA_TOKENS[token] = path
    return token


def _dlc_sample_media_url(job_id: str, metric_id: str, token: str) -> str:
    return (
        f"/dlc/jobs/{quote(job_id, safe='')}"
        f"/metrics/{quote(metric_id, safe='')}"
        f"/samples/media/{quote(token, safe='')}"
    )


def _extract_dlc_sample_media(
    item: dict[str, Any], *, sample_file: Path, job_id: str, metric_id: str
) -> list[dict[str, str]]:
    media: list[dict[str, str]] = []
    seen: set[str] = set()
    sample_dir = sample_file.parent
    for text in _iter_sample_strings(item):
        data_media_type = _data_image_media_type(text)
        if data_media_type is not None:
            if text not in seen:
                seen.add(text)
                media.append({"url": text, "label": "data image", "source": "data-url", "media_type": data_media_type})
            continue

        url_media_type = _url_image_media_type(text)
        if url_media_type is not None:
            if text not in seen:
                seen.add(text)
                label = Path(text.split("#", 1)[0].split("?", 1)[0]).name or "image"
                media.append({"url": text, "label": label, "source": text, "media_type": url_media_type})
            continue

        if not _looks_like_local_image_reference(text):
            continue
        local = _local_image_path(text, sample_dir)
        if local is None:
            continue
        path, media_type = local
        source = str(path)
        if source in seen:
            continue
        seen.add(source)
        token = _register_dlc_sample_media_path(path)
        media.append(
            {
                "url": _dlc_sample_media_url(job_id, metric_id, token),
                "label": path.name,
                "source": source,
                "media_type": media_type,
            }
        )
    return media


_CHOICE_ANSWER_PATTERN = re.compile(r"^\s*\(?([A-Z])\)?[.)]?\s*$", re.IGNORECASE)


def _literal_list_from_python_repr(value: str) -> list[Any] | None:
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        import ast

        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _extract_choice_answer(value: Any) -> str | None:
    parsed = _parse_json_maybe(value)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            return None
        return _extract_choice_answer(parsed[0])
    if not isinstance(parsed, str):
        return None

    python_list = _literal_list_from_python_repr(parsed)
    if python_list is not None:
        if len(python_list) != 1:
            return None
        return _extract_choice_answer(python_list[0])

    match = _CHOICE_ANSWER_PATTERN.fullmatch(parsed)
    return match.group(1).upper() if match else None


def _first_present_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _choice_target(row: dict[str, Any]) -> str | None:
    return _extract_choice_answer(_first_present_value(row, ("target", "answer", "gold", "ground_truth")))


def _choice_prediction(row: dict[str, Any]) -> str | None:
    return _extract_choice_answer(
        _first_present_value(
            row,
            (
                "filtered_resps",
                "extracted_answer",
                "prediction",
                "pred",
                "model_output",
                "response",
                "resps",
            ),
        )
    )


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return float(stripped)
            except ValueError:
                return None
    return None


def _correctness_from_score_value(value: Any) -> bool | None:
    score = _numeric_score(value)
    if score is None:
        return None
    if score == 0:
        return False
    if score == 1:
        return True
    return None


def _correctness_from_metrics(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    for key in ("exact_match", "score", "judge_score", "llm_judge_score"):
        if key in value:
            result = _correctness_from_score_value(value[key])
            if result is not None:
                return result
    for metric_value in value.values():
        if isinstance(metric_value, dict):
            result = _correctness_from_metrics(metric_value)
            if result is not None:
                return result
    return None


def _sample_correctness(
    row: dict[str, Any], metric_name: str, target_choice: str | None, prediction_choice: str | None
) -> bool | None:
    if metric_name in row:
        result = _correctness_from_score_value(row[metric_name])
        if result is not None:
            return result
    for key in ("exact_match", "score", "judge_score", "llm_judge_score"):
        if key in row:
            result = _correctness_from_score_value(row[key])
            if result is not None:
                return result
    result = _correctness_from_metrics(row.get("metrics"))
    if result is not None:
        return result
    if target_choice is not None and prediction_choice is not None:
        return target_choice == prediction_choice
    return None


def _answer_buckets(counter: Counter[str]) -> list[ChoiceAnswerBucket]:
    total = sum(counter.values())
    if total == 0:
        return []
    return [
        ChoiceAnswerBucket(option=option, count=count, ratio=count / total)
        for option, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def _make_answer_stats(
    *,
    correct_answers: Counter[str],
    target_answers: Counter[str],
    total: int,
    filtered_total: int,
    wrong_total: int,
    unknown_correctness_total: int,
) -> ChoiceAnswerStats:
    correct_answer_total = sum(correct_answers.values())
    target_answer_total = sum(target_answers.values())
    return ChoiceAnswerStats(
        is_multiple_choice=correct_answer_total > 0 and target_answer_total > 0,
        correct_answers=_answer_buckets(correct_answers),
        target_answers=_answer_buckets(target_answers),
        total=total,
        filtered_total=filtered_total,
        wrong_total=wrong_total,
        unknown_correctness_total=unknown_correctness_total,
        correct_answer_total=correct_answer_total,
        target_answer_total=target_answer_total,
    )


def _preferred_sample_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "_sample_file",
        "doc_id",
        "input",
        "target",
        "filtered_resps",
        "response",
        "model_output",
        "prediction",
        "pred",
        "extracted_answer",
        "answer",
        "exact_match",
        "score",
        "judge_score",
        "judge_reason",
        "token_counts",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(key for key in row.keys() if key != "_media")
    for key in preferred:
        if key in all_keys:
            ordered.append(key)
            seen.add(key)
    for key in sorted(all_keys):
        if key not in seen:
            ordered.append(key)
    return ordered


def _read_sample_jsonls(
    sample_files: list[str],
    *,
    job_id: str,
    metric_id: str,
    metric_name: str,
    offset: int,
    limit: int,
    only_wrong: bool,
) -> tuple[list[dict[str, Any]], int, ChoiceAnswerStats]:
    rows: list[dict[str, Any]] = []
    total = 0
    filtered_total = 0
    wrong_total = 0
    unknown_correctness_total = 0
    correct_answers: Counter[str] = Counter()
    target_answers: Counter[str] = Counter()
    for sample_file in sample_files:
        path = Path(sample_file)
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    total += 1
                    target_choice = _choice_target(item)
                    prediction_choice = _choice_prediction(item)
                    if target_choice is not None:
                        correct_answers[target_choice] += 1
                    if prediction_choice is not None:
                        target_answers[prediction_choice] += 1

                    correctness = _sample_correctness(item, metric_name, target_choice, prediction_choice)
                    if correctness is False:
                        wrong_total += 1
                    elif correctness is None:
                        unknown_correctness_total += 1

                    if only_wrong and correctness is not False:
                        continue
                    if filtered_total >= offset and len(rows) < limit:
                        row = {str(key): _stringify_sample_value(value) for key, value in item.items()}
                        row["_sample_file"] = path.name
                        media = _extract_dlc_sample_media(item, sample_file=path, job_id=job_id, metric_id=metric_id)
                        if media:
                            row["_media"] = media
                        rows.append(row)
                    filtered_total += 1
        except OSError:
            continue
    answer_stats = _make_answer_stats(
        correct_answers=correct_answers,
        target_answers=target_answers,
        total=total,
        filtered_total=filtered_total,
        wrong_total=wrong_total,
        unknown_correctness_total=unknown_correctness_total,
    )
    return rows, filtered_total, answer_stats


def _result_status_for_root(result_root: Path | None) -> str:
    if result_root is None:
        return "not_found"
    if not result_root.exists() or not result_root.is_dir():
        return "not_found"
    if (result_root / "summary.jsonl").is_file() or list(result_root.rglob("*_results.json")):
        return "has_results"
    if _judged_sample_metric_rows(idx_start=0, result_root=result_root):
        return "has_results"
    if (result_root / "config.json").is_file() or list(result_root.rglob("task_status.tsv")) or (result_root / "tasks").is_dir():
        return "partial"
    return "empty"


def _job_runtime_paths(detail: dict[str, Any]) -> tuple[Path | None, Path | None, str | None]:
    runtime_config = _runtime_config_path_from_detail(detail)
    result_root = _result_root_from_runtime_config(runtime_config)
    log_dir = None
    command = detail.get("UserCommand")
    if isinstance(command, str):
        command = str(_replace_default_user(command))
        log_dir = _extract_log_dir(command)
    if result_root is None:
        result_root = _result_root_from_command_output_path(detail)
    return runtime_config, result_root, log_dir


def _exact_result_resolution_error(detail: dict[str, Any]) -> str:
    job_id = str(detail.get("JobId") or "")
    display_name = str(detail.get("DisplayName") or "")
    command = detail.get("UserCommand")
    evidence: list[str] = []
    if isinstance(command, str) and command:
        command = str(_replace_default_user(command))
        runtime_path = _extract_command_path(command, "runtime_config.json")
        output_path = _extract_cli_option_value(command, "--output_path")
        if runtime_path:
            evidence.append(f"runtime_config.json path is not readable or its output path does not exist: {runtime_path}")
        if output_path:
            evidence.append(f"--output_path does not exist or is not a directory: {output_path}")
        if not runtime_path and not output_path:
            evidence.append("UserCommand contains neither a runtime_config.json path nor --output_path")
    else:
        evidence.append("DLC detail has no UserCommand")
    return (
        f"Cannot resolve exact local result directory for DLC job {job_id} ({display_name}). "
        f"{'; '.join(evidence)}. Refusing fuzzy result matching."
    )


def _no_metric_rows_error(result_root: Path) -> str:
    return (
        f"No metric rows found in exact local result directory: {result_root}. "
        "Expected at least one *_results.json with a non-empty results object, "
        "or a summary.jsonl entry whose result_json points inside this exact directory."
    )


# --- Endpoints ---


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": get_version(),
        "git": get_git_info(),
        "system": get_system_info(),
        "env_setup": _detect_env_setup(),
    }


@app.post("/auth/login", response_model=AuthUserResponse)
async def login(request: AuthLoginRequest, response: Response) -> AuthUserResponse:
    user = _build_authenticated_user(request.access_key_id, request.secret_access_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Access Key validation failed")

    session_id, session = _create_auth_session(user)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=session_id,
        max_age=_auth_session_ttl_seconds(),
        httponly=True,
        samesite="lax",
        path="/",
    )
    return AuthUserResponse(**_public_auth_user(session), expires_at=float(session["expires_at"]))


@app.get("/auth/me", response_model=AuthUserResponse)
async def get_current_user(request: Request) -> AuthUserResponse:
    user = _require_authenticated_user(request)
    return AuthUserResponse(**_public_auth_user(user), expires_at=float(user["expires_at"]))


@app.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, str]:
    _delete_auth_session(request.cookies.get(AUTH_COOKIE_NAME))
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    return {"status": "logged_out"}


@app.get("/defaults", response_model=DefaultsResponse)
async def get_defaults() -> DefaultsResponse:
    """Get the production DLC + vLLM defaults used by the Web UI."""
    eval_config = _default_eval_config()
    judge_config = _default_judge_config()
    model_config = eval_config["model"]
    eval_section = eval_config["eval"]
    judge_api = judge_config["judge"]["api"]
    dlc_path = DEFAULT_DLC_PATH_TEMPLATE
    dlc_config = _default_dlc_config()
    dlc_section = dlc_config.get("dlc")
    if not isinstance(dlc_section, dict):
        raise HTTPException(status_code=500, detail="Default DLC config must contain a dlc object")
    job_name = _validate_eval_job_name(str(dlc_section.get("job_name") or DEFAULT_EVAL_JOB_NAME))
    dlc_section["binary"] = dlc_path
    dlc_section["job_name"] = job_name
    eval_config["log"]["dir"] = _path_with_leaf(eval_config["log"]["dir"], job_name, field_name="log.dir")
    eval_section["output_path"] = _path_with_leaf(eval_section["output_path"], job_name, field_name="eval.output_path")

    return DefaultsResponse(
        user="",
        job_name=job_name,
        eval_inference_mode=DEFAULT_EVAL_INFERENCE_MODE,
        model=str(model_config["path"]),
        api_url=DEFAULT_API_EVAL_URL,
        api_key="",
        dlc_path=dlc_path,
        model_args="",
        tasks=_split_tasks(eval_section["tasks"]),
        judge_backend=DEFAULT_JUDGE_BACKEND,
        judge_api_url=str(judge_api["base_url"]),
        judge_api_key=str(judge_api.get("key") or ""),
        env_vars=_dict_to_env_vars({str(key): value for key, value in eval_config["env"].items()}),
        batch_size=1,
        limit=int(eval_section.get("limit", -1)),
        output_path=str(eval_section["output_path"]),
        log_samples=True,
        verbosity=str(eval_section.get("verbosity", "INFO")),
        device=None,
        env_setup="",
        run_mode=DEFAULT_RUN_MODE,
        dlc_config=dlc_config,
        model_tp=int(model_config["tp"]),
        max_model_len=int(model_config["max_model_len"]),
        gpu_memory_utilization=float(model_config["gpu_memory_utilization"]),
        max_num_seqs=int(model_config["max_num_seqs"]),
        base_port=int(model_config["base_port"]),
        concurrency=int(eval_section["concurrency"]),
        gen_kwargs=str(eval_section["gen_kwargs"]),
        enable_thinking=bool(model_config.get("enable_thinking", DEFAULT_ENABLE_THINKING)),
        debug=bool(eval_section.get("debug", False)),
    )


@app.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    """Get available models."""
    cache = get_discovery_cache()
    models = cache.get_models(include_all=True)
    return [ModelInfo(id=model_id, name=name) for model_id, name in models]


@app.get("/tasks", response_model=list[TaskInfo])
async def get_tasks() -> list[TaskInfo]:
    """Get available tasks."""
    cache = get_discovery_cache()
    tasks = cache.get_tasks(include_all=True)
    return [
        TaskInfo(
            id=task_id,
            name=name,
            group=name.startswith("[Group]"),
            requires_llm_judge=False if name.startswith("[Group]") else _task_requires_llm_judge(task_id),
        )
        for task_id, name in tasks
    ]


class _TaskYamlLoader(yaml.SafeLoader):
    pass


def _unknown_task_yaml_constructor(loader: _TaskYamlLoader, _tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise yaml.YAMLError(f"Unsupported YAML node: {type(node).__name__}")


_TaskYamlLoader.add_multi_constructor("", _unknown_task_yaml_constructor)


def _validate_task_id(task_id: str) -> str:
    normalized = task_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="task_id is required")
    if not TASK_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="task_id must match ^[A-Za-z0-9][A-Za-z0-9_-]*$",
        )
    return normalized


def _task_create_paths(task_id: str) -> tuple[Path, Path, Path]:
    task_dir = (TASKS_DIR / task_id).resolve()
    tasks_root = TASKS_DIR.resolve()
    if task_dir != tasks_root and tasks_root not in task_dir.parents:
        raise HTTPException(status_code=400, detail="Resolved task path escapes tasks directory")
    return task_dir, task_dir / f"{task_id}.yaml", task_dir / "utils.py"


def _validate_task_yaml(task_id: str, yaml_content: str) -> None:
    if not yaml_content.strip():
        raise HTTPException(status_code=400, detail="yaml_content is required")
    try:
        parsed = yaml.load(yaml_content, Loader=_TaskYamlLoader)
    except yaml.YAMLError as error:
        raise HTTPException(status_code=400, detail=f"Invalid task YAML: {error}") from error
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Task YAML must contain a mapping object")
    task_value = parsed.get("task")
    if task_value != task_id:
        raise HTTPException(status_code=400, detail=f"Task YAML field 'task' must equal '{task_id}'")


def _validate_task_python(python_content: str) -> None:
    if not python_content.strip():
        raise HTTPException(status_code=400, detail="python_content is required")


@app.post("/tasks/create", response_model=TaskCreateResponse)
async def create_task(request: TaskCreateRequest) -> TaskCreateResponse:
    task_id = _validate_task_id(request.task_id)
    _validate_task_yaml(task_id, request.yaml_content)
    _validate_task_python(request.python_content)

    task_dir, yaml_path, python_path = _task_create_paths(task_id)
    if not request.overwrite:
        existing_paths = [str(path) for path in (yaml_path, python_path) if path.exists()]
        if existing_paths:
            raise HTTPException(status_code=409, detail=f"Task files already exist: {', '.join(existing_paths)}")

    task_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(request.yaml_content.rstrip() + "\n", encoding="utf-8")
    python_path.write_text(request.python_content.rstrip() + "\n", encoding="utf-8")
    discovered_task_count, _model_count = get_discovery_cache().reload()

    return TaskCreateResponse(
        task_id=task_id,
        task_dir=str(task_dir),
        yaml_path=str(yaml_path),
        python_path=str(python_path),
        discovered_task_count=discovered_task_count,
    )


@app.get("/tasks/{task_id}/yaml")
async def get_task_yaml(task_id: str) -> dict[str, str]:
    tasks_dir = TASKS_DIR
    if not tasks_dir.exists():
        raise HTTPException(status_code=500, detail="Tasks directory not found")

    task_pattern = re.compile(rf"^\s*task\s*:\s*[\"']?{re.escape(task_id)}[\"']?\s*$", re.MULTILINE)
    yaml_files = sorted({*tasks_dir.rglob("*.yaml"), *tasks_dir.rglob("*.yml")})

    for yaml_file in yaml_files:
        try:
            yaml_content = yaml_file.read_text(encoding="utf-8")
        except OSError:
            continue

        if task_pattern.search(yaml_content):
            repo_root = Path(__file__).resolve().parents[2]
            try:
                relative_path = str(yaml_file.relative_to(repo_root))
            except ValueError:
                relative_path = str(yaml_file)

            return {
                "task_id": task_id,
                "yaml": yaml_content,
                "path": relative_path,
            }

    raise HTTPException(status_code=404, detail=f"Task YAML not found for '{task_id}'")


def _normalize_env_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        return stripped
    if "=" in stripped:
        return f"export {stripped}"
    return None


def _build_env_exports(env_vars: str) -> list[str]:
    exports: list[str] = []
    for line in env_vars.splitlines():
        export_line = _normalize_env_line(line)
        if export_line:
            exports.append(export_line)
    return exports


def _env_vars_to_dict(env_vars: str) -> dict[str, str]:
    """Convert env_vars multi-line string to a dict for YAML export."""
    env_dict: dict[str, str] = {}
    for line in env_vars.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:]
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            env_dict[key.strip()] = value.strip()
    return env_dict


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _dict_to_env_vars(env_dict: dict[str, Any]) -> str:
    """Convert env dict from YAML to env_vars multi-line string for UI."""
    lines = []
    for key, value in env_dict.items():
        lines.append(f"export {key}={_format_env_value(value)}")
    return "\n".join(lines)


def _validate_run_mode(run_mode: str) -> None:
    if run_mode not in {"dlc", "local"}:
        raise HTTPException(status_code=400, detail=f"Unsupported run_mode: {run_mode}")


def _validate_eval_inference_mode(eval_inference_mode: str) -> str:
    normalized = eval_inference_mode.strip().lower()
    if normalized not in {"ckpt", "api"}:
        raise HTTPException(status_code=400, detail=f"Unsupported eval_inference_mode: {eval_inference_mode}")
    return normalized


def _validate_judge_backend(judge_backend: str) -> str:
    normalized = judge_backend.strip().lower()
    if normalized not in {"vllm", "api"}:
        raise HTTPException(status_code=400, detail=f"Unsupported judge_backend: {judge_backend}")
    return normalized


def _validate_eval_job_name(job_name: str) -> str:
    normalized = job_name.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Job name is required")
    if not EVAL_JOB_NAME_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="DLC eval job_name must start with eval_ and contain only letters, numbers, '_' or '-'",
        )
    return normalized


def _path_with_leaf(path_value: Any, leaf: str, *, field_name: str) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    normalized = path_value.strip().rstrip("/")
    parent, separator, _current_leaf = normalized.rpartition("/")
    if not separator:
        return leaf
    return f"{parent}/{leaf}"


def _require_dlc_resource_id(resource_id: Any, *, field: str) -> str:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise HTTPException(status_code=400, detail=f"Missing {field} in dlc_config")
    normalized = resource_id.strip()
    if normalized != DEFAULT_DLC_RESOURCE_ID:
        raise HTTPException(status_code=400, detail=f"{field} must be {DEFAULT_DLC_RESOURCE_ID}")
    return normalized


def _require_dlc_nas_mount(data_source_uris: Any, *, field: str) -> str:
    if not isinstance(data_source_uris, str) or not data_source_uris.strip():
        raise HTTPException(status_code=400, detail=f"Missing {field} in dlc_config")
    normalized = data_source_uris.strip()
    if REQUIRED_NAS_MOUNT_URI not in [item.strip() for item in normalized.split(",")]:
        raise HTTPException(status_code=400, detail=f"{field} must include {REQUIRED_NAS_MOUNT_URI}")
    return normalized


def _apply_api_eval_dlc_resources(dlc: dict[str, Any]) -> None:
    judge = dlc.get("judge")
    if isinstance(judge, dict):
        for key in (
            "worker_cpu",
            "worker_memory",
            "worker_shared_memory",
            "worker_image",
            "data_source_uris",
            "resource_id",
            "workspace_id",
            "vpc_id",
            "switch_id",
            "security_group_id",
            "extended_cidrs",
        ):
            if judge.get(key) not in ("", None):
                dlc[key] = judge[key]
        if judge.get("job_max_running_time_minutes") not in ("", None):
            dlc["job_max_running_time_minutes"] = judge["job_max_running_time_minutes"]
        if judge.get("running_timeout") not in ("", None):
            dlc["running_timeout"] = judge["running_timeout"]
        if judge.get("priority") not in ("", None):
            dlc["priority"] = judge["priority"]

    dlc["workers"] = 1
    dlc["worker_gpu"] = 0
    if dlc.get("worker_cpu") in ("", None):
        dlc["worker_cpu"] = 8
    if dlc.get("worker_memory") in ("", None):
        dlc["worker_memory"] = "64Gi"
    if dlc.get("worker_shared_memory") in ("", None):
        dlc["worker_shared_memory"] = "16Gi"


def _request_dlc_config(request: EvalRequest | PreviewRequest | ExportYamlRequest) -> dict[str, Any]:
    config = copy.deepcopy(request.dlc_config) if request.dlc_config else _default_dlc_config()
    config = _replace_user_placeholder(config, request.user)
    dlc = config.get("dlc")
    if not isinstance(dlc, dict):
        raise HTTPException(status_code=400, detail="dlc_config must contain a dlc object")
    eval_inference_mode = _validate_eval_inference_mode(request.eval_inference_mode)
    judge_backend = _validate_judge_backend(request.judge_backend)
    inline_local_judge = judge_backend == "vllm" and bool(_llm_as_judge_tasks(request.tasks))
    if eval_inference_mode == "api" and not inline_local_judge:
        _apply_api_eval_dlc_resources(dlc)
    dlc_path = str(request.dlc_path or "").strip()
    if not dlc_path:
        raise HTTPException(status_code=400, detail="DLC path is required")
    dlc["binary"] = _replace_user_placeholder(dlc_path, request.user)
    for key in (
        "binary",
        "run_script",
        "workers",
        "worker_gpu",
        "worker_cpu",
        "worker_memory",
        "worker_shared_memory",
        "worker_image",
        "data_source_uris",
        "resource_id",
        "workspace_id",
        "vpc_id",
        "switch_id",
        "security_group_id",
        "extended_cidrs",
    ):
        if key not in dlc or dlc[key] in ("", None):
            raise HTTPException(status_code=400, detail=f"Missing dlc.{key} in dlc_config")
    if str(dlc["workspace_id"]) != DEFAULT_DLC_WORKSPACE_ID:
        raise HTTPException(status_code=400, detail=f"DLC workspace_id must be {DEFAULT_DLC_WORKSPACE_ID}")
    dlc["resource_id"] = _require_dlc_resource_id(dlc["resource_id"], field="dlc.resource_id")
    dlc["data_source_uris"] = _require_dlc_nas_mount(dlc["data_source_uris"], field="dlc.data_source_uris")
    judge = dlc.get("judge")
    if isinstance(judge, dict):
        judge["resource_id"] = _require_dlc_resource_id(judge.get("resource_id"), field="dlc.judge.resource_id")
        judge["data_source_uris"] = _require_dlc_nas_mount(
            judge.get("data_source_uris"), field="dlc.judge.data_source_uris"
        )
    run_script = str(dlc["run_script"])
    if Path(run_script).name != QWEN35_WORKER_BASENAME:
        raise HTTPException(
            status_code=400,
            detail=f"DLC run_script must point to {QWEN35_WORKER_BASENAME}, got: {run_script}",
        )
    dlc["run_script"] = run_script
    dlc["job_name"] = _validate_eval_job_name(request.job_name)
    return config


def _build_dlc_credential_args(
    auth_user: dict[str, Any],
    *,
    mask_secrets: bool = False,
) -> list[str]:
    access_key = str(auth_user.get("access_key_id") or "").strip()
    secret_access_key = str(auth_user.get("secret_access_key") or "").strip()
    if not access_key:
        raise HTTPException(status_code=500, detail="Authenticated WebUI user is missing access_key_id")
    if not secret_access_key:
        raise HTTPException(status_code=500, detail="Authenticated WebUI user is missing secret_access_key")

    display_access_key = MASKED_SECRET if mask_secrets else access_key
    display_secret_access_key = MASKED_SECRET if mask_secrets else secret_access_key
    return [
        DLC_ACCESS_ID_FLAG,
        display_access_key,
        DLC_ACCESS_KEY_FLAG,
        display_secret_access_key,
        DLC_IGNORE_LOCAL_CONFIG_FLAG,
    ]


def _build_vllm_eval_config(request: EvalRequest | PreviewRequest | ExportYamlRequest) -> dict[str, Any]:
    if not request.model:
        raise HTTPException(status_code=400, detail="Checkpoint path is required")
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No tasks specified")
    if request.batch_size != 1:
        raise HTTPException(status_code=400, detail="DLC vLLM evaluation requires batch_size=1")
    if request.model_args:
        raise HTTPException(status_code=400, detail="model_args is not used in DLC vLLM mode; leave it empty")

    job_name = _validate_eval_job_name(request.job_name)
    config = _replace_user_placeholder(_default_eval_config(), request.user)
    env_dict = _env_vars_to_dict(request.env_vars)
    if env_dict:
        config["env"].update(_replace_user_placeholder(env_dict, request.user))
    _sync_judge_api_to_eval_env(config, request)

    config["model"]["path"] = request.model
    config["model"]["tp"] = request.model_tp
    config["model"]["max_model_len"] = request.max_model_len
    config["model"]["gpu_memory_utilization"] = request.gpu_memory_utilization
    config["model"]["max_num_seqs"] = request.max_num_seqs
    config["model"]["base_port"] = request.base_port
    config["model"]["reasoning_parser"] = DEFAULT_REASONING_PARSER
    config["model"]["enable_thinking"] = request.enable_thinking
    config["model"]["is_qwen3_vl"] = False

    config["eval"]["tasks"] = ",".join(request.tasks)
    config["log"]["dir"] = _path_with_leaf(config["log"]["dir"], job_name, field_name="log.dir")
    config["eval"]["output_path"] = _path_with_leaf(request.output_path, job_name, field_name="eval.output_path")
    config["eval"]["concurrency"] = request.concurrency
    config["eval"]["gen_kwargs"] = request.gen_kwargs
    config["eval"]["limit"] = -1 if request.limit is None else request.limit
    config["eval"]["debug"] = request.debug
    config["eval"]["verbosity"] = request.verbosity
    return _replace_user_placeholder(config, request.user)


def _build_api_eval_config(request: EvalRequest | PreviewRequest | ExportYamlRequest) -> dict[str, Any]:
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No tasks specified")
    if request.model_args:
        raise HTTPException(status_code=400, detail="model_args is not used in DLC API mode; leave it empty")
    api_url = request.api_url.strip()
    api_key = request.api_key.strip()
    if not api_url:
        raise HTTPException(status_code=400, detail="API address is required for API evaluation")
    if not api_key:
        raise HTTPException(status_code=400, detail="API token is required for API evaluation")

    job_name = _validate_eval_job_name(request.job_name)
    config = _replace_user_placeholder(_default_eval_config(), request.user)
    env_dict = _env_vars_to_dict(request.env_vars)
    if env_dict:
        config["env"].update(_replace_user_placeholder(env_dict, request.user))

    api_model = os.getenv("LMMS_EVAL_WEBUI_API_MODEL", DEFAULT_API_EVAL_MODEL).strip() or DEFAULT_API_EVAL_MODEL
    config["env"]["api_type"] = DEFAULT_API_EVAL_TYPE
    config["env"]["openai_api_url"] = api_url
    config["env"]["openai_api_key"] = api_key
    config["model"]["backend"] = "openai"
    config["model"]["path"] = api_model
    config["model"]["is_qwen3_vl"] = False
    config["eval"]["tasks"] = ",".join(request.tasks)
    config["log"]["dir"] = _path_with_leaf(config["log"]["dir"], job_name, field_name="log.dir")
    config["eval"]["output_path"] = _path_with_leaf(request.output_path, job_name, field_name="eval.output_path")
    config["eval"]["concurrency"] = request.concurrency
    config["eval"]["batch_size"] = request.batch_size
    config["eval"]["gen_kwargs"] = request.gen_kwargs
    config["eval"]["limit"] = -1 if request.limit is None else request.limit
    config["eval"]["debug"] = request.debug
    config["eval"]["verbosity"] = request.verbosity
    return _replace_user_placeholder(config, request.user)


def _build_eval_config(request: EvalRequest | PreviewRequest | ExportYamlRequest) -> dict[str, Any]:
    eval_inference_mode = _validate_eval_inference_mode(request.eval_inference_mode)
    if eval_inference_mode == "ckpt":
        return _build_vllm_eval_config(request)
    return _build_api_eval_config(request)


def _build_judge_config(
    request: EvalRequest | PreviewRequest | ExportYamlRequest,
    eval_config: dict[str, Any],
) -> dict[str, Any] | None:
    judge_tasks = _llm_as_judge_tasks(request.tasks)
    if not judge_tasks:
        return None

    backend = _validate_judge_backend(request.judge_backend)
    config = _replace_user_placeholder(_default_judge_config(), request.user)
    config["env"].update(copy.deepcopy(eval_config["env"]))
    config["log"]["dir"] = str(eval_config["log"]["dir"])
    config["judge"]["backend"] = backend
    if backend == "api":
        api_url = request.judge_api_url.strip()
        api_key = request.judge_api_key.strip()
        if not api_url:
            raise HTTPException(status_code=400, detail="LLM API URL is required for selected LLM-as-judge tasks")
        if not api_key:
            raise HTTPException(status_code=400, detail="LLM API key is required for selected LLM-as-judge tasks")
        judge_model = os.getenv("LMMS_EVAL_WEBUI_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
        config["judge"]["model"] = judge_model
        config["judge"]["api"]["key"] = api_key
        config["judge"]["api"]["base_url"] = api_url
    else:
        vllm = config["judge"].get("vllm")
        if not isinstance(vllm, dict):
            raise HTTPException(status_code=500, detail="Default judge config must contain judge.vllm")
        config["judge"]["parallel"] = DEFAULT_LOCAL_JUDGE_PARALLEL
        config["judge"]["model"] = DEFAULT_LOCAL_JUDGE_MODEL
        config["judge"]["api"]["key"] = ""
        config["judge"]["api"]["base_url"] = ""
        vllm.update(
            {
                "model_path": DEFAULT_LOCAL_JUDGE_MODEL_PATH,
                "tp": DEFAULT_LOCAL_JUDGE_TP,
                "max_model_len": DEFAULT_LOCAL_JUDGE_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_LOCAL_JUDGE_GPU_MEMORY_UTILIZATION,
                "max_num_seqs": DEFAULT_LOCAL_JUDGE_MAX_NUM_SEQS,
                "port": DEFAULT_LOCAL_JUDGE_PORT,
            }
        )
    config["eval"]["input_result_path"] = JUDGE_INPUT_RESULT_PLACEHOLDER
    config["eval"]["tasks"] = ",".join(judge_tasks)
    config["eval"]["output_path"] = JUDGE_OUTPUT_PATH_PLACEHOLDER
    config["eval"]["debug"] = False
    return _replace_user_placeholder(config, request.user)


def _redact_judge_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(config)
    env = redacted.get("env")
    if isinstance(env, dict) and env.get("openai_api_key"):
        env["openai_api_key"] = MASKED_SECRET
    if isinstance(env, dict) and env.get("judge_api_key"):
        env["judge_api_key"] = MASKED_SECRET
    judge = redacted.get("judge")
    if isinstance(judge, dict) and isinstance(judge.get("api"), dict) and judge["api"].get("key"):
        judge["api"]["key"] = MASKED_SECRET
    return redacted


def _redact_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(config)
    env = redacted.get("env")
    if isinstance(env, dict) and env.get("openai_api_key"):
        env["openai_api_key"] = MASKED_SECRET
    if isinstance(env, dict) and env.get("judge_api_key"):
        env["judge_api_key"] = MASKED_SECRET
    return redacted


def _json_heredoc(path: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return f"cat > {path} <<'JSON'\n{payload}\nJSON"


def _build_dlc_command(
    request: EvalRequest | PreviewRequest | ExportYamlRequest,
    *,
    auth_user: dict[str, Any],
    mask_secrets: bool = False,
) -> str:
    dlc_config = _request_dlc_config(request)
    credential_args = _build_dlc_credential_args(auth_user, mask_secrets=mask_secrets)
    eval_config = _build_eval_config(request)
    judge_config = _build_judge_config(request, eval_config)
    unresolved_values = [dlc_config, eval_config]
    if judge_config is not None:
        unresolved_values.append(judge_config)
    if request.user.strip() and any(_contains_user_placeholder(value) for value in unresolved_values):
        raise HTTPException(status_code=400, detail="USER replacement left unresolved user placeholders")
    submit_script = str(DLC_SUBMIT_SCRIPT)
    if not DLC_SUBMIT_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"DLC submit script not found: {submit_script}")

    command_lines = [
        "set -euo pipefail",
        'RUN_DIR="${LMMS_EVAL_WEB_UI_RUN_DIR:-$(mktemp -d /tmp/lmms_eval_webui.XXXXXX)}"',
        'mkdir -p "${RUN_DIR}"',
        _json_heredoc('"${RUN_DIR}/config_dlc.json"', dlc_config),
        _json_heredoc('"${RUN_DIR}/config_eval.json"', _redact_eval_config(eval_config) if mask_secrets else eval_config),
    ]
    if mask_secrets:
        command_lines.append("# DLC Access Key values are redacted in preview; Start uses the unmasked in-memory values.")
        if _validate_eval_inference_mode(request.eval_inference_mode) == "api":
            command_lines.append("# API token is redacted in preview; Start uses the unmasked in-memory value.")
    submit_args = '"${RUN_DIR}/config_dlc.json" "${RUN_DIR}/config_eval.json"'
    if judge_config is not None:
        display_judge_config = _redact_judge_config(judge_config) if mask_secrets else judge_config
        command_lines.append(_json_heredoc('"${RUN_DIR}/config_judge.json"', display_judge_config))
        submit_args += ' "${RUN_DIR}/config_judge.json"'
        if mask_secrets:
            command_lines.append("# LLM API key is redacted in preview; Start uses the unmasked in-memory value.")
    credential_arg_text = " ".join(shlex.quote(arg) for arg in credential_args)
    command_lines.append(f"bash {shlex.quote(submit_script)} {submit_args} -- {credential_arg_text}")
    return "\n".join(command_lines)


def _build_command(request: EvalRequest | PreviewRequest, *, auth_user: dict[str, Any] | None = None) -> str:
    """Build the lmms_eval command string."""
    _validate_run_mode(request.run_mode)
    if request.run_mode == "dlc":
        if auth_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return _build_dlc_command(request, auth_user=auth_user, mask_secrets=True)

    parts = ["python -m lmms_eval"]
    parts.append(f"--model {request.model}")
    if request.model_args:
        parts.append(f"--model_args '{request.model_args}'")
    if request.tasks:
        parts.append(f"--tasks {','.join(request.tasks)}")
    parts.append(f"--batch_size {request.batch_size}")
    if request.limit is not None:
        parts.append(f"--limit {request.limit}")
    parts.append(f"--output_path {request.output_path}")
    if request.log_samples:
        parts.append("--log_samples")
    parts.append(f"--verbosity {request.verbosity}")
    if request.device:
        parts.append(f"--device {request.device}")
    command = " \\\n    ".join(parts)
    # Collect all prefix lines: env_setup first, then env_vars exports
    prefix_lines: list[str] = []
    env_setup = request.env_setup or _detect_env_setup()
    if env_setup:
        prefix_lines.append(env_setup)
    prefix_lines.extend(_build_env_exports(request.env_vars))
    if prefix_lines:
        return "\n".join([*prefix_lines, command])
    return command


def _build_shell_command(request: EvalRequest, *, auth_user: dict[str, Any] | None = None) -> str:
    """Build the shell command for execution."""
    _validate_run_mode(request.run_mode)
    if request.run_mode == "dlc":
        if auth_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return _build_dlc_command(request, auth_user=auth_user, mask_secrets=False)

    parts = ["python", "-m", "lmms_eval"]
    parts.extend(["--model", request.model])
    if request.model_args:
        parts.extend(["--model_args", request.model_args])
    if request.tasks:
        parts.extend(["--tasks", ",".join(request.tasks)])
    parts.extend(["--batch_size", str(request.batch_size)])
    if request.limit is not None:
        parts.extend(["--limit", str(request.limit)])
    parts.extend(["--output_path", request.output_path])
    if request.log_samples:
        parts.append("--log_samples")
    parts.extend(["--verbosity", request.verbosity])
    if request.device:
        parts.extend(["--device", request.device])
    command = " ".join(parts)
    # Collect all prefix commands: env_setup first, then env_vars exports
    prefix_parts: list[str] = []
    env_setup = request.env_setup or _detect_env_setup()
    if env_setup:
        prefix_parts.append(env_setup)
    prefix_parts.extend(_build_env_exports(request.env_vars))
    if prefix_parts:
        prefix = " && ".join(prefix_parts)
        return f"{prefix} && {command}"
    return command


@app.post("/eval/preview", response_model=PreviewResponse)
async def preview_command(request: PreviewRequest, http_request: Request) -> PreviewResponse:
    """Generate command preview without executing."""
    command = _build_command(request, auth_user=_require_authenticated_user(http_request))
    return PreviewResponse(command=command)


@app.post("/eval/export-yaml", response_model=ExportYamlResponse)
async def export_yaml(request: ExportYamlRequest) -> ExportYamlResponse:
    """Export current UI config as a YAML config file."""
    _validate_run_mode(request.run_mode)
    if request.run_mode == "dlc":
        eval_config = _build_eval_config(request)
        config = {
            "run_mode": "dlc",
            "user": request.user.strip(),
            "job_name": _validate_eval_job_name(request.job_name),
            "eval_inference_mode": _validate_eval_inference_mode(request.eval_inference_mode),
            "api_url": request.api_url.strip(),
            "api_key": request.api_key.strip(),
            "dlc_path": request.dlc_path.strip(),
            "judge_backend": _validate_judge_backend(request.judge_backend),
            "judge_api_url": request.judge_api_url.strip(),
            "judge_api_key": request.judge_api_key.strip(),
            **_request_dlc_config(request),
            **eval_config,
        }
        header = (
            "# LMMs-Eval DLC config exported from Web UI\n"
            "# Usage: the Web UI preview writes this config to JSON and calls qwen35_submit.sh.\n"
            "# In ckpt mode, keep all fields unchanged when comparing checkpoints; only change model.path.\n\n"
        )
        yaml_content = header + yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return ExportYamlResponse(yaml_content=yaml_content)

    config: dict[str, Any] = {}

    env_dict = _env_vars_to_dict(request.env_vars)
    if env_dict:
        config["env"] = env_dict

    config["model"] = request.model
    if request.model_args:
        config["model_args"] = request.model_args
    if request.tasks:
        config["tasks"] = ",".join(request.tasks)
    config["judge_backend"] = _validate_judge_backend(request.judge_backend)
    if request.judge_api_url.strip():
        config["judge_api_url"] = request.judge_api_url.strip()
    if request.judge_api_key.strip():
        config["judge_api_key"] = request.judge_api_key.strip()
    config["batch_size"] = request.batch_size
    if request.limit is not None:
        config["limit"] = request.limit
    config["output_path"] = request.output_path
    if request.log_samples:
        config["log_samples"] = True
    config["verbosity"] = request.verbosity
    if request.device:
        config["device"] = request.device
    config["run_mode"] = "local"

    header = "# LMMs-Eval config exported from Web UI\n" "# Usage: python -m lmms_eval --config <this_file>.yaml\n" "# CLI args override YAML values.\n\n"
    yaml_content = header + yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return ExportYamlResponse(yaml_content=yaml_content)


@app.post("/eval/import-yaml", response_model=ImportYamlResponse)
async def import_yaml(request: ImportYamlRequest) -> ImportYamlResponse:
    """Import a YAML config file into UI config values."""
    try:
        config = yaml.safe_load(request.yaml_content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="YAML must be a dict (not a list or scalar)")

    run_mode = str(config.get("run_mode", DEFAULT_RUN_MODE))
    if run_mode == "dlc":
        env_dict = config.get("env", {})
        if not isinstance(env_dict, dict):
            raise HTTPException(status_code=400, detail="env must be a dict")
        model_config = config.get("model", {})
        eval_config = config.get("eval", {})
        judge_api_url = str(config.get("judge_api_url") or "")
        judge_api_key = str(config.get("judge_api_key") or "")
        judge_backend = str(config.get("judge_backend") or DEFAULT_JUDGE_BACKEND)
        judge_config = config.get("judge_config", {})
        if isinstance(judge_config, dict):
            judge_section = judge_config.get("judge", {})
            if isinstance(judge_section, dict):
                judge_backend = str(judge_section.get("backend") or judge_backend)
                api_section = judge_section.get("api", {})
                if isinstance(api_section, dict):
                    judge_api_url = judge_api_url or str(api_section.get("base_url") or "")
                    judge_api_key = judge_api_key or str(api_section.get("key") or "")
        if not isinstance(model_config, dict):
            raise HTTPException(status_code=400, detail="model must be a dict in DLC mode")
        if not isinstance(eval_config, dict):
            raise HTTPException(status_code=400, detail="eval must be a dict in DLC mode")
        raw_eval_inference_mode = str(config.get("eval_inference_mode") or "")
        if not raw_eval_inference_mode:
            raw_eval_inference_mode = "api" if str(model_config.get("backend") or "").strip().lower() == "openai" else "ckpt"
        eval_inference_mode = _validate_eval_inference_mode(raw_eval_inference_mode)
        dlc_config = {"dlc": config.get("dlc", {})}
        if not isinstance(dlc_config["dlc"], dict):
            raise HTTPException(status_code=400, detail="dlc must be a dict in DLC mode")
        dlc_path = str(config.get("dlc_path") or dlc_config["dlc"].get("binary") or DEFAULT_DLC_PATH_TEMPLATE)
        job_name = _validate_eval_job_name(str(config.get("job_name") or dlc_config["dlc"].get("job_name") or DEFAULT_EVAL_JOB_NAME))
        dlc_config["dlc"]["job_name"] = job_name
        return ImportYamlResponse(
            user=str(config.get("user", "")),
            job_name=job_name,
            eval_inference_mode=eval_inference_mode,
            model=str(model_config.get("path", "")),
            api_url=str(config.get("api_url") or env_dict.get("openai_api_url") or DEFAULT_API_EVAL_URL),
            api_key=str(config.get("api_key") or env_dict.get("openai_api_key") or ""),
            dlc_path=dlc_path,
            model_args="",
            tasks=_split_tasks(eval_config.get("tasks", "")),
            judge_backend=_validate_judge_backend(judge_backend),
            judge_api_url=judge_api_url,
            judge_api_key=judge_api_key,
            env_vars=_dict_to_env_vars({str(key): value for key, value in env_dict.items()}),
            batch_size=1,
            limit=eval_config.get("limit", -1),
            output_path=_path_with_leaf(str(eval_config.get("output_path", "./logs/")), job_name, field_name="eval.output_path"),
            log_samples=True,
            verbosity=str(eval_config.get("verbosity", "INFO")),
            device=None,
            run_mode="dlc",
            dlc_config=dlc_config,
            model_tp=int(model_config.get("tp", DEFAULT_MODEL_TP)),
            max_model_len=int(model_config.get("max_model_len", DEFAULT_MAX_MODEL_LEN)),
            gpu_memory_utilization=float(model_config.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
            max_num_seqs=int(model_config.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS)),
            base_port=int(model_config.get("base_port", DEFAULT_BASE_PORT)),
            concurrency=int(eval_config.get("concurrency", DEFAULT_CONCURRENCY)),
            gen_kwargs=str(eval_config.get("gen_kwargs", DEFAULT_GEN_KWARGS)),
            enable_thinking=bool(model_config.get("enable_thinking", DEFAULT_ENABLE_THINKING)),
            debug=bool(eval_config.get("debug", False)),
        )

    env_dict = config.pop("env", {})
    env_vars = _dict_to_env_vars(env_dict) if env_dict else ""

    tasks_raw = config.get("tasks", "")
    if isinstance(tasks_raw, str):
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]
    elif isinstance(tasks_raw, list):
        tasks = tasks_raw
    else:
        tasks = []

    return ImportYamlResponse(
        job_name=DEFAULT_EVAL_JOB_NAME,
        eval_inference_mode=DEFAULT_EVAL_INFERENCE_MODE,
        model=config.get("model", ""),
        api_url=str(config.get("api_url") or DEFAULT_API_EVAL_URL),
        api_key=str(config.get("api_key") or ""),
        model_args=config.get("model_args", ""),
        tasks=tasks,
        judge_backend=_validate_judge_backend(str(config.get("judge_backend") or DEFAULT_JUDGE_BACKEND)),
        judge_api_url=str(config.get("judge_api_url") or ""),
        judge_api_key=str(config.get("judge_api_key") or ""),
        env_vars=env_vars,
        batch_size=config.get("batch_size", 1),
        limit=config.get("limit"),
        output_path=config.get("output_path", "./logs/"),
        log_samples=config.get("log_samples", False),
        verbosity=config.get("verbosity", "INFO"),
        device=config.get("device"),
        run_mode="local",
    )


@app.post("/eval/start", response_model=EvalStartResponse)
async def start_eval(request: EvalRequest, http_request: Request) -> EvalStartResponse:
    """Start an evaluation job."""
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No tasks specified")

    job_id = str(uuid.uuid4())
    auth_user = _require_authenticated_user(http_request)
    command = _build_command(request, auth_user=auth_user)
    shell_command = _build_shell_command(request, auth_user=auth_user)

    _jobs[job_id] = {
        "status": "starting",
        "command": shell_command,
        "process": None,
        "request": request,
    }

    return EvalStartResponse(job_id=job_id, command=command)


async def _stream_output(job_id: str):
    """Stream subprocess output as SSE events."""
    job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
        return

    shell_command = job["command"]

    try:
        process = await asyncio.create_subprocess_shell(
            shell_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            executable="/bin/bash",
        )
        job["process"] = process
        job["status"] = "running"

        if process.stdout:
            async for line in process.stdout:
                if job.get("stopped"):
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                yield f"data: {json.dumps({'type': 'output', 'line': decoded})}\n\n"

        await process.wait()
        exit_code = process.returncode

        if job.get("stopped"):
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done', 'exit_code': exit_code})}\n\n"

        job["status"] = "completed"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        job["status"] = "error"

    finally:
        job["process"] = None


@app.get("/eval/{job_id}/stream")
async def stream_eval(job_id: str):
    """Stream evaluation output via SSE."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        _stream_output(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/eval/{job_id}/stop")
async def stop_eval(job_id: str) -> dict[str, str]:
    """Stop a running evaluation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job["stopped"] = True
    process = job.get("process")

    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                process.terminate()
            except Exception:
                pass

    return {"status": "stopped"}


@app.get("/dlc/jobs", response_model=DlcJobsResponse)
async def list_dlc_jobs(
    http_request: Request,
    page_size: int = Query(100, ge=1, le=100),
    max_pages: int = Query(1, ge=1, le=20),
    status: str = Query(""),
    display_name: str = Query(_view_log_job_prefix_label()),
) -> DlcJobsResponse:
    """List DLC jobs from the configured DLC workspace."""
    auth_user = _require_authenticated_user(http_request)
    rows = await asyncio.to_thread(
        _list_dlc_jobs_from_cli,
        page_size=page_size,
        max_pages=max_pages,
        status=status.strip(),
        display_name=display_name.strip(),
    )
    jobs: list[DlcJobSummary] = []
    for row in rows:
        jobs.append(DlcJobSummary(**_with_dlc_job_kill_permission(row, auth_user)))
    return DlcJobsResponse(
        jobs=jobs,
        total=len(jobs),
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        source=f"{_resolve_dlc_binary()} workspace={DEFAULT_DLC_WORKSPACE_ID}",
    )


@app.get("/dlc/pool-usage", response_model=DlcPoolUsageResponse)
async def get_dlc_pool_usage() -> DlcPoolUsageResponse:
    """Return active DLC resource requests for the configured resource pool."""
    usage = await asyncio.to_thread(_build_dlc_pool_usage)
    return DlcPoolUsageResponse(**usage)


@app.post("/dlc/jobs/{job_id}/kill", response_model=DlcJobKillResponse)
async def kill_dlc_job(job_id: str, http_request: Request) -> DlcJobKillResponse:
    auth_user = _require_authenticated_user(http_request)
    detail = await asyncio.to_thread(_get_dlc_job_detail, job_id)
    _assert_dlc_job_kill_allowed(detail, auth_user)
    await asyncio.to_thread(
        _run_authenticated_dlc_command,
        ["stop", "job", job_id, "--force", "--quiet"],
        auth_user,
        timeout=DLC_STOP_TIMEOUT_SECONDS,
    )
    _clear_dlc_runtime_caches()
    return DlcJobKillResponse(
        job_id=job_id,
        status="kill_requested",
        message="DLC stop job command submitted",
    )


@app.get("/dlc/jobs/{job_id}", response_model=DlcJobDetailResponse)
async def get_dlc_job(job_id: str) -> DlcJobDetailResponse:
    detail = await asyncio.to_thread(_get_dlc_job_detail, job_id)
    if not _is_view_log_job_name(detail.get("DisplayName")):
        raise HTTPException(status_code=400, detail=f"DLC job name must start with {_view_log_job_prefix_label()}")
    runtime_config, result_root, log_dir = _job_runtime_paths(detail)
    return DlcJobDetailResponse(
        job=detail,
        result_root=str(result_root) if result_root else None,
        runtime_config_path=str(runtime_config) if runtime_config else None,
        log_dir=log_dir,
        result_status=_result_status_for_root(result_root),
    )


@app.get("/dlc/jobs/{job_id}/metrics", response_model=DlcMetricsResponse)
async def get_dlc_job_metrics(job_id: str) -> DlcMetricsResponse:
    detail = await asyncio.to_thread(_get_dlc_job_detail, job_id)
    if not _is_view_log_job_name(detail.get("DisplayName")):
        raise HTTPException(status_code=400, detail=f"DLC job name must start with {_view_log_job_prefix_label()}")
    _runtime_config, result_root, _log_dir = _job_runtime_paths(detail)
    if result_root is None:
        raise HTTPException(status_code=404, detail=_exact_result_resolution_error(detail))
    if not result_root.exists() or not result_root.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Exact local result directory does not exist for DLC job {job_id}: {result_root}",
        )
    metrics, summary_files = await asyncio.to_thread(_build_metric_rows, result_root)
    if not metrics:
        raise HTTPException(status_code=404, detail=_no_metric_rows_error(result_root))
    return DlcMetricsResponse(
        job_id=job_id,
        result_root=str(result_root),
        metrics=metrics,
        summary_files=summary_files,
        message="",
    )


@app.get("/dlc/jobs/{job_id}/metrics/{metric_id}/samples", response_model=DlcMetricSamplesResponse)
async def get_dlc_metric_samples(
    job_id: str,
    metric_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    only_wrong: bool = Query(False),
) -> DlcMetricSamplesResponse:
    detail = await asyncio.to_thread(_get_dlc_job_detail, job_id)
    if not _is_view_log_job_name(detail.get("DisplayName")):
        raise HTTPException(status_code=400, detail=f"DLC job name must start with {_view_log_job_prefix_label()}")
    _runtime_config, result_root, _log_dir = _job_runtime_paths(detail)
    if result_root is None or not result_root.exists() or not result_root.is_dir():
        raise HTTPException(status_code=404, detail=_exact_result_resolution_error(detail))

    metrics, _summary_files = await asyncio.to_thread(_build_metric_rows, result_root)
    if not metrics:
        raise HTTPException(status_code=404, detail=_no_metric_rows_error(result_root))
    selected = next((row for row in metrics if row.metric_id == metric_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Metric not found")
    if not selected.sample_jsonls:
        raise HTTPException(status_code=404, detail="Samples JSONL not found for this metric")

    rows, total, answer_stats = await asyncio.to_thread(
        _read_sample_jsonls,
        selected.sample_jsonls,
        job_id=job_id,
        metric_id=metric_id,
        metric_name=selected.metric_name,
        offset=offset,
        limit=limit,
        only_wrong=only_wrong,
    )
    columns = _preferred_sample_columns(rows)
    return DlcMetricSamplesResponse(
        job_id=job_id,
        metric_id=metric_id,
        columns=columns,
        rows=rows,
        total=total,
        offset=offset,
        limit=limit,
        sample_files=selected.sample_jsonls,
        answer_stats=answer_stats,
    )


@app.get("/dlc/jobs/{job_id}/metrics/{metric_id}/samples/media/{token}")
async def get_dlc_metric_sample_media(job_id: str, metric_id: str, token: str):
    _ = (job_id, metric_id)
    path = DLC_SAMPLE_MEDIA_TOKENS.get(token)
    if path is None:
        raise HTTPException(status_code=404, detail="Sample media token not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample media file not found")
    media_type = _image_file_media_type(path)
    if media_type is None:
        raise HTTPException(status_code=400, detail="Sample media is not a supported image")
    return FileResponse(
        path=str(path),
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/logs/runs", response_model=list[LogRunSummary])
async def list_log_runs(logs_path: str = Query("./logs/")) -> list[LogRunSummary]:
    logs_root = _resolve_logs_root(logs_path)
    if not logs_root.exists() or not logs_root.is_dir():
        return []

    runs: list[LogRunSummary] = []

    for results_file in logs_root.rglob("*_results.json"):
        resolved_file = results_file.resolve()
        try:
            _ensure_path_within_base(logs_root, resolved_file)
        except HTTPException:
            continue

        try:
            with resolved_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(result_data, dict):
            continue

        task_results = result_data.get("results")
        if not isinstance(task_results, dict):
            task_results = {}

        metrics: dict[str, dict[str, Any]] = {}
        for task_name, task_metrics in task_results.items():
            if not isinstance(task_metrics, dict):
                continue
            metrics[str(task_name)] = {str(metric_name): metric_value for metric_name, metric_value in task_metrics.items() if metric_name != "alias"}

        config = result_data.get("config")
        if not isinstance(config, dict):
            config = {}

        n_samples = result_data.get("n-samples")
        if not isinstance(n_samples, dict):
            n_samples = {}

        date = result_data.get("date")
        if date is None:
            date = resolved_file.stem.removesuffix("_results")

        model_name = result_data.get("model_name")
        if model_name is None:
            model_name = ""

        relative_path = resolved_file.relative_to(logs_root).as_posix()

        runs.append(
            LogRunSummary(
                run_id=quote(relative_path, safe=""),
                model_name=str(model_name),
                date=str(date),
                tasks=[str(task_name) for task_name in task_results.keys()],
                metrics=metrics,
                total_evaluation_time_seconds=result_data.get("total_evaluation_time_seconds"),
                config=config,
                n_samples=n_samples,
            )
        )

    runs.sort(key=lambda run: run.date, reverse=True)
    return runs


@app.get("/logs/runs/{run_id:path}/results")
async def get_log_run_results(
    run_id: str,
    logs_path: str = Query("./logs/"),
) -> dict[str, Any]:
    run_path = _resolve_run_results_path(logs_path, run_id)
    if not run_path.name.endswith("_results.json"):
        raise HTTPException(status_code=404, detail="Run results not found")
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(status_code=404, detail="Run results not found")

    try:
        with run_path.open("r", encoding="utf-8") as f:
            result_data = json.load(f)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Run results not found") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run results JSON is invalid") from exc

    if not isinstance(result_data, dict):
        raise HTTPException(status_code=500, detail="Run results must be a JSON object")

    return result_data


@app.get("/logs/runs/{run_id:path}/samples/{task_name}", response_model=LogSamplesResponse)
async def get_log_run_samples(
    run_id: str,
    task_name: str,
    logs_path: str = Query("./logs/"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> LogSamplesResponse:
    run_path = _resolve_run_results_path(logs_path, run_id)
    if not run_path.name.endswith("_results.json"):
        raise HTTPException(status_code=404, detail="Run results not found")
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(status_code=404, detail="Run results not found")
    if "/" in task_name or "\\" in task_name:
        raise HTTPException(status_code=400, detail="Invalid task name")

    run_stem = run_path.stem
    if not run_stem.endswith("_results"):
        raise HTTPException(status_code=404, detail="Run results not found")

    run_prefix = run_stem.removesuffix("_results")
    samples_path = run_path.with_name(f"{run_prefix}_samples_{task_name}.jsonl")

    if not samples_path.exists() or not samples_path.is_file():
        raise HTTPException(status_code=404, detail="Samples file not found")

    samples: list[dict[str, Any]] = []
    total = 0

    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(sample, dict):
                    continue

                if total >= offset and len(samples) < limit:
                    samples.append(sample)
                total += 1
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Samples file not found") from exc

    return LogSamplesResponse(samples=samples, total=total, offset=offset, limit=limit)


@app.get("/logs/runs/{run_id:path}/samples/{task_name}/media/{doc_id}")
async def get_log_run_sample_media(
    run_id: str,
    task_name: str,
    doc_id: int,
    logs_path: str = Query("./logs/"),
):
    run_path = _resolve_run_results_path(logs_path, run_id)
    if not run_path.name.endswith("_results.json"):
        raise HTTPException(status_code=404, detail="Run results not found")
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(status_code=404, detail="Run results not found")
    if "/" in task_name or "\\" in task_name:
        raise HTTPException(status_code=400, detail="Invalid task name")
    if doc_id < 0:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    mode, payload, media_type = await asyncio.to_thread(_resolve_dataset_media, task_name, doc_id)
    if mode == "file":
        return FileResponse(
            path=str(payload),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
