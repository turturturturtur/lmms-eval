"""Shared storage contract for the WebUI and its DLC submitters."""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any

QUOTA_ID = "quota1cauvwqouvn"
WORKSPACE_ID = "247143"
CPFSA_URI = "bmcpfs://cpfs-29001wm47tsv6qpysnl0y-vpc-tm0qfl.cn-wulanchabu.cpfs.aliyuncs.com/user/tianleniu/"
CPFSA_MOUNT = "/mnt/cpfsA/user/tianleniu"
CPFSA_ACCESS_POINT = "ap-1sqxfm3pf4vkkcqec"
CPFSA_ROLE = "acs:ram::1245621648665098:role/tianleniu-role"
NASA_URI = "nas://2926de4a653-hnb54.cn-wulanchabu.nas.aliyuncs.com/"
NASA_MOUNT = "/mnt/nasA"
BENCHMARK_CACHE = "/mnt/nasA/tianleniu/evaluation_cache/lmms_eval"
HF_HOME = "/mnt/nasA/tianleniu/evaluation_cache/huggingface"
DEFAULT_MOUNTS = f"{CPFSA_URI}::{CPFSA_MOUNT},{NASA_URI}::{NASA_MOUNT}"


def readable_path(value: str | Path) -> Path:
    """Resolve exact historical storage prefixes without changing saved records."""
    path = str(Path(value).expanduser())
    path = re.sub(r"^/mnt/nasB(?=/|$)", "/mnt/nasA", path)
    for legacy in ("/mnt/cpfsB", "/mnt/cpfs", "/mnt/workspace"):
        if not path.startswith(legacy + "/"):
            continue
        relative = path[len(legacy) + 1:]
        if relative.startswith("evaluation_cache/"):
            path = "/mnt/nasA/tianleniu/" + relative
        elif re.match(r"[^/]+/Innovator-Tune/lmms-eval/(?:eval_result|eval_logs|eval_model_views|judge_logs|judge_results)(?:/|$)", relative):
            path = "/mnt/nasA/" + relative
        elif re.match(r"[^/]+/(?:models|lmms_eval_views)(?:/|$)", relative):
            path = "/mnt/nasA/" + relative
        else:
            path = "/mnt/cpfsA/user/" + relative
        break
    result = Path(path).resolve()
    validate_paths(str(result), "read path")
    return result


def data_sources(value: str) -> list[dict[str, str]]:
    """Validate both the filesystem identity and the mount location.

    Only A storage is accepted, including when B is disguised under an A path.
    The access point exposes /user/tianleniu as its root, so mounting it at
    CPFSA_MOUNT preserves the exact same paths as the development machine.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("data_source_uris is required")
    expected = {CPFSA_URI: CPFSA_MOUNT, NASA_URI: NASA_MOUNT}
    result = []
    for item in value.split(","):
        uri, separator, mount = item.strip().partition("::")
        if not separator or ":" in mount or uri not in expected:
            raise ValueError("Only the configured CPFS A and NAS A data sources are allowed")
        mount = posixpath.normpath(mount)
        if mount != expected[uri] or any(row["Uri"] == uri for row in result):
            raise ValueError(f"Invalid or duplicate mount for {uri}; expected {expected[uri]}")
        row = {"Uri": uri, "MountPath": mount}
        if uri == CPFSA_URI:
            row.update(AccessPointId=CPFSA_ACCESS_POINT, RoleArn=CPFSA_ROLE)
        result.append(row)
    if {row["Uri"] for row in result} != set(expected):
        raise ValueError(f"Both CPFS A ({CPFSA_MOUNT}) and NAS A ({NASA_MOUNT}) are required")
    return result


def validate_paths(value: Any, field: str = "config") -> None:
    """Reject stale B paths, including aliases that resolve back to B storage."""
    if isinstance(value, dict):
        for key, child in value.items():
            validate_paths(child, f"{field}.{key}")
    elif isinstance(value, list):
        for child in value:
            validate_paths(child, field)
    elif isinstance(value, str):
        for match in re.finditer(r"/(?:mnt|tmp|home)/[^\s,\"'=;]+", value):
            path = match.group().rstrip("/")
            resolved = str(Path(path).resolve())
            for candidate in (posixpath.normpath(path), resolved):
                if re.match(r"^/mnt/(?:cpfsB|nasB|cpfs|workspace)(?:/|$)", candidate):
                    raise ValueError(f"{field} uses unavailable B/legacy storage: {path}; use CPFS A or NAS A")


def validate_config(config: dict) -> None:
    validate_paths(config)
    dlc = config.get("dlc")
    if not isinstance(dlc, dict):
        return
    for field, section in (("dlc", dlc), ("dlc.judge", dlc.get("judge"))):
        if not isinstance(section, dict):
            continue
        if section.get("resource_id") != QUOTA_ID:
            raise ValueError(f"{field}.resource_id must be {QUOTA_ID}")
        data_sources(section.get("data_source_uris"))


if __name__ == "__main__":
    import sys

    try:
        for filename in sys.argv[1:]:
            validate_config(json.loads(Path(filename).read_text()))
    except (ValueError, OSError) as error:
        sys.exit(f"[ERROR] {error}")
