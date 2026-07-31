"""Strict Qwen3.5 checkpoint compatibility checks for lmms-eval/vLLM.

Transformers 5.x can save the video processor inside ``processor_config.json``,
while the Transformers version used by the production lmms-eval environment
expects an independent ``video_preprocessor_config.json``.  This module creates
an immutable, symlink-based compatibility view without modifying the source
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
VIEW_MANIFEST_NAME = ".qwen35_processor_compat_manifest.json"
SOURCE_RECORD_NAME = "SOURCE_CHECKPOINT"
GENERATED_VIDEO_CONFIG_NAME = "video_preprocessor_config.json"
COMPATIBILITY_VIEW = "qwen35_video_processor_view"
COMPATIBILITY_NATIVE = "native_video_processor_config"
SOURCE_MANIFEST_KIND = "sha256-small-files-and-weight-stat-v1"
_MODELSCOPE_LOCK_NAME_OVERHEAD = 70
_FILESYSTEM_NAME_MAX = 255
_GENERATED_NAMES = frozenset(
    {VIEW_MANIFEST_NAME, SOURCE_RECORD_NAME, GENERATED_VIDEO_CONFIG_NAME}
)
_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class ModelCompatibilityError(ValueError):
    """Raised when a local Qwen3.5 checkpoint violates the runtime contract."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ModelCompatibilityError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ModelCompatibilityError(f"{label} is empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelCompatibilityError(
            f"{label} must contain a JSON object, got {type(payload).__name__}: {path}"
        )
    return payload


def _resolve_source(source: str | os.PathLike[str]) -> Path:
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ModelCompatibilityError(
            f"model source does not exist or cannot be resolved: {candidate}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise ModelCompatibilityError(f"model source must be a directory: {resolved}")
    return resolved


def _validate_weight_index(source: Path) -> tuple[Path, list[Path]]:
    index_path = source / "model.safetensors.index.json"
    index = _load_json_object(index_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelCompatibilityError(
            f"weight_map must be a non-empty object: {index_path}"
        )

    shard_names: set[str] = set()
    for tensor_name, raw_shard in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ModelCompatibilityError(
                f"weight_map contains an invalid tensor name: {tensor_name!r}"
            )
        if not isinstance(raw_shard, str) or not raw_shard:
            raise ModelCompatibilityError(
                f"weight_map[{tensor_name!r}] must reference a non-empty shard path"
            )
        relative = Path(raw_shard)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelCompatibilityError(
                f"weight_map[{tensor_name!r}] escapes the model directory: {raw_shard}"
            )
        shard_names.add(raw_shard)

    is_compatibility_view = (source / VIEW_MANIFEST_NAME).is_file()
    shards: list[Path] = []
    for shard_name in sorted(shard_names):
        shard = source / shard_name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise ModelCompatibilityError(
                f"missing or empty referenced weight shard: {shard}"
            )
        if shard.is_symlink():
            if not is_compatibility_view:
                raise ModelCompatibilityError(
                    "source checkpoint weight shards must not be symlinks; "
                    f"only generated compatibility views may link them: {shard}"
                )
        else:
            try:
                shard.resolve(strict=True).relative_to(source)
            except (OSError, ValueError) as exc:
                raise ModelCompatibilityError(
                    f"referenced weight shard escapes the model directory: {shard}"
                ) from exc
        shards.append(shard)
    return index_path, shards


def get_local_model_type(source: str | os.PathLike[str]) -> str:
    """Return the explicit model_type from a strict local model directory."""

    source_path = _resolve_source(source)
    config = _load_json_object(source_path / "config.json", "config.json")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ModelCompatibilityError(
            f"model_type must be a non-empty string: {source_path / 'config.json'}"
        )
    return model_type


def _source_manifest_sha256(
    source: Path,
    *,
    config_path: Path,
    tokenizer_config_path: Path,
    tokenizer_path: Path,
    weight_index_path: Path,
    shards: list[Path],
    processor_path: Path,
) -> str:
    digest = hashlib.sha256()
    content_paths = (
        config_path,
        tokenizer_config_path,
        tokenizer_path,
        weight_index_path,
        processor_path,
    )
    for path in content_paths:
        relative = path.relative_to(source).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    for shard in shards:
        stat = shard.stat()
        digest.update(shard.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        for value in (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ):
            digest.update(str(value).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def inspect_model(source: str | os.PathLike[str]) -> dict[str, Any]:
    """Inspect a local Qwen3.5 checkpoint without creating or changing files."""

    source_path = _resolve_source(source)
    config_path = source_path / "config.json"
    tokenizer_config_path = source_path / "tokenizer_config.json"
    tokenizer_path = source_path / "tokenizer.json"
    config = _load_json_object(config_path, "config.json")
    _load_json_object(tokenizer_config_path, "tokenizer_config.json")
    _load_json_object(tokenizer_path, "tokenizer.json")

    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ModelCompatibilityError(
            "Qwen3.5 compatibility handling only supports model_type "
            f"{sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}: {config_path}"
        )

    weight_index_path, shards = _validate_weight_index(source_path)
    native_video_path = source_path / GENERATED_VIDEO_CONFIG_NAME
    nested_processor_path = source_path / "processor_config.json"
    if native_video_path.is_file():
        video_processor = _load_json_object(
            native_video_path,
            GENERATED_VIDEO_CONFIG_NAME,
        )
        processor_path = native_video_path
        compatibility = COMPATIBILITY_NATIVE
    else:
        processor_config = _load_json_object(
            nested_processor_path,
            "processor_config.json",
        )
        video_processor = processor_config.get("video_processor")
        if not isinstance(video_processor, dict):
            raise ModelCompatibilityError(
                "processor_config.json must contain a video_processor object "
                f"when {GENERATED_VIDEO_CONFIG_NAME} is absent: {nested_processor_path}"
            )
        processor_path = nested_processor_path
        compatibility = COMPATIBILITY_VIEW

    processor_class = video_processor.get("video_processor_type")
    if not isinstance(processor_class, str) or not processor_class.strip():
        raise ModelCompatibilityError(
            f"video_processor_type must be a non-empty string: {processor_path}"
        )

    manifest_sha256 = _source_manifest_sha256(
        source_path,
        config_path=config_path,
        tokenizer_config_path=tokenizer_config_path,
        tokenizer_path=tokenizer_path,
        weight_index_path=weight_index_path,
        shards=shards,
        processor_path=processor_path,
    )
    return {
        "source_path": str(source_path),
        "model_type": model_type,
        "processor_class": processor_class,
        "processor_config_path": str(processor_path),
        "compatibility": compatibility,
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_manifest_sha256": manifest_sha256,
        "weight_shards": [str(path) for path in shards],
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify_compatibility_view(resolved_path: Path) -> dict[str, Any] | None:
    manifest_path = resolved_path / VIEW_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = _load_json_object(manifest_path, "compatibility view manifest")
    if manifest.get("compatibility") != COMPATIBILITY_VIEW:
        raise ModelCompatibilityError(
            f"invalid compatibility marker in {manifest_path}: "
            f"{manifest.get('compatibility')!r}"
        )
    if manifest.get("resolved_path") != str(resolved_path):
        raise ModelCompatibilityError(
            f"compatibility view resolved_path mismatch in {manifest_path}: "
            f"{manifest.get('resolved_path')!r} != {str(resolved_path)!r}"
        )

    raw_source = manifest.get("source_path")
    if not isinstance(raw_source, str) or not raw_source:
        raise ModelCompatibilityError(
            f"compatibility view manifest has invalid source_path: {manifest_path}"
        )
    source_path = _resolve_source(raw_source)
    source_record_path = resolved_path / SOURCE_RECORD_NAME
    if not source_record_path.is_file():
        raise ModelCompatibilityError(
            f"compatibility view is missing {SOURCE_RECORD_NAME}: {resolved_path}"
        )
    if source_record_path.read_text(encoding="utf-8").strip() != str(source_path):
        raise ModelCompatibilityError(
            f"compatibility view {SOURCE_RECORD_NAME} does not match manifest source_path: "
            f"{source_record_path}"
        )
    _verify_view_links(resolved_path, source_path)

    generated_path = resolved_path / GENERATED_VIDEO_CONFIG_NAME
    if not generated_path.is_file():
        raise ModelCompatibilityError(
            "compatibility view is missing generated video processor config: "
            f"{generated_path}"
        )
    generated_sha256 = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    if manifest.get("generated_video_config_sha256") != generated_sha256:
        raise ModelCompatibilityError(
            "compatibility view generated video processor hash mismatch: "
            f"{generated_path}"
        )
    source_inspection = inspect_model(source_path)
    if (
        manifest.get("source_manifest_sha256")
        != source_inspection["source_manifest_sha256"]
    ):
        raise ModelCompatibilityError(
            f"compatibility view source manifest hash mismatch: {manifest_path}"
        )
    return manifest


def check_model(resolved: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the video processor using the current evaluation environment."""

    resolved_path = _resolve_source(resolved)
    inspection = inspect_model(resolved_path)
    view_manifest = _verify_compatibility_view(resolved_path)
    if not (resolved_path / GENERATED_VIDEO_CONFIG_NAME).is_file():
        raise ModelCompatibilityError(
            "resolved Qwen3.5 model is not runtime-compatible: missing "
            f"{resolved_path / GENERATED_VIDEO_CONFIG_NAME}"
        )

    try:
        from transformers import AutoVideoProcessor

        processor = AutoVideoProcessor.from_pretrained(
            str(resolved_path),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise ModelCompatibilityError(
            "AutoVideoProcessor preflight failed for "
            f"{resolved_path} with transformers={_package_version('transformers')}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    loaded_class = processor.__class__.__name__
    if loaded_class != inspection["processor_class"]:
        raise ModelCompatibilityError(
            "video processor class mismatch for "
            f"{resolved_path}: declared={inspection['processor_class']!r}, "
            f"loaded={loaded_class!r}"
        )
    result = {
        **inspection,
        "resolved_path": str(resolved_path),
        "processor_class": loaded_class,
        "transformers_version": _package_version("transformers"),
        "vllm_version": _package_version("vllm"),
    }
    if view_manifest is not None:
        result.update(view_manifest)
    return result


def _normalise_run_id(run_id: str) -> str:
    cleaned = _SAFE_RUN_ID.sub("-", run_id.strip()).strip(".-")
    if not cleaned:
        raise ModelCompatibilityError(
            f"run_id must contain at least one safe character, got {run_id!r}"
        )
    if len(cleaned) > 80:
        raise ModelCompatibilityError(
            f"run_id must be at most 80 safe characters, got {len(cleaned)}"
        )
    return cleaned


def _read_view_manifest(target: Path) -> dict[str, Any]:
    return _load_json_object(target / VIEW_MANIFEST_NAME, "compatibility view manifest")


def _canonical_future_directory(path: Path) -> Path:
    missing_parts: list[str] = []
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise ModelCompatibilityError(
                f"cannot resolve an existing ancestor for view_root: {path}"
            )
        missing_parts.append(probe.name)
        probe = probe.parent
    if not probe.is_dir():
        raise ModelCompatibilityError(
            f"view_root has a non-directory ancestor: {probe}"
        )
    resolved = probe.resolve(strict=True)
    for part in reversed(missing_parts):
        resolved /= part
    return resolved


def _require_view_root_outside_source(root: Path, source_path: Path) -> None:
    try:
        root.relative_to(source_path)
    except ValueError:
        return
    raise ModelCompatibilityError(
        "view_root must not be inside the source checkpoint: "
        f"source={source_path}, view_root={root}"
    )


def _require_target_path_length(target: Path) -> None:
    if len(str(target)) + _MODELSCOPE_LOCK_NAME_OVERHEAD > _FILESYSTEM_NAME_MAX:
        raise ModelCompatibilityError(
            "resolved compatibility view path is too long for the vLLM/ModelScope "
            "cache lock filename; choose a shorter model.view_root: "
            f"path_length={len(str(target))}, maximum="
            f"{_FILESYSTEM_NAME_MAX - _MODELSCOPE_LOCK_NAME_OVERHEAD}, "
            f"target={target}"
        )


def _verify_view_links(resolved_path: Path, source_path: Path) -> None:
    source_entries = {entry.name: entry for entry in source_path.iterdir()}
    view_entries = {
        entry.name: entry
        for entry in resolved_path.iterdir()
        if entry.name not in _GENERATED_NAMES
    }
    if set(view_entries) != set(source_entries):
        raise ModelCompatibilityError(
            "compatibility view entry set does not match source checkpoint: "
            f"view={resolved_path}, source={source_path}, "
            f"missing={sorted(set(source_entries) - set(view_entries))}, "
            f"unexpected={sorted(set(view_entries) - set(source_entries))}"
        )
    for name, source_entry in source_entries.items():
        view_entry = view_entries[name]
        if not view_entry.is_symlink():
            raise ModelCompatibilityError(
                f"compatibility view entry must be a symlink: {view_entry}"
            )
        try:
            observed_target = view_entry.resolve(strict=True)
            expected_target = source_entry.resolve(strict=True)
        except OSError as exc:
            raise ModelCompatibilityError(
                f"compatibility view contains an unresolvable symlink: {view_entry}: {exc}"
            ) from exc
        if observed_target != expected_target:
            raise ModelCompatibilityError(
                "compatibility view symlink target mismatch: "
                f"entry={view_entry}, expected={expected_target}, "
                f"observed={observed_target}"
            )


def _validate_existing_view(
    target: Path,
    *,
    source_path: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = _read_view_manifest(target)
    expected = {
        "source_path": str(source_path),
        "source_manifest_sha256": source_manifest_sha256,
        "compatibility": COMPATIBILITY_VIEW,
    }
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise ModelCompatibilityError(
            "existing compatibility view does not match the requested source: "
            f"target={target}, expected={expected}, observed={observed}"
        )
    checked = check_model(target)
    return {**checked, **manifest}


def prepare_model(
    source: str | os.PathLike[str],
    view_root: str | os.PathLike[str],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Return a verified native checkpoint or create an immutable compat view."""

    inspection = inspect_model(source)
    source_path = Path(inspection["source_path"])
    if inspection["compatibility"] == COMPATIBILITY_NATIVE:
        checked = check_model(source_path)
        return {**inspection, **checked}

    root = Path(view_root).expanduser()
    if not root.is_absolute():
        raise ModelCompatibilityError(f"view_root must be an absolute path: {root}")
    root = Path(os.path.normpath(root))
    root = _canonical_future_directory(root)
    _require_view_root_outside_source(root, source_path)

    safe_run_id = _normalise_run_id(run_id)
    fingerprint = inspection["source_manifest_sha256"]
    run_fingerprint = hashlib.sha256(safe_run_id.encode("utf-8")).hexdigest()[:8]
    target_name = f"q35v-{fingerprint[:12]}-{run_fingerprint}"
    _require_target_path_length(root / target_name)

    root.mkdir(parents=True, exist_ok=True)
    root = _canonical_future_directory(root)
    _require_view_root_outside_source(root, source_path)
    target = root / target_name
    _require_target_path_length(target)
    if target.exists():
        return _validate_existing_view(
            target,
            source_path=source_path,
            source_manifest_sha256=fingerprint,
        )

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root)
    )
    try:
        for entry in source_path.iterdir():
            if entry.name in _GENERATED_NAMES:
                raise ModelCompatibilityError(
                    "source checkpoint contains a reserved compatibility-view file: "
                    f"{entry}"
                )
            if entry.is_symlink() and not entry.exists():
                raise ModelCompatibilityError(
                    f"source checkpoint contains a broken symlink: {entry}"
                )
            if not (entry.is_file() or entry.is_dir() or entry.is_symlink()):
                raise ModelCompatibilityError(
                    f"source checkpoint contains an unsupported special file: {entry}"
                )
            (temporary / entry.name).symlink_to(
                entry.resolve(strict=True),
                target_is_directory=entry.is_dir(),
            )

        nested_processor = _load_json_object(
            source_path / "processor_config.json",
            "processor_config.json",
        ).get("video_processor")
        if not isinstance(nested_processor, dict):
            raise ModelCompatibilityError(
                "processor_config.json must contain a video_processor object: "
                f"{source_path / 'processor_config.json'}"
            )
        generated_path = temporary / GENERATED_VIDEO_CONFIG_NAME
        generated_path.write_text(
            json.dumps(nested_processor, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / SOURCE_RECORD_NAME).write_text(
            f"{source_path}\n",
            encoding="utf-8",
        )
        manifest = {
            "source_path": str(source_path),
            "resolved_path": str(target),
            "model_type": inspection["model_type"],
            "processor_class": inspection["processor_class"],
            "compatibility": COMPATIBILITY_VIEW,
            "source_manifest_kind": SOURCE_MANIFEST_KIND,
            "source_manifest_sha256": fingerprint,
            "generated_video_config_sha256": hashlib.sha256(
                generated_path.read_bytes()
            ).hexdigest(),
        }
        (temporary / VIEW_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if target.exists():
            return _validate_existing_view(
                target,
                source_path=source_path,
                source_manifest_sha256=fingerprint,
            )
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            return _validate_existing_view(
                target,
                source_path=source_path,
                source_manifest_sha256=fingerprint,
            )
        temporary = None
        checked = check_model(target)
        return {**checked, **manifest}
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _json_dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, prepare, and verify Qwen3.5 processor compatibility."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--source", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--view-root", required=True)
    prepare_parser.add_argument("--run-id", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--model", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            payload = inspect_model(args.source)
        elif args.command == "prepare":
            payload = prepare_model(
                args.source,
                args.view_root,
                run_id=args.run_id,
            )
        elif args.command == "check":
            payload = check_model(args.model)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except ModelCompatibilityError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    _json_dump(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
