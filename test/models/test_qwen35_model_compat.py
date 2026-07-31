import json
import os
from pathlib import Path

import pytest

from lmms_eval.models.model_utils import qwen35_model_compat


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_qwen35_checkpoint(
    root: Path,
    *,
    nested_video_processor: bool,
    independent_video_processor: bool = False,
) -> Path:
    root.mkdir()
    _write_json(root / "config.json", {"model_type": "qwen3_5"})
    _write_json(root / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write_json(root / "tokenizer.json", {"version": "1.0"})
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}},
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"fixture-weights")
    processor = {"processor_class": "Qwen3_5Processor"}
    if nested_video_processor:
        processor["video_processor"] = {
            "video_processor_type": "Qwen3VLVideoProcessor",
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "size": {"shortest_edge": 4096, "longest_edge": 16384000},
        }
    _write_json(root / "processor_config.json", processor)
    if independent_video_processor:
        _write_json(
            root / "video_preprocessor_config.json",
            {
                "video_processor_type": "Qwen3VLVideoProcessor",
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 16384000},
            },
        )
    return root


def test_prepare_creates_read_only_view_from_nested_video_processor(tmp_path, monkeypatch):
    source = _write_qwen35_checkpoint(
        tmp_path / "checkpoint-227",
        nested_video_processor=True,
    )
    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in source.iterdir()
    }
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )

    result = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="regression",
    )

    resolved = Path(result["resolved_path"])
    assert resolved != source.resolve()
    assert result["source_path"] == str(source.resolve())
    assert result["compatibility"] == "qwen35_video_processor_view"
    assert resolved.name.startswith("q35v-")
    assert len(str(resolved)) + 70 <= 255
    assert result["processor_class"] == "Qwen3VLVideoProcessor"
    assert (resolved / "config.json").is_symlink()
    assert (resolved / "config.json").resolve() == (source / "config.json").resolve()
    generated = json.loads((resolved / "video_preprocessor_config.json").read_text())
    assert generated["video_processor_type"] == "Qwen3VLVideoProcessor"
    assert (resolved / "SOURCE_CHECKPOINT").read_text().strip() == str(source.resolve())
    manifest = json.loads(
        (resolved / qwen35_model_compat.VIEW_MANIFEST_NAME).read_text()
    )
    assert manifest["source_manifest_sha256"] == result["source_manifest_sha256"]
    after = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in source.iterdir()
    }
    assert after == before


def test_prepare_reuses_already_compatible_checkpoint(tmp_path, monkeypatch):
    source = _write_qwen35_checkpoint(
        tmp_path / "base",
        nested_video_processor=False,
        independent_video_processor=True,
    )
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )

    result = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="compatible",
    )

    assert result["resolved_path"] == str(source.resolve())
    assert result["compatibility"] == "native_video_processor_config"
    assert not (tmp_path / "views").exists()


def test_prepare_rejects_missing_video_processor_type(tmp_path):
    source = _write_qwen35_checkpoint(
        tmp_path / "invalid",
        nested_video_processor=False,
    )
    _write_json(source / "processor_config.json", {"video_processor": {}})

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="video_processor_type",
    ):
        qwen35_model_compat.prepare_model(
            source,
            tmp_path / "views",
            run_id="invalid",
        )


def test_inspect_rejects_missing_weight_shard(tmp_path):
    source = _write_qwen35_checkpoint(
        tmp_path / "missing-shard",
        nested_video_processor=True,
    )
    (source / "model-00001-of-00001.safetensors").unlink()

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="referenced weight shard",
    ):
        qwen35_model_compat.inspect_model(source)


def test_prepare_rejects_existing_mismatched_view(tmp_path, monkeypatch):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )
    first = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="same-run",
    )
    manifest_path = Path(first["resolved_path"]) / qwen35_model_compat.VIEW_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["source_path"] = "/different/source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="existing compatibility view",
    ):
        qwen35_model_compat.prepare_model(
            source,
            tmp_path / "views",
            run_id="same-run",
        )


def test_check_rejects_tampered_generated_video_config(tmp_path, monkeypatch):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    real_check = qwen35_model_compat.check_model
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )
    prepared = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="tamper",
    )
    resolved = Path(prepared["resolved_path"])
    (resolved / "video_preprocessor_config.json").write_text(
        '{"video_processor_type":"Tampered"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(qwen35_model_compat, "check_model", real_check)

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="hash mismatch",
    ):
        qwen35_model_compat.check_model(resolved)


def test_check_rejects_missing_generated_video_config_with_contract_error(
    tmp_path,
    monkeypatch,
):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )
    prepared = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="missing-generated",
    )
    resolved = Path(prepared["resolved_path"])
    (resolved / "video_preprocessor_config.json").unlink()
    monkeypatch.undo()

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="missing generated video processor config",
    ):
        qwen35_model_compat.check_model(resolved)


def test_prepare_rejects_view_root_inside_source_without_mutation(tmp_path):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    nested_root = source / "views"

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="must not be inside the source checkpoint",
    ):
        qwen35_model_compat.prepare_model(
            source,
            nested_root,
            run_id="nested",
        )

    assert not nested_root.exists()


def test_prepare_rejects_normalized_view_root_inside_source_without_mutation(
    tmp_path,
):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    nested_root = tmp_path / "missing-parent" / ".." / source.name / "views"

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="must not be inside the source checkpoint",
    ):
        qwen35_model_compat.prepare_model(
            source,
            nested_root,
            run_id="normalized-nested",
        )

    assert not (source / "views").exists()
    assert not (tmp_path / "missing-parent").exists()


def test_prepare_rechecks_view_root_after_creation_without_publishing_inside_source(
    tmp_path,
    monkeypatch,
):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    inside_root = source / "injected-views"
    inside_root.mkdir()
    alias_root = tmp_path / "outside-alias"
    alias_root.symlink_to(inside_root, target_is_directory=True)
    real_canonical = qwen35_model_compat._canonical_future_directory
    calls = 0

    def simulate_symlink_swap(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Path(path)
        return real_canonical(path)

    monkeypatch.setattr(
        qwen35_model_compat,
        "_canonical_future_directory",
        simulate_symlink_swap,
    )

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="must not be inside the source checkpoint",
    ):
        qwen35_model_compat.prepare_model(
            source,
            alias_root,
            run_id="symlink-swap",
        )

    assert calls >= 2
    assert list(inside_root.iterdir()) == []


def test_check_rejects_compatibility_view_symlink_target_drift(
    tmp_path,
    monkeypatch,
):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )
    prepared = qwen35_model_compat.prepare_model(
        source,
        tmp_path / "views",
        run_id="drift",
    )
    resolved = Path(prepared["resolved_path"])
    shard = resolved / "model-00001-of-00001.safetensors"
    evil = tmp_path / "same-size-evil.safetensors"
    evil.write_bytes(b"x" * len(b"fixture-weights"))
    shard.unlink()
    shard.symlink_to(evil)
    monkeypatch.undo()

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="symlink target mismatch",
    ):
        qwen35_model_compat.check_model(resolved)


def test_prepare_canonicalizes_symlinked_view_root(tmp_path, monkeypatch):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    real_root = tmp_path / "real-views"
    real_root.mkdir()
    alias_root = tmp_path / "alias-views"
    alias_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(
        qwen35_model_compat,
        "check_model",
        lambda path: {
            "processor_class": "Qwen3VLVideoProcessor",
            "resolved_path": str(Path(path).resolve()),
        },
    )

    first = qwen35_model_compat.prepare_model(
        source,
        alias_root,
        run_id="canonical",
    )
    second = qwen35_model_compat.prepare_model(
        source,
        real_root,
        run_id="canonical",
    )

    assert Path(first["resolved_path"]).parent == real_root.resolve()
    assert first["resolved_path"] == second["resolved_path"]


def test_source_manifest_changes_when_weight_stat_changes(tmp_path):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    before = qwen35_model_compat.inspect_model(source)
    shard = source / "model-00001-of-00001.safetensors"
    stat = shard.stat()
    shard.write_bytes(b"x" * stat.st_size)
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    after = qwen35_model_compat.inspect_model(source)

    assert before["source_manifest_kind"] == qwen35_model_compat.SOURCE_MANIFEST_KIND
    assert before["source_manifest_sha256"] != after["source_manifest_sha256"]


def test_prepare_rejects_view_path_too_long_for_modelscope_lock(tmp_path):
    source = _write_qwen35_checkpoint(
        tmp_path / "source",
        nested_video_processor=True,
    )
    long_root = tmp_path / ("v" * 180)

    with pytest.raises(
        qwen35_model_compat.ModelCompatibilityError,
        match="cache lock filename",
    ):
        qwen35_model_compat.prepare_model(
            source,
            long_root,
            run_id="long",
        )

    assert not long_root.exists()
