import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lmms_eval import __main__ as cli_main
from lmms_eval.models.model_utils import qwen35_model_compat


def _args(
    *,
    model: str = "vllm",
    model_args: str,
    mode: str,
    view_root: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        model_args=model_args,
        model_processor_compat=mode,
        model_view_root=str(view_root) if view_root is not None else None,
    )


def test_cli_compat_off_does_not_parse_or_touch_model_args(monkeypatch):
    monkeypatch.setattr(
        qwen35_model_compat,
        "prepare_model",
        lambda *_args, **_kwargs: pytest.fail("compat module must not be called"),
    )
    args = _args(
        model_args="this is deliberately not a valid key=value string",
        mode="off",
        view_root=None,
    )

    cli_main._resolve_cli_model_processor_compat(args)

    assert args.model_args == "this is deliberately not a valid key=value string"
    assert args.model_artifact is None


def test_cli_required_rewrites_model_as_dict_without_losing_json(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    resolved = tmp_path / "views" / "resolved"
    observed: dict[str, object] = {}

    def fake_prepare(source_arg, view_root_arg, *, run_id):
        observed.update(
            source=source_arg,
            view_root=view_root_arg,
            run_id=run_id,
        )
        return {
            "source_path": str(source.resolve()),
            "resolved_path": str(resolved.resolve()),
            "model_type": "qwen3_5",
            "processor_class": "Qwen3VLVideoProcessor",
            "source_manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr(qwen35_model_compat, "prepare_model", fake_prepare)
    args = _args(
        model_args=(
            f'model={source},'
            'hf_overrides={"architectures":["A","B"],"nested":{"x":1}},'
            "stop_token_ids=[1,2]"
        ),
        mode="required",
        view_root=tmp_path / "views",
    )

    cli_main._resolve_cli_model_processor_compat(args)

    assert args.model_args["model"] == str(resolved.resolve())
    assert (
        args.model_args["hf_overrides"]
        == '{"architectures":["A","B"],"nested":{"x":1}}'
    )
    assert args.model_args["stop_token_ids"] == "[1,2]"
    assert args.model_artifact["source_path"] == str(source.resolve())
    assert observed["run_id"] == "cli"


def test_cli_required_rejects_remote_model_id(tmp_path: Path):
    args = _args(
        model_args="model=Qwen/Qwen3.5-9B",
        mode="required",
        view_root=tmp_path / "views",
    )

    with pytest.raises(ValueError, match="local model directory"):
        cli_main._resolve_cli_model_processor_compat(args)


def test_cli_auto_leaves_non_qwen_local_model_unchanged(tmp_path: Path):
    source = tmp_path / "llama"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "llama"}),
        encoding="utf-8",
    )
    raw_model_args = f"model={source},dtype=bfloat16"
    args = _args(
        model_args=raw_model_args,
        mode="auto",
        view_root=tmp_path / "views",
    )

    cli_main._resolve_cli_model_processor_compat(args)

    assert args.model_args == raw_model_args
    assert args.model_artifact is None
    assert not (tmp_path / "views").exists()


def test_cli_compat_rejects_duplicate_model_key(tmp_path: Path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    args = _args(
        model_args=f"model={source},model={source}",
        mode="required",
        view_root=tmp_path / "views",
    )

    with pytest.raises(ValueError, match="duplicate model_args key"):
        cli_main._resolve_cli_model_processor_compat(args)


def test_cli_compat_requires_absolute_view_root(tmp_path: Path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    args = _args(
        model_args=f"model={source}",
        mode="required",
        view_root=Path("relative/views"),
    )

    with pytest.raises(ValueError, match="absolute path"):
        cli_main._resolve_cli_model_processor_compat(args)


def test_cli_required_preflight_failure_returns_nonzero(tmp_path: Path):
    missing_model = tmp_path / "missing-qwen35"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lmms_eval",
            "--model",
            "vllm",
            "--model_args",
            f"model={missing_model}",
            "--model_processor_compat",
            "required",
            "--model_view_root",
            str(tmp_path / "views"),
            "--tasks",
            "list",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "requires a local model directory" in result.stdout
