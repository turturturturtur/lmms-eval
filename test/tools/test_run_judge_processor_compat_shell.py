import json
import os
import subprocess
from pathlib import Path


LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
RUN_JUDGE = LMMS_EVAL_ROOT / "run_scripts" / "run_judge.sh"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_judge_rejects_raw_qwen35_checkpoint_before_vllm_launch(
    tmp_path: Path,
):
    model = tmp_path / "checkpoint-raw"
    model.mkdir()
    _write_json(model / "config.json", {"model_type": "qwen3_5"})
    _write_json(model / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write_json(model / "tokenizer.json", {"version": "1.0"})
    _write_json(
        model / "model.safetensors.index.json",
        {"weight_map": {"weight": "model-00001-of-00001.safetensors"}},
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _write_json(
        model / "processor_config.json",
        {
            "video_processor": {
                "video_processor_type": "Qwen3VLVideoProcessor",
            }
        },
    )

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("", encoding="utf-8")
    config = {
        "env": {
            "venv_path": str(venv),
            "hf_home": str(tmp_path / "hf"),
        },
        "log": {"dir": str(tmp_path / "logs")},
        "judge": {
            "backend": "vllm",
            "parallel": 1,
            "model": "checkpoint-raw",
            "vllm": {
                "model_path": str(model),
                "processor_compat": "required",
                "tp": 1,
                "max_model_len": 1024,
                "gpu_memory_utilization": "0.5",
                "max_num_seqs": 1,
                "port": 18002,
            },
        },
        "eval": {
            "input_result_path": str(tmp_path / "missing-input"),
            "tasks": "ai2d",
            "output_path": str(tmp_path / "output"),
        },
    }
    config_path = tmp_path / "judge.json"
    _write_json(config_path, config)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(LMMS_EVAL_ROOT)
    result = subprocess.run(
        ["bash", str(RUN_JUDGE), str(config_path)],
        cwd=LMMS_EVAL_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Qwen3.5 judge model check failed before vLLM launch" in result.stdout
    assert "missing" in result.stdout
    assert "video_preprocessor_config.json" in result.stdout
    assert "Starting vLLM judge backend" not in result.stdout
