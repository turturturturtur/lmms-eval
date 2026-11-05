import csv
import hashlib
import json
from pathlib import Path

import pytest

from lmms_eval.llm_judge.retry import prepare_retry, validate_source


@pytest.fixture
def inference(tmp_path):
    root = tmp_path / "source" / "timestamp"
    task_dir = root / "tasks" / "mmmu_val_qwen3_official"
    task_dir.mkdir(parents=True)
    task = task_dir.name
    (task_dir / "run_results.json").write_text(json.dumps({"n-samples": {task: {"original": 2, "effective": 2}}}))
    (task_dir / f"run_samples_{task}.jsonl").write_text('\n'.join(json.dumps({"doc_id": i, "resps": [["A"]]}) for i in range(2)) + '\n')
    with (root / "task_status.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "status", "exit_code", "output_path"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"task": task, "status": "success", "exit_code": 0, "output_path": str(task_dir)})
    config = {"model": {"path": "/model", "source_path": "/model"},
              "eval": {"tasks": task, "limit": -1, "gen_kwargs": "", "output_path": str(root.parent), "timestamp": root.name}}
    path = tmp_path / "source_runtime.json"
    path.write_text(json.dumps(config))
    return path, config, root


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_retry_copies_complete_inference_without_old_judge_or_source_changes(inference, tmp_path):
    source_path, source, root = inference
    (root / "judge").mkdir()
    (root / "judge" / "incomplete.json").write_text('{}')
    before = fingerprint(root)
    source["eval"].update(output_path=str(tmp_path / "retry"), judge_only_source_config=str(source_path), judge_only_source_job_id="dlcsource")
    runtime = tmp_path / "retry_runtime.json"
    runtime.write_text(json.dumps(source))
    target = prepare_retry(runtime)
    assert fingerprint(root) == before
    assert not (target / "judge").exists()
    copied, copied_root, rows = validate_source(runtime)
    assert copied_root == target
    assert rows[0]["output_path"].startswith(str(target))
    assert json.loads((target / "inference_source.json").read_text())["source_job_id"] == "dlcsource"
    with pytest.raises(ValueError, match="already exists"):
        prepare_retry(runtime)


@pytest.mark.parametrize("corruption", ["failed", "missing_task", "duplicate_sample", "missing_sample", "outside_path"])
def test_retry_rejects_incomplete_or_unrelated_artifacts(inference, tmp_path, corruption):
    path, config, root = inference
    manifest = root / "task_status.tsv"
    sample = next(root.rglob("*_samples_*.jsonl"))
    if corruption == "failed":
        manifest.write_text(manifest.read_text().replace("success", "failed"))
    elif corruption == "missing_task":
        config["eval"]["tasks"] += ",ocrbench"
        path.write_text(json.dumps(config))
    elif corruption == "duplicate_sample":
        sample.write_text('{"doc_id": 0}\n{"doc_id": 0}\n')
    elif corruption == "missing_sample":
        sample.write_text('{"doc_id": 0}\n')
    else:
        manifest.write_text(manifest.read_text().replace(str(root / "tasks"), str(tmp_path / "unrelated")))
    with pytest.raises(ValueError):
        validate_source(path)


def test_retry_preserves_generation_recipe(inference, tmp_path):
    path, config, _ = inference
    config["eval"].update(output_path=str(tmp_path / "retry"), judge_only_source_config=str(path), limit=1)
    retry_path = tmp_path / "retry.json"
    retry_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="preserve source eval.limit"):
        prepare_retry(retry_path)
