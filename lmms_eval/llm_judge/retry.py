"""Validate and copy completed inference into a separate judge retry run."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


def validate_source(runtime_path: Path) -> tuple[dict, Path, list[dict]]:
    config = json.loads(runtime_path.read_text())
    settings = config["eval"]
    root = (Path(settings["output_path"]) / settings["timestamp"]).resolve()
    tasks = settings["tasks"].split(",")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Source task list is empty or contains duplicates")
    with (root / "task_status.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(tasks) or {row["task"] for row in rows} != set(tasks):
        raise ValueError("Source inference manifest does not cover every task exactly once")
    for row in rows:
        task = row["task"]
        if row["status"] != "success" or row["exit_code"] != "0":
            raise ValueError(f"Source inference is incomplete: {task}")
        directory = Path(row["output_path"]).resolve()
        if not directory.is_relative_to(root / "tasks") or directory == root / "tasks":
            raise ValueError(f"Source task path is outside its run: {task}")
        files = list(directory.rglob("*_results.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one inference result for {task}")
        counts = json.loads(files[0].read_text())["n-samples"][task]
        original = int(counts["original"])
        limit = settings.get("limit", -1)
        expected = original if limit in (None, -1) else min(int(limit), original)
        sample_path = files[0].with_name(files[0].name.replace("_results.json", f"_samples_{task}.jsonl"))
        with sample_path.open() as stream:
            ids = [json.loads(line)["doc_id"] for line in stream if line.strip()]
        if expected < 1 or counts["effective"] != expected or len(ids) != expected or set(ids) != set(range(expected)):
            raise ValueError(f"Source sample coverage is incomplete: {task}")
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError(f"Source task artifacts must not contain symlinks: {task}")
    return config, root, rows


def prepare_retry(runtime_path: Path) -> Path:
    config = json.loads(runtime_path.read_text())
    source_path = Path(config["eval"]["judge_only_source_config"])
    source, source_root, rows = validate_source(source_path)
    for key in ("tasks", "limit", "gen_kwargs"):
        if source["eval"].get(key) != config["eval"].get(key):
            raise ValueError(f"Judge retry must preserve source eval.{key}")
    model = lambda data: data["model"].get("source_path") or data["model"]["path"]
    if model(source) != model(config):
        raise ValueError("Judge retry must preserve the source model")
    target = (Path(config["eval"]["output_path"]) / config["eval"]["timestamp"]).resolve()
    if target == source_root or target.is_relative_to(source_root) or source_root.is_relative_to(target):
        raise ValueError("Judge retry output must be separate from source inference")
    if target.exists():
        raise ValueError("Judge retry output already exists; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".judge-retry-", dir=target.parent))
    try:
        for row in rows:
            old = Path(row["output_path"]).resolve()
            relative = old.relative_to(source_root)
            shutil.copytree(old, staging / relative)
            row["output_path"] = str(target / relative)
        with (staging / "task_status.tsv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "inference_source.json").write_text(json.dumps({
            "source_job_id": config["eval"].get("judge_only_source_job_id"),
            "runtime_config": str(source_path), "result_root": str(source_root),
            "tasks": [row["task"] for row in rows],
        }, indent=2) + "\n")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_config", type=Path)
    args = parser.parse_args()
    print(f"Reused validated inference at {prepare_retry(args.runtime_config)}")
