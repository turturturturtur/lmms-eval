"""lmms-eval judge subcommand: standalone judging from JSONL files.

This module provides a CLI interface for judging model outputs from JSONL files
without regeneration. It separates the generation and judging phases completely.

Usage:
    lmms-eval judge --input results.jsonl --task mathvision_reason_testmini
    lmms-eval judge -i "*.jsonl" --judge-model gpt-4o
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

from loguru import logger as eval_logger

SCIVQR_SUBJECTS = ["math", "physics", "chemistry", "biology", "geography", "astronomy"]
SCIVQR_TASKS = {"scivqr_mcq", "scivqr_open", "scivqr_reasoning"}
SCIVQR_DEFAULT_TESTED_MODEL = "InternVL3-8B-Instruct"
SCIVQR_DEFAULT_REASONING_PREDICTION_MODEL = "o1"
JudgeRunner = None
Aggregator = None
score_file = None


def _load_judge_runtime_objects(include_score: bool = False):
    global JudgeRunner, Aggregator, score_file
    if JudgeRunner is None:
        from lmms_eval.llm_judge.standalone import JudgeRunner as _JudgeRunner
        JudgeRunner = _JudgeRunner
    if Aggregator is None:
        from lmms_eval.llm_judge.aggregator import Aggregator as _Aggregator
        Aggregator = _Aggregator
    if include_score and score_file is None:
        from lmms_eval.llm_judge.scorer import score_file as _score_file
        score_file = _score_file
    return JudgeRunner, Aggregator, score_file


def _make_table(results: dict, key: str = "results") -> str:
    from lmms_eval.utils import make_table
    return make_table(results, key)


def _normalize_judge_mode(mode: Optional[str]) -> str:
    mode = (mode or "auto").lower()
    if mode in {"llm", "rule", "api"}:
        return "judge"
    if mode in {"auto", "judge", "score"}:
        return mode
    return "auto"


def add_judge_parser(subparsers):
    """Add judge subcommand to CLI."""
    parser = subparsers.add_parser(
        "judge",
        help="Judge model outputs from JSONL files without regeneration",
        description="""
Standalone judge command for evaluating model outputs from JSONL files.

This command separates the generation and judging phases, allowing you to:
1. Re-judge existing results with different criteria
2. Use LLM-as-judge for tasks that normally use rule-based judging
3. Batch process multiple result files

Examples:
    # Basic usage with auto-detected task from a single file
    lmms-eval judge --input_result results/model_samples_task.jsonl

    # Specify single task explicitly
    lmms-eval judge --input_result results.jsonl -t mathvision_reason_testmini

    # Judge multiple tasks from a directory
    lmms-eval judge -i /path/to/results/ -t mathvision_test,wemath_testmini_reasoning

    # Use LLM judge
    lmms-eval judge -i results.jsonl --judge-model gpt-4o

    # Batch process with output directory
    lmms-eval judge --input_result "results/*.jsonl" -d judged/ --parallel 8
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_result", "-i",
        required=True,
        help="Path to JSONL result file(s). Supports wildcards (*.jsonl)",
    )
    parser.add_argument(
        "--task", "-t",
        default="auto-detect",
        help="Task name(s) for loading process_results. Use comma-separated list for multiple tasks (e.g., 'task1,task2'). Use 'auto-detect' to infer from filename(s). When multiple tasks are given, --input_result should be a directory.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file path (single file mode)",
    )
    parser.add_argument(
        "--output-dir", "-d",
        help="Output directory (batch mode)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        help="Judge model name (default: from JUDGE_MODEL env var or gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_API_KEY"),
        help="API key for judge model (default: from JUDGE_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_API_URL") or "",
        help=(
            "Base URL for judge API. "
            "For local vLLM/SGLang: http://localhost:8000/v1 "
            "(default: from JUDGE_BASE_URL env or OpenAI default)"
        ),
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=int(os.getenv("JUDGE_MAX_CONCURRENT", "1")),
        help="Number of parallel judge workers (default: from JUDGE_MAX_CONCURRENT env or 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "judge", "score"],
        default=_normalize_judge_mode(os.getenv("JUDGE_MODE", "auto")),
        help=(
            "Operation mode. 'judge' runs LLM-as-judge (legacy behavior). "
            "'score' re-runs task.process_results + aggregation from saved JSONL without inference. "
            "'auto' detects request-failure markers and falls back to 'score' when needed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without saving results",
    )
    parser.add_argument(
        "--scivqr-reasoning-batch",
        action="store_true",
        default=os.getenv("SCIVQR_REASONING_BATCH", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the official SciVQR reasoning OpenAI Batch workflow. "
            "Writes uploads/{model}/requests_chunk*.jsonl, "
            "results/{model}_results/output_chunk*.ndjson, and "
            "results/{model}_results/Evaluation-Chunk*.json under --output-dir."
        ),
    )
    parser.add_argument(
        "--scivqr-split-id",
        type=int,
        default=int(os.getenv("SCIVQR_SPLIT_ID", "0")),
        help="SciVQR reasoning split id for official chunked Batch evaluation.",
    )
    parser.add_argument(
        "--scivqr-num-chunk",
        type=int,
        default=int(os.getenv("SCIVQR_NUM_CHUNK", "1")),
        help="SciVQR reasoning total chunk count for official Batch evaluation.",
    )
    parser.add_argument(
        "--scivqr-reasoning-prediction-model",
        default=os.getenv("SCIVQR_REASONING_PREDICTION_MODEL", os.getenv("SCIVQR_TESTED_MODEL", SCIVQR_DEFAULT_REASONING_PREDICTION_MODEL)),
        help="Prediction model directory/name used in official SciVQR reasoning uploads/results paths.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.set_defaults(func=run_judge, subcommand="judge", judge_mode=os.getenv("JUDGE_MODE", "auto"))


def _detect_task_from_filename(filename: str) -> str:
    """Extract task name from samples filename.
    
    Example patterns:
        - '20240328_samples_mathvision_reason_testmini.jsonl' -> 'mathvision_reason_testmini'
        - 'model_Qwen_samples_mmmu_val.jsonl' -> 'mmmu_val'
        - 'samples_wemath_testmini.jsonl' -> 'wemath_testmini'
    """
    # Remove .jsonl extension
    name = filename.replace(".jsonl", "")
    
    # Try to find _samples_ pattern
    if "_samples_" in name:
        parts = name.split("_samples_")
        if len(parts) >= 2:
            return parts[1]
    
    # Try to find just samples_ pattern
    if "samples_" in name:
        parts = name.split("samples_")
        if len(parts) >= 2:
            return parts[1]
    
    raise ValueError(
        f"Cannot auto-detect task from filename: {filename}. "
        f"Expected pattern: *_samples_{{task}}.jsonl"
    )


def _is_scivqr_task(task_name: str) -> bool:
    return task_name in SCIVQR_TASKS


def _scivqr_subject_from_path(path: Path) -> Optional[str]:
    suffix = "_results"
    stem = path.stem
    if stem.endswith(suffix):
        subject = stem[: -len(suffix)]
        if subject in SCIVQR_SUBJECTS:
            return subject
    return None


def _resolve_scivqr_official_files(input_path: Path) -> List[Path]:
    """Return official SciVQR subject result files in official script order."""
    return [
        input_path / f"{subject}_results.jsonl"
        for subject in SCIVQR_SUBJECTS
        if (input_path / f"{subject}_results.jsonl").is_file()
    ]


def _detect_mode_from_files(judge_items: List[Tuple[str, Path]]) -> str:
    """Auto-detect whether we should run in judge or score mode.
    
    If any sample contains a request-failure marker (e.g. HTTP 404),
    we need to recompute metrics from scratch => score mode.
    Otherwise we assume the JSONL already has valid per-sample metrics
    and we can proceed with judge/aggregation => judge mode.
    """
    import json
    for _task_name, input_file in judge_items:
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for _ in range(5):  # inspect first 5 lines only
                    line = f.readline()
                    if not line:
                        break
                    sample = json.loads(line)
                    resp = sample.get("filtered_resps", "")
                    if isinstance(resp, str) and "[LMMS_EVAL_REQUEST_FAILED" in resp:
                        return "score"
                    # Also switch to score if all existing metrics are 0/NaN for multiple samples
        except Exception:
            continue
    return "judge"


def _expand_group_tasks(task_list: List[str]) -> List[str]:
    """Expand lmms-eval group names into leaf tasks."""
    if task_list == ["auto-detect"]:
        return task_list
    try:
        from lmms_eval.tasks import get_task_dict
        from lmms_eval.evaluator_utils import get_subtask_list

        def _collect_leaf_tasks(subtasks):
            leaves = []
            for name, children in subtasks.items():
                if not children:
                    leaves.append(name)
                else:
                    leaves.extend(children)
            return leaves

        expanded_task_list = []
        for task_name in task_list:
            try:
                task_dict = get_task_dict(task_name)
                subtasks = get_subtask_list(task_dict)
                leaves = _collect_leaf_tasks(subtasks)
                if leaves:
                    expanded_task_list.extend(leaves)
                else:
                    expanded_task_list.append(task_name)
            except Exception:
                expanded_task_list.append(task_name)
        return expanded_task_list
    except Exception as e:
        eval_logger.debug(f"Failed to expand group tasks: {e}")
        return task_list


def _resolve_input_files(input_result: str, task_list: List[str]) -> List[Tuple[str, Path]]:
    """Resolve input files for given tasks.
    
    Mimics the evaluation framework's multi-task selection by allowing
    comma-separated task names. When multiple tasks are provided,
    --input_result is treated as a directory and matching files are
    auto-discovered using the *samples_<task>.jsonl pattern.
    
    Returns:
        List of (task_name, input_file_path) tuples.
        task_name may be "auto-detect" for wildcard/directory modes.
    """
    input_path = Path(input_result)

    # Case 1: Wildcard pattern
    if "*" in input_result:
        files = [Path(p) for p in sorted(glob.glob(input_result))]
        if not files:
            raise ValueError(f"No files found matching pattern: {input_result}")
        if task_list != ["auto-detect"]:
            if len(task_list) != 1:
                raise ValueError(
                    "Wildcard input with explicit tasks supports exactly one task. "
                    "Please use a directory for multiple tasks."
                )
            return [(task_list[0], f) for f in files]
        # Auto-detect task from filename when wildcards are used without --task
        return [("auto-detect", f) for f in files]

    # Case 2: Single file
    if input_path.is_file():
        if len(task_list) > 1:
            raise ValueError(
                f"Multiple tasks specified but --input_result is a single file. "
                f"Please provide a directory or use a single task."
            )
        return [(task_list[0], input_path)]

    # Case 3: Directory
    if input_path.is_dir():
        result = []
        if task_list == ["auto-detect"]:
            files = sorted(input_path.rglob("*samples_*.jsonl"))
            if not files:
                if _resolve_scivqr_official_files(input_path):
                    raise ValueError(
                        f"Found SciVQR official *_results.jsonl files in {input_path}. "
                        "Please specify --task scivqr_mcq, scivqr_open, or scivqr_reasoning."
                    )
                raise ValueError(f"No *samples_*.jsonl files found in directory: {input_path}")
            for f in files:
                try:
                    task = _detect_task_from_filename(f.name)
                    result.append((task, f))
                except ValueError:
                    eval_logger.warning(f"Skipping file with unrecognized pattern: {f.name}")
            return result
        else:
            for task in task_list:
                if _is_scivqr_task(task):
                    official_files = _resolve_scivqr_official_files(input_path)
                    if official_files:
                        result.extend((task, f) for f in official_files)
                        continue
                pattern = f"*samples_{task}.jsonl"
                files = sorted(input_path.rglob(pattern))
                if not files:
                    raise ValueError(
                        f"No file found for task: {task} (pattern: {pattern}) in directory: {input_path}"
                    )
                # Pick the latest file by mtime (same logic as shell script)
                latest = max(files, key=lambda p: p.stat().st_mtime)
                result.append((task, latest))
            return result

    raise ValueError(f"Input path not found: {input_path}")


def _scivqr_tested_model() -> str:
    return os.getenv("SCIVQR_TESTED_MODEL", SCIVQR_DEFAULT_TESTED_MODEL)


def _scivqr_reasoning_chunk_name() -> str:
    split_id = os.getenv("SCIVQR_SPLIT_ID", "0")
    return f"Evaluation-Chunk{split_id}.json"


def _get_output_path(input_file: Path, output: Optional[str], output_dir: Optional[str], task_name: Optional[str] = None) -> Path:
    """Determine output file path."""
    if output:
        return Path(output)
    if output_dir:
        out_dir = Path(output_dir)
        if task_name == "scivqr_open":
            out_dir = out_dir / _scivqr_tested_model()
        elif task_name == "scivqr_reasoning":
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / _scivqr_reasoning_chunk_name()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / input_file.name
    # Default: add _judged suffix
    return input_file.parent / f"{input_file.stem}_judged.jsonl"


def _result_json_path_for_output(output_path: Path) -> Path:
    """Return the WebUI-readable result JSON path next to a judged samples JSONL."""
    stem = output_path.stem
    if "_samples_" in stem:
        prefix = stem.split("_samples_", 1)[0]
    else:
        prefix = stem
    return output_path.with_name(f"{prefix}_results.json")


def _write_judge_results_json(
    *,
    output_path: Path,
    task_name: str,
    summary: dict,
    n_samples: int,
    input_file: Path,
    judge_model: str,
    effective_mode: str,
) -> Path:
    """Write aggregate judge metrics in the same shape as lmms-eval result JSON."""
    result_json = _result_json_path_for_output(output_path)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            task_name: {str(key): value for key, value in summary.items()}
        },
        "n-shot": {task_name: " "},
        "higher_is_better": {task_name: {}},
        "n-samples": {
            task_name: {
                "original": n_samples,
                "effective": n_samples,
            }
        },
        "judge": {
            "mode": effective_mode,
            "model": judge_model,
            "input_file": str(input_file),
            "samples_file": str(output_path),
        },
    }
    with result_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    eval_logger.info(f"Saved judge aggregate results to {result_json}")
    return result_json


def _defer_scivqr_reasoning_save(task_name: str, output: Optional[str], output_dir: Optional[str], item_counts: Counter) -> bool:
    return task_name == "scivqr_reasoning" and output is None and output_dir is not None and item_counts[task_name] > 1


def _display_task_name(task_name: str, input_file: Path, item_counts: Counter) -> str:
    if item_counts[task_name] <= 1:
        return task_name
    subject = _scivqr_subject_from_path(input_file)
    suffix = subject if subject else input_file.stem
    return f"{task_name}/{suffix}"


def _expand_scivqr_subject_summary(task_name: str, summary: dict) -> dict:
    if task_name != "scivqr_mcq":
        return summary
    subject_accuracy = summary.get("subject_accuracy")
    if isinstance(subject_accuracy, dict):
        summary = dict(summary)
        for subject in SCIVQR_SUBJECTS:
            if subject in subject_accuracy:
                summary[subject] = subject_accuracy[subject]
        summary.pop("subject_accuracy", None)
    return summary


def _write_scivqr_mcq_metrics_json(input_result: str, output_dir: Optional[str], summary: dict) -> None:
    subject_accuracy = summary.get("subject_accuracy")
    if not isinstance(subject_accuracy, dict):
        return
    metrics = {
        subject: subject_accuracy[subject]
        for subject in SCIVQR_SUBJECTS
        if subject in subject_accuracy
    }
    if not metrics:
        return

    out_dir = Path(output_dir) if output_dir else Path(input_result)
    if output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.is_dir():
        return
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    eval_logger.info(f"Wrote SciVQR official metrics to {out_file}")


def _load_jsonl_rows(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_openai_chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url[: -len("/chat/completions")]
    return base_url


def _scivqr_reasoning_batch_paths(output_dir: Optional[str], prediction_model: str, split_id: int):
    root = Path(output_dir) if output_dir else Path(".")
    requests_jsonl = root / "uploads" / prediction_model / f"requests_chunk{split_id}.jsonl"
    result_ndjson = root / "results" / f"{prediction_model}_results" / f"output_chunk{split_id}.ndjson"
    result_json = root / "results" / f"{prediction_model}_results" / f"Evaluation-Chunk{split_id}.json"
    return requests_jsonl, result_ndjson, result_json


def _run_scivqr_reasoning_batch(args: argparse.Namespace, judge_items: List[Tuple[str, Path]]) -> None:
    """Run the official SciVQR reasoning OpenAI Batch workflow."""
    from openai import OpenAI

    from lmms_eval.tasks import TaskManager
    from lmms_eval.tasks.scivqr.reasoning import utils as reasoning_utils

    batch_items = []
    for task_name, input_file in judge_items:
        if task_name == "auto-detect":
            task_name = _detect_task_from_filename(input_file.name)
        if task_name != "scivqr_reasoning":
            raise ValueError("--scivqr-reasoning-batch can only be used with --task scivqr_reasoning")
        batch_items.append((task_name, input_file))

    task_manager = TaskManager(verbosity="DEBUG" if args.verbose else "INFO")
    task_dict = task_manager.load_task_or_group("scivqr_reasoning")
    task = list(task_dict.values())[0]
    if hasattr(task, "eval_docs_no_media"):
        dataset_docs = list(task.eval_docs_no_media)
    elif task.has_test_docs():
        dataset_docs = list(task.test_docs())
    else:
        dataset_docs = list(task.validation_docs())

    samples = []
    for _task_name, input_file in batch_items:
        subject = _scivqr_subject_from_path(input_file)
        for row in _load_jsonl_rows(input_file):
            if subject and not row.get("subject"):
                row = dict(row)
                row["subject"] = subject
                row["__scivqr_subject_from_path"] = True
            samples.append(row)

    data = reasoning_utils.build_official_reasoning_items(samples, dataset_docs)
    prediction_model = getattr(args, "scivqr_reasoning_prediction_model", SCIVQR_DEFAULT_REASONING_PREDICTION_MODEL)
    split_id = getattr(args, "scivqr_split_id", int(os.getenv("SCIVQR_SPLIT_ID", "0")))
    num_chunk = getattr(args, "scivqr_num_chunk", int(os.getenv("SCIVQR_NUM_CHUNK", "1")))
    requests_jsonl, result_ndjson, result_json = _scivqr_reasoning_batch_paths(args.output_dir, prediction_model, split_id)

    meta = reasoning_utils.write_official_reasoning_batch_requests(
        data,
        requests_jsonl,
        split_id=split_id,
        num_chunk=num_chunk,
    )
    eval_logger.info(
        f"Wrote SciVQR reasoning Batch requests to {requests_jsonl} "
        f"({len(meta['split_data'])}/{len(data)} samples)"
    )

    if args.dry_run:
        return

    api_key = args.judge_api_key or os.getenv("OPENAI_API_KEY") or ""
    base_url = _strip_openai_chat_completions_url(args.judge_base_url or os.getenv("OPENAI_API_URL") or "")
    client = OpenAI(api_key=api_key, base_url=base_url)

    batch_id = reasoning_utils.submit_official_reasoning_batch(client, requests_jsonl)
    eval_logger.info(f"Submitted SciVQR reasoning Batch job: {batch_id}")
    batch = reasoning_utils.wait_for_official_reasoning_batch(
        client,
        batch_id,
        interval=int(os.getenv("SCIVQR_BATCH_POLL_INTERVAL", "10")),
    )
    reasoning_utils.download_official_reasoning_batch_results(client, batch, result_ndjson)
    eval_logger.info(f"Downloaded SciVQR reasoning Batch results to {result_ndjson}")
    official_results = reasoning_utils.write_official_reasoning_results_from_ndjson(
        meta["data"],
        meta["id_mapping"],
        result_ndjson,
        result_json,
        start=meta["start"],
        end=meta["end"],
    )
    eval_logger.info(f"Wrote SciVQR official reasoning results to {result_json} ({len(official_results)} rows)")


def _build_results_dict(task_name: str, summary: dict) -> dict:
    """Build a results dict compatible with make_table."""
    return {
        "results": {
            task_name: {
                f"{k}": v for k, v in summary.items()
            }
        },
        "n-shot": {task_name: " "},
        "higher_is_better": {task_name: {}},
    }


def run_judge(args: argparse.Namespace) -> None:
    """Execute judge command."""

    def _setup_logger():
        """Configure logging to match the framework style."""
        eval_logger.remove()
        # Check if colors should be disabled (for clean log files)
        use_color = os.environ.get('NO_COLOR', '') == '' and os.environ.get('LOGURU_NO_COLOR', '') == ''
        if use_color:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
        else:
            log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        log_level = "DEBUG" if args.verbose else "INFO"
        eval_logger.add(sys.stdout, colorize=use_color, level=log_level, format=log_format)

    _setup_logger()

    # Import here to avoid heavy imports during CLI parsing
    try:
        JudgeRunner, Aggregator, _score_file = _load_judge_runtime_objects()
    except ImportError as e:
        eval_logger.error(f"Failed to import JudgeRunner: {e}")
        eval_logger.error("Please ensure lmms-eval is installed: pip install -e .")
        sys.exit(1)

    # Some sub-modules (e.g. lmms_eval.models) reset the global loguru logger
    # during their first import. Re-configure after heavy imports are done.
    _setup_logger()

    # Parse task list (mimics evaluation framework's --tasks comma separation)
    if args.task == "auto-detect":
        task_list = ["auto-detect"]
    else:
        task_list = [t.strip() for t in args.task.split(",") if t.strip()]
    if not task_list:
        eval_logger.error("No tasks specified.")
        sys.exit(1)

    task_list = _expand_group_tasks(task_list)

    # Resolve input files for the requested tasks
    try:
        judge_items = _resolve_input_files(args.input_result, task_list)
    except ValueError as e:
        eval_logger.error(str(e))
        sys.exit(1)

    if not judge_items:
        eval_logger.error("No files to judge.")
        sys.exit(1)

    eval_logger.info(f"Found {len(judge_items)} file(s) to judge")
    for task_name, input_file in judge_items:
        eval_logger.info(f"  [{task_name}] -> {input_file}")
    item_counts = Counter(task_name for task_name, _ in judge_items)

    if getattr(args, "scivqr_reasoning_batch", False):
        try:
            _run_scivqr_reasoning_batch(args, judge_items)
        except Exception as e:
            eval_logger.error(f"SciVQR reasoning Batch evaluation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        return

    # Determine effective mode for auto
    effective_mode = _normalize_judge_mode(getattr(args, "mode", getattr(args, "judge_mode", "auto")))
    if effective_mode == "auto":
        effective_mode = _detect_mode_from_files(judge_items)
        eval_logger.info(f"Auto-detected mode: {effective_mode}")

    runner = None
    if effective_mode == "judge":
        runner = JudgeRunner(
            judge_mode="judge",
            judge_model=args.judge_model,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            parallel=args.parallel,
        )
        eval_logger.info(
            f"judge ({args.input_result}), judge_mode: (judge), "
            f"judge_model: ({args.judge_model}), parallel: {args.parallel}"
        )
    else:
        eval_logger.info(
            f"score ({args.input_result}), output_dir: ({args.output_dir})"
        )

    # Process each file
    success_count = 0
    error_count = 0
    all_summaries = []
    merged_judge_results = defaultdict(list)

    for task_name, input_file in judge_items:
        # Auto-detect task from filename if needed
        if task_name == "auto-detect":
            try:
                task_name = _detect_task_from_filename(input_file.name)
            except ValueError as e:
                eval_logger.error(f"{e}. Use --task to specify explicitly.")
                error_count += 1
                continue

        try:
            if effective_mode == "score":
                _JudgeRunner, _Aggregator, score_file = _load_judge_runtime_objects(include_score=True)
                # Offline re-scoring: re-run process_results + aggregation
                output_dir = Path(args.output_dir) if args.output_dir else None
                results_dict, _ = score_file(
                    input_file,
                    task_name,
                    output_path=output_dir,
                    verbose=args.verbose,
                )
                if results_dict and task_name in results_dict.get("results", {}):
                    summary = dict(results_dict["results"][task_name])
                    display_name = _display_task_name(task_name, input_file, item_counts)
                    all_summaries.append((display_name, summary))
                success_count += 1
            else:
                # Run judging
                results = runner.judge_file(input_file, task_name)
                if item_counts[task_name] > 1:
                    merged_judge_results[task_name].extend(results)

                # Compute summary
                summary = runner.compute_summary(results)

                # For tasks with special aggregation (e.g. MMBench), run the
                # task-specific aggregator so that metrics like accuracy reflect
                # the true scoring logic rather than the generic per-sample mean.
                try:
                    agg = Aggregator()
                    agg_summary = agg.aggregate(results, task_name)
                    if agg_summary:
                        summary.update(agg_summary)
                except Exception as e:
                    eval_logger.debug(f"Task-specific aggregation failed for {task_name}: {e}")

                if summary:
                    display_name = _display_task_name(task_name, input_file, item_counts)
                    all_summaries.append((display_name, _expand_scivqr_subject_summary(task_name, summary)))

                # Save results
                if not args.dry_run and not _defer_scivqr_reasoning_save(task_name, args.output, args.output_dir, item_counts):
                    output_path = _get_output_path(input_file, args.output, args.output_dir, task_name)
                    runner.save_results(results, output_path)
                    if summary:
                        _write_judge_results_json(
                            output_path=output_path,
                            task_name=task_name,
                            summary=_expand_scivqr_subject_summary(task_name, summary),
                            n_samples=len(results),
                            input_file=input_file,
                            judge_model=args.judge_model,
                            effective_mode=effective_mode,
                        )

                success_count += 1

        except Exception as e:
            eval_logger.error(f"Error processing {input_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1

    if effective_mode == "judge":
        for task_name, merged_results in merged_judge_results.items():
            if not merged_results:
                continue
            try:
                summary = runner.compute_summary(merged_results)
                try:
                    agg = Aggregator()
                    agg_summary = agg.aggregate(merged_results, task_name)
                    if agg_summary:
                        summary.update(agg_summary)
                except Exception as e:
                    eval_logger.debug(f"Merged task-specific aggregation failed for {task_name}: {e}")
                metrics_summary = dict(summary)
                display_summary = _expand_scivqr_subject_summary(task_name, summary)
                all_summaries.append((task_name, display_summary))
                if task_name == "scivqr_mcq" and not args.dry_run:
                    _write_scivqr_mcq_metrics_json(args.input_result, args.output_dir, metrics_summary)
                elif task_name == "scivqr_reasoning" and not args.dry_run and args.output is None and args.output_dir:
                    runner._current_task_name = task_name
                    runner._current_task = runner._load_task(task_name)
                    output_path = _get_output_path(Path(_scivqr_reasoning_chunk_name()), args.output, args.output_dir, task_name)
                    runner.save_results(merged_results, output_path)
                    if summary:
                        _write_judge_results_json(
                            output_path=output_path,
                            task_name=task_name,
                            summary=display_summary,
                            n_samples=len(merged_results),
                            input_file=Path(args.input_result),
                            judge_model=args.judge_model,
                            effective_mode=effective_mode,
                        )
            except Exception as e:
                eval_logger.debug(f"Merged summary failed for {task_name}: {e}")

    # Inject group-level summaries for hierarchical display
    def _load_group_map():
        import yaml
        tasks_dir = Path(__file__).parent.parent / "tasks"
        group_map = {}
        for yaml_file in tasks_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict) and "group" in data and "task" in data:
                    members = [str(t) for t in data["task"] if isinstance(t, str)]
                    if members:
                        group_map[str(data["group"])] = members
            except Exception:
                continue
        return group_map

    def _inject_group_rows(summaries):
        group_map = _load_group_map()
        task_index = {name: (idx, summary) for idx, (name, summary) in enumerate(summaries)}
        grouped_tasks = set()
        new_rows = []

        for group_name, members in group_map.items():
            member_present = []
            for m in members:
                if m in task_index:
                    member_present.append(m)
            if not member_present:
                continue

            # Aggregate numeric metrics from member summaries.
            # We look for the first usable float per summary (prefer exact "score",
            # then keys ending with ".score", then any numeric value).
            scores = []
            for m in member_present:
                s = task_index[m][1]
                val = None
                if "score" in s and isinstance(s["score"], (int, float)):
                    val = float(s["score"])
                else:
                    for k, v in s.items():
                        if k.endswith(".score") and isinstance(v, (int, float)):
                            val = float(v)
                            break
                        elif isinstance(v, (int, float)):
                            val = float(v)
                            break
                if val is not None:
                    scores.append(val)
            if scores:
                group_summary = {"score": round(sum(scores) / len(scores), 4)}
            else:
                group_summary = {}

            # Insert group header + indented members + group total
            new_rows.append((group_name, group_summary))
            for m in member_present:
                grouped_tasks.add(m)
                orig_summary = task_index[m][1]
                new_rows.append((f"  {m}", orig_summary))

        # Append any tasks that are not part of a group
        for name, summary in summaries:
            if name not in grouped_tasks:
                new_rows.append((name, summary))

        return new_rows

    all_summaries = _inject_group_rows(all_summaries)

    # Log results in the same style as normal evaluation
    if all_summaries:
        combined_results = {}
        combined_nshot = {}
        combined_hib = {}
        for task_name, summary in all_summaries:
            combined_results[task_name] = {f"{k}": v for k, v in summary.items()}
            combined_nshot[task_name] = " "
            combined_hib[task_name] = {}
        combined_dict = {
            "results": combined_results,
            "n-shot": combined_nshot,
            "higher_is_better": combined_hib,
        }
        eval_logger.info("\n" + _make_table(combined_dict))

    if error_count > 0:
        sys.exit(1)
