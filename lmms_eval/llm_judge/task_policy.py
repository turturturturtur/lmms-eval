"""Explicit task policy for the two-stage judge pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskJudgePolicy:
    execution_stage: str
    standalone_task: str


_GROUP_EXPANSIONS = {
    "mathvista_testmini": (
        "mathvista_testmini_cot",
        "mathvista_testmini_solution",
        "mathvista_testmini_format",
    ),
    "mathvista_testmini_qwen3": (
        "mathvista_testmini_cot_qwen3",
        "mathvista_testmini_solution_qwen3",
        "mathvista_testmini_format_qwen3",
    ),
}

_POST_EVAL_TASKS = {
    "MolParse",
    "OpenRxn",
    "ocrbench",
    "simplevqa",
    "mmbench_en_dev",
    "mmbench_cn_cc",
    "mmmu_val_qwen3_official",
    "mmmu_pro_qwen3_official",
    "sfe-en",
    "sfe-zh",
    "scivqr_open",
    "scivqr_reasoning",
    "mathverse_testmini_reasoning",
    "mathverse_testmini_reasoning_qwen3",
    "mathvista_testmini_cot",
    "mathvista_testmini_solution",
    "mathvista_testmini_format",
    "mathvista_testmini_cot_qwen3",
    "mathvista_testmini_solution_qwen3",
    "mathvista_testmini_format_qwen3",
    "mathvision_reason_test_reasoning",
    "mathvision_reason_test_reasoning_qwen3",
    "wemath_testmini_reasoning",
    "wemath_testmini_reasoning_qwen3",
}

TASK_JUDGE_POLICIES = {
    task.lower(): TaskJudgePolicy(execution_stage="post_eval", standalone_task=task)
    for task in _POST_EVAL_TASKS
}


def expand_task(task: str) -> tuple[str, ...]:
    normalized = task.strip().lower()
    if not normalized:
        raise ValueError("Task name must not be empty")
    return _GROUP_EXPANSIONS.get(normalized, (task.strip(),))


def task_policy(task: str) -> TaskJudgePolicy | None:
    normalized = task.strip().lower()
    if not normalized:
        raise ValueError("Task name must not be empty")
    return TASK_JUDGE_POLICIES.get(normalized)


def requires_post_eval_judge(task: str) -> bool:
    return any(task_policy(leaf) is not None for leaf in expand_task(task))


def resolve_post_eval_judge_tasks(tasks: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        for leaf in expand_task(task):
            policy = task_policy(leaf)
            if policy is None:
                continue
            canonical = policy.standalone_task
            if canonical.lower() not in seen:
                resolved.append(canonical)
                seen.add(canonical.lower())
    return resolved
