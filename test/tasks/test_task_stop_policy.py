from pathlib import Path

import pytest

from lmms_eval.api.task import TaskConfig
from lmms_eval.utils import load_yaml_config


TASK_ROOT = Path(__file__).resolve().parents[2] / "lmms_eval/tasks"


@pytest.mark.parametrize(
    ("relative_path", "expected_max_new_tokens"),
    [
        ("microvqa/microvqa.yaml", 4096),
        ("simplevqa/simplevqa.yaml", 32),
        ("mmmu/mmmu_val_qwen3_official.yaml", 128),
        (
            "mmmu_pro_qwen3_official/reasoning/mmmu_pro_standard_reasoning_qwen3_official.yaml",
            49152,
        ),
    ],
)
def test_reasoning_tasks_disable_synthetic_paragraph_stop(
    relative_path,
    expected_max_new_tokens,
):
    config = load_yaml_config(
        yaml_path=str(TASK_ROOT / relative_path),
        mode="simple",
    )
    task_config = TaskConfig(**config)

    assert task_config.generation_kwargs["until"] == []
    assert task_config.generation_kwargs["max_new_tokens"] == expected_max_new_tokens
