from pathlib import Path

from lmms_eval.api.task import TaskConfig
from lmms_eval.utils import load_yaml_config


TASK_ROOT = Path(__file__).resolve().parents[2] / "lmms_eval/tasks"


def test_ocrbench_disables_synthetic_paragraph_stop():
    config = load_yaml_config(
        yaml_path=str(TASK_ROOT / "ocrbench/ocrbench.yaml"),
        mode="simple",
    )
    task_config = TaskConfig(**config)

    assert "until" in task_config.generation_kwargs
    assert task_config.generation_kwargs["until"] is None
    assert task_config.generation_kwargs["max_new_tokens"] == 128
