from pathlib import Path

from lmms_eval.api.task import TaskConfig
from lmms_eval.utils import load_yaml_config


TASK_ROOT = Path(__file__).resolve().parents[2] / "lmms_eval/tasks"


def test_ocrbench_keeps_default_until_for_backend_contract_test():
    config = load_yaml_config(
        yaml_path=str(TASK_ROOT / "ocrbench/ocrbench.yaml"),
        mode="simple",
    )
    assert "until" not in config["generation_kwargs"]

    task_config = TaskConfig(**config)

    assert "until" in task_config.generation_kwargs
    assert task_config.generation_kwargs["until"] == ["\n\n"]
    assert task_config.generation_kwargs["max_new_tokens"] == 128
