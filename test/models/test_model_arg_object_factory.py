from lmms_eval.api.model import lmms


class _DummyModel(lmms):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def loglikelihood(self, requests):
        return []

    def generate_until(self, requests):
        return []

    def generate_until_multi_round(self, requests):
        return []


def test_create_from_arg_object_copies_mapping_and_merges_additional_config():
    original = {"model": "/resolved/model", "nested": {"value": 1}}

    instance = _DummyModel.create_from_arg_object(
        original,
        {"batch_size": 1, "device": None},
    )

    assert instance.kwargs == {
        "model": "/resolved/model",
        "nested": {"value": 1},
        "batch_size": 1,
    }
    assert original == {"model": "/resolved/model", "nested": {"value": 1}}


def test_create_from_arg_object_rejects_duplicate_additional_config_key():
    try:
        _DummyModel.create_from_arg_object(
            {"model": "/resolved/model", "batch_size": 2},
            {"batch_size": 1},
        )
    except ValueError as exc:
        assert "duplicate model argument keys" in str(exc)
    else:
        raise AssertionError("expected duplicate-key validation to fail")
