import pytest

from lmms_eval.tui import server


@pytest.mark.parametrize("empty_value", [None, "", [], {}])
def test_primary_metrics_skips_empty_values(empty_value):
    result_data = {
        "results": {
            "vstar_bench": {
                "alias": "vstar_bench",
                "vstar_direct_attributes_acc,none": empty_value,
                "vstar_direct_attributes_acc_stderr,none": [],
                "vstar_overall_acc,none": 78.53403141361257,
                "vstar_overall_acc_stderr,none": "N/A",
            }
        }
    }

    assert server._primary_metrics(result_data) == [
        ("vstar_bench", "vstar_overall_acc", 78.53403141361257, "N/A")
    ]


@pytest.mark.parametrize("valid_value", [0, 0.0, False])
def test_primary_metrics_keeps_falsey_scores(valid_value):
    result_data = {
        "results": {
            "zero_score_bench": {
                "alias": "zero_score_bench",
                "accuracy,none": valid_value,
                "fallback_accuracy,none": 99.0,
            }
        }
    }

    assert server._primary_metrics(result_data) == [
        ("zero_score_bench", "accuracy", valid_value, None)
    ]
