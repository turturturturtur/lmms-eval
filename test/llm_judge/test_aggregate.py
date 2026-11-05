"""Tests for the aggregate module.

This module tests the Aggregator class and aggregate CLI command
for aggregating judged results with task-specific logic.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


from lmms_eval.llm_judge.aggregator import Aggregator
from lmms_eval.cli.aggregate_cmd import _load_jsonl, add_aggregate_parser, run_aggregate


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def wemath_samples():
    """Create sample WeMath judged data for testing."""
    return [
        {
            "doc_id": 0,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q1",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_1",
                "question number": 1,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q1",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_1",
                "question number": 1,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "metrics": {"acc_score": 1.0},
        },
        {
            "doc_id": 1,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q2",
                "option": "A. 1; B. 2",
                "answer": "B",
                "key": "2steps_2",
                "question number": 2,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q2",
                "option": "A. 1; B. 2",
                "answer": "B",
                "key": "2steps_2",
                "question number": 2,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "metrics": {"acc_score": 1.0},
        },
        {
            "doc_id": 2,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q3",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_multi",
                "question number": 3,
                "knowledge concept description": "Test",
                "acc_score": 0.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q3",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_multi",
                "question number": 3,
                "knowledge concept description": "Test",
                "acc_score": 0.0,
            },
            "metrics": {"acc_score": 0.0},
        },
    ]


@pytest.fixture
def mathvision_samples():
    """Create sample MathVision judged data for testing."""
    return [
        {
            "doc_id": 0,
            "mathvision_standard_eval": {
                "response": ["Answer 1"],
                "scores": [True],
            },
            "metrics": {"llm_judge_score": 1},
        },
        {
            "doc_id": 1,
            "mathvision_standard_eval": {
                "response": ["Answer 2"],
                "scores": [False],
            },
            "metrics": {"llm_judge_score": 0},
        },
        {
            "doc_id": 2,
            "mathvision_standard_eval": {
                "response": ["Answer 3"],
                "scores": [True],
            },
            "metrics": {"llm_judge_score": 1},
        },
    ]


@pytest.fixture
def generic_samples():
    """Create sample generic judged data for testing."""
    return [
        {"doc_id": 0, "metrics": {"accuracy": 0.8, "score": 1.0}},
        {"doc_id": 1, "metrics": {"accuracy": 0.6, "score": 0.0}},
        {"doc_id": 2, "metrics": {"accuracy": 0.9, "score": 1.0}},
    ]


@pytest.fixture
def sample_jsonl(tmp_path, generic_samples):
    """Create a sample JSONL file for testing."""
    file_path = tmp_path / "test_samples.jsonl"
    with open(file_path, "w") as f:
        for sample in generic_samples:
            f.write(json.dumps(sample) + "\n")
    return file_path


# -----------------------------------------------------------------------------
# Test Aggregator class
# -----------------------------------------------------------------------------

class TestAggregator:
    """Tests for the Aggregator class."""
    
    def test_init(self):
        """Test Aggregator initialization."""
        aggregator = Aggregator()
        
        assert aggregator._task_manager is None
        assert aggregator._cache == {}
    
    def test_get_special_config_wemath(self):
        """Test detection of WeMath special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("wemath_testmini_reasoning")
        assert config is not None
        assert "wemath" in config["module"]
        assert "loose_func" in config
        assert "strict_func" in config
    
    def test_get_special_config_mathvision(self):
        """Test detection of MathVision special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("mathvision_test")
        assert config is not None
        assert "mathvision" in config["module"]
        assert "accuracy_func" in config
    
    def test_get_special_config_generic(self):
        """Test that generic tasks return None."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("some_generic_task")
        assert config is None
    
    def test_get_special_config_no_false_positive(self):
        """Test that substring matches inside other words are rejected."""
        aggregator = Aggregator()
        
        # Should NOT match because "wemath" is embedded inside another word without word boundaries
        config = aggregator._get_special_config("mywemathtask")
        assert config is None
        
        # Should match because "wemath" is a proper prefix with underscore boundary
        config = aggregator._get_special_config("wemath_testmini")
        assert config is not None
    
    def test_get_special_config_mmbench(self):
        """Test detection of MMBench special aggregations."""
        aggregator = Aggregator()
        
        for task in ["mmbench_en_dev", "mmbench_en_test", "mmbench_cn_dev", "mmbench_cn_test", "mmbench_ru_dev", "mmbench_ko_dev", "mmbench_cn_cc"]:
            config = aggregator._get_special_config(task)
            assert config is not None, f"{task} should have special aggregation"
            assert "mmbench" in config["module"]
            assert "accuracy_func" in config
    
    def test_get_special_config_sfe(self):
        """Test detection of SFE special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("sfe")
        assert config is not None
        assert "sfe" in config["module"]
        assert config["data_key"] == "sfe_info"
    
    def test_get_special_config_mathvision_qwen3(self):
        """Test detection of MathVision Qwen3 special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("mathvision_testmini_qwen3")
        assert config is not None
        assert "utils_qwen3" in config["module"]
        assert config["data_key"] == "mathvision_qwen3_eval"
    
    def test_get_special_config_mmmu_pro(self):
        """Test detection of MMMU-Pro special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("mmmu_pro_standard_qwen3_official")
        assert config is not None
        assert "mmmu_pro" in config["module"]
        assert config["data_key"] == "mmmu_pro_acc_official"
    
    def test_get_special_config_exclude_patterns(self):
        """Test that mathvision_reason is excluded from mathvision match."""
        aggregator = Aggregator()
        
        # mathvision_reason should NOT match the generic mathvision config
        config = aggregator._get_special_config("mathvision_reason_testmini_qwen3")
        assert config is None
        
        # But mathvision_test should still match
        config = aggregator._get_special_config("mathvision_test")
        assert config is not None
    
    def test_aggregate_generic_task(self, generic_samples):
        """Test aggregation for generic tasks."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate(generic_samples, "generic_task")
        
        assert "accuracy" in results
        assert "score" in results
        # (0.8 + 0.6 + 0.9) / 3 = 0.7667
        assert results["accuracy"] == pytest.approx(0.7667, abs=0.001)
        # (1.0 + 0.0 + 1.0) / 3 = 0.6667
        assert results["score"] == pytest.approx(0.6667, abs=0.001)
    
    def test_aggregate_generic_task_with_metric_filter(self, generic_samples):
        """Test aggregation with specific metric filter."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate(generic_samples, "generic_task", metric_name="accuracy")
        
        assert "accuracy" in results
        assert "score" not in results  # Should be filtered out
    
    def test_aggregate_empty_samples(self):
        """Test aggregation with empty samples."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate([], "generic_task")
        
        assert results == {}
    
    def test_aggregate_samples_without_metrics(self):
        """Test aggregation with samples missing metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0}},
            {"doc_id": 1},  # No metrics
            {"doc_id": 2, "metrics": {"accuracy": 0.0}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        # Should only count samples with the metric
        assert results["accuracy"] == 0.5  # (1.0 + 0.0) / 2
    
    def test_aggregate_with_nested_metrics(self):
        """Test aggregation handles nested metric structures correctly."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0, "nested": {"value": 1}}},
            {"doc_id": 1, "metrics": {"accuracy": 0.0}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        # Should only include numeric metrics, not nested dicts
        assert "accuracy" in results
        assert "nested" not in results


class TestAggregatorWeMath:
    """Tests for WeMath-specific aggregation."""
    
    @patch("lmms_eval.llm_judge.aggregator._get_task_manager")
    def test_aggregate_wemath_loose(self, mock_tm, wemath_samples):
        """Test WeMath loose aggregation."""
        aggregator = Aggregator()
        
        pytest.importorskip("lmms_eval.tasks.wemath.reasoning.utils", exc_type=ImportError)
        results = aggregator.aggregate(wemath_samples, "wemath_testmini_reasoning", metric_name="wemath_loose")
        assert "Score (Loose)" in results
    
    @patch("lmms_eval.llm_judge.aggregator._get_task_manager")
    def test_aggregate_wemath_strict(self, mock_tm, wemath_samples):
        """Test WeMath strict aggregation."""
        aggregator = Aggregator()
        
        pytest.importorskip("lmms_eval.tasks.wemath.reasoning.utils", exc_type=ImportError)
        results = aggregator.aggregate(wemath_samples, "wemath_testmini_reasoning", metric_name="wemath_strict")
        assert "Score (Strict)" in results
    
    def test_wemath_data_extraction(self, wemath_samples):
        """Test extraction of WeMath data from samples."""
        aggregator = Aggregator()
        config = aggregator._get_special_config("wemath_testmini_reasoning")
        
        # Manually extract data like the aggregator does
        data_key = config["data_key"]
        extracted_data = []
        
        for sample in wemath_samples:
            if data_key in sample:
                extracted_data.append(sample[data_key])
        
        assert len(extracted_data) == 3
        assert all("ID" in d for d in extracted_data)
        assert all("key" in d for d in extracted_data)


class TestAggregatorMathVision:
    """Tests for MathVision-specific aggregation."""
    
    def test_aggregate_mathvision(self, mathvision_samples):
        """Test MathVision aggregation."""
        aggregator = Aggregator()
        
        try:
            results = aggregator.aggregate(mathvision_samples, "mathvision_test")
            
            if results:
                assert "accuracy" in results
        except (ImportError, AttributeError):
            pytest.skip("MathVision utils not available")
    
    def test_mathvision_data_extraction(self, mathvision_samples):
        """Test extraction of MathVision data from samples."""
        aggregator = Aggregator()
        config = aggregator._get_special_config("mathvision_test")
        
        data_key = config["data_key"]
        extracted_data = []
        
        for sample in mathvision_samples:
            if data_key in sample:
                extracted_data.append(sample[data_key])
        
        assert len(extracted_data) == 3
        assert all("scores" in d for d in extracted_data)


# -----------------------------------------------------------------------------
# Test CLI functions
# -----------------------------------------------------------------------------

class TestAggregatorDictAccuracy:
    """Tests for accuracy_func returning dict or None."""
    
    def test_aggregate_dict_accuracy(self, monkeypatch):
        """When accuracy_func returns a dict, results should be updated with its contents."""
        aggregator = Aggregator()
        
        # Mock a module with a dict-returning accuracy function
        mock_module = MagicMock()
        mock_module.mock_sfe_aggregate = MagicMock(return_value={
            "exact_match": 0.75,
            "rouge_score": 0.5,
        })
        
        # Temporarily register a fake special aggregation
        fake_config = {
            "module": "mock_sfe_module",
            "accuracy_func": "mock_sfe_aggregate",
            "data_key": "sfe_info",
        }
        aggregator.SPECIAL_AGGREGATIONS["__test_sfe__"] = fake_config
        
        samples = [
            {"doc_id": 0, "sfe_info": {"id": "1"}},
            {"doc_id": 1, "sfe_info": {"id": "2"}},
        ]
        
        with patch.object(importlib, "import_module", return_value=mock_module):
            results = aggregator.aggregate(samples, "__test_sfe__")
        
        assert "exact_match" in results
        assert results["exact_match"] == 0.75
        assert "rouge_score" in results
        assert results["rouge_score"] == 0.5
        # Clean up
        del aggregator.SPECIAL_AGGREGATIONS["__test_sfe__"]
    
    def test_aggregate_none_accuracy(self, monkeypatch):
        """When accuracy_func returns None (e.g., test splits), results should be empty."""
        aggregator = Aggregator()
        
        mock_module = MagicMock()
        mock_module.mock_test_aggregate = MagicMock(return_value=None)
        
        fake_config = {
            "module": "mock_test_module",
            "accuracy_func": "mock_test_aggregate",
            "data_key": "submission",
        }
        aggregator.SPECIAL_AGGREGATIONS["__test_none__"] = fake_config
        
        samples = [
            {"doc_id": 0, "submission": {"index": 1}},
        ]
        
        with patch.object(importlib, "import_module", return_value=mock_module):
            results = aggregator.aggregate(samples, "__test_none__")
        
        assert results == {}
        del aggregator.SPECIAL_AGGREGATIONS["__test_none__"]


class TestArgumentParser:
    """Tests for argument parsing."""

    @pytest.mark.parametrize("arguments", [["--input", "samples.jsonl"], ["--task", "mmbench_en_dev"]])
    def test_input_and_task_are_required(self, arguments):
        import argparse

        parser = argparse.ArgumentParser()
        add_aggregate_parser(parser.add_subparsers())
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["aggregate", *arguments])
        assert error.value.code == 2
    
    def test_add_aggregate_parser(self):
        """Test that aggregate parser is added correctly."""
        import argparse
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")
        
        add_aggregate_parser(subparsers)
        
        # Test parsing basic args
        args = parser.parse_args(["aggregate", "--input", "test.jsonl", "--task", "wemath"])
        assert args.subcommand == "aggregate"
        assert args.input == "test.jsonl"
        assert args.task == "wemath"
        assert args.metric is None  # Default
    
    def test_parser_with_metric(self):
        """Test parsing with metric argument."""
        import argparse
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        add_aggregate_parser(subparsers)
        
        args = parser.parse_args([
            "aggregate",
            "--input", "test.jsonl",
            "--task", "wemath",
            "--metric", "wemath_loose"
        ])
        
        assert args.metric == "wemath_loose"


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

class TestAggregateIntegration:
    """Integration tests for the aggregate command."""

    def test_real_aggregation_writes_score_to_requested_path(self, tmp_path, generic_samples):
        import argparse

        input_path = tmp_path / "judged.jsonl"
        input_path.write_text("".join(json.dumps(sample) + "\n" for sample in generic_samples))
        output_path = tmp_path / "new_directory/results.json"
        parser = argparse.ArgumentParser()
        add_aggregate_parser(parser.add_subparsers())
        args = parser.parse_args([
            "aggregate", "--input", str(input_path), "--task", "generic_task",
            "--metric", "accuracy", "--output", str(output_path),
        ])
        args.func(args)

        assert json.loads(output_path.read_text()) == {"accuracy": 0.7667}
    
    @patch("lmms_eval.llm_judge.aggregator.Aggregator")
    def test_run_aggregate_single_file(self, mock_aggregator_class, tmp_path, generic_samples):
        """Test aggregating a single file."""
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = {"accuracy": 0.75}
        mock_aggregator_class.return_value = mock_aggregator
        
        # Create test file
        test_file = tmp_path / "test_samples.jsonl"
        with open(test_file, "w") as f:
            for sample in generic_samples:
                f.write(json.dumps(sample) + "\n")
        
        import argparse
        args = argparse.Namespace(
            input=str(test_file),
            task="test_task",
            metric=None,
            output=None,
            verbose=False,
        )
        
        run_aggregate(args)
        
        mock_aggregator.aggregate.assert_called_once()
        call_args = mock_aggregator.aggregate.call_args
        assert call_args.args[1] == "test_task"
        assert list(tmp_path.iterdir()) == [test_file]
    
    @patch("lmms_eval.llm_judge.aggregator.Aggregator")
    def test_run_aggregate_file_not_found(self, mock_aggregator_class, tmp_path):
        """Test handling of file not found."""
        import argparse
        
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.jsonl"),
            task="test_task",
            metric=None,
            output=None,
            verbose=False,
        )
        
        with pytest.raises(SystemExit) as error:
            run_aggregate(args)
        assert error.value.code == 1
        mock_aggregator_class.assert_not_called()
    
    @patch("lmms_eval.llm_judge.aggregator.Aggregator")
    def test_run_aggregate_with_output(self, mock_aggregator_class, tmp_path, generic_samples):
        """Test aggregating with output file."""
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = {"accuracy": 0.75}
        mock_aggregator_class.return_value = mock_aggregator
        
        # Create test file
        test_file = tmp_path / "test_samples.jsonl"
        with open(test_file, "w") as f:
            for sample in generic_samples:
                f.write(json.dumps(sample) + "\n")
        
        output_file = tmp_path / "results.json"
        
        import argparse
        args = argparse.Namespace(
            input=str(test_file),
            task="test_task",
            metric=None,
            output=str(output_file),
            verbose=False,
        )
        
        run_aggregate(args)
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            content = json.load(f)
        assert content["accuracy"] == 0.75


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_aggregate_with_invalid_json(self, tmp_path):
        """Test handling of invalid JSON in input file."""
        # Create file with some invalid JSON
        test_file = tmp_path / "invalid.jsonl"
        with open(test_file, "w") as f:
            f.write('{"doc_id": 0, "metrics": {"accuracy": 1.0}}\n')
            f.write('invalid json line\n')
            f.write('{"doc_id": 1, "metrics": {"accuracy": 0.0}}\n')
        
        samples = _load_jsonl(test_file)
        
        # Should skip invalid line and load 2 valid samples
        assert len(samples) == 2
    
    def test_aggregate_with_empty_metrics(self):
        """Test aggregation when samples have empty metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {}},
            {"doc_id": 1, "metrics": {}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        assert results == {}
    
    def test_aggregate_with_non_numeric_metrics(self):
        """Test aggregation filters out non-numeric metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0, "detail": "some string"}},
            {"doc_id": 1, "metrics": {"accuracy": 0.0, "detail": "another string"}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        assert "accuracy" in results
        assert "detail" not in results
