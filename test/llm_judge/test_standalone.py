"""Tests for the standalone judge module.

This module tests the JudgeRunner class and related functionality
for judging JSONL files without regeneration.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


# Skip all tests if judge dependencies are not available
try:
    from lmms_eval.llm_judge.standalone import JudgeRunner
    JUDGE_AVAILABLE = True
except ImportError:
    JUDGE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not JUDGE_AVAILABLE, reason="Judge dependencies not available")


def _slow_process_results(doc, results):
    import time

    time.sleep(float(doc.get("sleep_seconds", 1.0)))
    return {"acc_score": 1.0}


def _timeout_once_process_results(doc, results):
    import time
    from pathlib import Path

    counter_path = Path(doc["counter_path"])
    count = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
    counter_path.write_text(str(count + 1), encoding="utf-8")
    if count == 0:
        time.sleep(float(doc.get("sleep_seconds", 1.0)))
    return {"acc_score": 1.0}


def _make_task(process_results_fn):
    return SimpleNamespace(
        config=SimpleNamespace(
            process_results=process_results_fn,
            dataset_path="unit-test",
            num_fewshot=0,
            metric_list=[],
        )
    )


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file for testing."""
    samples = [
        {
            "doc_id": 0,
            "doc": {
                "question": "What is 2+2?",
                "answer": "4",
                "options": ["2", "3", "4", "5"],
            },
            "filtered_resps": "The answer is 4.",
            "target": "4",
        },
        {
            "doc_id": 1,
            "doc": {
                "question": "What is the capital of France?",
                "answer": "Paris",
            },
            "filtered_resps": "Paris",
            "target": "Paris",
        },
        {
            "doc_id": 2,
            "doc": {
                "question": "Solve for x: 2x = 10",
                "answer": "5",
            },
            "filtered_resps": "x = 5",
            "target": "5",
        },
    ]
    
    file_path = tmp_path / "test_samples.jsonl"
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    return file_path


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = MagicMock()
    task.config.process_results = MagicMock(return_value={
        "acc_score": 1.0,
        "format_score": 1.0,
    })
    return task


class TestJudgeRunner:
    """Tests for the JudgeRunner class."""
    
    def test_init(self, monkeypatch):
        """Test JudgeRunner initialization."""
        monkeypatch.delenv("JUDGE_DATAPOINT_TIMEOUT", raising=False)
        monkeypatch.delenv("JUDGE_DATAPOINT_MAX_ATTEMPTS", raising=False)
        monkeypatch.delenv("JUDGE_DATAPOINT_MAX_RETRIES", raising=False)
        runner = JudgeRunner(
            judge_mode="auto",
            judge_model="gpt-4o-mini",
            parallel=4,
        )
        
        assert runner.judge_mode == "auto"
        assert runner.judge_model == "gpt-4o-mini"
        assert runner.parallel == 4
        assert runner.datapoint_timeout == 30.0
        assert runner.datapoint_max_attempts == 5
        assert runner._judge_provider is None
    
    def test_init_with_env_vars(self, monkeypatch):
        """Test initialization with environment variables."""
        monkeypatch.setenv("JUDGE_API_KEY", "test-key")
        monkeypatch.setenv("JUDGE_BASE_URL", "https://test.api.com")
        
        runner = JudgeRunner()
        
        assert runner.judge_api_key == "test-key"
        assert runner.judge_base_url == "https://test.api.com"
    
    def test_load_jsonl(self, sample_jsonl):
        """Test loading JSONL file."""
        runner = JudgeRunner()
        samples = runner._load_jsonl(sample_jsonl)
        
        assert len(samples) == 3
        assert samples[0]["doc_id"] == 0
        assert samples[0]["doc"]["question"] == "What is 2+2?"
    
    def test_load_jsonl_not_found(self):
        """Test loading non-existent file."""
        runner = JudgeRunner()
        
        with pytest.raises(FileNotFoundError):
            runner._load_jsonl(Path("/nonexistent/file.jsonl"))
    
    def test_load_jsonl_invalid_json(self, tmp_path):
        """Test loading file with invalid JSON."""
        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, "w") as f:
            f.write('{"valid": true}\n')
            f.write('invalid json\n')
            f.write('{"valid": false}\n')
        
        runner = JudgeRunner()
        samples = runner._load_jsonl(file_path)
        
        assert len(samples) == 2  # One line should be skipped
    
    def test_extract_question(self):
        """Test question extraction from doc."""
        runner = JudgeRunner()
        
        # Test various field names
        assert runner._extract_question({"question": "Q1"}) == "Q1"
        assert runner._extract_question({"problem": "Q2"}) == "Q2"
        assert runner._extract_question({"query": "Q3"}) == "Q3"
        assert runner._extract_question({"query_wo": "Q4"}) == "Q4"
        
        # Test fallback
        doc = {"other": "field"}
        result = runner._extract_question(doc)
        assert "other" in result
    
    def test_needs_llm_judge(self):
        """Test detection of when LLM judge is needed."""
        runner = JudgeRunner()
        
        # Should need LLM judge
        assert runner._needs_llm_judge({"acc_score": 0}) is True
        assert runner._needs_llm_judge({"accuracy": 0.0}) is True
        assert runner._needs_llm_judge({"correct": False}) is True
        assert runner._needs_llm_judge({"exact_match": 0}) is True
        
        # Should not need LLM judge
        assert runner._needs_llm_judge({"acc_score": 1.0}) is False
        assert runner._needs_llm_judge({"accuracy": 0.9}) is False
        assert runner._needs_llm_judge({"correct": True}) is False
        
        # Edge cases
        assert runner._needs_llm_judge({}) is False
        assert runner._needs_llm_judge({"other_metric": 0}) is False
    
    def test_clean_for_json(self):
        """Test JSON cleaning."""
        runner = JudgeRunner()
        
        # Test basic types
        assert runner._clean_for_json(123) == 123
        assert runner._clean_for_json("str") == "str"
        assert runner._clean_for_json(None) is None
        
        # Test nested dict
        data = {
            "a": 1,
            "b": [1, 2, 3],
            "c": {"d": "value"},
        }
        result = runner._clean_for_json(data)
        assert result == data
        
        # Test non-serializable
        obj = {"key": MagicMock()}
        result = runner._clean_for_json(obj)
        assert isinstance(result["key"], str)
    
    def test_save_results(self, tmp_path):
        """Test saving results."""
        runner = JudgeRunner()
        
        results = [
            {"id": 1, "metrics": {"acc": 1.0}},
            {"id": 2, "metrics": {"acc": 0.0}},
        ]
        output_path = tmp_path / "output.jsonl"
        
        runner.save_results(results, output_path)
        
        assert output_path.exists()
        
        # Verify contents
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1
    
    @patch("lmms_eval.llm_judge.standalone.TaskManager")
    def test_load_task(self, mock_tm_class):
        """Test loading task."""
        mock_tm = MagicMock()
        mock_task = MagicMock()
        mock_tm.load_task_or_group.return_value = {"task": mock_task}
        mock_tm_class.return_value = mock_tm
        
        runner = JudgeRunner()
        task = runner._load_task("test_task")
        
        assert task == mock_task
        mock_tm.load_task_or_group.assert_called_once_with("test_task")
    
    @patch("lmms_eval.llm_judge.standalone.TaskManager")
    def test_load_task_error(self, mock_tm_class):
        """Test loading task with error."""
        mock_tm = MagicMock()
        mock_tm.load_task_or_group.side_effect = Exception("Task not found")
        mock_tm_class.return_value = mock_tm
        
        runner = JudgeRunner()
        
        with pytest.raises(ValueError):
            runner._load_task("invalid_task")

    def test_datapoint_timeout_marks_sample_failed(self):
        """A hung datapoint should be killed and marked failed after max attempts."""
        runner = JudgeRunner(
            judge_mode="rule",
            parallel=1,
            datapoint_timeout=0.2,
            datapoint_max_attempts=2,
        )
        task = _make_task(_slow_process_results)
        sample = {
            "doc_id": 7,
            "doc": {"question": "Q", "answer": "A", "sleep_seconds": 2.0},
            "filtered_resps": "A",
        }

        results = runner._judge_samples_with_datapoint_timeout(
            [sample],
            task,
            task.config.process_results,
            "timeout_unit_test",
        )

        assert len(results) == 1
        assert results[0]["judge_mode"] == "timeout"
        assert results[0]["metrics"]["judge_timeout"] is True
        assert results[0]["metrics"]["judge_attempts"] == 2

    def test_datapoint_timeout_retries_until_success(self, tmp_path):
        """A datapoint can time out once and still succeed on a later attempt."""
        counter_path = tmp_path / "attempts.txt"
        runner = JudgeRunner(
            judge_mode="rule",
            parallel=1,
            datapoint_timeout=0.2,
            datapoint_max_attempts=2,
        )
        task = _make_task(_timeout_once_process_results)
        sample = {
            "doc_id": 8,
            "doc": {
                "question": "Q",
                "answer": "A",
                "counter_path": str(counter_path),
                "sleep_seconds": 2.0,
            },
            "filtered_resps": "A",
        }

        results = runner._judge_samples_with_datapoint_timeout(
            [sample],
            task,
            task.config.process_results,
            "retry_unit_test",
        )

        assert results[0]["judge_mode"] == "rule"
        assert results[0]["metrics"]["acc_score"] == 1.0
        assert results[0]["judge_attempts"] == 2
        assert counter_path.read_text(encoding="utf-8") == "2"


class TestJudgeSample:
    """Tests for the _judge_sample method."""
    
    def test_judge_sample_rule_mode(self, mock_task):
        """Test judging in rule mode."""
        runner = JudgeRunner(judge_mode="rule")
        
        sample = {
            "doc_id": 0,
            "doc": {"question": "Q", "answer": "A"},
            "filtered_resps": "Response",
        }
        
        result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
        
        assert "metrics" in result
        assert result["judge_mode"] == "rule"
        assert result["metrics"]["acc_score"] == 1.0
    
    def test_judge_sample_string_response(self, mock_task):
        """Test judging with string response (not list)."""
        runner = JudgeRunner(judge_mode="rule")
        
        sample = {
            "doc_id": 0,
            "doc": {"question": "Q", "answer": "A"},
            "filtered_resps": "Response",  # String, not list
        }
        
        result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
        
        assert "metrics" in result
        # Should convert string to list before passing to process_results
        mock_task.config.process_results.assert_called_once()
        call_args = mock_task.config.process_results.call_args
        assert call_args[0][1] == ["Response"]  # Should be wrapped in list
    
    def test_judge_sample_error_handling(self, mock_task):
        """Test error handling in judging."""
        runner = JudgeRunner(judge_mode="rule")
        mock_task.config.process_results.side_effect = Exception("Processing error")
        
        sample = {
            "doc_id": 0,
            "doc": {"question": "Q", "answer": "A"},
            "filtered_resps": "Response",
        }
        
        result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
        
        assert "metrics" in result
        assert result["judge_mode"] == "error"
        assert "error" in result["metrics"]


class TestJudgeFile:
    """Tests for the judge_file method."""
    
    @patch("lmms_eval.llm_judge.standalone.TaskManager")
    def test_judge_file_success(self, mock_tm_class, sample_jsonl):
        """Test successful judging of file."""
        mock_tm = MagicMock()
        mock_task = MagicMock()
        mock_task.config.process_results = MagicMock(return_value={
            "acc_score": 1.0,
        })
        mock_tm.load_task_or_group.return_value = {"task": mock_task}
        mock_tm_class.return_value = mock_tm
        
        runner = JudgeRunner(judge_mode="rule")
        results = runner.judge_file(sample_jsonl, "test_task")
        
        assert len(results) == 3
        assert all("metrics" in r for r in results)
        assert all(r["judge_mode"] == "rule" for r in results)
    
    @patch("lmms_eval.llm_judge.standalone.TaskManager")
    def test_judge_file_no_process_results(self, mock_tm_class, sample_jsonl):
        """Test judging when task has no process_results."""
        mock_tm = MagicMock()
        mock_task = MagicMock()
        mock_task.config.process_results = None
        mock_tm.load_task_or_group.return_value = {"task": mock_task}
        mock_tm_class.return_value = mock_tm
        
        runner = JudgeRunner()
        
        with pytest.raises(ValueError, match="no process_results"):
            runner.judge_file(sample_jsonl, "test_task")


class TestExtractExistingMetrics:
    """Tests for _extract_existing_metrics."""
    
    def test_extracts_common_scalars(self):
        runner = JudgeRunner()
        sample = {
            "doc_id": 0,
            "acc_score": 0.5,
            "llm_judge_score": 0.0,
            "format_score": 1.0,
            "exact_match": 0.0,
        }
        metrics = runner._extract_existing_metrics(sample)
        assert metrics["acc_score"] == 0.5
        assert metrics["llm_judge_score"] == 0.0
        assert metrics["format_score"] == 1.0
        assert metrics["exact_match"] == 0.0
    
    def test_extracts_needs_llm_judge(self):
        """needs_llm_judge must be extracted so that tasks like SFE can enter dedicated scoring."""
        runner = JudgeRunner()
        sample = {
            "doc_id": 0,
            "needs_llm_judge": True,
            "gpt_eval_score": 0,
        }
        metrics = runner._extract_existing_metrics(sample)
        assert metrics["needs_llm_judge"] is True
        assert "gpt_eval_score" not in metrics
    
    def test_extracts_sfe_fields(self):
        runner = JudgeRunner()
        sample = {
            "doc_id": 0,
            "sfe_info": {"id": "x"},
            "formatted_question": "Q",
            "answer": "A",
        }
        metrics = runner._extract_existing_metrics(sample)
        assert metrics["sfe_info"]["id"] == "x"
        assert metrics["formatted_question"] == "Q"
        assert metrics["answer"] == "A"

    def test_backward_compat_api_judge_accuracy(self):
        """Old JSONL files with api_judge_accuracy should be mapped to llm_judge_score + needs_llm_judge."""
        runner = JudgeRunner()
        sample = {
            "doc_id": 0,
            "api_judge_accuracy": 0,
            "question": "Q",
        }
        metrics = runner._extract_existing_metrics(sample)
        assert metrics["llm_judge_score"] == 0
        assert metrics["needs_llm_judge"] is True
        assert "api_judge_accuracy" not in metrics


class TestMetricSync:
    """Tests for llm_judge_score sync back to trigger keys in auto mode."""
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_auto_fallback_syncs_llm_judge_score(self, mock_factory_class):
        """When process_results returns llm_judge_score with needs_llm_judge, fallback updates it directly."""
        mock_provider = MagicMock()
        mock_provider.evaluate_binary.return_value = {
            "result": 1,
            "raw_response": "Correct",
            "model": "gpt-4o",
            "success": True,
        }
        mock_factory_class.create_provider.return_value = mock_provider
        
        runner = JudgeRunner(judge_mode="auto", judge_api_key="test-key")
        mock_task = MagicMock()
        mock_task.config.process_results = MagicMock(return_value={
            "llm_judge_score": 0,
            "needs_llm_judge": True,
            "question": "Q",
        })
        
        sample = {
            "doc_id": 0,
            "doc": {"question": "Q", "answer": "A"},
            "filtered_resps": "A",
            "target": "A",
        }
        result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
        
        assert result["judge_mode"] == "llm_fallback"
        assert result["metrics"]["llm_judge_score"] == 1
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_auto_fallback_skips_sync_for_sfe(self, mock_factory_class):
        """SFE should not sync llm_judge_score to exact_match because it handles its own normalization."""
        mock_provider = MagicMock()
        mock_provider.evaluate_score.return_value = {
            "result": 7,
            "raw_response": "7",
            "model": "gpt-4o",
            "success": True,
        }
        mock_factory_class.create_provider.return_value = mock_provider
        
        runner = JudgeRunner(judge_mode="auto", judge_api_key="test-key")
        runner._current_task_name = "sfe"
        mock_task = MagicMock()
        mock_task.config.process_results = MagicMock(return_value={
            "exact_match": 0.0,
            "needs_llm_judge": True,
            "formatted_question": "Q",
            "answer": "A",
            "sfe_info": {"id": "1"},
        })
        
        sample = {
            "doc_id": 0,
            "doc": {"question": "Q", "answer": "A"},
            "filtered_resps": "wrong",
            "target": "A",
        }
        result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
        
        assert result["judge_mode"] == "llm_fallback"
        # exact_match should be set by SFE-specific block (7/10=0.7), not sync logic
        assert result["metrics"]["exact_match"] == 0.7


class TestLLMJudge:
    """Tests for LLM judge functionality."""
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_init_judge_provider(self, mock_factory_class):
        """Test initializing judge provider."""
        mock_provider = MagicMock()
        mock_factory_class.create_provider.return_value = mock_provider
        
        runner = JudgeRunner(
            judge_mode="llm",
            judge_model="gpt-4o",
            judge_api_key="test-key",
            judge_base_url="https://test.api.com",
        )
        
        runner._init_judge_provider()
        
        assert runner._judge_provider == mock_provider
        mock_factory_class.create_provider.assert_called_once()
    
    def test_init_judge_provider_no_key(self):
        """Test initializing without API key uses dummy key for local vLLM."""
        runner = JudgeRunner(judge_mode="llm")
        runner.judge_api_key = None
        
        # Should not raise; uses dummy key for local servers
        with patch("lmms_eval.llm_judge.standalone.ProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_provider.return_value = mock_provider
            runner._init_judge_provider()
            assert runner._judge_provider is not None
            mock_factory.create_provider.assert_called_once()
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_apply_llm_judge(self, mock_factory_class):
        """Test applying LLM judge."""
        mock_provider = MagicMock()
        mock_provider.evaluate_binary.return_value = {
            "result": 1,
            "raw_response": "Correct",
            "model": "gpt-4o",
            "success": True,
        }
        mock_factory_class.create_provider.return_value = mock_provider
        
        runner = JudgeRunner(
            judge_mode="llm",
            judge_api_key="test-key",
        )
        
        doc = {"question": "What is 2+2?", "answer": "4"}
        results = ["The answer is 4."]
        fallback = {"acc_score": 0}
        
        metrics = runner._apply_llm_judge(doc, results, fallback)
        
        assert metrics["llm_judge_score"] == 1
        assert metrics["llm_judge_raw"] == "Correct"
        assert metrics["llm_judge_model"] == "gpt-4o"
        assert metrics["acc_score"] == 0  # Fallback preserved
        
        mock_provider.evaluate_binary.assert_called_once_with(
            question="What is 2+2?",
            answer="4",
            prediction="The answer is 4.",
            output_format="0/1",
            custom_prompt=None,
        )
    
    def test_apply_llm_judge_with_custom_prompt(self):
        """Test that get_judge_prompt hook is used when present in task module."""
        runner = JudgeRunner(judge_mode="llm", judge_api_key="test-key")
        mock_provider = MagicMock()
        mock_provider.evaluate_binary.return_value = {
            "result": 1,
            "raw_response": "Correct",
            "model": "gpt-4o",
            "success": True,
        }
        runner._judge_provider = mock_provider
        
        mock_process_results = MagicMock()
        mock_process_results.__globals__ = {
            "get_judge_prompt": lambda doc, pred, target: "Custom prompt for {question}".format(question=doc.get("question", ""))
        }
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        runner._current_task = mock_task
        
        doc = {"question": "What is 2+2?", "answer": "4"}
        results = ["The answer is 4."]
        metrics = runner._apply_llm_judge(doc, results, {"acc_score": 0})
        
        mock_provider.evaluate_binary.assert_called_once()
        call_kwargs = mock_provider.evaluate_binary.call_args[1]
        assert call_kwargs["custom_prompt"] == "Custom prompt for What is 2+2?"
    
    def test_apply_llm_judge_sfe_scoring(self):
        """Test SFE-specific 0-10 scoring path in _apply_llm_judge."""
        runner = JudgeRunner(judge_mode="llm", judge_api_key="test-key")
        runner._current_task_name = "sfe"
        mock_provider = MagicMock()
        mock_provider.evaluate_score.return_value = {
            "result": 8,
            "raw_response": "8",
            "model": "gpt-4o",
            "success": True,
        }
        runner._judge_provider = mock_provider
        
        doc = {"question": "Q", "answer": "A"}
        results = ["pred"]
        fallback = {
            "needs_llm_judge": True,
            "formatted_question": "Q",
            "answer": "A",
            "exact_match": 0.0,
            "sfe_info": {"id": "1"},
        }
        metrics = runner._apply_llm_judge(doc, results, fallback)
        
        assert metrics["llm_judge_score"] == 8
        assert metrics["exact_match"] == 0.8
        assert metrics["sfe_info"]["llm_score"] == ["8"]
        mock_provider.evaluate_score.assert_called_once()
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_apply_llm_judge_no_answer(self, mock_factory_class):
        """Test LLM judge when no ground truth answer."""
        runner = JudgeRunner(judge_api_key="test-key")
        runner._judge_provider = MagicMock()
        
        doc = {"question": "What is 2+2?"}  # No answer field
        results = ["The answer is 4."]
        
        metrics = runner._apply_llm_judge(doc, results, {})
        
        assert metrics["llm_judge_skipped"] is True
        runner._judge_provider.evaluate_binary.assert_not_called()
    
    @patch("lmms_eval.llm_judge.standalone.ProviderFactory")
    def test_apply_llm_judge_error(self, mock_factory_class):
        """Test LLM judge when API fails."""
        mock_provider = MagicMock()
        mock_provider.evaluate_binary.side_effect = Exception("API Error")
        mock_factory_class.create_provider.return_value = mock_provider
        
        runner = JudgeRunner(judge_api_key="test-key")
        
        doc = {"question": "Q", "answer": "A"}
        results = ["Response"]
        fallback = {"acc_score": 0}
        
        metrics = runner._apply_llm_judge(doc, results, fallback)
        
        assert metrics["llm_judge_error"] == "API Error"
        assert metrics["llm_judge_failed"] is True
        assert metrics["acc_score"] == 0  # Fallback preserved


class TestComputeSummary:
    """Tests for compute_summary."""
    
    def test_compute_summary_sfe(self):
        """SFE summary should average exact_match and llm_judge_score."""
        runner = JudgeRunner()
        results = [
            {"metrics": {"exact_match": 0.5, "llm_judge_score": 5, "sfe_info": {"id": "1"}}},
            {"metrics": {"exact_match": 0.8, "llm_judge_score": 8, "sfe_info": {"id": "2"}}},
        ]
        summary = runner.compute_summary(results)
        assert summary["exact_match"] == 0.65
        assert summary["llm_judge_score_avg"] == 6.5
        assert summary["total_acc"] == 0.65
    
    def test_compute_summary_flat_scores(self):
        """Standard flat-score summary with rule_acc + llm_fallback_acc."""
        runner = JudgeRunner()
        results = [
            {"metrics": {"acc_score": 1.0, "llm_judge_score": 0}},
            {"metrics": {"acc_score": 0.0, "llm_judge_score": 1}},
        ]
        summary = runner.compute_summary(results)
        assert summary["rule_acc"] == 0.5
        assert summary["llm_fallback_acc"] == 0.25
        assert summary["total_acc"] == 0.75
    
    def test_compute_summary_pure_llm_judge_tasks(self):
        """Pure LLM-judge tasks (e.g. MolParse, OpenRxn) should report total_acc."""
        runner = JudgeRunner()
        results = [
            {"metrics": {"llm_judge_score": 0, "llm_judge_success": True}},
            {"metrics": {"llm_judge_score": 1, "llm_judge_success": True}},
            {"metrics": {"llm_judge_score": 1, "llm_judge_success": True}},
        ]
        summary = runner.compute_summary(results)
        assert "rule_acc" not in summary
        assert "llm_fallback_acc" not in summary
        assert summary["total_acc"] == 0.6667
    
    def test_compute_summary_llm_judge_score(self):
        """Averages llm_judge_score directly and exposes total_acc for pure LLM-judge tasks."""
        runner = JudgeRunner()
        results = [
            {"metrics": {"llm_judge_score": 0.0}},
            {"metrics": {"llm_judge_score": 1.0}},
        ]
        summary = runner.compute_summary(results)
        assert summary["llm_judge_score"] == 0.5
        assert "rule_acc" not in summary
        assert summary["total_acc"] == 0.5
