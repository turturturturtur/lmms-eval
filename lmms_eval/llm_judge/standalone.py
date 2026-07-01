"""Standalone judge runner for JSONL files.

This module provides the core logic for judging model outputs from JSONL files
without regeneration, completely separating the generation and judging phases.
"""

import asyncio
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from tqdm import tqdm

# Delay imports to avoid loading heavy dependencies
ProviderFactory = None
ServerConfig = None
SCIVQR_SUBJECTS = {"math", "physics", "chemistry", "biology", "geography", "astronomy"}

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)

def _get_provider_factory():
    global ProviderFactory
    if ProviderFactory is None:
        from lmms_eval.llm_judge.factory import ProviderFactory as PF
        ProviderFactory = PF
    return ProviderFactory

def _get_server_config():
    global ServerConfig
    if ServerConfig is None:
        from lmms_eval.llm_judge.protocol import ServerConfig as SC
        ServerConfig = SC
    return ServerConfig

# Delay import of TaskManager to avoid heavy imports
TaskManager = None

def _get_task_manager():
    global TaskManager
    if TaskManager is None:
        from lmms_eval.tasks import TaskManager as TM
        TaskManager = TM
    return TaskManager()


class JudgeRunner:
    """Runner for standalone judging from JSONL files.
    
    This class handles:
    1. Loading JSONL files with model outputs
    2. Loading the appropriate task's process_results function
    3. Applying rule-based or LLM-based judging
    4. Saving results with updated metrics
    
    Example:
        >>> runner = JudgeRunner(judge_mode="auto", judge_model="gpt-4o-mini")
        >>> results = runner.judge_file(Path("results.jsonl"), "mathvision_reason_testmini")
        >>> runner.save_results(results, Path("judged.jsonl"))
    """
    
    def __init__(
        self,
        judge_mode: str = "auto",
        judge_model: str = "gpt-4o-mini",
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        parallel: int = 1,
    ):
        """Initialize the judge runner.
        
        Args:
            judge_mode: Judging mode - always "auto" (rule first, LLM fallback)
            judge_model: Name of the LLM to use for judging
            judge_api_key: API key for the judge model
            judge_base_url: Base URL for the judge API
            parallel: Number of parallel workers (currently only for LLM judge)
        """
        self.judge_mode = judge_mode
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.getenv("JUDGE_API_KEY")
        self.judge_base_url = judge_base_url or os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_API_URL") or ""
        self.parallel = parallel
        self._task_manager = None
        self._judge_provider = None
        self._provider_lock = threading.Lock()
        self._current_task_name = None
        self._current_task = None
        self._scivqr_doc_lookup_cache = {}
        logger.info(
            f"JudgeRunner initialized: mode={self.judge_mode}, model={self.judge_model}, "
            f"base_url={self.judge_base_url}, parallel={self.parallel}"
        )
        
    def judge_file(self, input_path: Path, task_name: str) -> List[Dict[str, Any]]:
        """Judge all samples in a JSONL file.
        
        Args:
            input_path: Path to JSONL file with model outputs
            task_name: Task name for loading process_results function
            
        Returns:
            List of judged samples with updated metrics
            
        Raises:
            ValueError: If task cannot be loaded
            FileNotFoundError: If input file doesn't exist
        """
        self._current_task_name = task_name
        
        # Load task and get process_results function
        logger.debug(f"Loading task: {task_name}")
        task = self._load_task(task_name)
        self._current_task = task
        process_results_fn = task.config.process_results
        
        if process_results_fn is None:
            raise ValueError(f"Task {task_name} has no process_results function")
        
        logger.info(
            f"Task config [{task_name}]: dataset_path={getattr(task.config, 'dataset_path', 'N/A')}, "
            f"fewshot={getattr(task.config, 'num_fewshot', 'N/A')}, "
            f"metric_list={[m.get('metric') for m in getattr(task.config, 'metric_list', [])]}"
        )
        
        # Load samples
        logger.debug(f"Loading samples from {input_path}")
        samples = self._load_jsonl(input_path)
        samples = self._attach_scivqr_subject_from_path(samples, input_path)
        logger.info(f"Loaded {len(samples)} samples")
        
        # Judge each sample concurrently
        logger.info(f"Starting concurrent judging with max_workers={self.parallel}")
        judged_samples = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {
                executor.submit(self._judge_sample, sample, task, process_results_fn): i
                for i, sample in enumerate(samples)
            }
            for future in tqdm(
                as_completed(futures),
                desc=f"Judging {task_name}",
                total=len(samples),
                miniters=10,
            ):
                idx = futures[future]
                judged_samples[idx] = future.result()

        logger.info(f"Finished judging {len(judged_samples)} samples")
        return judged_samples
    
    def _judge_sample(
        self,
        sample: Dict[str, Any],
        task: Any,
        process_results_fn: Callable,
    ) -> Dict[str, Any]:
        """Judge a single sample.
        
        Args:
            sample: Sample dict from JSONL (contains doc, filtered_resps, etc.)
            task: Task object with config
            process_results_fn: The task's process_results function
            
        Returns:
            Sample dict with added/updated metrics
        """
        doc = sample.get("doc", {})
        filtered_resps = sample.get("filtered_resps", [])
        target = sample.get("target", None)
        doc_id = sample.get("doc_id", "unknown")

        if self._is_scivqr_task() and not filtered_resps and "response" in sample:
            filtered_resps = [sample.get("response", "")]
        if self._is_scivqr_task() and target is None:
            if self._current_task_name == "scivqr_reasoning":
                target = sample.get("gt_reason", sample.get("solution", sample.get("answer")))
            else:
                target = sample.get("answer")
        
        # Detect whether the original doc was dropped during serialization.
        # The evaluation tracker pops "doc" before saving JSONL, so if doc is
        # empty we should try to reuse pre-computed metrics instead of calling
        # process_results with an incomplete doc.
        doc_was_dropped = not sample.get("doc")
        
        # For tasks that need full sample context (like mmmu_val_qwen3_official),
        # merge sample data into doc if doc is empty or minimal
        if not doc or (isinstance(doc, dict) and len(doc) < 3):
            # Some tasks (e.g. mathverse) store the original doc fields in submission
            if "submission" in sample and isinstance(sample["submission"], dict):
                doc = dict(sample["submission"])
                doc["__sample_context__"] = sample
            else:
                # Create a merged doc with sample context
                doc = {"__sample_context__": sample}
        doc = self._enrich_scivqr_reasoning_doc(doc, sample, task)
        
        # Convert to list if single response
        if isinstance(filtered_resps, str):
            filtered_resps = [filtered_resps]
        
        # Initialize result
        result_sample = sample.copy()
        
        try:
            # Step 1: rule-based judging (reuse pre-computed metrics if doc was dropped)
            if doc_was_dropped:
                process_globals = getattr(process_results_fn, "__globals__", {})
                force_reprocess = process_globals.get("FORCE_REPROCESS_FROM_SAMPLE") is True
                existing_metrics = {} if force_reprocess else self._extract_existing_metrics(sample)
                if existing_metrics:
                    metrics = existing_metrics
                    result_sample["metrics"] = metrics
                    result_sample["judge_mode"] = "rule_existing"
                else:
                    metrics = process_results_fn(doc, filtered_resps)
                    result_sample["metrics"] = metrics
                    result_sample["judge_mode"] = "rule"
            else:
                try:
                    metrics = process_results_fn(doc, filtered_resps)
                    result_sample["metrics"] = metrics
                    result_sample["judge_mode"] = "rule"
                except Exception as rule_err:
                    if self.judge_mode == "auto":
                        logger.debug(f"Sample {doc_id}: rule-based failed ({rule_err}), trying LLM judge")
                        metrics = {"rule_error": str(rule_err)}
                        result_sample["metrics"] = metrics
                        result_sample["judge_mode"] = "rule_error"
                    else:
                        raise rule_err
            
            # Step 2: LLM fallback for low-scoring / failed rule-based samples
            if self._needs_llm_judge(metrics):
                logger.debug(f"Sample {doc_id}: rule-based score low, trying LLM judge")
                llm_metrics = self._apply_llm_judge(doc, filtered_resps, metrics, target=target, sample=sample)
                # Sync llm_judge_score back to the original trigger key so that
                # task-specific metrics reflect the final result.
                trigger_key = None
                for key in ["acc_score", "accuracy", "correct", "score", "exact_match", "gpt_eval_score"]:
                    if key in metrics and (metrics[key] == 0 or metrics[key] is False or (isinstance(metrics[key], float) and metrics[key] < 0.1)):
                        trigger_key = key
                        break
                # Skip sync for SFE because its _apply_llm_judge block already normalizes exact_match
                is_sfe = self._current_task_name and "sfe" in self._current_task_name.lower()
                if trigger_key is not None and "llm_judge_score" in llm_metrics and not is_sfe:
                    llm_metrics[trigger_key] = llm_metrics["llm_judge_score"]
                result_sample["metrics"] = llm_metrics
                result_sample["judge_mode"] = "llm_judge" if self.judge_mode == "judge" else "llm_fallback"
                
        except Exception as e:
            logger.warning(f"Error judging sample {doc_id}: {e}")
            result_sample["metrics"] = {"error": str(e), "judge_failed": True}
            result_sample["judge_mode"] = "error"
        
        return result_sample
    
    def _apply_llm_judge(
        self,
        doc: Dict[str, Any],
        results: List[str],
        fallback_metrics: Dict[str, Any],
        target: Optional[str] = None,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply LLM judge when rule-based fails or for LLM-only mode.
        
        Args:
            doc: Document dict with question, answer, etc.
            results: List of model responses
            fallback_metrics: Existing metrics from rule-based judging
            
        Returns:
            Updated metrics with LLM judge results
        """
        # Initialize judge provider if needed
        if self._judge_provider is None:
            self._init_judge_provider()
        
        # Extract question and answer
        question = self._extract_question(doc, sample)
        answer = str(doc.get("answer", ""))
        if not answer and target is not None:
            answer = str(target)
        if not answer:
            answer = self._extract_answer_from_metrics(fallback_metrics)
        prediction = results[0] if results else ""
        
        if not answer:
            logger.warning("No ground truth answer found in doc, skipping LLM judge")
            return {**fallback_metrics, "llm_judge_skipped": True}
        
        # Enrich doc for custom prompt generation when it was dropped during serialization
        if not doc.get("question"):
            doc = dict(doc)
            doc["question"] = self._extract_question(doc, sample)
        if not doc.get("answer") and answer:
            doc = dict(doc)
            doc["answer"] = answer
        
        # Load task-specific custom judge hooks if available
        task_module_globals = {}
        custom_prompt = None
        custom_messages = None
        parse_judge_response = None
        update_metrics_from_judge = None
        if self._current_task is not None:
            try:
                process_results_fn = getattr(self._current_task.config, "process_results", None)
                if process_results_fn is not None:
                    task_module_globals = getattr(process_results_fn, "__globals__", {})
                    parse_judge_response = task_module_globals.get("parse_judge_response")
                    update_metrics_from_judge = task_module_globals.get("update_metrics_from_judge")
                    get_judge_messages = task_module_globals.get("get_judge_messages")
                    if get_judge_messages is not None:
                        custom_messages = get_judge_messages(doc, prediction, target)
                        logger.debug("Using custom judge messages from task module")
                    get_judge_prompt = task_module_globals.get("get_judge_prompt")
                    if get_judge_prompt is not None:
                        custom_prompt = get_judge_prompt(doc, prediction, target)
                        logger.debug("Using custom judge prompt from task module")
            except Exception as e:
                logger.debug(f"Failed to load custom judge prompt: {e}")

        if parse_judge_response is not None and (custom_messages is not None or custom_prompt is not None):
            try:
                from lmms_eval.llm_judge.protocol import Request

                SC = _get_server_config()
                base_config = self._judge_provider.config
                config = SC(
                    model_name=task_module_globals.get("JUDGE_MODEL", self.judge_model),
                    temperature=task_module_globals.get("JUDGE_TEMPERATURE", base_config.temperature),
                    max_tokens=task_module_globals.get("JUDGE_MAX_TOKENS", base_config.max_tokens),
                    top_p=task_module_globals.get("JUDGE_TOP_P", base_config.top_p),
                    timeout=task_module_globals.get("JUDGE_TIMEOUT", base_config.timeout),
                    num_retries=task_module_globals.get("JUDGE_NUM_RETRIES", base_config.num_retries),
                    retry_delay=task_module_globals.get("JUDGE_RETRY_DELAY", base_config.retry_delay),
                    max_concurrent=task_module_globals.get("JUDGE_MAX_CONCURRENT", base_config.max_concurrent),
                    response_format=task_module_globals.get("JUDGE_RESPONSE_FORMAT", base_config.response_format),
                )
                messages = custom_messages if custom_messages is not None else [{"role": "user", "content": custom_prompt}]
                response = self._judge_provider.evaluate(
                    Request(
                        messages=messages,
                        question=question,
                        answer=answer,
                        prediction=prediction,
                        config=config,
                    )
                )
                parsed = parse_judge_response(response.content)
                if update_metrics_from_judge is not None:
                    return update_metrics_from_judge(doc, results, fallback_metrics, parsed, response.content, response.model_used)

                metrics = fallback_metrics.copy()
                metrics["llm_judge_raw"] = response.content
                metrics["llm_judge_model"] = response.model_used
                metrics["llm_judge_success"] = parsed is not None
                if isinstance(parsed, bool):
                    metrics["llm_judge_score"] = int(parsed)
                    metrics["correct"] = parsed
                    self._sync_ocrbench_metric(
                        metrics,
                        score=int(parsed),
                        raw=response.content,
                        model=response.model_used,
                        success=parsed is not None,
                    )
                else:
                    metrics["llm_judge_result"] = parsed
                return metrics
            except Exception as e:
                logger.error(f"Custom LLM judge failed: {e}")
                return {
                    **fallback_metrics,
                    "llm_judge_error": str(e),
                    "llm_judge_failed": True,
                    "llm_judge_success": False,
                }
        
        # SFE-specific 0-10 scoring for mcq/exact_match questions
        is_sfe = self._current_task_name and "sfe" in self._current_task_name.lower()
        needs_score = fallback_metrics.get("needs_llm_judge") is True
        
        if is_sfe and needs_score:
            try:
                formatted_q = fallback_metrics.get("formatted_question", question)
                ans = fallback_metrics.get("answer", answer)
                sfe_template = """You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the ground truth, either by option letters or content. Include partial credit for answers that are close in content.
- For exact match and open-ended questions:
  * Assign a high score if the prediction matches the answer semantically, considering variations in format.
  * Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 10 to reflect your judgment of the answer's correctness.

# Strict Output format example
10"""
                # Use replace() instead of format() to safely handle literal braces in content
                sfe_prompt = sfe_template.replace("{question}", formatted_q).replace("{answer}", ans).replace("{pred}", prediction)
                
                judge_result = self._judge_provider.evaluate_score(
                    question=formatted_q,
                    answer=ans,
                    prediction=prediction,
                    score_range=(0, 10),
                    custom_prompt=sfe_prompt,
                )
                
                metrics = fallback_metrics.copy()
                score = float(judge_result.get("result", 0))
                metrics["llm_judge_score"] = score
                metrics["llm_judge_raw"] = judge_result.get("raw_response", "")
                metrics["llm_judge_model"] = judge_result.get("model", self.judge_model)
                metrics["llm_judge_success"] = judge_result.get("success", False)
                metrics["llm_judge_failed"] = False
                metrics["exact_match"] = score / 10.0
                
                # Update sfe_info so aggregate sees the new LLM score
                if "sfe_info" in metrics:
                    metrics["sfe_info"] = dict(metrics["sfe_info"])
                    metrics["sfe_info"]["llm_score"] = [str(int(score))]
                
                return metrics
                
            except Exception as e:
                logger.error(f"SFE LLM judge failed: {e}")
                return {
                    **fallback_metrics,
                    "llm_judge_error": str(e),
                    "llm_judge_failed": True,
                }
        
        try:
            # Call LLM judge
            judge_result = self._judge_provider.evaluate_binary(
                question=question,
                answer=answer,
                prediction=prediction,
                output_format="0/1",
                custom_prompt=custom_prompt,
            )
            
            # Merge with fallback metrics
            metrics = fallback_metrics.copy()
            metrics["llm_judge_score"] = int(judge_result.get("result", 0))
            metrics["llm_judge_raw"] = judge_result.get("raw_response", "")
            metrics["llm_judge_model"] = judge_result.get("model", self.judge_model)
            metrics["llm_judge_success"] = judge_result.get("success", False)
            metrics["llm_judge_failed"] = False
            self._sync_ocrbench_metric(
                metrics,
                score=metrics["llm_judge_score"],
                raw=metrics["llm_judge_raw"],
                model=metrics["llm_judge_model"],
                success=metrics["llm_judge_success"],
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return {
                **fallback_metrics,
                "llm_judge_error": str(e),
                "llm_judge_failed": True,
                "llm_judge_success": False,
            }
    
    def _init_judge_provider(self) -> None:
        """Initialize the LLM judge provider.
        
        Supports OpenAI, Azure, and local vLLM/SGLang servers via OpenAI-compatible API.
        For local vLLM, set JUDGE_BASE_URL to your vLLM endpoint (e.g., http://localhost:8000/v1).
        """
        if self._judge_provider is not None:
            return
        
        with self._provider_lock:
            if self._judge_provider is not None:
                return
            
            # Delayed imports
            PF = _get_provider_factory()
            SC = _get_server_config()
            
            # For local vLLM/SGLang, use a dummy key if not provided
            # OpenAI client requires a key, but local servers often don't validate it
            api_key = self.judge_api_key or "dummy-key-for-local-vllm"
            
            # Detect if using local vLLM/SGLang
            is_local = any(x in self.judge_base_url for x in ["localhost", "127.0.0.1", ":8000", ":30000"])
            if is_local:
                logger.info(f"Detected local LLM server at {self.judge_base_url}")
            
            # Set env vars for the provider.
            # Note: do NOT overwrite OPENAI_API_URL with a stripped URL,
            # because downstream task evaluators (e.g. MMBench) may read it
            # directly and expect the full /chat/completions endpoint.
            os.environ["OPENAI_API_KEY"] = api_key
            if not os.getenv("OPENAI_API_URL"):
                os.environ["OPENAI_API_URL"] = self.judge_base_url
            
            # Allow overriding API type for local servers
            api_type = os.getenv("JUDGE_API_TYPE", "openai")
            
            config = SC(
                model_name=self.judge_model,
                temperature=_env_float("JUDGE_TEMPERATURE", 0.0),
                max_tokens=_env_int("JUDGE_MAX_TOKENS", 16),
                timeout=_env_int("JUDGE_TIMEOUT", 60),
                num_retries=_env_int("JUDGE_NUM_RETRIES", 5),
                retry_delay=_env_float("JUDGE_RETRY_DELAY", 10.0),
                max_concurrent=self.parallel,
            )
            logger.info(
                f"Judge ServerConfig: model={config.model_name}, temperature={config.temperature}, "
                f"max_tokens={config.max_tokens}, timeout={config.timeout}, "
                f"num_retries={config.num_retries}, retry_delay={config.retry_delay}, "
                f"max_concurrent={config.max_concurrent}"
            )
            
            self._judge_provider = PF.create_provider(api_type=api_type, config=config)
            
            if is_local:
                logger.info(f"Initialized local LLM judge provider at {self.judge_base_url}")
            else:
                logger.info(f"Initialized cloud LLM judge provider with model: {self.judge_model}")
    
    def _extract_question(self, doc: Dict[str, Any], sample: Optional[Dict[str, Any]] = None) -> str:
        """Extract question text from doc.
        
        Tries multiple common field names. Falls back to sample input if doc is empty.
        """
        # Try common field names
        for key in ["question", "problem", "query", "prompt", "text"]:
            if key in doc and doc[key]:
                return str(doc[key])
        
        # Try to construct from available fields
        if "query_wo" in doc:  # MathVerse specific
            return str(doc["query_wo"])
        
        # Fallback to sample fields if doc is empty (common for offline JSONL judging)
        if sample is not None:
            for key in ["input", "prompt"]:
                if key in sample and sample[key]:
                    return str(sample[key])
        
        # Last resort: return string representation
        return str(doc)[:500]  # Limit length
    
    def _extract_existing_metrics(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pre-computed metrics from a serialized sample.
        
        The evaluation tracker drops the ``doc`` field before saving JSONL,
        but it preserves top-level metric keys. We harvest them here so that
        standalone judging can reuse the original rule-based scores instead of
        re-running ``process_results`` with an incomplete doc.
        
        Returns:
            Dict of metrics if any are found, otherwise an empty dict.
        """
        metrics = {}
        # Common scalar metrics
        for key in ["acc_score", "format_score", "accuracy", "correct", "score", "exact_match", "mathvision_qwen3_eval", "llm_judge_score"]:
            if key in sample:
                metrics[key] = sample[key]
        # Backward compatibility: old JSONL files use api_judge_accuracy as a placeholder
        if "api_judge_accuracy" in sample:
            metrics["llm_judge_score"] = sample["api_judge_accuracy"]
            metrics["needs_llm_judge"] = True
        # Nested metric objects (e.g. wemath_loose / wemath_strict)
        for key in ["wemath_loose", "wemath_strict", "scivqr_acc", "scivqr_open", "scivqr_reasoning", "ocrbench_accuracy"]:
            if key in sample:
                metrics[key] = sample[key]
        # SFE-specific fields needed for standalone judge and aggregation
        for key in ["formatted_question", "answer", "question_type",
                    "sfe_info", "raw_output", "rouge_score", "bertscore", "bleu_score",
                    "meteor_score", "execute_success_rate", "iou_score", "field", "id"]:
            if key in sample:
                metrics[key] = sample[key]
        # needs_llm_judge must be extracted so that tasks like SFE can enter their
        # dedicated scoring branch (e.g. 0-10 LLM scoring) during standalone judging.
        if "needs_llm_judge" in sample:
            metrics["needs_llm_judge"] = sample["needs_llm_judge"]
        return metrics
    
    def _needs_llm_judge(self, metrics: Dict[str, Any]) -> bool:
        """Check if rule-based judge failed and LLM judge is needed.
        
        Args:
            metrics: Metrics dict from rule-based judging
            
        Returns:
            True if LLM judge should be applied
        """
        # Explicit judge mode means re-judge every sample with LLM-as-judge.
        if self.judge_mode == "judge":
            return True
        # Check explicit flag from tasks that defer LLM judging to standalone phase
        if metrics.get("needs_llm_judge") is True:
            return True
        # OCRBench stores the per-sample score inside a nested metric object.
        ocrbench = metrics.get("ocrbench_accuracy")
        if isinstance(ocrbench, dict):
            val = ocrbench.get("score")
            if val == 0 or val is False:
                return True
            if isinstance(val, float) and val < 0.1:
                return True
        # Check common accuracy metrics
        for key in ["acc_score", "accuracy", "correct", "score", "exact_match", "gpt_eval_score"]:
            if key in metrics:
                val = metrics[key]
                # Consider 0 or False as needing fallback
                if val == 0 or val is False:
                    return True
                # For float values, check if very low
                if isinstance(val, float) and val < 0.1:
                    return True
        return False

    @staticmethod
    def _extract_answer_from_metrics(metrics: Dict[str, Any]) -> str:
        """Recover ground truth from serialized nested metrics when doc is absent."""
        ocrbench = metrics.get("ocrbench_accuracy")
        if isinstance(ocrbench, dict):
            answer = ocrbench.get("ground_truth")
            if answer is not None:
                return str(answer)
        return ""

    @staticmethod
    def _sync_ocrbench_metric(
        metrics: Dict[str, Any],
        score: int,
        raw: str,
        model: str,
        success: bool,
    ) -> None:
        """Write LLM judge output back to OCRBench's nested metric shape."""
        ocrbench = metrics.get("ocrbench_accuracy")
        if not isinstance(ocrbench, dict):
            return
        updated = dict(ocrbench)
        updated["score"] = int(score)
        updated["llm_judge_score"] = int(score)
        updated["llm_judge_raw"] = raw
        updated["llm_judge_model"] = model
        updated["llm_judge_success"] = bool(success)
        metrics["ocrbench_accuracy"] = updated

    def _is_scivqr_task(self) -> bool:
        return bool(self._current_task_name and self._current_task_name.startswith("scivqr"))

    def _attach_scivqr_subject_from_path(self, samples: List[Dict[str, Any]], input_path: Path) -> List[Dict[str, Any]]:
        if not self._is_scivqr_task():
            return samples
        suffix = "_results"
        stem = input_path.stem
        if not stem.endswith(suffix):
            return samples
        subject = stem[: -len(suffix)]
        if subject not in SCIVQR_SUBJECTS:
            return samples
        enriched_samples = []
        for sample in samples:
            if isinstance(sample, dict) and not sample.get("subject"):
                sample = dict(sample)
                sample["subject"] = subject
                sample["__scivqr_subject_from_path"] = True
            enriched_samples.append(sample)
        return enriched_samples

    def _enrich_scivqr_reasoning_doc(self, doc: Dict[str, Any], sample: Dict[str, Any], task: Any) -> Dict[str, Any]:
        """Attach the official SciVQR reasoning reference solution when possible.

        The official reasoning evaluator rebuilds ``gt_reason`` from the SciVQR
        dataset, keyed by the predicted row's question text. Standalone judging
        of official ``*_results.jsonl`` files should do the same when those rows
        contain only ``prompt``/``response``/``answer``.
        """
        if self._current_task_name != "scivqr_reasoning":
            return doc
        if not isinstance(doc, dict) or "__sample_context__" not in doc:
            return doc
        context = doc.get("__sample_context__") or {}
        if context.get("gt_reason") or context.get("solution"):
            return doc

        dataset_doc = self._lookup_scivqr_dataset_doc(task, context)
        if not dataset_doc or not dataset_doc.get("solution"):
            return doc

        enriched = dict(dataset_doc)
        enriched.setdefault("gt_reason", dataset_doc.get("solution", ""))
        return enriched

    def _lookup_scivqr_dataset_doc(self, task: Any, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        by_pid, by_question, by_subject_question = self._get_scivqr_doc_lookup(task)

        question = self._extract_scivqr_question_text(sample)
        normalized = self._normalize_scivqr_question(question)
        subject = sample.get("subject")
        if subject and normalized:
            subject_key = (str(subject), normalized)
            if subject_key in by_subject_question:
                return by_subject_question[subject_key]
            if sample.get("__scivqr_subject_from_path"):
                return None

        for key in ["pid", "question_id", "doc_id"]:
            value = sample.get(key)
            if value is not None and str(value) in by_pid:
                return by_pid[str(value)]

        if normalized and normalized in by_question:
            return by_question[normalized]
        return None

    def _get_scivqr_doc_lookup(self, task: Any):
        cache_key = id(task)
        if cache_key in self._scivqr_doc_lookup_cache:
            return self._scivqr_doc_lookup_cache[cache_key]

        docs = []
        try:
            if hasattr(task, "eval_docs_no_media"):
                docs = list(task.eval_docs_no_media)
            elif task.has_test_docs():
                docs = list(task.test_docs())
            else:
                docs = list(task.validation_docs())
        except Exception as e:
            logger.debug(f"Could not preload SciVQR docs for reasoning lookup: {e}")

        by_pid = {}
        by_question = {}
        by_subject_question = {}
        for item in docs:
            if not isinstance(item, dict):
                continue
            pid = item.get("pid")
            if pid is not None:
                by_pid[str(pid)] = item
            question_key = self._normalize_scivqr_question(item.get("question", ""))
            if question_key:
                by_question[question_key] = item
                subject = item.get("subject")
                if subject:
                    by_subject_question[(str(subject), question_key)] = item

        self._scivqr_doc_lookup_cache[cache_key] = (by_pid, by_question, by_subject_question)
        return by_pid, by_question, by_subject_question

    @staticmethod
    def _extract_scivqr_question_text(sample: Dict[str, Any]) -> str:
        text = str(sample.get("question") or sample.get("prompt") or sample.get("input") or "")
        match = re.search(r"\\boxed\{\}.*?\n(.*?)(?:\nChoices:|\Z)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        text = re.split(r"\n[A-E]\.\s+", text, maxsplit=1)[0]
        text = text.split("\nChoices:")[0]
        for suffix in [
            "\nAnswer with the option's letter from the given choices directly.",
            "\nAnswer the question directly.",
        ]:
            text = text.replace(suffix, "")
        return text.strip()

    @staticmethod
    def _normalize_scivqr_question(question: Any) -> str:
        text = str(question or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\s+", " ", text).strip()
    
    def _load_task(self, task_name: str) -> Any:
        """Load task by name.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task object
            
        Raises:
            ValueError: If task cannot be loaded
        """
        try:
            task_manager = _get_task_manager()
            task_dict = task_manager.load_task_or_group(task_name)
            tasks = list(task_dict.values())
            if not tasks:
                raise ValueError(f"Task {task_name} not found")
            return tasks[0]
        except Exception as e:
            raise ValueError(f"Failed to load task {task_name}: {e}")
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file.
        
        Args:
            path: Path to JSONL file
            
        Returns:
            List of sample dicts
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        
        return samples
    
    def compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate summary from judged results.
        
        Args:
            results: List of judged samples with metrics
            
        Returns:
            Dict mapping metric names to aggregated values (mean for numeric).
        """
        if not results:
            return {}
        
        # Whitelist of meaningful nested fields to extract
        MEANINGFUL_NESTED_FIELDS = {'hit', 'score', 'correct', 'accuracy'}
        
        metric_values: Dict[str, List[float]] = {}
        for sample in results:
            metrics = sample.get("metrics", {})
            for key, val in metrics.items():
                if isinstance(val, bool):
                    metric_values.setdefault(key, []).append(float(val))
                elif isinstance(val, (int, float)):
                    metric_values.setdefault(key, []).append(float(val))
                elif isinstance(val, dict):
                    # Only extract meaningful nested fields (hit, score, correct, accuracy)
                    for nested_key in MEANINGFUL_NESTED_FIELDS:
                        if nested_key in val and isinstance(val[nested_key], (int, float)):
                            nested_metric_name = f"{key}.{nested_key}"
                            metric_values.setdefault(nested_metric_name, []).append(float(val[nested_key]))
                    # Special case: MMBench gpt_eval_score dict contains answer/prediction
                    if key == "gpt_eval_score" and "answer" in val and "prediction" in val:
                        ans = val["answer"]
                        pred = val["prediction"]
                        if ans is not None and pred is not None:
                            metric_values.setdefault("accuracy", []).append(1.0 if str(ans) == str(pred) else 0.0)
        
        summary = {}
        for key, values in metric_values.items():
            if values:
                summary[key] = round(sum(values) / len(values), 4)
        
        # Compute combined accuracy metrics for two common patterns:
        # 1) wemath/mathvision: flat acc_score + llm_judge_score
        # 2) mmmu/mmmu_pro: nested dict with hit + extraction_method + extraction_flag
        
        # Detect SFE (uses 0-10 scoring, not binary 0/1)
        is_sfe = any("sfe_info" in r.get("metrics", {}) for r in results[:1] if r)
        
        # --- Case 1: flat scores ---
        acc_scores = []
        llm_scores = []
        for sample in results:
            metrics = sample.get("metrics", {})
            if "acc_score" in metrics:
                acc_scores.append(float(metrics["acc_score"]))

            elif "exact_match" in metrics:
                acc_scores.append(float(metrics["exact_match"]))
            elif "gpt_eval_score" in metrics and isinstance(metrics["gpt_eval_score"], dict):
                # MMBench-style dict with answer/prediction
                gpt_val = metrics["gpt_eval_score"]
                ans = gpt_val.get("answer")
                pred = gpt_val.get("prediction")
                if ans is not None and pred is not None:
                    acc_scores.append(1.0 if str(ans) == str(pred) else 0.0)
            if "llm_judge_score" in metrics and metrics["llm_judge_score"] >= 0:
                llm_scores.append(float(metrics["llm_judge_score"]))
        
        if is_sfe and acc_scores:
            # For SFE, exact_match already encodes the final score
            # (rouge for open_ended, iou for bbox, llm_score/10 for mcq/exact_match)
            summary["exact_match"] = round(sum(acc_scores) / len(acc_scores), 4)
            if llm_scores:
                summary["llm_judge_score_avg"] = round(sum(llm_scores) / len(llm_scores), 4)
            # Unify final metric name across all tasks
            summary["total_acc"] = summary["exact_match"]
        elif acc_scores and llm_scores:
            rule_acc = sum(acc_scores) / len(acc_scores)
            llm_fallback_acc_raw = sum(llm_scores) / len(llm_scores)
            llm_fallback_acc = (1 - rule_acc) * llm_fallback_acc_raw
            total_acc = rule_acc + llm_fallback_acc
            summary["rule_acc"] = round(rule_acc, 4)
            summary["llm_fallback_acc"] = round(llm_fallback_acc, 4)
            summary["total_acc"] = round(total_acc, 4)
        elif llm_scores:
            # Pure LLM-judge tasks (e.g. MolParse, OpenRxn) have no rule-based score
            summary["total_acc"] = round(sum(llm_scores) / len(llm_scores), 4)
        
        # --- Case 2: nested official metrics (mmmu / mmmu_pro) ---
        rule_hits = 0.0
        fallback_hits = 0.0
        total_count = 0
        for sample in results:
            metrics = sample.get("metrics", {})
            for val in metrics.values():
                if isinstance(val, dict) and "hit" in val and "extraction_method" in val:
                    hit = float(val["hit"])
                    method = val.get("extraction_method", "")
                    flag = val.get("extraction_success", False)
                    
                    total_count += 1
                    if method == "rule" and flag:
                        rule_hits += hit
                    else:
                        fallback_hits += hit
        
        if total_count > 0:
            rule_acc = rule_hits / total_count
            llm_fallback_acc = fallback_hits / total_count
            total_acc = rule_acc + llm_fallback_acc
            summary["rule_acc"] = round(rule_acc, 4)
            summary["llm_fallback_acc"] = round(llm_fallback_acc, 4)
            summary["total_acc"] = round(total_acc, 4)
        
        return summary
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Save judged results to JSONL.
        
        Args:
            results: List of judged samples
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_hook = self._get_task_save_hook()
        if save_hook is not None:
            save_hook(results, output_path)
            logger.debug(f"Saved {len(results)} task-formatted results to {output_path}")
            return
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in results:
                # Handle non-serializable objects
                cleaned = self._clean_for_json(sample)
                f.write(json.dumps(cleaned, ensure_ascii=False, default=str) + "\n")
        
        logger.debug(f"Saved {len(results)} results to {output_path}")

    def _get_task_save_hook(self) -> Optional[Callable]:
        if self._current_task is None:
            return None
        try:
            process_results_fn = getattr(self._current_task.config, "process_results", None)
            if process_results_fn is None:
                return None
            task_module_globals = getattr(process_results_fn, "__globals__", {})
            hook = task_module_globals.get("save_judged_results")
            return hook if callable(hook) else None
        except Exception as e:
            logger.debug(f"Failed to load task save hook: {e}")
            return None
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization.
        
        Recursively converts non-serializable objects to strings.
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
