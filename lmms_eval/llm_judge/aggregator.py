"""Aggregator for judged results with task-specific logic.

This module handles the final aggregation step that converts per-sample judged
results into task-level metrics, supporting complex aggregation logic like
WeMath's multi-step analysis.
"""

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# Delay import of TaskManager to avoid heavy imports
TaskManager = None


def _get_task_manager():
    global TaskManager
    if TaskManager is None:
        from lmms_eval.tasks import TaskManager as TM
        TaskManager = TM
    return TaskManager()


class Aggregator:
    """Aggregator for converting per-sample results to task-level metrics.
    
    This class handles:
    1. Loading task-specific aggregation functions
    2. Preparing data for aggregation (handling WeMath's multi-step structure)
    3. Executing aggregation and returning final metrics
    
    Example:
        >>> aggregator = Aggregator()
        >>> samples = [...]  # Judged samples from JSONL
        >>> results = aggregator.aggregate(samples, "wemath_testmini_reasoning")
        >>> print(results["Score (Loose)"])
        '65.43%'
    """
    
    # Registry of tasks that require special aggregation logic
    # Maps task name patterns to their aggregation module paths
    SPECIAL_AGGREGATIONS = {
        "wemath": {
            "module": "lmms_eval.tasks.wemath.reasoning.utils",
            "loose_func": "wemath_aggregate_results_loose",
            "strict_func": "wemath_aggregate_results_strict",
            "data_key": "wemath_loose",  # Key in metrics to extract data from
        },
        "mathvision": {
            "module": "lmms_eval.tasks.mathvision.utils",
            "accuracy_func": "mathvision_aggregate_results_eval",
            "data_key": "mathvision_standard_eval",
            "score_key": "scores",  # Key within data to extract scores
            "exclude_patterns": ["mathvision_reason"],
        },
        "mathvision_testmini_qwen3": {
            "module": "lmms_eval.tasks.mathvision.utils_qwen3",
            "accuracy_func": "mathvision_aggregate_results_qwen3",
            "data_key": "mathvision_qwen3_eval",
            "score_key": "scores",
        },
        "mmmu_val_qwen3_official": {
            "module": "lmms_eval.tasks.mmmu.utils_qwen3_official",
            "accuracy_func": "mmmu_qwen3_official_aggregate_accuracy",
            "data_key": "mmmu_acc_official",  # Key in metrics to extract data from
        },
        "mmmu_pro": {
            "module": "lmms_eval.tasks.mmmu_pro_qwen3_official.utils_qwen3_official",
            "accuracy_func": "mmmu_pro_qwen3_official_aggregate_accuracy",
            "data_key": "mmmu_pro_acc_official",
        },
        "scivqr_reasoning": {
            "module": "lmms_eval.tasks.scivqr.reasoning.utils",
            "accuracy_func": "scivqr_reasoning_aggregate_scores",
            "data_key": "scivqr_reasoning",
            "prefer_metrics": True,
        },
        "scivqr_open": {
            "module": "lmms_eval.tasks.scivqr.utils",
            "accuracy_func": "scivqr_standalone_aggregate_accuracy",
            "data_key": "scivqr_open",
            "prefer_metrics": True,
        },
        "scivqr_mcq": {
            "module": "lmms_eval.tasks.scivqr.utils",
            "accuracy_func": "scivqr_standalone_aggregate_accuracy",
            "data_key": "scivqr_acc",
            "prefer_metrics": True,
        },
        "mmbench_en_dev": {
            "module": "lmms_eval.tasks.mmbench.en_utils",
            "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
            "data_key": "gpt_eval_score",
        },
        "mmbench_en_test": {
            "module": "lmms_eval.tasks.mmbench.en_utils",
            "accuracy_func": "mmbench_aggregate_test_results_standalone",
            "data_key": "submission",
        },
        "mmbench_cn_dev": {
            "module": "lmms_eval.tasks.mmbench.cn_utils",
            "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
            "data_key": "gpt_eval_score",
        },
        "mmbench_cn_test": {
            "module": "lmms_eval.tasks.mmbench.cn_utils",
            "accuracy_func": "mmbench_aggregate_test_results_standalone",
            "data_key": "submission",
        },
        "mmbench_ru_dev": {
            "module": "lmms_eval.tasks.mmbench.ru_utils",
            "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
            "data_key": "gpt_eval_score",
        },
        "mmbench_ko_dev": {
            "module": "lmms_eval.tasks.mmbench.ko_utils",
            "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
            "data_key": "gpt_eval_score",
        },
        "mmbench_cn_cc": {
            "module": "lmms_eval.tasks.mmbench.cc_utils",
            "accuracy_func": "mmbench_cn_cc_aggregate_dev_results_eval_standalone",
            "data_key": "gpt_eval_score",
        },
        "sfe": {
            "module": "lmms_eval.tasks.sfe.utils",
            "accuracy_func": "sfe_standalone_aggregate",
            "data_key": "sfe_info",
        },
        "simplevqa": {
            "module": "lmms_eval.tasks.simplevqa.utils",
            "accuracy_func": "simplevqa_standalone_aggregate",
            "data_key": "simplevqa_judge",
        },
    }
    
    def __init__(self):
        self._task_manager = None
        self._cache = {}  # Cache for loaded aggregation functions
    
    def aggregate(
        self,
        samples: List[Dict[str, Any]],
        task_name: str,
        metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-sample results into task-level metrics.
        
        Args:
            samples: List of judged samples from JSONL
            task_name: Task name for loading aggregation function
            metric_name: Specific metric to aggregate (if None, aggregates all)
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Check if this is a special aggregation task
        special_config = self._get_special_config(task_name)
        
        if special_config:
            return self._aggregate_special(samples, task_name, special_config, metric_name)
        else:
            return self._aggregate_generic(samples, task_name, metric_name)
    
    def _get_special_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Check if task requires special aggregation.
        
        Args:
            task_name: Task name to check
            
        Returns:
            Special config dict if task needs special handling, None otherwise
        """
        import re
        task_lower = task_name.lower()
        # Sort by pattern length descending so more specific patterns match first
        for pattern, config in sorted(self.SPECIAL_AGGREGATIONS.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"(^|_){re.escape(pattern)}(_|$)", task_lower):
                # Check exclusion patterns if defined
                exclude_patterns = config.get("exclude_patterns", [])
                if any(exc in task_lower for exc in exclude_patterns):
                    continue
                return config
        return None
    
    def _aggregate_special(
        self,
        samples: List[Dict[str, Any]],
        task_name: str,
        config: Dict[str, Any],
        metric_name: Optional[str],
    ) -> Dict[str, Any]:
        """Handle special aggregation for tasks like WeMath.
        
        Args:
            samples: Judged samples
            task_name: Task name
            config: Special aggregation config
            metric_name: Specific metric to aggregate
            
        Returns:
            Aggregated metrics
        """
        logger.info(f"Using special aggregation for {task_name}")
        
        # Extract data from samples based on config
        # Try both top-level and nested in metrics
        data_key = config["data_key"]
        extracted_data = []
        
        for sample in samples:
            data_dict = None
            
            if config.get("prefer_metrics") and "metrics" in sample and isinstance(sample["metrics"], dict):
                if data_key in sample["metrics"]:
                    data_dict = sample["metrics"][data_key]
            # First try top-level (where WeMath data is stored)
            if data_dict is None and data_key in sample:
                data_dict = sample[data_key]
            # Then try nested in metrics
            elif data_dict is None and "metrics" in sample and isinstance(sample["metrics"], dict):
                if data_key in sample["metrics"]:
                    data_dict = sample["metrics"][data_key]
            
            if data_dict is not None:
                if isinstance(data_dict, dict):
                    extracted_data.append(data_dict)
                else:
                    logger.warning(f"Unexpected data type for {data_key}: {type(data_dict)}")
            else:
                logger.debug(f"Sample missing {data_key}")
        
        if not extracted_data:
            logger.warning(f"No data extracted for aggregation from {len(samples)} samples")
            return {}
        
        logger.info(f"Extracted {len(extracted_data)} data records for aggregation")
        
        # Load and execute aggregation functions
        results = {}
        
        try:
            module = importlib.import_module(config["module"])
            
            # Handle WeMath-style aggregation (loose/strict)
            if "loose_func" in config and "strict_func" in config:
                # Aggregate loose metric if requested
                if metric_name is None or metric_name == "wemath_loose":
                    loose_func = getattr(module, config["loose_func"])
                    loose_score = loose_func(extracted_data)
                    results["Score (Loose)"] = loose_score
                    logger.info(f"Loose Score: {loose_score}")
                
                # Aggregate strict metric if requested
                if metric_name is None or metric_name == "wemath_strict":
                    strict_func = getattr(module, config["strict_func"])
                    strict_score = strict_func(extracted_data)
                    results["Score (Strict)"] = strict_score
                    logger.info(f"Strict Score: {strict_score}")
            
            # Handle MathVision-style aggregation (accuracy from scores array)
            elif "accuracy_func" in config:
                accuracy_func = getattr(module, config["accuracy_func"])
                accuracy = accuracy_func(extracted_data)
                if isinstance(accuracy, dict):
                    results.update(accuracy)
                    if "exact_match" in accuracy:
                        logger.info(f"exact_match: {accuracy['exact_match']}")
                    elif "accuracy" in accuracy:
                        logger.info(f"Accuracy: {accuracy['accuracy']}%")
                elif accuracy is None:
                    logger.info("Accuracy function returned None (expected for test splits)")
                else:
                    results["accuracy"] = accuracy
                    logger.info(f"Accuracy: {accuracy}%")
                
        except ImportError as e:
            logger.error(f"Failed to import aggregation module: {e}")
            raise
        except Exception as e:
            logger.error(f"Aggregation function failed: {e}")
            raise
        
        return results
    
    def _aggregate_generic(
        self,
        samples: List[Dict[str, Any]],
        task_name: str,
        metric_name: Optional[str],
    ) -> Dict[str, Any]:
        """Handle generic aggregation for simple tasks.
        
        For tasks without special aggregation requirements, this computes
        simple averages for numeric metrics.
        
        Args:
            samples: Judged samples
            task_name: Task name
            metric_name: Specific metric to aggregate
            
        Returns:
            Aggregated metrics
        """
        logger.info(f"Using generic aggregation for {task_name}")
        
        # Collect all numeric metrics
        metric_values: Dict[str, List[float]] = {}
        
        for sample in samples:
            metrics = sample.get("metrics", {})
            for key, value in metrics.items():
                # Skip non-numeric and complex nested structures
                if isinstance(value, bool):
                    metric_values.setdefault(key, []).append(float(value))
                elif isinstance(value, (int, float)):
                    metric_values.setdefault(key, []).append(float(value))
                # Skip dict/list values (like WeMath's per-sample data dicts)
        
        # Compute averages
        results = {}
        for key, values in metric_values.items():
            if metric_name and key != metric_name:
                continue
            if values:
                results[key] = round(sum(values) / len(values), 4)
        
        return results
    
    def get_available_metrics(self, task_name: str) -> List[str]:
        """Get list of available metrics for a task.
        
        Args:
            task_name: Task name
            
        Returns:
            List of metric names
        """
        special_config = self._get_special_config(task_name)
        
        if special_config:
            metrics = []
            if "loose_func" in special_config:
                metrics.append("wemath_loose")
            if "strict_func" in special_config:
                metrics.append("wemath_strict")
            return metrics
        
        # For generic tasks, we'd need to load the task config
        # This is a simplified version
        return []
