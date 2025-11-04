"""
Base classes for benchmark evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    benchmark_name: str
    model_name: str
    score: float
    details: Dict[str, Any]
    num_samples: int
    timestamp: datetime
    config: Dict[str, Any]


@dataclass
class EvaluationSample:
    """Container for a single evaluation sample."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, config: Dict[str, Any], data_path: str):
        self.config = config
        self.data_path = data_path
        self.name = self.__class__.__name__.lower().replace('benchmark', '')

    @abstractmethod
    async def load_data(self) -> List[EvaluationSample]:
        """Load benchmark data."""
        pass

    @abstractmethod
    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        model_response: str
    ) -> Dict[str, Any]:
        """Evaluate a single sample."""
        pass

    @abstractmethod
    def aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all samples."""
        pass

    def _combine_responses(
        self,
        sample: EvaluationSample,
        responses: List[str],
        eval_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Combine multiple responses for a single sample.
        Default implementation uses majority voting for correct answers.
        Subclasses can override for custom combination strategies.

        Returns:
            Tuple of (final_evaluation_result, best_response)
        """
        if not responses or not eval_results:
            return {"correct": False, "score": 0.0, "error": "No responses"}, ""

        # Count correct responses
        correct_results = [result for result in eval_results if result.get("correct", False)]
        correct_count = len(correct_results)

        if correct_count == 0:
            # No correct responses - return the first one
            return eval_results[0], responses[0]
        elif correct_count == 1:
            # One correct response - use it
            correct_idx = next(i for i, result in enumerate(eval_results) if result.get("correct", False))
            return eval_results[correct_idx], responses[correct_idx]
        else:
            # Multiple correct responses - use majority voting on predicted answers
            return self._majority_vote_responses(sample, responses, eval_results)

    def _majority_vote_responses(
        self,
        sample: EvaluationSample,
        responses: List[str],
        eval_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Use majority voting to select the best response.
        """
        from collections import Counter

        # Count predicted answers
        predicted_answers = []
        for result in eval_results:
            if result.get("correct", False):
                predicted_answers.append(result.get("predicted", ""))

        if not predicted_answers:
            # No correct predictions - return first response
            return eval_results[0], responses[0]

        # Find most common correct answer
        answer_counts = Counter(predicted_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]

        # Find the first response that gave this answer
        for i, result in enumerate(eval_results):
            if result.get("predicted") == most_common_answer and result.get("correct", False):
                # Enhance the result with majority voting info
                enhanced_result = result.copy()
                enhanced_result["majority_vote_count"] = answer_counts[most_common_answer]
                enhanced_result["total_responses"] = len(responses)
                enhanced_result["correct_responses"] = len([r for r in eval_results if r.get("correct", False)])
                return enhanced_result, responses[i]

        # Fallback - return first correct response
        correct_idx = next(i for i, result in enumerate(eval_results) if result.get("correct", False))
        return eval_results[correct_idx], responses[correct_idx]

    async def run_evaluation(
        self,
        inference_client,
        model_name: str,
        num_samples: Optional[int] = None,
        num_responses: int = 1,
        save_predictions: bool = True,
        output_dir: str = "results"
    ) -> EvaluationResult:
        """Run the complete evaluation."""
        logger.info(f"Starting {self.name} evaluation for model: {model_name}")

        # Load data
        samples = await self.load_data()
        if num_samples:
            samples = samples[:num_samples]

        logger.info(f"Evaluating {len(samples)} samples")

        # Get concurrency settings from config
        max_concurrent = self.config.get("max_concurrent", 4)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"Using {max_concurrent} concurrent workers for evaluation")

        # Run inference and evaluation concurrently
        async def process_sample(i: int, sample: EvaluationSample) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
            """Process a single sample with concurrency control."""
            async with semaphore:
                logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.id}")

                try:
                    # Generate multiple responses for this sample
                    responses = []
                    eval_results = []

                    # Generate responses (can be done concurrently for multiple responses)
                    if num_responses == 1:
                        # Single response
                        response = await inference_client.generate(
                            prompt=sample.input_text,
                            model_name=model_name
                        )
                        model_output = response["choices"][0]["message"]["content"]
                        responses.append(model_output)

                        eval_result = await self.evaluate_sample(sample, model_output)
                        eval_results.append(eval_result)
                    else:
                        # Multiple responses - generate concurrently
                        async def generate_single_response(response_idx: int):
                            logger.debug(f"Generating response {response_idx+1}/{num_responses} for sample {sample.id}")
                            response = await inference_client.generate(
                                prompt=sample.input_text,
                                model_name=model_name
                            )
                            model_output = response["choices"][0]["message"]["content"]
                            eval_result = await self.evaluate_sample(sample, model_output)
                            return model_output, eval_result

                        # Generate multiple responses concurrently
                        response_tasks = [generate_single_response(idx) for idx in range(num_responses)]
                        response_results = await asyncio.gather(*response_tasks)

                        responses = [result[0] for result in response_results]
                        eval_results = [result[1] for result in response_results]

                    # Combine multiple responses using the appropriate strategy
                    if num_responses == 1:
                        # Single response - use as-is
                        final_eval_result = eval_results[0]
                        best_response = responses[0]
                    else:
                        # Multiple responses - use majority voting or best response
                        final_eval_result, best_response = self._combine_responses(
                            sample, responses, eval_results
                        )

                    # Create prediction data for saving
                    prediction_data = None
                    if save_predictions:
                        prediction_data = {
                            "id": sample.id,
                            "input": sample.input_text,
                            "expected": sample.expected_output,
                            "predicted": best_response,
                            "evaluation": final_eval_result,
                            "metadata": sample.metadata
                        }

                        # Add all responses if multiple were generated
                        if num_responses > 1:
                            prediction_data["all_responses"] = responses
                            prediction_data["all_evaluations"] = eval_results
                            prediction_data["num_responses"] = num_responses

                    return i, final_eval_result, prediction_data

                except Exception as e:
                    error_msg = f"Error processing sample {sample.id}: {e}"
                    logger.error(error_msg)

                    # Add more specific error context for HTTP errors
                    if "HTTP" in str(e) or "Connection" in str(e) or "Timeout" in str(e):
                        logger.error(f"Network/HTTP error details: {type(e).__name__}: {e}")
                        error_type = "network_error"
                    else:
                        error_type = "processing_error"

                    # Return failed sample result
                    failed_result = {
                        "correct": False,
                        "error": str(e),
                        "error_type": error_type,
                        "score": 0.0
                    }
                    return i, failed_result, None

        # Process all samples concurrently
        logger.info("Starting concurrent processing...")
        tasks = [process_sample(i, sample) for i, sample in enumerate(samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and maintain order
        sample_results = [None] * len(samples)
        predictions = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                # Add a default failed result
                sample_results.append({
                    "correct": False,
                    "error": str(result),
                    "error_type": "task_error",
                    "score": 0.0
                })
            else:
                index, eval_result, prediction_data = result
                sample_results[index] = eval_result

                if prediction_data:
                    predictions.append(prediction_data)

        # Filter out None results (shouldn't happen, but be safe)
        sample_results = [r for r in sample_results if r is not None]

        logger.info(f"Completed processing {len(sample_results)} samples")

        # Aggregate results
        aggregated = self.aggregate_results(sample_results)

        # Create evaluation result
        result = EvaluationResult(
            benchmark_name=self.name,
            model_name=model_name,
            score=aggregated.get("overall_score", 0.0),
            details=aggregated,
            num_samples=len(samples),
            timestamp=datetime.now(),
            config=self.config
        )

        # Save predictions if requested
        if save_predictions and predictions:
            await self._save_predictions(predictions, model_name, output_dir)

        logger.info(f"Completed {self.name} evaluation. Score: {result.score:.4f}")
        return result

    async def _save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        model_name: str,
        output_dir: str
    ):
        """Save predictions to file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{model_name.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved predictions to {filepath}")


class MultipleChoiceBenchmark(BaseBenchmark):
    """Base class for multiple choice benchmarks."""

    def extract_choice(self, response: str) -> str:
        """Extract choice from model response."""
        response = response.strip().upper()

        # Look for patterns like "A)", "(A)", "A.", "A:", "The answer is A"
        import re

        patterns = [
            r'\b([A-Z])\)',  # A)
            r'\(([A-Z])\)',  # (A)
            r'\b([A-Z])\.',  # A.
            r'\b([A-Z]):',   # A:
            r'answer is ([A-Z])',  # answer is A
            r'Answer: ([A-Z])',    # Answer: A
            r'^([A-Z])\b',   # A at start
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # If no clear pattern, try to find any single letter A-Z
        letters = re.findall(r'\b[A-Z]\b', response)
        if letters:
            return letters[0]

        return "UNKNOWN"

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        model_response: str
    ) -> Dict[str, Any]:
        """Evaluate multiple choice sample."""
        predicted_choice = self.extract_choice(model_response)
        expected_choice = sample.expected_output

        correct = predicted_choice == expected_choice

        return {
            "correct": correct,
            "predicted": predicted_choice,
            "expected": expected_choice,
            "score": 1.0 if correct else 0.0,
            "raw_response": model_response
        }

    def aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple choice results."""
        total_samples = len(sample_results)
        correct_samples = sum(1 for r in sample_results if r.get("correct", False))

        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        return {
            "overall_score": accuracy,
            "accuracy": accuracy,
            "correct": correct_samples,
            "total": total_samples,
            "error_rate": 1.0 - accuracy
        }