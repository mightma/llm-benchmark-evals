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
                        # Multiple responses - generate with throttling to prevent server congestion
                        # Get max concurrent responses from config, default to 10
                        config_max_concurrent = self.config.get("inference", {}).get("max_concurrent_responses", 10)
                        max_concurrent_responses = min(num_responses, config_max_concurrent)

                        if max_concurrent_responses < num_responses:
                            logger.info(f"Throttling {num_responses} responses to max {max_concurrent_responses} concurrent requests for sample {sample.id}")

                        response_semaphore = asyncio.Semaphore(max_concurrent_responses)

                        async def generate_single_response(response_idx: int):
                            async with response_semaphore:  # Throttle concurrent requests
                                logger.debug(f"Generating response {response_idx+1}/{num_responses} for sample {sample.id}")
                                response = await inference_client.generate(
                                    prompt=sample.input_text,
                                    model_name=model_name
                                )
                                model_output = response["choices"][0]["message"]["content"]
                                eval_result = await self.evaluate_sample(sample, model_output)
                                return model_output, eval_result

                        # Generate multiple responses with throttling
                        response_tasks = [generate_single_response(idx) for idx in range(num_responses)]
                        response_results = await asyncio.gather(*response_tasks)

                        responses = [result[0] for result in response_results]
                        eval_results = [result[1] for result in response_results]

                    # Handle multiple responses as separate samples
                    if num_responses == 1:
                        # Single response - use as-is
                        final_eval_result = eval_results[0]
                        best_response = responses[0]

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

                        return i, [final_eval_result], [prediction_data] if save_predictions else [None]
                    else:
                        # Multiple responses - treat each as a separate sample
                        prediction_data_list = []
                        if save_predictions:
                            for response_idx, (response, eval_result) in enumerate(zip(responses, eval_results)):
                                prediction_data = {
                                    "id": f"{sample.id}_response_{response_idx + 1}",
                                    "original_id": sample.id,
                                    "response_index": response_idx + 1,
                                    "total_responses": num_responses,
                                    "input": sample.input_text,
                                    "expected": sample.expected_output,
                                    "predicted": response,
                                    "evaluation": eval_result,
                                    "metadata": sample.metadata
                                }
                                prediction_data_list.append(prediction_data)
                        else:
                            prediction_data_list = [None] * num_responses

                        return i, eval_results, prediction_data_list

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
                    return i, [failed_result], [None]

        # Process all samples concurrently
        logger.info("Starting concurrent processing...")
        tasks = [process_sample(i, sample) for i, sample in enumerate(samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and flatten multiple responses into individual samples
        sample_results = []
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
                index, eval_results_list, prediction_data_list = result

                # Add each evaluation result as a separate sample
                for eval_result in eval_results_list:
                    sample_results.append(eval_result)

                # Add each prediction data (if not None)
                for prediction_data in prediction_data_list:
                    if prediction_data:
                        predictions.append(prediction_data)

        logger.info(f"Completed processing {len(sample_results)} samples")

        # Aggregate results
        aggregated = self.aggregate_results(sample_results)

        # Create evaluation result
        # Update num_samples to reflect actual samples evaluated (including multiple responses)
        original_samples_count = len(samples)
        total_evaluated_samples = len(sample_results)

        result = EvaluationResult(
            benchmark_name=self.name,
            model_name=model_name,
            score=aggregated.get("overall_score", 0.0),
            details={
                **aggregated,
                "original_samples": original_samples_count,
                "responses_per_sample": num_responses,
                "total_evaluated_samples": total_evaluated_samples
            },
            num_samples=total_evaluated_samples,  # Total evaluated samples (original * num_responses)
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
        """Aggregate multiple choice results, excluding inference failures."""
        total_samples = len(sample_results)

        # Separate successful and failed samples
        successful_samples = [r for r in sample_results if not r.get("error")]
        failed_samples = [r for r in sample_results if r.get("error")]

        # Count correct among successful samples
        correct_samples = sum(1 for r in successful_samples if r.get("correct", False))
        successful_count = len(successful_samples)
        failed_count = len(failed_samples)

        # Calculate accuracy based on successful samples only
        accuracy = correct_samples / successful_count if successful_count > 0 else 0.0

        # Categorize failure types
        failure_types = {}
        for failed in failed_samples:
            error_type = failed.get("error_type", "unknown_error")
            failure_types[error_type] = failure_types.get(error_type, 0) + 1

        return {
            "overall_score": accuracy,
            "accuracy": accuracy,
            "correct": correct_samples,
            "successful_samples": successful_count,
            "failed_samples": failed_count,
            "total_samples": total_samples,
            "success_rate": successful_count / total_samples if total_samples > 0 else 0.0,
            "failure_rate": failed_count / total_samples if total_samples > 0 else 0.0,
            "error_rate": 1.0 - accuracy,  # Among successful samples
            "failure_types": failure_types,
            # Legacy fields for compatibility
            "total": successful_count  # For backward compatibility, use successful count
        }