"""
Base classes for benchmark evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import logging
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

        # Run inference and evaluation
        sample_results = []
        predictions = []

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}")

            try:
                # Get model response
                response = await inference_client.generate(
                    prompt=sample.input_text,
                    model_name=model_name
                )

                model_output = response["choices"][0]["message"]["content"]

                # Evaluate the response
                eval_result = await self.evaluate_sample(sample, model_output)
                sample_results.append(eval_result)

                # Store prediction for saving
                if save_predictions:
                    predictions.append({
                        "id": sample.id,
                        "input": sample.input_text,
                        "expected": sample.expected_output,
                        "predicted": model_output,
                        "evaluation": eval_result,
                        "metadata": sample.metadata
                    })

            except Exception as e:
                error_msg = f"Error processing sample {sample.id}: {e}"
                logger.error(error_msg)

                # Add more specific error context for HTTP errors
                if "HTTP" in str(e) or "Connection" in str(e) or "Timeout" in str(e):
                    logger.error(f"Network/HTTP error details: {type(e).__name__}: {e}")
                    error_type = "network_error"
                else:
                    error_type = "processing_error"

                # Add failed sample result
                sample_results.append({
                    "correct": False,
                    "error": str(e),
                    "error_type": error_type,
                    "score": 0.0
                })

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