"""
AIME 2025 benchmark implementation.

AIME (American Invitational Mathematics Examination) 2025 problems.
These are challenging mathematics problems requiring exact integer answers.
"""

import json
import os
import logging
import re
from typing import List, Dict, Any

from .base import EvaluationSample, BaseBenchmark

logger = logging.getLogger(__name__)


class AIME25Benchmark(BaseBenchmark):
    """AIME 2025 benchmark implementation."""

    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config, data_path)

    async def load_data(self) -> List[EvaluationSample]:
        """Load AIME25 data."""
        logger.info("Loading AIME25 dataset...")

        samples = []

        # Try to load from local JSON file
        json_path = os.path.join(self.data_path, "aime25.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                sample = self._create_sample(item)
                samples.append(sample)

        else:
            # Create sample problems if no data file exists
            logger.warning(f"No data file found at {json_path}. Creating sample problems.")
            samples = self._create_sample_problems()

        logger.info(f"Loaded {len(samples)} AIME25 problems")
        return samples

    def _create_sample(self, item: Dict[str, Any]) -> EvaluationSample:
        """Create evaluation sample from data item."""
        problem_id = item.get("id", item.get("problem_number", "unknown"))
        question = item.get("problem", item.get("question", ""))
        answer = str(item.get("answer", ""))

        # Format the prompt
        prompt = f"""Problem: {question}

Please solve this step by step and provide your final answer as an integer between 000 and 999.

Your response should end with "The answer is: [your answer]" where [your answer] is a 3-digit number (use leading zeros if necessary, e.g., 007, 042, 123).
"""

        return EvaluationSample(
            id=f"aime25_{problem_id}",
            input_text=prompt,
            expected_output=answer.zfill(3),  # Ensure 3-digit format
            metadata={
                "problem_number": problem_id,
                "question": question,
                "type": "mathematics"
            }
        )

    def _create_sample_problems(self) -> List[EvaluationSample]:
        """Create sample AIME-style problems for testing."""
        sample_problems = [
            {
                "id": 1,
                "problem": "Find the number of positive integers n ≤ 1000 such that n and n+1 are both perfect squares.",
                "answer": "0"
            },
            {
                "id": 2,
                "problem": "Let S be the set of all positive integers n such that n² + 19n + 99 is a perfect square. Find the sum of all elements in S.",
                "answer": "17"
            },
            {
                "id": 3,
                "problem": "A regular hexagon with side length 10 is inscribed in a circle. Find the area of the region inside the circle but outside the hexagon.",
                "answer": "41"
            }
        ]

        return [self._create_sample(problem) for problem in sample_problems]

    def extract_answer(self, response: str) -> str:
        """Extract numerical answer from model response."""
        response = response.strip()

        # Look for "The answer is: XXX" pattern
        patterns = [
            r"The answer is:?\s*(\d{1,3})",
            r"answer is:?\s*(\d{1,3})",
            r"Answer:?\s*(\d{1,3})",
            r"Final answer:?\s*(\d{1,3})",
            r"Therefore,?\s*(\d{1,3})",
            r"Thus,?\s*(\d{1,3})",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1)
                return answer.zfill(3)  # Ensure 3-digit format

        # Look for any 3-digit number at the end
        match = re.search(r'(\d{3})(?:\s*$|\s*\.)', response)
        if match:
            return match.group(1)

        # Look for any 1-3 digit number and pad
        match = re.search(r'(\d{1,3})(?:\s*$|\s*\.)', response)
        if match:
            return match.group(1).zfill(3)

        # Last resort: look for any digits
        digits = re.findall(r'\d+', response)
        if digits:
            # Take the last number found
            last_number = digits[-1]
            if len(last_number) <= 3:
                return last_number.zfill(3)
            else:
                # Take last 3 digits
                return last_number[-3:]

        return "UNKNOWN"

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        model_response: str
    ) -> Dict[str, Any]:
        """Evaluate AIME sample."""
        predicted_answer = self.extract_answer(model_response)
        expected_answer = sample.expected_output

        correct = predicted_answer == expected_answer

        return {
            "correct": correct,
            "predicted": predicted_answer,
            "expected": expected_answer,
            "score": 1.0 if correct else 0.0,
            "raw_response": model_response,
            "metadata": sample.metadata
        }

    def aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate AIME results."""
        total_samples = len(sample_results)
        correct_samples = sum(1 for r in sample_results if r.get("correct", False))

        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        # Calculate average score (same as accuracy for AIME)
        total_score = sum(r.get("score", 0.0) for r in sample_results)
        avg_score = total_score / total_samples if total_samples > 0 else 0.0

        return {
            "overall_score": accuracy,
            "accuracy": accuracy,
            "average_score": avg_score,
            "correct": correct_samples,
            "total": total_samples,
            "error_rate": 1.0 - accuracy,
            "problems_solved": correct_samples,
            "problems_attempted": total_samples
        }