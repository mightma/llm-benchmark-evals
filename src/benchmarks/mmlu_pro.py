"""
MMLU-Pro benchmark implementation.

MMLU-Pro is an enhanced version of MMLU with more challenging questions
and extended multiple choice options (up to 10 choices).
"""

import json
import os
import logging
from typing import List, Dict, Any
from datasets import load_dataset

from .base import EvaluationSample, MultipleChoiceBenchmark

logger = logging.getLogger(__name__)


class MMLUProBenchmark(MultipleChoiceBenchmark):
    """MMLU-Pro benchmark implementation."""

    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config, data_path)
        self.subjects = config.get("subjects", None)  # None means all subjects

    async def load_data(self) -> List[EvaluationSample]:
        """Load MMLU-Pro data."""
        logger.info("Loading MMLU-Pro dataset...")

        try:
            # Try to load from local path first
            samples = await self._load_local_data()
            if samples:
                return samples
        except Exception as e:
            logger.warning(f"Could not load local data: {e}")

        # Fall back to loading from HuggingFace
        return await self._load_from_huggingface()

    async def _load_local_data(self) -> List[EvaluationSample]:
        """Load data from local files."""
        samples = []

        if not os.path.exists(self.data_path):
            return samples

        # Look for JSON files in the data path
        for filename in os.listdir(self.data_path):
            if not filename.endswith('.json'):
                continue

            # Extract subject from filename
            subject = filename.replace('.json', '')
            if self.subjects and subject not in self.subjects:
                continue

            filepath = os.path.join(self.data_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                sample = self._create_sample(item, subject)
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from local files")
        return samples

    async def _load_from_huggingface(self) -> List[EvaluationSample]:
        """Load data from HuggingFace datasets."""
        logger.info("Loading MMLU-Pro from HuggingFace...")

        try:
            # Load the dataset
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            samples = []

            for item in dataset:
                subject = item.get("category", "unknown")

                # Filter by subjects if specified
                if self.subjects and subject not in self.subjects:
                    continue

                sample = self._create_sample(item, subject)
                samples.append(sample)

            logger.info(f"Loaded {len(samples)} samples from HuggingFace")
            return samples

        except Exception as e:
            logger.error(f"Failed to load MMLU-Pro from HuggingFace: {e}")
            raise

    def _create_sample(self, item: Dict[str, Any], subject: str) -> EvaluationSample:
        """Create evaluation sample from data item."""
        # Build the prompt
        question = item.get("question", "")
        options = item.get("options", [])

        prompt = f"Subject: {subject}\n\nQuestion: {question}\n\nOptions:\n"

        # Add options (A, B, C, ...)
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            prompt += f"{letter}) {option}\n"

        prompt += "\nPlease select the correct answer by providing only the letter (A, B, C, etc.):"

        # Get the correct answer
        answer_idx = item.get("answer", 0)
        if isinstance(answer_idx, str):
            # If answer is already a letter
            correct_answer = answer_idx.upper()
        else:
            # Convert index to letter
            correct_answer = chr(ord('A') + answer_idx)

        return EvaluationSample(
            id=f"{subject}_{item.get('question_id', len(prompt))}",
            input_text=prompt,
            expected_output=correct_answer,
            metadata={
                "subject": subject,
                "question": question,
                "options": options,
                "answer_index": answer_idx
            }
        )

    def extract_choice(self, response: str) -> str:
        """Extract choice from model response (override to handle more options)."""
        response = response.strip().upper()

        import re

        # MMLU-Pro can have up to 10 options (A-J)
        patterns = [
            r'\b([A-J])\)',  # A)
            r'\(([A-J])\)',  # (A)
            r'\b([A-J])\.',  # A.
            r'\b([A-J]):',   # A:
            r'answer is ([A-J])',  # answer is A
            r'Answer: ([A-J])',    # Answer: A
            r'^([A-J])\b',   # A at start
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # If no clear pattern, try to find any single letter A-J
        letters = re.findall(r'\b[A-J]\b', response)
        if letters:
            return letters[0]

        return "UNKNOWN"

    def aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate MMLU-Pro results with subject breakdown, excluding failed samples."""
        base_results = super().aggregate_results(sample_results)

        # Only process successful samples for subject analysis
        successful_samples = [r for r in sample_results if not r.get("error")]
        failed_samples = [r for r in sample_results if r.get("error")]

        # Add subject-specific analysis if metadata is available
        subject_stats = {}
        subject_failures = {}

        # Process successful samples
        for result in successful_samples:
            if "metadata" in result:
                subject = result["metadata"].get("subject", "unknown")
                if subject not in subject_stats:
                    subject_stats[subject] = {"correct": 0, "total": 0}

                subject_stats[subject]["total"] += 1
                if result.get("correct", False):
                    subject_stats[subject]["correct"] += 1

        # Process failed samples by subject
        for result in failed_samples:
            if "metadata" in result:
                subject = result["metadata"].get("subject", "unknown")
                if subject not in subject_failures:
                    subject_failures[subject] = {"failed": 0, "total_attempted": 0}

                subject_failures[subject]["failed"] += 1
                subject_failures[subject]["total_attempted"] += 1

        # Calculate per-subject accuracy (among successful samples only)
        subject_accuracy = {}
        for subject, stats in subject_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

            # Add failure information if any
            failure_info = subject_failures.get(subject, {"failed": 0, "total_attempted": 0})
            total_attempted = stats["total"] + failure_info["failed"]

            subject_accuracy[subject] = {
                "accuracy": accuracy,  # Among successful samples
                "correct": stats["correct"],
                "successful": stats["total"],
                "failed": failure_info["failed"],
                "total_attempted": total_attempted,
                "success_rate": stats["total"] / total_attempted if total_attempted > 0 else 0.0
            }

        # Add overall subject statistics
        total_subjects_with_failures = len(set(list(subject_stats.keys()) + list(subject_failures.keys())))

        base_results.update({
            "subjects": subject_accuracy,
            "num_subjects": len(subject_accuracy),
            "total_subjects_attempted": total_subjects_with_failures,
            "subjects_with_failures": len(subject_failures)
        })

        return base_results