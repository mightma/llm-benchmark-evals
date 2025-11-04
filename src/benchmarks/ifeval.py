"""
IFEval benchmark implementation.

IFEval tests instruction-following capabilities by evaluating whether
models can follow specific formatting and content instructions.
"""

import json
import os
import logging
import re
from typing import List, Dict, Any, Callable
from datasets import load_dataset

from .base import EvaluationSample, BaseBenchmark

logger = logging.getLogger(__name__)


class IFEvalBenchmark(BaseBenchmark):
    """IFEval benchmark implementation."""

    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config, data_path)
        self.instruction_checkers = self._initialize_checkers()

    def _initialize_checkers(self) -> Dict[str, Callable]:
        """Initialize instruction checking functions."""
        return {
            "length_constraints:number_words": self._check_word_count,
            "length_constraints:number_sentences": self._check_sentence_count,
            "length_constraints:number_paragraphs": self._check_paragraph_count,
            "keywords:existence": self._check_keyword_existence,
            "keywords:frequency": self._check_keyword_frequency,
            "keywords:forbidden_words": self._check_forbidden_words,
            "language:response_language": self._check_response_language,
            "format:number_sections": self._check_number_sections,
            "format:multiple_sections": self._check_multiple_sections,
            "format:json_format": self._check_json_format,
            "format:title": self._check_title_format,
            "punctuation:no_comma": self._check_no_comma,
            "case:frequency_capital_words": self._check_capital_word_frequency,
            "case:all_capital": self._check_all_capital,
            "combination:two_responses": self._check_two_responses,
        }

    async def load_data(self) -> List[EvaluationSample]:
        """Load IFEval data."""
        logger.info("Loading IFEval dataset...")

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

        json_path = os.path.join(self.data_path, "ifeval.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                sample = self._create_sample(item)
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from local files")
        return samples

    async def _load_from_huggingface(self) -> List[EvaluationSample]:
        """Load data from HuggingFace datasets."""
        logger.info("Loading IFEval from HuggingFace...")

        try:
            dataset = load_dataset("google/IFEval", split="train")
            samples = []

            for item in dataset:
                sample = self._create_sample(item)
                samples.append(sample)

            logger.info(f"Loaded {len(samples)} samples from HuggingFace")
            return samples

        except Exception as e:
            logger.error(f"Failed to load IFEval from HuggingFace: {e}")
            # Create sample data for testing
            return self._create_sample_problems()

    def _create_sample(self, item: Dict[str, Any]) -> EvaluationSample:
        """Create evaluation sample from data item."""
        prompt = item.get("prompt", "")
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs = item.get("kwargs", [])

        return EvaluationSample(
            id=item.get("key", str(hash(prompt))),
            input_text=prompt,
            expected_output=None,  # IFEval doesn't have expected outputs
            metadata={
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "prompt": prompt
            }
        )

    def _create_sample_problems(self) -> List[EvaluationSample]:
        """Create sample IFEval problems for testing."""
        sample_problems = [
            {
                "key": "sample_1",
                "prompt": "Write a short story about a robot. Your response should contain exactly 3 paragraphs.",
                "instruction_id_list": ["length_constraints:number_paragraphs"],
                "kwargs": [{"num_paragraphs": 3}]
            },
            {
                "key": "sample_2",
                "prompt": "Explain photosynthesis in simple terms. Use the word 'energy' at least 3 times.",
                "instruction_id_list": ["keywords:frequency"],
                "kwargs": [{"keyword": "energy", "frequency": 3, "relation": "at least"}]
            },
            {
                "key": "sample_3",
                "prompt": "Write a poem about the ocean. Your response should not contain the word 'blue'.",
                "instruction_id_list": ["keywords:forbidden_words"],
                "kwargs": [{"forbidden_words": ["blue"]}]
            }
        ]

        return [self._create_sample(problem) for problem in sample_problems]

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        model_response: str
    ) -> Dict[str, Any]:
        """Evaluate IFEval sample."""
        instruction_id_list = sample.metadata.get("instruction_id_list", [])
        kwargs_list = sample.metadata.get("kwargs", [])

        instruction_results = []
        total_score = 0.0

        for i, instruction_id in enumerate(instruction_id_list):
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}

            if instruction_id in self.instruction_checkers:
                checker = self.instruction_checkers[instruction_id]
                passed = checker(model_response, kwargs)
                instruction_results.append({
                    "instruction_id": instruction_id,
                    "passed": passed,
                    "kwargs": kwargs
                })
                total_score += 1.0 if passed else 0.0
            else:
                logger.warning(f"Unknown instruction type: {instruction_id}")
                instruction_results.append({
                    "instruction_id": instruction_id,
                    "passed": False,
                    "kwargs": kwargs,
                    "error": "Unknown instruction type"
                })

        # Calculate overall score for this sample
        num_instructions = len(instruction_id_list)
        sample_score = total_score / num_instructions if num_instructions > 0 else 0.0

        return {
            "correct": sample_score == 1.0,  # All instructions followed
            "score": sample_score,
            "instruction_results": instruction_results,
            "num_instructions": num_instructions,
            "instructions_passed": int(total_score),
            "raw_response": model_response,
            "metadata": sample.metadata
        }

    def aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate IFEval results."""
        total_samples = len(sample_results)
        total_instructions = sum(r.get("num_instructions", 0) for r in sample_results)
        total_instructions_passed = sum(r.get("instructions_passed", 0) for r in sample_results)

        # Overall accuracy (samples with all instructions followed)
        perfect_samples = sum(1 for r in sample_results if r.get("correct", False))
        sample_accuracy = perfect_samples / total_samples if total_samples > 0 else 0.0

        # Instruction-level accuracy
        instruction_accuracy = total_instructions_passed / total_instructions if total_instructions > 0 else 0.0

        # Average sample score
        avg_sample_score = sum(r.get("score", 0.0) for r in sample_results) / total_samples if total_samples > 0 else 0.0

        return {
            "overall_score": instruction_accuracy,  # Use instruction accuracy as main score
            "sample_accuracy": sample_accuracy,
            "instruction_accuracy": instruction_accuracy,
            "average_sample_score": avg_sample_score,
            "perfect_samples": perfect_samples,
            "total_samples": total_samples,
            "total_instructions": total_instructions,
            "instructions_passed": total_instructions_passed,
            "instructions_failed": total_instructions - total_instructions_passed
        }

    # Instruction checker functions
    def _check_word_count(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has the required number of words."""
        target_words = kwargs.get("num_words", 0)
        relation = kwargs.get("relation", "equal to")

        word_count = len(response.split())

        if relation == "less than":
            return word_count < target_words
        elif relation == "at least":
            return word_count >= target_words
        else:  # equal to
            return word_count == target_words

    def _check_sentence_count(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has the required number of sentences."""
        target_sentences = kwargs.get("num_sentences", 0)

        # Simple sentence counting by splitting on sentence endings
        sentences = re.split(r'[.!?]+', response)
        sentence_count = len([s for s in sentences if s.strip()])

        return sentence_count == target_sentences

    def _check_paragraph_count(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has the required number of paragraphs."""
        target_paragraphs = kwargs.get("num_paragraphs", 0)

        # Count paragraphs by splitting on double newlines
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]

        return len(paragraphs) == target_paragraphs

    def _check_keyword_existence(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if required keywords exist in response."""
        keywords = kwargs.get("keywords", [])
        response_lower = response.lower()

        for keyword in keywords:
            if keyword.lower() not in response_lower:
                return False
        return True

    def _check_keyword_frequency(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if keywords appear with required frequency."""
        keyword = kwargs.get("keyword", "")
        frequency = kwargs.get("frequency", 1)
        relation = kwargs.get("relation", "at least")

        count = response.lower().count(keyword.lower())

        if relation == "at least":
            return count >= frequency
        elif relation == "less than":
            return count < frequency
        else:  # equal to
            return count == frequency

    def _check_forbidden_words(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check that forbidden words don't appear in response."""
        forbidden_words = kwargs.get("forbidden_words", [])
        response_lower = response.lower()

        for word in forbidden_words:
            if word.lower() in response_lower:
                return False
        return True

    def _check_response_language(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response is in the required language."""
        # This is a simplified check - in practice, you'd use a language detection library
        required_language = kwargs.get("language", "english")

        # For now, just check for non-ASCII characters as a proxy for non-English
        if required_language.lower() == "english":
            return all(ord(char) < 128 or char.isspace() for char in response)
        else:
            # For other languages, just return True (would need proper language detection)
            return True

    def _check_number_sections(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has the required number of sections."""
        num_sections = kwargs.get("num_sections", 1)

        # Look for section markers (simple heuristic)
        section_markers = len(re.findall(r'^#+\s|^\d+\.\s|^[A-Z][a-z]*:\s', response, re.MULTILINE))

        return section_markers >= num_sections

    def _check_multiple_sections(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has multiple sections."""
        return self._check_number_sections(response, {"num_sections": 2})

    def _check_json_format(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response is valid JSON."""
        try:
            json.loads(response.strip())
            return True
        except json.JSONDecodeError:
            return False

    def _check_title_format(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response has a proper title."""
        # Look for title-like patterns at the beginning
        lines = response.split('\n')
        if not lines:
            return False

        first_line = lines[0].strip()
        # Check if first line looks like a title (capitalized, not too long)
        return len(first_line) > 0 and len(first_line) < 100 and first_line[0].isupper()

    def _check_no_comma(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response contains no commas."""
        return ',' not in response

    def _check_capital_word_frequency(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check frequency of capitalized words."""
        capital_freq = kwargs.get("capital_frequency", 0)
        relation = kwargs.get("relation", "at least")

        words = response.split()
        capital_words = sum(1 for word in words if word and word[0].isupper())

        if relation == "at least":
            return capital_words >= capital_freq
        elif relation == "less than":
            return capital_words < capital_freq
        else:  # equal to
            return capital_words == capital_freq

    def _check_all_capital(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if entire response is in capital letters."""
        return response.isupper()

    def _check_two_responses(self, response: str, kwargs: Dict[str, Any]) -> bool:
        """Check if response contains two separate responses."""
        # Look for separators or multiple distinct sections
        separators = [
            "response 1:",
            "response 2:",
            "first response:",
            "second response:",
            "---",
            "***"
        ]

        response_lower = response.lower()
        separator_count = sum(1 for sep in separators if sep in response_lower)

        return separator_count >= 1 or len(response.split('\n\n')) >= 2