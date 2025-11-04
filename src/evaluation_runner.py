"""
Evaluation runner that orchestrates the entire evaluation process.
"""

import asyncio
import logging
import yaml
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from .vllm_manager import VLLMServerManager, VLLMInferenceClient
from .benchmarks.mmlu_pro import MMLUProBenchmark
from .benchmarks.aime25 import AIME25Benchmark
from .benchmarks.ifeval import IFEvalBenchmark
from .benchmarks.base import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main evaluation runner class."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.server_manager = VLLMServerManager(self.config)
        self.inference_client = None
        self.benchmarks = {}
        self._initialize_benchmarks()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _initialize_benchmarks(self):
        """Initialize benchmark instances."""
        benchmark_config = self.config.get("benchmarks", {})

        # Get evaluation config for concurrency settings
        evaluation_config = self.config.get("evaluation", {})

        # MMLU-Pro
        if benchmark_config.get("mmlu_pro", {}).get("enabled", False):
            mmlu_config = benchmark_config["mmlu_pro"]
            # Merge benchmark config with evaluation config
            merged_config = {**mmlu_config, **evaluation_config}
            self.benchmarks["mmlu_pro"] = MMLUProBenchmark(
                config=merged_config,
                data_path=mmlu_config.get("data_path", "data/mmlu_pro")
            )

        # AIME25
        if benchmark_config.get("aime25", {}).get("enabled", False):
            aime_config = benchmark_config["aime25"]
            # Merge benchmark config with evaluation config
            merged_config = {**aime_config, **evaluation_config}
            self.benchmarks["aime25"] = AIME25Benchmark(
                config=merged_config,
                data_path=aime_config.get("data_path", "data/aime25")
            )

        # IFEval
        if benchmark_config.get("ifeval", {}).get("enabled", False):
            ifeval_config = benchmark_config["ifeval"]
            # Merge benchmark config with evaluation config
            merged_config = {**ifeval_config, **evaluation_config}
            self.benchmarks["ifeval"] = IFEvalBenchmark(
                config=merged_config,
                data_path=ifeval_config.get("data_path", "data/ifeval")
            )

    async def start_server(self) -> str:
        """Start the VLLM server."""
        logger.info("Starting VLLM server...")
        server_url = await self.server_manager.start_server()

        # Initialize inference client
        self.inference_client = VLLMInferenceClient(server_url, self.config)

        return server_url

    async def stop_server(self):
        """Stop the VLLM server."""
        if self.inference_client:
            await self.inference_client.close()
            self.inference_client = None

        await self.server_manager.stop_server()

    async def run_single_benchmark(
        self,
        benchmark_name: str,
        model_name: str,
        num_samples: Optional[int] = None,
        num_responses: int = 1,
        **inference_params
    ) -> EvaluationResult:
        """Run a single benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found or not enabled")

        if not self.inference_client:
            raise RuntimeError("Server not started. Call start_server() first.")

        benchmark = self.benchmarks[benchmark_name]
        evaluation_config = self.config.get("evaluation", {})

        # Override inference parameters if provided
        if inference_params:
            # Create a temporary client with custom parameters
            temp_client = VLLMInferenceClient(
                self.server_manager.get_server_url(),
                self._merge_inference_params(inference_params)
            )
            client_to_use = temp_client
        else:
            client_to_use = self.inference_client

        try:
            result = await benchmark.run_evaluation(
                inference_client=client_to_use,
                model_name=model_name,
                num_samples=num_samples,
                num_responses=num_responses,
                save_predictions=evaluation_config.get("save_predictions", True),
                output_dir=evaluation_config.get("output_dir", "results")
            )
            return result
        finally:
            if inference_params and client_to_use != self.inference_client:
                await client_to_use.close()

    async def run_all_benchmarks(
        self,
        model_name: str,
        num_samples: Optional[int] = None,
        num_responses: int = 1,
        **inference_params
    ) -> List[EvaluationResult]:
        """Run all enabled benchmarks."""
        if not self.inference_client:
            raise RuntimeError("Server not started. Call start_server() first.")

        results = []
        enabled_benchmarks = [name for name in self.benchmarks.keys()]

        logger.info(f"Running {len(enabled_benchmarks)} benchmarks: {enabled_benchmarks}")

        for benchmark_name in enabled_benchmarks:
            try:
                logger.info(f"Starting benchmark: {benchmark_name}")
                result = await self.run_single_benchmark(
                    benchmark_name=benchmark_name,
                    model_name=model_name,
                    num_samples=num_samples,
                    num_responses=num_responses,
                    **inference_params
                )
                results.append(result)
                logger.info(f"Completed benchmark: {benchmark_name} (Score: {result.score:.4f})")
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")

        return results

    def _merge_inference_params(self, custom_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom inference parameters with config."""
        config_copy = self.config.copy()
        inference_config = config_copy.get("inference", {})
        inference_config.update(custom_params)
        config_copy["inference"] = inference_config
        return config_copy

    async def get_server_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return await self.server_manager.get_server_info()

    def list_available_benchmarks(self) -> List[str]:
        """List all available benchmarks."""
        return list(self.benchmarks.keys())

    def get_benchmark_info(self, benchmark_name: str) -> Dict[str, Any]:
        """Get information about a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            return {"error": f"Benchmark '{benchmark_name}' not found"}

        benchmark = self.benchmarks[benchmark_name]
        return {
            "name": benchmark_name,
            "class": benchmark.__class__.__name__,
            "data_path": benchmark.data_path,
            "config": benchmark.config
        }


class InferenceParameterManager:
    """Helper class for managing inference parameters."""

    @staticmethod
    def validate_parameters(**params) -> Dict[str, Any]:
        """Validate and clean inference parameters."""
        valid_params = {}

        # Temperature
        if "temperature" in params:
            temp = float(params["temperature"])
            if 0.0 <= temp <= 2.0:
                valid_params["temperature"] = temp
            else:
                logger.warning(f"Temperature {temp} out of range [0.0, 2.0], ignoring")

        # Top-p
        if "top_p" in params:
            top_p = float(params["top_p"])
            if 0.0 <= top_p <= 1.0:
                valid_params["top_p"] = top_p
            else:
                logger.warning(f"Top-p {top_p} out of range [0.0, 1.0], ignoring")

        # Top-k
        if "top_k" in params:
            top_k = int(params["top_k"])
            if top_k >= -1:  # -1 means disabled
                valid_params["top_k"] = top_k
            else:
                logger.warning(f"Top-k {top_k} invalid (must be >= -1), ignoring")

        # Max tokens
        if "max_tokens" in params:
            max_tokens = int(params["max_tokens"])
            if max_tokens > 0:
                valid_params["max_tokens"] = max_tokens
            else:
                logger.warning(f"Max tokens {max_tokens} invalid (must be > 0), ignoring")

        # Repetition penalty
        if "repetition_penalty" in params:
            rep_penalty = float(params["repetition_penalty"])
            if rep_penalty > 0:
                valid_params["repetition_penalty"] = rep_penalty
            else:
                logger.warning(f"Repetition penalty {rep_penalty} invalid (must be > 0), ignoring")

        return valid_params

    @staticmethod
    def get_parameter_info() -> Dict[str, Dict[str, Any]]:
        """Get information about available inference parameters."""
        return {
            "temperature": {
                "type": "float",
                "range": [0.0, 2.0],
                "default": 0.0,
                "description": "Controls randomness in generation. 0.0 = deterministic, higher = more random"
            },
            "top_p": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 1.0,
                "description": "Nucleus sampling parameter. Lower values = more focused responses"
            },
            "top_k": {
                "type": "int",
                "range": [-1, "inf"],
                "default": -1,
                "description": "Top-k sampling parameter. -1 = disabled, positive int = limit vocabulary"
            },
            "max_tokens": {
                "type": "int",
                "range": [1, "model_max"],
                "default": 2048,
                "description": "Maximum number of tokens to generate"
            },
            "repetition_penalty": {
                "type": "float",
                "range": [0.0, "inf"],
                "default": 1.0,
                "description": "Penalty for repeating tokens. 1.0 = no penalty, > 1.0 = discourage repetition"
            }
        }