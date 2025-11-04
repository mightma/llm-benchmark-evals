#!/usr/bin/env python3
"""
VLLM Server Monitor Script

Continuously monitors and displays information about the currently running VLLM server,
including model details, server status, and performance metrics.
"""

import asyncio
import httpx
import json
import time
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for using evaluation framework components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.evaluation_runner import EvaluationRunner
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

class VLLMMonitor:
    """Monitor for VLLM server status and information."""

    def __init__(self, server_url: str = "http://localhost:8000", config_path: Optional[str] = None, enable_runtime_detection: bool = True, base_timeout: int = 30, patient_mode: bool = False):
        self.server_url = server_url.rstrip('/')
        self.config_path = config_path
        self.enable_runtime_detection = enable_runtime_detection
        self.base_timeout = base_timeout
        self.patient_mode = patient_mode

        # Configure timeouts based on mode
        if patient_mode:
            # Very long timeouts for slow servers
            default_timeout = httpx.Timeout(base_timeout * 3, connect=20.0, read=base_timeout * 6)
        else:
            # Normal timeouts
            default_timeout = httpx.Timeout(base_timeout, connect=10.0, read=base_timeout * 2)

        self.client = httpx.AsyncClient(timeout=default_timeout)
        self.start_time = time.time()
        self.server_ready = False
        self.consecutive_timeouts = 0

    async def get_server_info(self) -> Dict[str, Any]:
        """Get basic server information with progressive timeout handling."""
        try:
            # Use adaptive timeout based on server readiness, consecutive timeouts, and mode
            base = self.base_timeout
            multiplier = 10 if self.patient_mode else 2

            if self.consecutive_timeouts > 3:
                # Server seems very slow, use much longer timeout
                timeout = httpx.Timeout(base * 4, connect=20.0, read=base * 6)
            elif self.consecutive_timeouts > 1:
                # Server is slow, increase timeout
                timeout = httpx.Timeout(base * 2, connect=15.0, read=base * 4)
            elif not self.server_ready:
                # First connection or server not ready, use longer timeout for model loading
                timeout = httpx.Timeout(base * 3, connect=15.0, read=base * 5)
            else:
                # Server was ready before, use normal timeout
                timeout = httpx.Timeout(base, connect=10.0, read=base * multiplier)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.server_url}/v1/models")
                response.raise_for_status()

                # Server responded successfully
                self.server_ready = True
                self.consecutive_timeouts = 0

                return {
                    "status": "running",
                    "models": response.json(),
                    "error": None,
                    "timeout_used": timeout.read
                }

        except httpx.ConnectError:
            self.consecutive_timeouts = 0  # Connection error is different from timeout
            return {
                "status": "not_running",
                "models": None,
                "error": "Connection refused - server not running or wrong URL"
            }
        except httpx.TimeoutException as e:
            self.consecutive_timeouts += 1
            timeout_msg = f"Server timeout after {timeout.read}s"

            if self.consecutive_timeouts == 1:
                timeout_msg += " - server may be loading model or processing requests"
            elif self.consecutive_timeouts <= 3:
                timeout_msg += f" - consecutive timeout #{self.consecutive_timeouts}, increasing wait time"
            else:
                timeout_msg += f" - server appears very slow or stuck (timeout #{self.consecutive_timeouts})"

            return {
                "status": "timeout",
                "models": None,
                "error": timeout_msg,
                "consecutive_timeouts": self.consecutive_timeouts
            }
        except httpx.HTTPStatusError as e:
            self.consecutive_timeouts = 0
            return {
                "status": "http_error",
                "models": None,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            }
        except Exception as e:
            self.consecutive_timeouts = 0
            return {
                "status": "error",
                "models": None,
                "error": f"Unexpected error: {str(e)}"
            }

    async def get_model_details(self) -> Dict[str, Any]:
        """Get detailed model information including quantization status."""
        try:
            # Try to get model info from /v1/models endpoint
            models_response = await self.client.get(f"{self.server_url}/v1/models")
            models_response.raise_for_status()
            models_data = models_response.json()

            model_details = {}
            if models_data and "data" in models_data:
                for model in models_data["data"]:
                    model_id = model.get("id", "unknown")

                    # Detect quantization from model ID/path
                    quantization_info = self._detect_quantization(model_id)

                    model_details[model_id] = {
                        "id": model_id,
                        "object": model.get("object", "model"),
                        "created": model.get("created"),
                        "owned_by": model.get("owned_by", "unknown"),
                        "quantization": quantization_info
                    }

            # Try to get additional VLLM server info including runtime detection
            try:
                server_info = await self._get_vllm_server_info()
                if server_info:
                    # Add runtime quantization info to model details
                    runtime_quant = server_info.get("runtime_quantization", {})
                    if runtime_quant:
                        for model_id in model_details:
                            model_details[model_id]["runtime_quantization"] = runtime_quant
            except Exception as e:
                logger.debug(f"Could not get VLLM server info: {e}")

            return model_details

        except Exception as e:
            return {"error": f"Failed to get model details: {e}"}

    def _detect_quantization(self, model_path: str) -> Dict[str, Any]:
        """Detect quantization information from model path/name."""
        model_path_lower = model_path.lower()

        quantization_info = {
            "is_quantized": False,
            "method": "none",
            "precision": None,
            "confidence": "low"
        }

        # Check for common quantization patterns
        quantization_patterns = [
            # GPTQ patterns
            ("gptq", ["gptq", "-gptq-", "_gptq_", "gptq-"]),
            # AWQ patterns
            ("awq", ["awq", "-awq-", "_awq_", "awq-"]),
            # GGUF patterns
            ("gguf", ["gguf", ".gguf", "-gguf-", "_gguf_"]),
            ("ggml", ["ggml", ".ggml", "-ggml-", "_ggml_"]),
            # SqueezeLLM
            ("squeezellm", ["squeezellm", "squeeze"]),
            # FP8 quantization (8-bit floating point)
            ("fp8", ["fp8", "-fp8", "_fp8_", "fp8-"]),
            # Bitsandbytes and other integer quantization
            ("bitsandbytes", ["int8", "int4", "nf4", "fp4"]),
            # Additional quantization methods
            ("smoothquant", ["smoothquant", "smooth-quant"]),
            ("qlora", ["qlora", "q-lora"]),
            ("blockwise", ["blockwise", "block-wise"]),
        ]

        for method, patterns in quantization_patterns:
            for pattern in patterns:
                if pattern in model_path_lower:
                    quantization_info["is_quantized"] = True
                    quantization_info["method"] = method
                    quantization_info["confidence"] = "high"

                    # Try to extract precision information
                    if "fp8" in model_path_lower:
                        quantization_info["precision"] = "fp8"
                    elif "int8" in model_path_lower or "-8bit" in model_path_lower:
                        quantization_info["precision"] = "int8"
                    elif "int4" in model_path_lower or "-4bit" in model_path_lower:
                        quantization_info["precision"] = "int4"
                    elif "fp16" in model_path_lower:
                        quantization_info["precision"] = "fp16"
                    elif "bf16" in model_path_lower:
                        quantization_info["precision"] = "bf16"
                    elif "fp4" in model_path_lower:
                        quantization_info["precision"] = "fp4"
                    elif "nf4" in model_path_lower:
                        quantization_info["precision"] = "nf4"

                    break

            if quantization_info["is_quantized"]:
                break

        return quantization_info

    async def _get_vllm_server_info(self) -> Dict[str, Any]:
        """Try to get additional VLLM server information including runtime quantization."""
        try:
            server_info = {}

            # Try VLLM-specific endpoints that might provide more details
            endpoints_to_try = [
                "/stats",
                "/health",
                "/metrics",
                "/version",
                "/v1/models"  # Re-check models endpoint for additional info
            ]

            for endpoint in endpoints_to_try:
                try:
                    response = await self.client.get(f"{self.server_url}{endpoint}")
                    if response.status_code == 200:
                        data = response.json()
                        server_info[endpoint] = data
                except:
                    continue

            # Try to get runtime model information via inference test (if enabled)
            if self.enable_runtime_detection:
                runtime_info = await self._detect_runtime_quantization()
                if runtime_info:
                    server_info["runtime_quantization"] = runtime_info

            return server_info

        except Exception as e:
            logger.debug(f"Could not get VLLM server info: {e}")
            return {}

    async def _detect_runtime_quantization(self) -> Dict[str, Any]:
        """Detect actual runtime quantization by analyzing inference behavior."""
        try:
            # Simple inference to analyze response patterns
            test_payload = {
                "model": "default",
                "messages": [{"role": "user", "content": "Hi"}],  # Shorter prompt for faster response
                "max_tokens": 5,  # Fewer tokens for faster response
                "temperature": 0.0,
                "logprobs": True,  # Request logprobs if available
                "top_logprobs": 1
            }

            # Use longer timeout for inference requests
            inference_timeout = httpx.Timeout(60.0, connect=15.0, read=90.0)

            start_time = time.time()
            async with httpx.AsyncClient(timeout=inference_timeout) as client:
                response = await client.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=test_payload
                )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()

                # Analyze response for quantization hints
                runtime_info = {
                    "response_time": end_time - start_time,
                    "detected_from_response": False,
                    "hints": []
                }

                # Check usage information
                usage = data.get("usage", {})
                if usage:
                    runtime_info["token_usage"] = usage

                # Check if logprobs are available (some quantized models don't support this)
                choices = data.get("choices", [])
                if choices:
                    choice = choices[0]
                    logprobs = choice.get("logprobs")
                    if logprobs is None and test_payload.get("logprobs"):
                        runtime_info["hints"].append("logprobs_not_supported")

                # Try a precision-specific test
                precision_test = await self._test_precision_behavior()
                if precision_test:
                    runtime_info.update(precision_test)

                return runtime_info

        except Exception as e:
            logger.debug(f"Runtime quantization detection failed: {e}")
            return {}

    async def _test_precision_behavior(self) -> Dict[str, Any]:
        """Test model behavior to infer precision."""
        try:
            # Test with temperature=0 for deterministic output
            test_cases = [
                {"content": "1+1=", "expected_pattern": "2"},
                {"content": "The capital of France is", "expected_pattern": "Paris"}
            ]

            precision_hints = {
                "consistency_score": 0,
                "response_quality": "unknown",
                "likely_precision": "unknown"
            }

            consistent_responses = 0
            total_tests = len(test_cases)

            for test_case in test_cases:
                # Run the same test multiple times to check consistency
                responses = []
                for _ in range(2):  # Limited runs to avoid too many requests
                    payload = {
                        "model": "default",
                        "messages": [{"role": "user", "content": test_case["content"]}],
                        "max_tokens": 5,
                        "temperature": 0.0
                    }

                    response = await self.client.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=payload
                    )

                    if response.status_code == 200:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        responses.append(content.strip())

                # Check consistency
                if len(responses) >= 2 and responses[0] == responses[1]:
                    consistent_responses += 1

            precision_hints["consistency_score"] = consistent_responses / total_tests if total_tests > 0 else 0

            # High consistency might indicate full precision, while some inconsistency might suggest quantization
            if precision_hints["consistency_score"] >= 0.8:
                precision_hints["likely_precision"] = "high (fp16/fp32)"
            elif precision_hints["consistency_score"] >= 0.5:
                precision_hints["likely_precision"] = "medium (int8/fp8)"
            else:
                precision_hints["likely_precision"] = "low (int4 or heavily quantized)"

            return precision_hints

        except Exception as e:
            logger.debug(f"Precision behavior test failed: {e}")
            return {}

    async def test_inference(self, model_name: str = None) -> Dict[str, Any]:
        """Test basic inference capability."""
        try:
            # Use the first available model if none specified
            if not model_name:
                server_info = await self.get_server_info()
                if server_info["status"] == "running" and server_info["models"]:
                    models = server_info["models"].get("data", [])
                    if models:
                        model_name = models[0]["id"]
                    else:
                        return {"error": "No models available"}
                else:
                    return {"error": "Server not running"}

            # Simple test inference
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello! Please respond with 'VLLM server is working correctly.'"}],
                "max_tokens": 50,
                "temperature": 0.1
            }

            start_time = time.time()
            response = await self.client.post(
                f"{self.server_url}/v1/chat/completions",
                json=test_payload
            )
            end_time = time.time()

            response.raise_for_status()
            data = response.json()

            return {
                "status": "success",
                "response_time": round(end_time - start_time, 3),
                "model_used": model_name,
                "response": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {}),
                "error": None
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "response_time": None,
                "model_used": model_name
            }

    async def get_framework_config(self) -> Dict[str, Any]:
        """Get configuration from evaluation framework if available."""
        if not FRAMEWORK_AVAILABLE or not self.config_path:
            return {"error": "Framework not available or no config path"}

        try:
            runner = EvaluationRunner(self.config_path)
            models_config = runner.config.get("models", [])
            vllm_config = runner.config.get("vllm_server", {})

            return {
                "models_config": models_config,
                "vllm_config": vllm_config,
                "config_path": self.config_path
            }
        except Exception as e:
            return {"error": f"Failed to load config: {e}"}

    def format_uptime(self, start_time: float) -> str:
        """Format uptime duration."""
        uptime_seconds = time.time() - start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def print_header(self):
        """Print monitoring header."""
        print("=" * 80)
        print("🚀 VLLM Server Monitor")
        print("=" * 80)
        print(f"Server URL: {self.server_url}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show timeout configuration
        if self.patient_mode:
            print(f"⏳ Patient Mode: Base timeout {self.base_timeout}s (up to {self.base_timeout * 6}s)")
        else:
            print(f"⏱️  Timeout: {self.base_timeout}s base (adaptive up to {self.base_timeout * 6}s)")

        if not self.enable_runtime_detection:
            print("🔧 Runtime detection disabled (faster monitoring)")

        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def print_status(self, info: Dict[str, Any]):
        """Print current server status."""
        status = info.get("status", "unknown")

        # Status indicator
        status_indicators = {
            "running": "🟢 RUNNING",
            "not_running": "🔴 NOT RUNNING",
            "timeout": "🟡 TIMEOUT",
            "http_error": "🔴 HTTP ERROR",
            "error": "🔴 ERROR"
        }

        print(f"\n📊 Status: {status_indicators.get(status, f'🟡 {status.upper()}')}")

        # Show timeout information if available
        if status == "running":
            timeout_used = info.get("timeout_used")
            if timeout_used:
                print(f"⏱️  Response time: {timeout_used}s timeout")

        if info.get("error"):
            print(f"❌ Error: {info['error']}")

            # Show helpful suggestions for timeout errors
            if status == "timeout":
                consecutive = info.get("consecutive_timeouts", 0)
                print(f"💡 Suggestions:")
                if consecutive == 1:
                    print(f"   • Server may be loading the model (first startup can take 1-5 minutes)")
                    print(f"   • Check server logs for model loading progress")
                elif consecutive <= 3:
                    print(f"   • Server is responding slowly, will increase timeout automatically")
                    print(f"   • Check GPU memory usage and server load")
                else:
                    print(f"   • Server appears stuck or extremely overloaded")
                    print(f"   • Consider restarting the VLLM server")
                    print(f"   • Check server process and GPU status")

                print(f"   • Use --no-runtime-detection to reduce server load")
                print(f"   • Try connecting to server directly: curl {self.server_url}/v1/models")

            return

        # Model information
        models = info.get("models", {})
        if models and "data" in models:
            print(f"🤖 Models: {len(models['data'])} available")
            for i, model in enumerate(models["data"][:3]):  # Show first 3 models
                model_id = model.get("id", "unknown")
                print(f"   {i+1}. {model_id}")
            if len(models["data"]) > 3:
                print(f"   ... and {len(models['data']) - 3} more")

    async def print_detailed_models(self):
        """Print detailed model information including quantization."""
        try:
            model_details = await self.get_model_details()

            if model_details.get("error"):
                print(f"❌ Could not get detailed model info: {model_details['error']}")
                return

            if not model_details:
                print("ℹ️  No detailed model information available")
                return

            print(f"\n🔍 Model Details:")
            for model_id, details in model_details.items():
                print(f"   📦 {model_id}")

                # Show model name-based quantization detection
                quantization = details.get("quantization", {})
                if quantization.get("is_quantized"):
                    method = quantization.get("method", "unknown")
                    precision = quantization.get("precision")
                    confidence = quantization.get("confidence", "low")

                    # Choose appropriate icon based on quantization method
                    if method in ["gptq", "awq"]:
                        quant_icon = "⚡"  # Lightning for efficient quantization
                    elif method == "fp8":
                        quant_icon = "🔥"  # Fire for FP8 (newer, high-performance)
                    elif method in ["gguf", "ggml"]:
                        quant_icon = "📦"  # Package for GGUF/GGML formats
                    else:
                        quant_icon = "🔧"  # Wrench for other methods
                    quant_text = f"{quant_icon} Model Name Suggests: {method.upper()}"

                    if precision:
                        quant_text += f" ({precision})"

                    if confidence == "high":
                        quant_text += " ✅"
                    else:
                        quant_text += " ⚠️"

                    print(f"      {quant_text}")
                else:
                    print(f"      🎯 Model Name Suggests: Non-quantized")

                # Show runtime quantization detection
                runtime_quant = details.get("runtime_quantization", {})
                if runtime_quant:
                    likely_precision = runtime_quant.get("likely_precision", "unknown")
                    consistency_score = runtime_quant.get("consistency_score", 0)
                    hints = runtime_quant.get("hints", [])

                    if likely_precision != "unknown":
                        if "high" in likely_precision:
                            runtime_icon = "🎯"
                            runtime_text = f"{runtime_icon} Runtime Detection: {likely_precision}"
                        elif "medium" in likely_precision:
                            runtime_icon = "🔶"
                            runtime_text = f"{runtime_icon} Runtime Detection: {likely_precision}"
                        else:
                            runtime_icon = "🔸"
                            runtime_text = f"{runtime_icon} Runtime Detection: {likely_precision}"

                        print(f"      {runtime_text}")
                        print(f"      📊 Consistency Score: {consistency_score:.2f}")

                        if hints:
                            hint_text = ", ".join(hints)
                            print(f"      💡 Hints: {hint_text}")

                        # Compare model name vs runtime detection
                        if quantization.get("is_quantized") and "high" in likely_precision:
                            print(f"      ⚠️  MISMATCH: Model name suggests quantized, but runtime behaves like full precision")
                        elif not quantization.get("is_quantized") and ("medium" in likely_precision or "low" in likely_precision):
                            print(f"      ⚠️  MISMATCH: Model name suggests full precision, but runtime shows quantization signs")

                # Show additional info if available
                owned_by = details.get("owned_by")
                if owned_by and owned_by != "unknown":
                    print(f"      👤 Owner: {owned_by}")

        except Exception as e:
            print(f"❌ Error getting model details: {e}")

    async def show_recent_logs(self, lines: int = 20):
        """Show recent VLLM server logs."""
        print(f"\n📋 Recent VLLM Server Logs (last {lines} lines):")

        # Try to get logs from VLLMManager if available
        if FRAMEWORK_AVAILABLE and self.config_path:
            try:
                from src.evaluation_runner import EvaluationRunner
                runner = EvaluationRunner(self.config_path)
                if hasattr(runner, 'vllm_manager') and runner.vllm_manager:
                    logs = runner.vllm_manager.get_recent_logs(lines)

                    if logs['stdout']:
                        print(f"📤 STDOUT:")
                        print("-" * 50)
                        print(logs['stdout'])

                    if logs['stderr']:
                        print(f"❌ STDERR:")
                        print("-" * 50)
                        print(logs['stderr'])

                    if logs['stdout'] or logs['stderr']:
                        return
            except Exception as e:
                logger.debug(f"Could not get logs from manager: {e}")

        # Fallback: Look for log files in logs directory
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            log_files = sorted([f for f in os.listdir(logs_dir) if f.startswith("vllm_server") and not f.endswith("_error.log")])
            error_files = sorted([f for f in os.listdir(logs_dir) if f.startswith("vllm_server_error")])

            if log_files:
                latest_log = os.path.join(logs_dir, log_files[-1])
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        recent_lines = f.readlines()[-lines:]
                        if recent_lines:
                            print(f"📄 From {latest_log}:")
                            print("-" * 50)
                            print(''.join(recent_lines))
                        else:
                            print("📄 Log file is empty")
                except Exception as e:
                    print(f"❌ Could not read log file: {e}")

            if error_files:
                latest_error = os.path.join(logs_dir, error_files[-1])
                try:
                    with open(latest_error, 'r', encoding='utf-8') as f:
                        error_lines = f.readlines()[-lines:]
                        if error_lines:
                            print(f"\n❌ Errors from {latest_error}:")
                            print("-" * 50)
                            print(''.join(error_lines))
                except Exception as e:
                    print(f"❌ Could not read error log: {e}")

            if not log_files and not error_files:
                print("ℹ️  No VLLM server log files found")
        else:
            print("ℹ️  No logs directory found")

    def print_performance(self, test_result: Dict[str, Any]):
        """Print performance test results."""
        if test_result.get("status") == "success":
            response_time = test_result.get("response_time", 0)
            usage = test_result.get("usage", {})

            print(f"\n⚡ Performance:")
            print(f"   Response time: {response_time:.3f}s")
            print(f"   Model: {test_result.get('model_used', 'unknown')}")

            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                print(f"   Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")

                if response_time > 0:
                    tokens_per_sec = completion_tokens / response_time
                    print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")

            response = test_result.get("response", "").strip()
            if response:
                print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        else:
            print(f"\n❌ Performance test failed: {test_result.get('error', 'Unknown error')}")

    def print_config(self, config: Dict[str, Any]):
        """Print framework configuration."""
        if config.get("error"):
            return

        print(f"\n⚙️  Configuration:")
        models_config = config.get("models_config", [])
        if models_config:
            for i, model in enumerate(models_config[:2]):  # Show first 2 models
                name = model.get("name", "unnamed")
                path = model.get("model_path", "unknown")
                quantization = model.get("quantization", "none")
                print(f"   Model {i+1}: {name}")
                print(f"      Path: {path}")
                print(f"      Quantization: {quantization}")

        vllm_config = config.get("vllm_config", {})
        if vllm_config:
            host = vllm_config.get("host", "unknown")
            port = vllm_config.get("port", "unknown")
            deploy_locally = vllm_config.get("deploy_locally", True)
            print(f"   VLLM: {host}:{port} (local: {deploy_locally})")

    async def monitor_once(self, test_inference: bool = True, show_config: bool = True, detailed_models: bool = True, show_logs: bool = False, log_lines: int = 10):
        """Run one monitoring cycle."""
        # Get server information
        server_info = await self.get_server_info()
        self.print_status(server_info)

        # Show detailed model information including quantization
        if detailed_models and server_info.get("status") == "running":
            await self.print_detailed_models()

        # Test inference if server is running
        if test_inference and server_info.get("status") == "running":
            test_result = await self.test_inference()
            self.print_performance(test_result)

        # Show configuration if available
        if show_config:
            config = await self.get_framework_config()
            self.print_config(config)

        # Show recent logs if requested
        if show_logs:
            await self.show_recent_logs(log_lines)

        print(f"\n🕐 Uptime: {self.format_uptime(self.start_time)}")
        print(f"📅 Last updated: {datetime.now().strftime('%H:%M:%S')}")

    async def monitor_continuous(self, interval: int = 5, test_inference: bool = True, show_config: bool = True, detailed_models: bool = True, show_logs: bool = False, log_lines: int = 10):
        """Continuously monitor server."""
        self.print_header()

        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')

                self.print_header()
                await self.monitor_once(test_inference, show_config, detailed_models, show_logs, log_lines)

                print(f"\n⏱️  Refreshing in {interval} seconds... (Ctrl+C to stop)")
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped.")
        finally:
            await self.client.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Monitor VLLM server information")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                       help="VLLM server URL (default: http://localhost:8000)")
    parser.add_argument("--config", "-c", default="config.yaml",
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--interval", "-i", type=int, default=5,
                       help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--once", action="store_true",
                       help="Run once instead of continuous monitoring")
    parser.add_argument("--no-test", action="store_true",
                       help="Skip inference testing")
    parser.add_argument("--no-config", action="store_true",
                       help="Skip configuration display")
    parser.add_argument("--no-model-details", action="store_true",
                       help="Skip detailed model information (including quantization)")
    parser.add_argument("--no-runtime-detection", action="store_true",
                       help="Skip runtime quantization detection (faster, but less accurate)")
    parser.add_argument("--timeout", "-t", type=int, default=30,
                       help="Base timeout in seconds for server requests (default: 30)")
    parser.add_argument("--patient", action="store_true",
                       help="Use very long timeouts for slow servers (up to 5 minutes)")
    parser.add_argument("--show-logs", action="store_true",
                       help="Show recent VLLM server logs during monitoring")
    parser.add_argument("--log-lines", type=int, default=10,
                       help="Number of log lines to show (default: 10)")
    parser.add_argument("--logs-only", action="store_true",
                       help="Only show logs and exit (no continuous monitoring)")

    args = parser.parse_args()

    # Check if config file exists
    config_path = args.config if os.path.exists(args.config) else None
    if not config_path and not args.no_config:
        print(f"⚠️  Config file '{args.config}' not found, configuration display disabled")

    monitor = VLLMMonitor(
        server_url=args.url,
        config_path=config_path,
        enable_runtime_detection=not args.no_runtime_detection,
        base_timeout=args.timeout,
        patient_mode=args.patient
    )

    # Handle logs-only mode
    if args.logs_only:
        await monitor.show_recent_logs(args.log_lines)
        return

    if args.once:
        await monitor.monitor_once(
            test_inference=not args.no_test,
            show_config=not args.no_config and config_path is not None,
            detailed_models=not args.no_model_details,
            show_logs=args.show_logs,
            log_lines=args.log_lines
        )
    else:
        await monitor.monitor_continuous(
            interval=args.interval,
            test_inference=not args.no_test,
            show_config=not args.no_config and config_path is not None,
            detailed_models=not args.no_model_details,
            show_logs=args.show_logs,
            log_lines=args.log_lines
        )

if __name__ == "__main__":
    asyncio.run(main())