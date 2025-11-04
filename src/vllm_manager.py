"""
VLLM Server Management Module

Handles deployment and management of VLLM servers for LLM inference.
"""

import asyncio
import logging
import subprocess
import time
import httpx
from typing import Optional, Dict, Any
import psutil
import signal
import os

logger = logging.getLogger(__name__)


class VLLMServerManager:
    """Manages VLLM server deployment and lifecycle."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vllm_config = config.get("vllm_server", {})
        self.model_config = config.get("models", [{}])[0]  # Use first model by default
        self.process: Optional[subprocess.Popen] = None
        self.server_url = None

    async def start_server(self) -> str:
        """Start VLLM server if configured to deploy locally."""
        if not self.vllm_config.get("deploy_locally", True):
            # Use external server
            external_host = self.vllm_config.get("external_host", "localhost")
            external_port = self.vllm_config.get("external_port", 8000)
            self.server_url = f"http://{external_host}:{external_port}"

            # Test connection to external server
            if await self._test_server_connection():
                logger.info(f"Connected to external VLLM server at {self.server_url}")
                return self.server_url
            else:
                raise ConnectionError(f"Cannot connect to external VLLM server at {self.server_url}")

        # Deploy server locally
        host = self.vllm_config.get("host", "0.0.0.0")
        port = self.vllm_config.get("port", 8000)
        self.server_url = f"http://{host}:{port}"

        logger.info(f"Starting VLLM server for model: {self.model_config.get('model_path')}")

        # Build VLLM command
        cmd = self._build_vllm_command(host, port)
        logger.info(f"VLLM command: {' '.join(cmd)}")

        # Start the server process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )

        # Wait for server to be ready
        await self._wait_for_server_ready()

        logger.info(f"VLLM server started successfully at {self.server_url}")
        return self.server_url

    def _build_vllm_command(self, host: str, port: int) -> list:
        """Build VLLM server command."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_config.get("model_path"),
            "--host", host,
            "--port", str(port),
        ]

        # Add optional parameters
        if gpu_mem := self.model_config.get("gpu_memory_utilization"):
            cmd.extend(["--gpu-memory-utilization", str(gpu_mem)])

        if max_len := self.model_config.get("max_model_len"):
            cmd.extend(["--max-model-len", str(max_len)])

        if quantization := self.model_config.get("quantization"):
            cmd.extend(["--quantization", quantization])

        # Add trust remote code flag for most models
        cmd.append("--trust-remote-code")

        return cmd

    async def _wait_for_server_ready(self, timeout: int = 300) -> None:
        """Wait for VLLM server to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self._test_server_connection():
                return

            # Check if process is still running
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"VLLM server process died. STDERR: {stderr}")

            logger.info("Waiting for VLLM server to be ready...")
            await asyncio.sleep(5)

        raise TimeoutError(f"VLLM server did not start within {timeout} seconds")

    async def _test_server_connection(self, max_retries: int = 3) -> bool:
        """Test connection to VLLM server with retries."""
        for attempt in range(max_retries):
            try:
                timeout = httpx.Timeout(30.0)  # Longer timeout for server health checks
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"{self.server_url}/v1/models")
                    if response.status_code == 200:
                        return True
                    else:
                        logger.debug(f"Server health check returned status {response.status_code}")
            except httpx.ConnectError as e:
                logger.debug(f"Connection failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)
            except httpx.TimeoutException as e:
                logger.debug(f"Timeout during health check on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)
            except Exception as e:
                logger.debug(f"Server connection test failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)

        return False

    async def stop_server(self) -> None:
        """Stop the VLLM server."""
        if self.process:
            logger.info("Stopping VLLM server...")

            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Forcing VLLM server shutdown...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()

                logger.info("VLLM server stopped")
            except Exception as e:
                logger.error(f"Error stopping VLLM server: {e}")
            finally:
                self.process = None

    def get_server_url(self) -> Optional[str]:
        """Get the server URL."""
        return self.server_url

    def is_running(self) -> bool:
        """Check if server is running."""
        if not self.vllm_config.get("deploy_locally", True):
            # For external servers, we assume they're running
            return self.server_url is not None

        return self.process is not None and self.process.poll() is None

    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and available models."""
        # Determine the expected server URL
        if not self.vllm_config.get("deploy_locally", True):
            # External server
            external_host = self.vllm_config.get("external_host", "localhost")
            external_port = self.vllm_config.get("external_port", 8000)
            expected_url = f"http://{external_host}:{external_port}"
        else:
            # Local server
            host = self.vllm_config.get("host", "0.0.0.0")
            port = self.vllm_config.get("port", 8000)
            # Use localhost for connection even if host is 0.0.0.0
            display_host = "localhost" if host == "0.0.0.0" else host
            expected_url = f"http://{display_host}:{port}"

        # Use current server_url if available, otherwise use expected URL
        check_url = self.server_url if self.server_url else expected_url

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                models_response = await client.get(f"{check_url}/v1/models")
                models_response.raise_for_status()

                return {
                    "server_url": check_url,
                    "expected_url": expected_url,
                    "status": "running",
                    "models": models_response.json(),
                    "deploy_locally": self.vllm_config.get("deploy_locally", True),
                    "process_running": self.is_running() if self.vllm_config.get("deploy_locally", True) else "N/A"
                }
        except Exception as e:
            return {
                "server_url": check_url,
                "expected_url": expected_url,
                "status": "not_running" if "Connection" in str(e) else "error",
                "error": str(e),
                "deploy_locally": self.vllm_config.get("deploy_locally", True),
                "process_running": self.is_running() if self.vllm_config.get("deploy_locally", True) else "N/A"
            }


class VLLMInferenceClient:
    """Client for making inference requests to VLLM server."""

    def __init__(self, server_url: str, config: Dict[str, Any]):
        self.server_url = server_url
        self.inference_config = config.get("inference", {})
        # Get retry configuration from config
        self.max_retries = self.inference_config.get("max_retries", 3)
        self.retry_delay = self.inference_config.get("retry_delay", 2.0)
        self.request_timeout = self.inference_config.get("request_timeout", 300.0)
        self.client = httpx.AsyncClient(timeout=self.request_timeout)

    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from the model with retry logic."""
        # Use provided parameters or fall back to config/instance defaults
        max_retries = max_retries if max_retries is not None else self.max_retries
        retry_delay = retry_delay if retry_delay is not None else self.retry_delay

        params = {
            "model": model_name or "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.inference_config.get("temperature", 0.0),
            "top_p": top_p if top_p is not None else self.inference_config.get("top_p", 1.0),
            "max_tokens": max_tokens if max_tokens is not None else self.inference_config.get("max_tokens", 2048),
        }

        # Add top_k if specified (not all models support it)
        if top_k is not None or self.inference_config.get("top_k", -1) > 0:
            params["top_k"] = top_k if top_k is not None else self.inference_config.get("top_k")

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Increase timeout for each retry
                timeout = httpx.Timeout(self.request_timeout + (attempt * 60.0))

                async with httpx.AsyncClient(timeout=timeout) as retry_client:
                    response = await retry_client.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=params
                    )
                    response.raise_for_status()
                    return response.json()

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Timeout during inference attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500:
                    # Server error - retry
                    logger.warning(f"Server error {e.response.status_code} during inference attempt {attempt + 1}/{max_retries + 1}: {e}")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                    continue
                elif e.response.status_code == 429:
                    # Rate limit - longer retry delay
                    logger.warning(f"Rate limit hit during inference attempt {attempt + 1}/{max_retries + 1}: {e}")
                    if attempt < max_retries:
                        rate_limit_delay = retry_delay * 2
                        logger.info(f"Retrying in {rate_limit_delay} seconds due to rate limit...")
                        await asyncio.sleep(rate_limit_delay)
                    continue
                else:
                    # Client error - don't retry
                    logger.error(f"Client error during inference: {e.response.status_code} - {e}")
                    raise

            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(f"Connection error during inference attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during inference attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise

        # If we get here, all retries failed
        logger.error(f"All {max_retries + 1} inference attempts failed. Last error: {last_exception}")
        raise last_exception

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()