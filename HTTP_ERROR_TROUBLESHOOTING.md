# HTTP Error Troubleshooting Guide

## Overview

The evaluation framework has been enhanced with robust HTTP error handling and retry mechanisms to address "HTTP error during inference" issues that may occur during model evaluation.

## Improvements Made

### 1. Enhanced Retry Logic (`src/vllm_manager.py`)

**Features:**
- **Automatic retries**: Up to 3 retries by default for failed requests
- **Exponential backoff**: Retry delays increase progressively (2s, 3s, 4.5s, etc.)
- **Intelligent error handling**: Different strategies for different error types
- **Configurable timeouts**: Base timeout increases with each retry attempt

**Error Types Handled:**
- **Connection errors**: Network connectivity issues
- **Timeout errors**: Requests taking too long
- **Server errors (5xx)**: VLLM server internal errors
- **Rate limiting (429)**: Server overload protection
- **Client errors (4xx)**: Malformed requests (no retry for these)

### 2. Configuration Options (`config.yaml`)

Added new configuration parameters under `inference` section:

```yaml
inference:
  # ... existing parameters ...
  max_retries: 3           # Number of retries for failed requests
  retry_delay: 2.0         # Initial delay between retries (seconds)
  request_timeout: 300     # Base timeout for requests (seconds)
```

### 3. Better Error Reporting (`src/benchmarks/base.py`)

- **Detailed error logging**: Specific context for HTTP/network errors
- **Error categorization**: Distinguish between network and processing errors
- **Better failure tracking**: Failed samples are properly recorded and reported

### 4. Improved Server Health Checks

- **Retry logic for health checks**: Multiple attempts to verify server status
- **Longer timeouts**: More time for server startup and health verification
- **Better connection testing**: More reliable server readiness detection

## Common HTTP Error Scenarios and Solutions

### 1. Connection Refused
```
ConnectionError: [Errno 111] Connection refused
```

**Causes:**
- VLLM server not started or crashed
- Wrong host/port configuration
- Firewall blocking connections

**Solutions:**
- Check if VLLM server is running: `python main.py server-status`
- Verify server configuration in `config.yaml`
- Check server logs for startup errors
- Ensure sufficient GPU memory is available

### 2. Timeout Errors
```
TimeoutException: Request timed out
```

**Causes:**
- Model inference taking too long
- Server overloaded
- Network latency issues

**Solutions:**
- Increase `request_timeout` in config.yaml
- Reduce batch size or concurrent requests
- Use smaller model or optimize inference parameters
- Check GPU memory usage

### 3. Server Errors (500, 502, 503)
```
HTTPStatusError: 500 Internal Server Error
```

**Causes:**
- VLLM server internal errors
- Out of memory errors
- Model loading failures

**Solutions:**
- Check VLLM server logs for detailed error messages
- Reduce `gpu_memory_utilization` in config.yaml
- Ensure model path is correct and accessible
- Restart the server: stop evaluation and run again

### 4. Rate Limiting (429)
```
HTTPStatusError: 429 Too Many Requests
```

**Causes:**
- Making requests too quickly
- Server has rate limiting enabled

**Solutions:**
- The retry logic automatically handles this with longer delays
- Reduce concurrent requests in evaluation settings
- Consider using a more powerful server setup

## Troubleshooting Steps

### Step 1: Check Server Status
```bash
python main.py server-status
```

### Step 2: Review Configuration
Verify these settings in `config.yaml`:
- Model path is correct
- GPU memory utilization is appropriate (try 0.8 instead of 0.9)
- Inference parameters are reasonable
- Retry settings are appropriate for your setup

### Step 3: Test with Minimal Sample
```bash
python main.py evaluate --benchmark mmlu_pro --num-samples 1
```

### Step 4: Check Logs
- Evaluation logs in `logs/` directory
- VLLM server output (if running locally)
- System memory and GPU usage

### Step 5: Adjust Configuration
If errors persist, try:
```yaml
inference:
  max_retries: 5           # More retries
  retry_delay: 5.0         # Longer delays
  request_timeout: 600     # Longer timeout

models:
  - gpu_memory_utilization: 0.8  # Reduce memory usage
```

## Advanced Configuration

### For High-Latency Networks
```yaml
inference:
  max_retries: 5
  retry_delay: 5.0
  request_timeout: 600
```

### For Unstable Connections
```yaml
inference:
  max_retries: 10
  retry_delay: 3.0
  request_timeout: 300
```

### For Resource-Constrained Environments
```yaml
models:
  - gpu_memory_utilization: 0.7
    max_model_len: 16384

evaluation:
  batch_size: 1
  max_concurrent: 1
```

## Monitoring and Debugging

### Enable Debug Logging
```bash
python main.py evaluate --verbose
```

### Monitor Resource Usage
- GPU memory: `nvidia-smi`
- System memory: `htop` or `free -h`
- Network connectivity: `ping` or `telnet` to server

### Check VLLM Server Health
```bash
curl http://localhost:8000/v1/models
```

## Getting Help

If you continue to experience HTTP errors after trying these solutions:

1. Check the GitHub issues for similar problems
2. Provide detailed error logs when reporting issues
3. Include your configuration and system specifications
4. Mention the specific model and benchmark being used