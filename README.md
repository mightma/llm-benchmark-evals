# LLM Evaluation Framework

A comprehensive framework for evaluating Large Language Models (LLMs) on various benchmarks including MMLU-Pro, AIME25, and IFEval using VLLM for efficient inference.

## Features

- **VLLM Integration**: Deploy LLM locally or connect to external VLLM server
- **Multiple Benchmarks**: Support for MMLU-Pro, AIME25, IFEval, and extensible for more
- **Flexible Model Support**: Works with various models including quantized versions (GPTQ, AWQ)
- **Configurable Inference**: Adjustable temperature, top-p, top-k, and other parameters
- **Result Analysis**: Comprehensive result summarization and model comparison tools
- **CLI Interface**: Easy-to-use command line interface with rich output formatting

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/ubuntu/efs/aifl-autel-llm-eval/evals
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Configuration

The framework uses a YAML configuration file (`config.yaml`) to define models, server settings, and benchmark parameters.

### Key Configuration Sections:

#### Models
```yaml
models:
  - name: "Qwen3-30B-GPTQ-Int8"
    model_path: "QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8"
    quantization: "gptq"
    gpu_memory_utilization: 0.9
    max_model_len: 32768
```

#### VLLM Server
```yaml
vllm_server:
  host: "0.0.0.0"
  port: 8000
  deploy_locally: true  # Set to false for external server
  external_host: null   # Set if using external server
  external_port: null   # Set if using external server
```

#### Inference Parameters
```yaml
inference:
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  max_tokens: 2048
  repetition_penalty: 1.0
```

## Usage

### Basic Commands

#### 1. Run Evaluation
```bash
# Run all enabled benchmarks
python main.py evaluate

# Run specific benchmark
python main.py evaluate --benchmark mmlu_pro

# Run with custom model name
python main.py evaluate --model "my-model"

# Limit number of samples
python main.py evaluate --num-samples 100
```

#### 2. Custom Inference Parameters
```bash
# Adjust temperature and sampling
python main.py evaluate \
  --temperature 0.1 \
  --top-p 0.9 \
  --top-k 50 \
  --max-tokens 1024
```

#### 3. Multiple Responses (Self-Consistency)
```bash
# Generate multiple responses per question for improved accuracy
python main.py evaluate \
  --benchmark aime25 \
  --num-responses 5 \
  --temperature 0.7

# Useful for mathematical problems and reasoning tasks
python main.py evaluate \
  --benchmark mmlu_pro \
  --num-responses 3 \
  --num-samples 100
```

#### 4. Concurrent Processing
```bash
# Use concurrent requests for faster evaluation (default: 4 concurrent)
python main.py evaluate \
  --benchmark mmlu_pro \
  --max-concurrent 8 \
  --num-samples 100

# Sequential processing (useful for debugging or rate-limited servers)
python main.py evaluate \
  --benchmark ifeval \
  --max-concurrent 1
```

#### 5. Compare Models
```bash
# Compare results from multiple evaluation runs
python main.py compare results/model1_results.json results/model2_results.json

# Export comparison to Excel
python main.py compare \
  results/model1_results.json \
  results/model2_results.json \
  --format excel \
  --output comparison.xlsx
```

#### 6. Server Management
```bash
# Check server status
python main.py server-status

# List available benchmarks
python main.py list-benchmarks

# Show inference parameters
python main.py parameters
```

### Advanced Usage

#### Using External VLLM Server

If you already have a VLLM server running, configure it in `config.yaml`:

```yaml
vllm_server:
  deploy_locally: false
  external_host: "192.168.1.100"
  external_port: 8000
```

#### Custom Data Paths

The framework will automatically download benchmark data from HuggingFace when needed. You can also place local data in the specified directories:

```
data/
├── mmlu_pro/          # MMLU-Pro data files (optional - auto-downloads from HuggingFace)
├── aime25/            # AIME25 problems (optional - auto-downloads from HuggingFace math-ai/aime25)
└── ifeval/            # IFEval data (optional - auto-downloads from HuggingFace)
```

#### Batch Evaluation

For multiple models or parameter combinations:

```bash
# Create a script for batch evaluation
#!/bin/bash
for temp in 0.1 0.3 0.7; do
  python main.py evaluate \
    --temperature $temp \
    --run-id "temp_${temp}" \
    --output-dir "results/temperature_sweep"
done
```

## Advanced Features

### Multiple Responses (Self-Consistency)

The framework supports generating multiple responses per question to improve accuracy through self-consistency:

#### How It Works
1. **Multiple Generation**: Generate N responses for each question (default: 1)
2. **Individual Evaluation**: Each response is evaluated independently
3. **Majority Voting**: If multiple responses are correct, use majority voting on predicted answers
4. **Best Response Selection**: Select the most frequent correct answer as the final result

#### Benefits
- **Improved Accuracy**: Especially effective for mathematical and reasoning problems
- **Reduced Variance**: Multiple samples help avoid single bad generations
- **Statistical Confidence**: Provides insights into model consistency

#### Usage Examples
```bash
# Generate 5 responses per AIME problem
python main.py evaluate --benchmark aime25 --num-responses 5

# Self-consistency for MMLU-Pro with temperature sampling
python main.py evaluate --benchmark mmlu_pro --num-responses 3 --temperature 0.7
```

#### Output Format
Results include additional metadata for multiple responses:
- `majority_vote_count`: Number of responses that agreed with the final answer
- `total_responses`: Total number of responses generated
- `correct_responses`: Number of responses that were individually correct
- `all_responses`: Full list of generated responses (in prediction files)

### Concurrent Processing

The framework supports concurrent processing to significantly speed up evaluations by making multiple VLLM server requests in parallel.

#### How It Works
1. **Semaphore Control**: Uses asyncio.Semaphore to limit concurrent requests
2. **Parallel Sample Processing**: Multiple samples processed simultaneously
3. **Nested Concurrency**: When using multiple responses, those are also generated concurrently
4. **Order Preservation**: Results maintain original sample order regardless of completion timing

#### Benefits
- **Significant Speed Improvement**: 2-4x faster evaluation times
- **Configurable Concurrency**: Adjust based on server capacity and rate limits
- **Resource Efficient**: Optimal utilization of VLLM server capacity
- **Error Isolation**: Individual sample failures don't affect other concurrent requests

#### Configuration Options
```yaml
evaluation:
  max_concurrent: 4  # Number of concurrent requests (default: 4)
```

#### Performance Considerations
- **VLLM Server Capacity**: Higher concurrency requires more server resources
- **Rate Limiting**: Some servers may have rate limits requiring lower concurrency
- **Memory Usage**: More concurrent requests use more memory
- **Network Latency**: High-latency connections benefit more from concurrency

#### Usage Examples
```bash
# High-performance evaluation with 8 concurrent requests
python main.py evaluate --max-concurrent 8 --benchmark mmlu_pro

# Conservative approach for rate-limited servers
python main.py evaluate --max-concurrent 2 --benchmark aime25

# Sequential processing for debugging
python main.py evaluate --max-concurrent 1 --benchmark ifeval
```

## Benchmark Details

### MMLU-Pro
- **Description**: Enhanced MMLU with more challenging questions and up to 10 answer choices
- **Metric**: Accuracy
- **Data**: Loads from HuggingFace or local JSON files
- **Format**: Multiple choice (A-J)

### AIME25
- **Description**: American Invitational Mathematics Examination 2025 problems
- **Metric**: Exact match accuracy
- **Data**: Auto-loads from HuggingFace `math-ai/aime25` dataset (30 problems) or local JSON files
- **Format**: Integer answers (000-999)

### IFEval
- **Description**: Instruction-following evaluation with specific formatting requirements
- **Metric**: Instruction compliance rate
- **Data**: Auto-loads from HuggingFace `google/IFEval` dataset (541 samples) or local JSON files
- **Format**: Complete coverage of all 25 instruction types from the original Google IFEval benchmark

## Output Files

### Result Files
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Raw evaluation results
- `evaluation_results_YYYYMMDD_HHMMSS_summary.json`: Summary report
- Individual benchmark predictions (if enabled)

### Result Structure
```json
{
  "benchmark_name": "mmlu_pro",
  "model_name": "Qwen3-30B-GPTQ-Int8",
  "score": 0.7543,
  "details": {
    "accuracy": 0.7543,
    "correct": 1508,
    "total": 2000
  },
  "num_samples": 2000,
  "timestamp": "2024-01-15T10:30:00",
  "config": {...}
}
```

## Example Workflows

### 1. Quick Model Evaluation
```bash
# Evaluate with default settings
python main.py evaluate --model "QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8"
```

### 2. Comprehensive Evaluation
```bash
# Run all benchmarks with custom parameters
python main.py evaluate \
  --model "my-model" \
  --temperature 0.0 \
  --top-p 1.0 \
  --num-samples 500 \
  --output-dir "results/comprehensive" \
  --run-id "comprehensive_eval"
```

### 3. Parameter Sweep
```bash
# Evaluate different temperature settings
for temp in 0.0 0.3 0.7 1.0; do
  python main.py evaluate \
    --temperature $temp \
    --benchmark mmlu_pro \
    --run-id "mmlu_temp_${temp}"
done

# Compare results
python main.py compare results/mmlu_temp_*.json --format markdown
```

### 4. Model Comparison
```bash
# Evaluate multiple models
python main.py evaluate --model "model1" --run-id "model1_eval"
python main.py evaluate --model "model2" --run-id "model2_eval"

# Compare and generate report
python main.py compare \
  results/model1_eval.json \
  results/model2_eval.json \
  --output model_comparison.json \
  --format table
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `gpu_memory_utilization` in config
   - Decrease `max_model_len`
   - Use smaller batch sizes

2. **Server Won't Start**
   - Check if port is already in use
   - Verify model path is correct
   - Ensure sufficient GPU memory

3. **Evaluation Fails**
   - Check data file paths
   - Verify model is compatible with benchmark format
   - Review server logs for inference errors

### Debug Mode
```bash
# Enable verbose logging
python main.py --verbose evaluate --benchmark mmlu_pro
```

### Manual Server Testing
```bash
# Test external server connection
curl http://localhost:8000/v1/models

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

## Extensions

### Adding New Benchmarks

1. Create a new benchmark class inheriting from `BaseBenchmark`
2. Implement required methods: `load_data()`, `evaluate_sample()`, `aggregate_results()`
3. Register in `evaluation_runner.py`
4. Add configuration to `config.yaml`

### Custom Metrics

Extend the `aggregate_results()` method to include custom metrics:

```python
def aggregate_results(self, sample_results):
    base_results = super().aggregate_results(sample_results)
    # Add custom metrics
    base_results["custom_metric"] = calculate_custom_metric(sample_results)
    return base_results
```

## Performance Tips

1. **Use quantized models** for faster inference and lower memory usage
2. **Adjust batch size** based on available GPU memory
3. **Enable tensor parallelism** for multi-GPU setups
4. **Use appropriate precision** (FP16/BF16) for speed gains
5. **Limit sample counts** during development and testing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration files
3. Enable verbose logging for detailed error information
4. Consult VLLM documentation for server-related issues