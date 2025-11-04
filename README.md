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

#### 3. Compare Models
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

#### 4. Server Management
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

Place your benchmark data in the specified directories:

```
data/
├── mmlu_pro/          # MMLU-Pro data files
├── aime25/            # AIME25 problems
└── ifeval/            # IFEval data
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

## Benchmark Details

### MMLU-Pro
- **Description**: Enhanced MMLU with more challenging questions and up to 10 answer choices
- **Metric**: Accuracy
- **Data**: Loads from HuggingFace or local JSON files
- **Format**: Multiple choice (A-J)

### AIME25
- **Description**: American Invitational Mathematics Examination 2025 problems
- **Metric**: Exact match accuracy
- **Data**: JSON file with mathematical problems
- **Format**: Integer answers (000-999)

### IFEval
- **Description**: Instruction-following evaluation with specific formatting requirements
- **Metric**: Instruction compliance rate
- **Data**: Loads from HuggingFace or local JSON files
- **Format**: Various instruction types

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