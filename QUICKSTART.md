# Quick Start Guide

Get started with the LLM Evaluation Framework in 5 minutes!

## 1. Installation

```bash
# Navigate to the project directory
cd /home/ubuntu/efs/aifl-autel-llm-eval/evals

# Install dependencies
pip install -r requirements.txt
```

## 2. Basic Configuration

The default `config.yaml` is ready to use with Qwen3-30B-GPTQ-Int8 model. Key settings:

```yaml
models:
  - name: "Qwen3-30B-GPTQ-Int8"
    model_path: "QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8"

vllm_server:
  deploy_locally: true  # Framework will start VLLM server
  port: 8000

benchmarks:
  mmlu_pro:
    enabled: true
  aime25:
    enabled: true
  ifeval:
    enabled: true
```

## 3. Run Your First Evaluation

```bash
# Run all benchmarks with default settings
python main.py evaluate

# Or run a specific benchmark
python main.py evaluate --benchmark mmlu_pro
```

## 4. View Results

Results are automatically saved to the `results/` directory:
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Raw results
- `evaluation_results_YYYYMMDD_HHMMSS_summary.json` - Summary report

## 5. Quick Commands

```bash
# Check what benchmarks are available
python main.py list-benchmarks

# See available inference parameters
python main.py parameters

# Check server status
python main.py server-status

# Compare two evaluation runs
python main.py compare results/file1.json results/file2.json
```

## Example: Evaluate with Custom Parameters

```bash
# Run MMLU-Pro with conservative settings
python main.py evaluate \
  --benchmark mmlu_pro \
  --temperature 0.1 \
  --top-p 0.9 \
  --num-samples 100 \
  --run-id "conservative_test"
```

## Using External VLLM Server

If you already have a VLLM server running:

1. Edit `config.yaml`:
```yaml
vllm_server:
  deploy_locally: false
  external_host: "your-server-ip"
  external_port: 8000
```

2. Run evaluation:
```bash
python main.py evaluate
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize `config.yaml` for your specific models and requirements
- Add your own benchmark data to the `data/` directory
- Explore different inference parameters to optimize performance

## Common First-Time Issues

1. **CUDA Out of Memory**: Reduce `gpu_memory_utilization` in config
2. **Model Not Found**: Ensure model path is correct and accessible
3. **Port Already in Use**: Change port in config or stop existing services

## Need Help?

- Check the troubleshooting section in [README.md](README.md)
- Use `--verbose` flag for detailed logging
- Ensure all dependencies are installed correctly