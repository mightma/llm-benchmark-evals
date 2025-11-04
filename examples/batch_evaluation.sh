#!/bin/bash
# Batch evaluation script for multiple configurations

# Set base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Configuration
MODEL_NAME="QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8"
OUTPUT_DIR="results/batch_evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting batch evaluation at $(date)"
echo "Results will be saved to: $OUTPUT_DIR"

# Function to run evaluation
run_eval() {
    local config_name="$1"
    shift
    local args="$@"

    echo "Running evaluation: $config_name"
    python main.py evaluate \
        --model "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --run-id "${config_name}_${TIMESTAMP}" \
        $args

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $config_name"
    else
        echo "✗ Failed: $config_name"
    fi
    echo ""
}

# Temperature sweep
echo "=== Temperature Sweep ==="
for temp in 0.0 0.1 0.3 0.5 0.7 1.0; do
    run_eval "temp_${temp}" \
        --temperature "$temp" \
        --benchmark mmlu_pro \
        --num-samples 100
done

# Top-p sweep
echo "=== Top-p Sweep ==="
for top_p in 0.5 0.7 0.9 0.95 1.0; do
    run_eval "top_p_${top_p}" \
        --top-p "$top_p" \
        --temperature 0.7 \
        --benchmark mmlu_pro \
        --num-samples 100
done

# Benchmark comparison
echo "=== Benchmark Comparison ==="
for benchmark in mmlu_pro aime25 ifeval; do
    run_eval "benchmark_${benchmark}" \
        --benchmark "$benchmark" \
        --temperature 0.1 \
        --num-samples 50
done

# Conservative vs Creative settings
echo "=== Conservative vs Creative Settings ==="

run_eval "conservative" \
    --temperature 0.1 \
    --top-p 0.9 \
    --top-k 10 \
    --num-samples 100

run_eval "creative" \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 50 \
    --num-samples 100

# Generate comparison reports
echo "=== Generating Comparison Reports ==="

# Find all result files from this batch
RESULT_FILES=$(find "$OUTPUT_DIR" -name "*${TIMESTAMP}.json" | grep -v summary | head -10)

if [ ! -z "$RESULT_FILES" ]; then
    echo "Comparing batch results..."
    python main.py compare \
        $RESULT_FILES \
        --output "$OUTPUT_DIR/batch_comparison_${TIMESTAMP}.json" \
        --format table

    # Export to Excel if multiple files
    FILE_COUNT=$(echo "$RESULT_FILES" | wc -l)
    if [ "$FILE_COUNT" -gt 1 ]; then
        python main.py compare \
            $RESULT_FILES \
            --output "$OUTPUT_DIR/batch_comparison_${TIMESTAMP}.xlsx" \
            --format excel
    fi
fi

echo "Batch evaluation completed at $(date)"
echo "Check results in: $OUTPUT_DIR"