#!/usr/bin/env python3
"""
Example usage scripts for the LLM Evaluation Framework
"""

import subprocess
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run_basic_evaluation():
    """Run a basic evaluation with all benchmarks."""
    print("=== Running Basic Evaluation ===")
    cmd = [
        "python", "../main.py", "evaluate",
        "--num-samples", "10",  # Small sample for demo
        "--run-id", "basic_demo"
    ]
    subprocess.run(cmd)


def run_mmlu_pro_only():
    """Run only MMLU-Pro benchmark."""
    print("=== Running MMLU-Pro Only ===")
    cmd = [
        "python", "../main.py", "evaluate",
        "--benchmark", "mmlu_pro",
        "--num-samples", "50",
        "--temperature", "0.1",
        "--run-id", "mmlu_demo"
    ]
    subprocess.run(cmd)


def run_parameter_sweep():
    """Run evaluation with different temperature settings."""
    print("=== Running Parameter Sweep ===")

    temperatures = [0.0, 0.3, 0.7, 1.0]

    for temp in temperatures:
        print(f"Testing temperature: {temp}")
        cmd = [
            "python", "../main.py", "evaluate",
            "--benchmark", "mmlu_pro",
            "--temperature", str(temp),
            "--num-samples", "20",
            "--run-id", f"temp_sweep_{temp}"
        ]
        subprocess.run(cmd)


def run_comparison():
    """Compare results from different runs."""
    print("=== Running Model Comparison ===")

    # First, check if we have result files
    results_dir = "../results"
    if not os.path.exists(results_dir):
        print("No results directory found. Run evaluations first.")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'summary' not in f]

    if len(result_files) < 2:
        print("Need at least 2 result files for comparison. Run more evaluations first.")
        return

    # Take first two result files
    file1 = os.path.join(results_dir, result_files[0])
    file2 = os.path.join(results_dir, result_files[1])

    cmd = [
        "python", "../main.py", "compare",
        file1, file2,
        "--format", "table"
    ]
    subprocess.run(cmd)


def show_server_info():
    """Show server status and available commands."""
    print("=== Server Information ===")

    commands = [
        ["python", "../main.py", "server-status"],
        ["python", "../main.py", "list-benchmarks"],
        ["python", "../main.py", "parameters"]
    ]

    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd)


def main():
    """Run example demonstrations."""
    print("LLM Evaluation Framework - Examples")
    print("=" * 50)

    examples = {
        "1": ("Basic Evaluation", run_basic_evaluation),
        "2": ("MMLU-Pro Only", run_mmlu_pro_only),
        "3": ("Parameter Sweep", run_parameter_sweep),
        "4": ("Compare Results", run_comparison),
        "5": ("Server Info", show_server_info),
        "all": ("Run All Examples", None)
    }

    print("\nAvailable examples:")
    for key, (desc, _) in examples.items():
        print(f"  {key}: {desc}")

    choice = input("\nSelect example to run (or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        return

    if choice == "all":
        print("Running all examples in sequence...")
        for key, (_, func) in examples.items():
            if func and key != "all":
                print(f"\n{'='*20} {examples[key][0]} {'='*20}")
                func()
                input("Press Enter to continue to next example...")
    elif choice in examples:
        func = examples[choice][1]
        if func:
            func()
        else:
            print("Invalid selection")
    else:
        print("Invalid selection")


if __name__ == "__main__":
    main()