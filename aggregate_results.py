#!/usr/bin/env python3
"""
Results Aggregation Script

Aggregates all evaluation results from the results/ directory and generates
comprehensive statistics about model performance across different benchmarks.
"""

import os
import json
import csv
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse

@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""
    model_name: str
    benchmark_name: str
    score: float
    accuracy: float
    total_samples: int
    successful_samples: int
    failed_samples: int
    success_rate: float
    failure_rate: float
    original_samples: Optional[int] = None
    responses_per_sample: Optional[int] = None
    timestamp: Optional[str] = None
    config_info: Optional[Dict[str, Any]] = None

    # Benchmark-specific metrics
    correct_answers: Optional[int] = None
    subjects: Optional[int] = None
    problems_solved: Optional[int] = None
    instruction_accuracy: Optional[float] = None
    instructions_passed: Optional[int] = None
    total_instructions: Optional[int] = None
    failure_types: Optional[Dict[str, int]] = None

class ResultsAggregator:
    """Aggregates and analyzes evaluation results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results: List[BenchmarkResult] = []

    def scan_results_directory(self) -> List[str]:
        """Scan results directory for JSON result files."""
        if not os.path.exists(self.results_dir):
            print(f"❌ Results directory not found: {self.results_dir}")
            return []

        # Look for JSON files that are not summaries
        pattern = os.path.join(self.results_dir, "*.json")
        json_files = glob.glob(pattern)

        # Filter out summary files
        result_files = [f for f in json_files if not f.endswith('_summary.json')]

        print(f"📁 Found {len(result_files)} result files in {self.results_dir}/")
        return sorted(result_files)

    def parse_result_file(self, file_path: str) -> List[BenchmarkResult]:
        """Parse a single result file and extract benchmark results."""
        results = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both single result and list of results
            if isinstance(data, list):
                result_list = data
            else:
                result_list = [data]

            for result_data in result_list:
                result = self._parse_single_result(result_data, file_path)
                if result:
                    results.append(result)

        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")

        return results

    def _parse_single_result(self, data: Dict[str, Any], file_path: str) -> Optional[BenchmarkResult]:
        """Parse a single result entry."""
        try:
            details = data.get('details', {})

            # Extract basic metrics
            benchmark_result = BenchmarkResult(
                model_name=data.get('model_name', 'Unknown'),
                benchmark_name=data.get('benchmark_name', 'Unknown'),
                score=float(data.get('score', 0.0)),
                accuracy=float(details.get('accuracy', details.get('overall_score', 0.0))),
                total_samples=int(details.get('total_samples', data.get('num_samples', 0))),
                successful_samples=int(details.get('successful_samples', details.get('total', 0))),
                failed_samples=int(details.get('failed_samples', 0)),
                success_rate=float(details.get('success_rate', 1.0)),
                failure_rate=float(details.get('failure_rate', 0.0)),
                timestamp=data.get('timestamp', os.path.basename(file_path)),
                config_info=data.get('config', {}),
                failure_types=details.get('failure_types', {})
            )

            # Extract multiple response info if available
            benchmark_result.original_samples = details.get('original_samples')
            benchmark_result.responses_per_sample = details.get('responses_per_sample')

            # Extract benchmark-specific metrics
            benchmark_result.correct_answers = details.get('correct')

            # MMLU-Pro specific
            if benchmark_result.benchmark_name == 'mmlu_pro':
                benchmark_result.subjects = details.get('num_subjects')

            # AIME25 specific
            elif benchmark_result.benchmark_name == 'aime25':
                benchmark_result.problems_solved = details.get('problems_solved')

            # IFEval specific
            elif benchmark_result.benchmark_name == 'ifeval':
                benchmark_result.instruction_accuracy = details.get('instruction_accuracy')
                benchmark_result.instructions_passed = details.get('instructions_passed')
                benchmark_result.total_instructions = details.get('total_instructions')

            return benchmark_result

        except Exception as e:
            print(f"⚠️  Error parsing result data: {e}")
            return None

    def load_all_results(self) -> None:
        """Load and parse all result files."""
        print("🔍 Scanning results directory...")

        result_files = self.scan_results_directory()
        if not result_files:
            print("❌ No result files found!")
            return

        self.results = []

        print("📊 Parsing result files...")
        for file_path in result_files:
            file_results = self.parse_result_file(file_path)
            self.results.extend(file_results)
            if file_results:
                print(f"   ✅ {os.path.basename(file_path)}: {len(file_results)} results")
            else:
                print(f"   ⚠️  {os.path.basename(file_path)}: no valid results")

        print(f"\n📈 Total results loaded: {len(self.results)}")

    def generate_summary_table(self) -> List[Dict[str, Any]]:
        """Generate a comprehensive summary table."""
        if not self.results:
            print("❌ No results available for summary")
            return []

        # Group results by model and benchmark
        grouped = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            grouped[result.model_name][result.benchmark_name].append(result)

        summary_rows = []

        for model_name in sorted(grouped.keys()):
            model_results = grouped[model_name]

            for benchmark_name in sorted(model_results.keys()):
                benchmark_results = model_results[benchmark_name]

                # If there are multiple results for the same model/benchmark,
                # use the most recent one (or aggregate them)
                if len(benchmark_results) > 1:
                    # Use the most recent result (by timestamp if available)
                    result = max(benchmark_results, key=lambda x: x.timestamp or "")
                    print(f"ℹ️  Multiple results for {model_name}/{benchmark_name}, using most recent")
                else:
                    result = benchmark_results[0]

                # Create summary row
                row = {
                    'Model': model_name,
                    'Benchmark': benchmark_name,
                    'Score': f"{result.score:.4f}",
                    'Accuracy': f"{result.accuracy:.4f}",
                    'Total_Samples': result.total_samples,
                    'Successful_Samples': result.successful_samples,
                    'Failed_Samples': result.failed_samples,
                    'Success_Rate': f"{result.success_rate:.1%}",
                    'Failure_Rate': f"{result.failure_rate:.1%}",
                    'Dataset_Size': result.original_samples or result.total_samples,
                    'Responses_Per_Sample': result.responses_per_sample or 1,
                    'Timestamp': result.timestamp
                }

                # Add benchmark-specific metrics
                if result.correct_answers is not None:
                    row['Correct_Answers'] = result.correct_answers

                if result.benchmark_name == 'mmlu_pro' and result.subjects:
                    row['Subjects'] = result.subjects
                elif result.benchmark_name == 'aime25' and result.problems_solved is not None:
                    row['Problems_Solved'] = result.problems_solved
                elif result.benchmark_name == 'ifeval':
                    if result.instruction_accuracy is not None:
                        row['Instruction_Accuracy'] = f"{result.instruction_accuracy:.4f}"
                    if result.instructions_passed is not None and result.total_instructions is not None:
                        row['Instructions_Passed'] = f"{result.instructions_passed}/{result.total_instructions}"

                # Add failure breakdown
                if result.failure_types:
                    failure_summary = []
                    for error_type, count in result.failure_types.items():
                        failure_summary.append(f"{error_type}:{count}")
                    row['Failure_Types'] = ", ".join(failure_summary)
                else:
                    row['Failure_Types'] = "None"

                summary_rows.append(row)

        return summary_rows

    def print_summary_table(self, summary_data: List[Dict[str, Any]]) -> None:
        """Print a formatted summary table to console."""
        if not summary_data:
            print("❌ No data to display")
            return

        print("\n" + "="*120)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*120)

        # Print header
        headers = [
            "Model", "Benchmark", "Score", "Accuracy", "Success/Total",
            "Success Rate", "Dataset Size", "Responses/Sample", "Specific Metrics"
        ]

        # Calculate column widths dynamically based on content
        col_widths = []

        # Model column width
        model_width = max(len("Model"), max(len(str(row.get("Model", ""))) for row in summary_data))
        col_widths.append(model_width)

        # Benchmark column width
        benchmark_width = max(len("Benchmark"), max(len(str(row.get("Benchmark", ""))) for row in summary_data))
        col_widths.append(benchmark_width)

        # Score column width
        score_width = max(len("Score"), 8)  # At least 8 for "0.0000" format
        col_widths.append(score_width)

        # Accuracy column width
        accuracy_width = max(len("Accuracy"), 8)  # At least 8 for "0.0000" format
        col_widths.append(accuracy_width)

        # Success/Total column width - calculate based on actual data
        success_total_width = max(len("Success/Total"),
                                 max(len(f"{row.get('Successful_Samples', 0)}/{row.get('Total_Samples', 0)}")
                                     for row in summary_data))
        col_widths.append(success_total_width)

        # Success Rate column width
        success_rate_width = max(len("Success Rate"), 12)  # At least 12 for "100.0%" format
        col_widths.append(success_rate_width)

        # Dataset Size column width
        dataset_size_width = max(len("Dataset Size"),
                                max(len(str(row.get("Dataset_Size", 0))) for row in summary_data))
        col_widths.append(dataset_size_width)

        # Responses/Sample column width
        responses_width = max(len("Responses/Sample"),
                             max(len(str(row.get("Responses_Per_Sample", 1))) for row in summary_data))
        col_widths.append(responses_width)

        # Specific Metrics column width - calculate based on actual content
        specific_metrics_contents = []
        for row in summary_data:
            specific_metrics = ""
            if row['Benchmark'] == 'mmlu_pro':
                if 'Subjects' in row:
                    specific_metrics = f"Subjects: {row['Subjects']}"
                if 'Correct_Answers' in row:
                    specific_metrics += f", Correct: {row['Correct_Answers']}"
            elif row['Benchmark'] == 'aime25':
                if 'Problems_Solved' in row:
                    specific_metrics = f"Solved: {row['Problems_Solved']}"
            elif row['Benchmark'] == 'ifeval':
                if 'Instruction_Accuracy' in row:
                    specific_metrics = f"Inst Acc: {row['Instruction_Accuracy']}"
                if 'Instructions_Passed' in row:
                    specific_metrics += f", Passed: {row['Instructions_Passed']}"
            specific_metrics_contents.append(specific_metrics)

        specific_metrics_width = max(len("Specific Metrics"),
                                   max(len(content) for content in specific_metrics_contents) if specific_metrics_contents else 15)
        col_widths.append(specific_metrics_width)

        # Print header
        header_row = ""
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            header_row += f"{header:<{width}} "
        print(header_row)
        print("-" * len(header_row))

        # Print data rows
        for row in summary_data:
            if row["Model"] == "Unknown":
                continue
            # Format success/total
            success_total = f"{row['Successful_Samples']}/{row['Total_Samples']}"

            # Format specific metrics based on benchmark
            specific_metrics = ""
            if row['Benchmark'] == 'mmlu_pro':
                if 'Subjects' in row:
                    specific_metrics = f"Subjects: {row['Subjects']}"
                if 'Correct_Answers' in row:
                    specific_metrics += f", Correct: {row['Correct_Answers']}"
            elif row['Benchmark'] == 'aime25':
                if 'Problems_Solved' in row:
                    specific_metrics = f"Solved: {row['Problems_Solved']}"
            elif row['Benchmark'] == 'ifeval':
                if 'Instruction_Accuracy' in row:
                    specific_metrics = f"Inst Acc: {row['Instruction_Accuracy']}"
                if 'Instructions_Passed' in row:
                    specific_metrics += f", Passed: {row['Instructions_Passed']}"

            # Create row
            data_row = ""
            values = [
                row['Model'], row['Benchmark'], row['Score'], row['Accuracy'],
                success_total, row['Success_Rate'], str(row['Dataset_Size']),
                str(row['Responses_Per_Sample']), specific_metrics
            ]

            for value, width in zip(values, col_widths):
                data_row += f"{str(value):<{width}} "
            print(data_row)

        print("-" * len(header_row))
        print(f"Total evaluations: {len(summary_data)}")

        # Print model summary
        models = set(row['Model'] for row in summary_data)
        benchmarks = set(row['Benchmark'] for row in summary_data)
        print(f"Models evaluated: {len(models)} ({', '.join(sorted(models))})")
        print(f"Benchmarks used: {len(benchmarks)} ({', '.join(sorted(benchmarks))})")

        # Print failure statistics
        total_samples = sum(row['Total_Samples'] for row in summary_data)
        total_failures = sum(row['Failed_Samples'] for row in summary_data)
        if total_failures > 0:
            print(f"⚠️  Total inference failures: {total_failures}/{total_samples} ({total_failures/total_samples:.1%})")
        else:
            print("✅ No inference failures detected")

    def save_to_csv(self, summary_data: List[Dict[str, Any]], output_file: str = None) -> str:
        """Save summary data to CSV file."""
        if not summary_data:
            print("❌ No data to save")
            return ""

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_summary_{timestamp}.csv"

        # Ensure it's in the results directory
        if not os.path.dirname(output_file):
            output_file = os.path.join(self.results_dir, output_file)

        try:
            # Get all possible columns
            all_columns = set()
            for row in summary_data:
                all_columns.update(row.keys())

            # Define column order
            priority_columns = [
                'Model', 'Benchmark', 'Score', 'Accuracy', 'Total_Samples',
                'Successful_Samples', 'Failed_Samples', 'Success_Rate', 'Failure_Rate',
                'Dataset_Size', 'Responses_Per_Sample', 'Correct_Answers',
                'Subjects', 'Problems_Solved', 'Instruction_Accuracy', 'Instructions_Passed',
                'Failure_Types', 'Timestamp'
            ]

            # Arrange columns: priority first, then others
            ordered_columns = []
            for col in priority_columns:
                if col in all_columns:
                    ordered_columns.append(col)
                    all_columns.remove(col)
            ordered_columns.extend(sorted(all_columns))

            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_columns)
                writer.writeheader()
                writer.writerows(summary_data)

            print(f"💾 Results saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return ""

    def generate_model_comparison(self, summary_data: List[Dict[str, Any]]) -> None:
        """Generate a model comparison matrix."""
        if not summary_data:
            return

        print("\n" + "="*80)
        print("🔄 MODEL COMPARISON MATRIX")
        print("="*80)

        # Group by benchmark, then by model
        benchmark_data = defaultdict(dict)

        for row in summary_data:
            benchmark = row['Benchmark']
            model = row['Model']
            score = float(row['Score'])
            benchmark_data[benchmark][model] = score

        # Print comparison table
        models = sorted(set(row['Model'] for row in summary_data if row['Model'] != 'Unknown'))
        benchmarks = sorted(benchmark_data.keys())

        if len(models) > 1:
            # Calculate column widths for comparison matrix
            benchmark_col_width = max(len("Benchmark"), max(len(benchmark) for benchmark in benchmarks))
            model_col_width = max(len(model) for model in models) if models else 10
            best_model_col_width = max(len("Best Model"), model_col_width)

            # Print header
            print(f"{'Benchmark':<{benchmark_col_width}} ", end="")
            for model in models:
                print(f"{model:<{model_col_width + 2}} ", end="")  # +2 for spacing
            print(f"{'Best Model':<{best_model_col_width}}")

            # Print separator line
            total_width = benchmark_col_width + 1 + len(models) * (model_col_width + 3) + best_model_col_width
            print("-" * total_width)

            # Print data rows
            for benchmark in benchmarks:
                if benchmark == 'Unknown':  # Skip Unknown benchmark
                    continue

                print(f"{benchmark:<{benchmark_col_width}} ", end="")
                best_score = -1
                best_model = ""

                for model in models:
                    score = benchmark_data[benchmark].get(model, 0.0)
                    if score > best_score:
                        best_score = score
                        best_model = model
                    print(f"{score:<{model_col_width + 2}.4f} ", end="")

                print(f"{best_model:<{best_model_col_width}}")
        else:
            print("Only one model found, no comparison available.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--results-dir", "-d", default="results",
                       help="Directory containing result files (default: results)")
    parser.add_argument("--output", "-o",
                       help="Output CSV filename (default: auto-generated)")
    parser.add_argument("--no-csv", action="store_true",
                       help="Don't save CSV file")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output")

    args = parser.parse_args()

    if not args.quiet:
        print("📊 Evaluation Results Aggregator")
        print("=" * 50)

    # Initialize aggregator
    aggregator = ResultsAggregator(args.results_dir)

    # Load all results
    aggregator.load_all_results()

    if not aggregator.results:
        print("❌ No evaluation results found!")
        return

    # Generate summary
    summary_data = aggregator.generate_summary_table()

    if not summary_data:
        print("❌ Failed to generate summary data!")
        return

    # Print summary table
    if not args.quiet:
        aggregator.print_summary_table(summary_data)
        aggregator.generate_model_comparison(summary_data)

    # Save to CSV
    if not args.no_csv:
        csv_file = aggregator.save_to_csv(summary_data, args.output)
        if csv_file and not args.quiet:
            print(f"\n✅ Summary saved to: {csv_file}")

    if not args.quiet:
        print("\n🎉 Results aggregation completed!")

if __name__ == "__main__":
    main()