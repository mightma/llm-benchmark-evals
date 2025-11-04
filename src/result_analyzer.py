"""
Result analysis and comparison tools for LLM evaluations.
"""

import json
import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict
import numpy as np

from .benchmarks.base import EvaluationResult

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyzes and compares evaluation results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_results(self, results: List[EvaluationResult], run_id: Optional[str] = None) -> str:
        """Save evaluation results to JSON file."""
        if not results:
            raise ValueError("No results provided")

        # Extract model name and benchmark names for filename
        model_name = results[0].model_name
        benchmark_names = [result.benchmark_name for result in results]

        # Clean model name for filename (remove path separators and special chars)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')

        # Create benchmark part of filename
        if len(benchmark_names) == 1:
            benchmark_part = benchmark_names[0]
        else:
            benchmark_part = "multi_benchmark"

        # Generate timestamp part
        if not run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = run_id

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            result_dict["timestamp"] = result.timestamp.isoformat()
            serializable_results.append(result_dict)

        # Save to JSON with descriptive filename
        filename = f"{clean_model_name}_{benchmark_part}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def load_results(self, filepath: str) -> List[EvaluationResult]:
        """Load evaluation results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data:
            # Convert timestamp back to datetime
            item["timestamp"] = datetime.fromisoformat(item["timestamp"])

            # Create EvaluationResult object
            result = EvaluationResult(**item)
            results.append(result)

        return results

    def generate_summary_report(
        self,
        results: List[EvaluationResult],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        if not results:
            return {"error": "No results provided"}

        summary = {
            "evaluation_summary": {
                "total_benchmarks": len(results),
                "model_name": results[0].model_name,
                "evaluation_date": results[0].timestamp.isoformat(),
                "benchmarks": []
            },
            "overall_performance": {},
            "benchmark_details": {},
            "recommendations": []
        }

        # Process each benchmark
        benchmark_scores = {}
        total_samples = 0

        for result in results:
            benchmark_info = {
                "name": result.benchmark_name,
                "score": result.score,
                "num_samples": result.num_samples,
                "config": result.config
            }
            summary["evaluation_summary"]["benchmarks"].append(benchmark_info)

            # Store for overall calculations
            benchmark_scores[result.benchmark_name] = result.score
            total_samples += result.num_samples

            # Detailed results
            summary["benchmark_details"][result.benchmark_name] = {
                "score": result.score,
                "details": result.details,
                "num_samples": result.num_samples,
                "timestamp": result.timestamp.isoformat()
            }

        # Overall performance metrics
        if benchmark_scores:
            summary["overall_performance"] = {
                "average_score": np.mean(list(benchmark_scores.values())),
                "weighted_average_score": self._calculate_weighted_average(results),
                "best_benchmark": max(benchmark_scores, key=benchmark_scores.get),
                "worst_benchmark": min(benchmark_scores, key=benchmark_scores.get),
                "score_variance": np.var(list(benchmark_scores.values())),
                "total_samples_evaluated": total_samples
            }

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(results)

        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary report saved to {output_path}")

        return summary

    def compare_models(
        self,
        model_results: Dict[str, List[EvaluationResult]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare results across multiple models."""
        if len(model_results) < 2:
            return {"error": "Need at least 2 models for comparison"}

        comparison = {
            "models_compared": list(model_results.keys()),
            "comparison_date": datetime.now().isoformat(),
            "benchmark_comparison": {},
            "overall_comparison": {},
            "detailed_analysis": {}
        }

        # Get all benchmarks that appear in results
        all_benchmarks = set()
        for results in model_results.values():
            all_benchmarks.update(result.benchmark_name for result in results)

        # Compare each benchmark
        for benchmark in all_benchmarks:
            benchmark_data = {}

            for model_name, results in model_results.items():
                # Find result for this benchmark
                benchmark_result = next(
                    (r for r in results if r.benchmark_name == benchmark),
                    None
                )

                if benchmark_result:
                    benchmark_data[model_name] = {
                        "score": benchmark_result.score,
                        "num_samples": benchmark_result.num_samples,
                        "details": benchmark_result.details
                    }
                else:
                    benchmark_data[model_name] = None

            comparison["benchmark_comparison"][benchmark] = benchmark_data

            # Find best performing model for this benchmark
            valid_scores = {
                model: data["score"]
                for model, data in benchmark_data.items()
                if data is not None
            }

            if valid_scores:
                best_model = max(valid_scores, key=valid_scores.get)
                worst_model = min(valid_scores, key=valid_scores.get)

                comparison["benchmark_comparison"][benchmark]["best_model"] = best_model
                comparison["benchmark_comparison"][benchmark]["worst_model"] = worst_model
                comparison["benchmark_comparison"][benchmark]["score_gap"] = (
                    valid_scores[best_model] - valid_scores[worst_model]
                )

        # Overall comparison
        model_averages = {}
        for model_name, results in model_results.items():
            if results:
                scores = [r.score for r in results]
                model_averages[model_name] = {
                    "average_score": np.mean(scores),
                    "num_benchmarks": len(scores),
                    "score_std": np.std(scores),
                    "total_samples": sum(r.num_samples for r in results)
                }

        if model_averages:
            overall_best = max(model_averages, key=lambda m: model_averages[m]["average_score"])
            overall_worst = min(model_averages, key=lambda m: model_averages[m]["average_score"])

            comparison["overall_comparison"] = {
                "best_overall_model": overall_best,
                "worst_overall_model": overall_worst,
                "model_averages": model_averages,
                "performance_gap": (
                    model_averages[overall_best]["average_score"] -
                    model_averages[overall_worst]["average_score"]
                )
            }

        # Detailed analysis
        comparison["detailed_analysis"] = self._detailed_model_analysis(model_results)

        # Save comparison if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            logger.info(f"Model comparison saved to {output_path}")

        return comparison

    def create_performance_table(
        self,
        model_results: Dict[str, List[EvaluationResult]],
        output_format: str = "markdown"
    ) -> str:
        """Create a formatted performance table."""
        # Collect data
        all_benchmarks = set()
        for results in model_results.values():
            all_benchmarks.update(result.benchmark_name for result in results)

        all_benchmarks = sorted(all_benchmarks)

        # Build table data
        table_data = []
        for model_name, results in model_results.items():
            row = {"Model": model_name}
            total_score = 0
            valid_benchmarks = 0

            for benchmark in all_benchmarks:
                result = next(
                    (r for r in results if r.benchmark_name == benchmark),
                    None
                )
                if result:
                    row[benchmark] = f"{result.score:.4f}"
                    total_score += result.score
                    valid_benchmarks += 1
                else:
                    row[benchmark] = "N/A"

            # Add average
            if valid_benchmarks > 0:
                row["Average"] = f"{total_score / valid_benchmarks:.4f}"
            else:
                row["Average"] = "N/A"

            table_data.append(row)

        # Format as requested
        if output_format.lower() == "markdown":
            return self._format_markdown_table(table_data, ["Model"] + all_benchmarks + ["Average"])
        elif output_format.lower() == "csv":
            return self._format_csv_table(table_data, ["Model"] + all_benchmarks + ["Average"])
        else:
            return str(table_data)

    def _calculate_weighted_average(self, results: List[EvaluationResult]) -> float:
        """Calculate weighted average score based on number of samples."""
        total_weighted_score = 0
        total_samples = 0

        for result in results:
            total_weighted_score += result.score * result.num_samples
            total_samples += result.num_samples

        return total_weighted_score / total_samples if total_samples > 0 else 0.0

    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Performance-based recommendations
        scores = [r.score for r in results]
        avg_score = np.mean(scores)

        if avg_score < 0.3:
            recommendations.append(
                "Overall performance is low. Consider fine-tuning or using a more capable model."
            )
        elif avg_score < 0.6:
            recommendations.append(
                "Performance is moderate. Consider prompt engineering or parameter tuning."
            )
        else:
            recommendations.append(
                "Good overall performance. Consider optimizing for specific weak benchmarks."
            )

        # Benchmark-specific recommendations
        benchmark_scores = {r.benchmark_name: r.score for r in results}

        if "mmlu_pro" in benchmark_scores and benchmark_scores["mmlu_pro"] < 0.4:
            recommendations.append(
                "MMLU-Pro performance is low. The model may benefit from broader knowledge training."
            )

        if "aime25" in benchmark_scores and benchmark_scores["aime25"] < 0.2:
            recommendations.append(
                "AIME25 performance is low. Consider models with stronger mathematical reasoning capabilities."
            )

        if "ifeval" in benchmark_scores and benchmark_scores["ifeval"] < 0.5:
            recommendations.append(
                "IFEval performance is low. The model may need better instruction-following training."
            )

        # Variance-based recommendations
        if np.var(scores) > 0.1:
            recommendations.append(
                "High variance in benchmark performance. Consider specialized models for different tasks."
            )

        return recommendations

    def _detailed_model_analysis(
        self,
        model_results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Any]:
        """Perform detailed analysis of model performance."""
        analysis = {}

        for model_name, results in model_results.items():
            if not results:
                continue

            scores = [r.score for r in results]

            analysis[model_name] = {
                "strengths": [],
                "weaknesses": [],
                "consistency": {
                    "score_std": np.std(scores),
                    "score_range": max(scores) - min(scores),
                    "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                },
                "benchmark_rankings": {}
            }

            # Identify strengths and weaknesses
            for result in results:
                if result.score > 0.7:
                    analysis[model_name]["strengths"].append(result.benchmark_name)
                elif result.score < 0.4:
                    analysis[model_name]["weaknesses"].append(result.benchmark_name)

        return analysis

    def _format_markdown_table(self, data: List[Dict], columns: List[str]) -> str:
        """Format data as markdown table."""
        if not data:
            return ""

        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        # Rows
        rows = []
        for row in data:
            row_str = "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
            rows.append(row_str)

        return "\n".join([header, separator] + rows)

    def _format_csv_table(self, data: List[Dict], columns: List[str]) -> str:
        """Format data as CSV table."""
        if not data:
            return ""

        lines = []
        # Header
        lines.append(",".join(columns))

        # Rows
        for row in data:
            row_values = [str(row.get(col, "")).replace(",", ";") for col in columns]
            lines.append(",".join(row_values))

        return "\n".join(lines)

    def export_to_excel(
        self,
        model_results: Dict[str, List[EvaluationResult]],
        output_path: str
    ):
        """Export results to Excel file."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for model_name, results in model_results.items():
                    for result in results:
                        summary_data.append({
                            "Model": model_name,
                            "Benchmark": result.benchmark_name,
                            "Score": result.score,
                            "Samples": result.num_samples,
                            "Timestamp": result.timestamp
                        })

                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name="Summary", index=False)

                # Comparison sheet
                comparison_data = []
                all_benchmarks = set()
                for results in model_results.values():
                    all_benchmarks.update(r.benchmark_name for r in results)

                for model_name, results in model_results.items():
                    row = {"Model": model_name}
                    for benchmark in sorted(all_benchmarks):
                        result = next((r for r in results if r.benchmark_name == benchmark), None)
                        row[benchmark] = result.score if result else None
                    comparison_data.append(row)

                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    df_comparison.to_excel(writer, sheet_name="Comparison", index=False)

            logger.info(f"Results exported to Excel: {output_path}")

        except ImportError:
            logger.error("pandas and openpyxl required for Excel export")
            raise