#!/usr/bin/env python3
"""
LLM Evaluation Framework

A comprehensive framework for evaluating LLMs on various benchmarks including
MMLU-Pro, AIME25, and IFEval using VLLM for efficient inference.
"""

import asyncio
import click
import logging
import os
import sys
import json
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation_runner import EvaluationRunner, InferenceParameterManager
from src.result_analyzer import ResultAnalyzer

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """LLM Evaluation Framework - Evaluate language models on various benchmarks."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)



@cli.command()
@click.option('--model', '-m', help='Model path to use (overrides config model_path, e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")')
@click.option('--benchmark', '-b', help='Specific benchmark to run (mmlu_pro, aime25, ifeval)')
@click.option('--num-samples', '-n', type=int, help='Number of samples to evaluate')
@click.option('--num-responses', type=int, default=1, help='Number of responses to generate per question (for self-consistency)')
@click.option('--max-concurrent', type=int, help='Maximum number of concurrent requests (overrides config)')
@click.option('--temperature', '-t', type=float, help='Temperature for generation')
@click.option('--top-p', type=float, help='Top-p for nucleus sampling')
@click.option('--top-k', type=int, help='Top-k for sampling')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--presence-penalty', type=float, help='Presence penalty for generation (affects repetition)')
@click.option('--output-dir', '-o', default='results', help='Output directory for results')
@click.option('--run-id', help='Custom run ID for saving results')
@click.pass_context
def evaluate(ctx, model, benchmark, num_samples, num_responses, max_concurrent, temperature, top_p, top_k, max_tokens, presence_penalty, output_dir, run_id):
    """Run evaluation on specified benchmarks.

    The --model argument will override the model_path in your config file, allowing you to
    evaluate different models without modifying the config. For example:

    python main.py evaluate --model "Qwen/Qwen3-30B-A3B-Instruct-2507"
    python main.py evaluate --model "microsoft/DialoGPT-medium" --benchmark mmlu_pro
    """

    async def run_evaluation():
        config_path = ctx.obj['config']

        if not os.path.exists(config_path):
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            return

        # Initialize runner
        runner = EvaluationRunner(config_path)

        # Override model configuration if --model is provided
        if model:
            runner.override_model(model)
            console.print(f"[green]Model overridden: {model}[/green]")

        # Override max_concurrent if provided
        if max_concurrent is not None:
            if "evaluation" not in runner.config:
                runner.config["evaluation"] = {}
            runner.config["evaluation"]["max_concurrent"] = max_concurrent
            # Reinitialize benchmarks with updated config
            runner._initialize_benchmarks()

        # Get model name for display/results
        if not model:
            models = runner.config.get('models', [])
            if models:
                configured_model = models[0].get('model') or models[0].get('model_path')  # Support legacy config
                if configured_model:
                    model_name = configured_model.split('/')[-1] if '/' in configured_model else configured_model
                else:
                    console.print("[red]No model configured in config file[/red]")
                    return
            else:
                console.print("[red]No model specified and none found in config[/red]")
                return
        else:
            model_name = model.split('/')[-1] if '/' in model else model

        # Validate inference parameters
        inference_params = {}
        if temperature is not None:
            inference_params['temperature'] = temperature
        if top_p is not None:
            inference_params['top_p'] = top_p
        if top_k is not None:
            inference_params['top_k'] = top_k
        if max_tokens is not None:
            inference_params['max_tokens'] = max_tokens
        if presence_penalty is not None:
            inference_params['presence_penalty'] = presence_penalty

        # Validate parameters
        inference_params = InferenceParameterManager.validate_parameters(**inference_params)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # Start server
                start_task = progress.add_task("Starting VLLM server...", total=None)
                server_url = await runner.start_server()
                progress.update(start_task, description=f"Server started at {server_url}")
                progress.remove_task(start_task)

                # Run evaluation
                if benchmark:
                    # Single benchmark
                    eval_task = progress.add_task(f"Running {benchmark} evaluation...", total=None)
                    results = [await runner.run_single_benchmark(
                        benchmark_name=benchmark,
                        model_name=model_name,
                        num_samples=num_samples,
                        num_responses=num_responses,
                        **inference_params
                    )]
                    progress.remove_task(eval_task)
                else:
                    # All benchmarks
                    eval_task = progress.add_task("Running all benchmarks...", total=None)
                    results = await runner.run_all_benchmarks(
                        model_name=model_name,
                        num_samples=num_samples,
                        num_responses=num_responses,
                        **inference_params
                    )
                    progress.remove_task(eval_task)

            # Display results
            display_results(results)

            # Save results
            analyzer = ResultAnalyzer(output_dir)
            results_file = analyzer.save_results(results, run_id)

            # Generate and save summary
            summary = analyzer.generate_summary_report(results)
            summary_file = results_file.replace('.json', '_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            console.print(f"\n[green]Results saved to {results_file}[/green]")
            console.print(f"[green]Summary saved to {summary_file}[/green]")

        except Exception as e:
            console.print(f"[red]Evaluation failed: {e}[/red]")
            logger.exception("Evaluation error")
        finally:
            # Cleanup
            try:
                await runner.stop_server()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")

    # Run the async function
    asyncio.run(run_evaluation())


@cli.command()
@click.argument('result_files', nargs=-1, required=True)
@click.option('--output', '-o', help='Output file for comparison report')
@click.option('--format', 'output_format', default='table',
              type=click.Choice(['table', 'markdown', 'csv', 'excel']),
              help='Output format')
@click.pass_context
def compare(ctx, result_files, output, output_format):
    """Compare results from multiple evaluation runs."""
    analyzer = ResultAnalyzer()

    # Load results from files
    model_results = {}

    for file_path in result_files:
        if not os.path.exists(file_path):
            console.print(f"[red]File not found: {file_path}[/red]")
            continue

        try:
            results = analyzer.load_results(file_path)
            if results:
                model_name = results[0].model_name
                model_results[model_name] = results
                console.print(f"[green]Loaded results for {model_name}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {e}[/red]")

    if len(model_results) < 2:
        console.print("[red]Need at least 2 result files for comparison[/red]")
        return

    # Generate comparison
    comparison = analyzer.compare_models(model_results, output)

    # Display comparison based on format
    if output_format == 'table':
        display_comparison_table(comparison)
    elif output_format in ['markdown', 'csv']:
        table_str = analyzer.create_performance_table(model_results, output_format)
        console.print(table_str)
    elif output_format == 'excel' and output:
        analyzer.export_to_excel(model_results, output)
        console.print(f"[green]Comparison exported to {output}[/green]")

    if output and output_format != 'excel':
        console.print(f"[green]Detailed comparison saved to {output}[/green]")


@cli.command()
@click.pass_context
def server_status(ctx):
    """Check VLLM server status."""

    async def check_status():
        config_path = ctx.obj['config']

        if not os.path.exists(config_path):
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            return

        runner = EvaluationRunner(config_path)

        try:
            status = await runner.get_server_status()

            console.print("[bold]VLLM Server Status:[/bold]")

            # Server configuration
            deploy_locally = status.get('deploy_locally', True)
            console.print(f"Deployment: {'Local' if deploy_locally else 'External'}")
            console.print(f"Expected URL: {status.get('expected_url', 'N/A')}")
            console.print(f"Current URL: {status.get('server_url', 'N/A')}")

            # Status
            server_status = status.get('status', 'Unknown')
            status_color = {
                'running': 'green',
                'not_running': 'yellow',
                'error': 'red'
            }.get(server_status, 'white')

            console.print(f"Status: [{status_color}]{server_status.replace('_', ' ').title()}[/{status_color}]")

            # Process info for local deployments
            if deploy_locally:
                process_running = status.get('process_running', 'Unknown')
                console.print(f"Process Running: {process_running}")

            # Models (if server is running)
            if 'models' in status and status['status'] == 'running':
                console.print("\n[bold]Available Models:[/bold]")
                for model in status['models'].get('data', []):
                    console.print(f"  - {model.get('id', 'Unknown')}")

            # Error details
            if 'error' in status:
                console.print(f"\n[red]Error Details:[/red]")
                console.print(f"  {status['error']}")

                if server_status == 'not_running':
                    console.print(f"\n[yellow]Tip:[/yellow] Start the server with:")
                    console.print(f"  python main.py evaluate")

        except Exception as e:
            console.print(f"[red]Failed to get server status: {e}[/red]")

    # Run the async function
    asyncio.run(check_status())


@cli.command()
@click.pass_context
def list_benchmarks(ctx):
    """List available benchmarks."""
    config_path = ctx.obj['config']

    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        return

    runner = EvaluationRunner(config_path)
    benchmarks = runner.list_available_benchmarks()

    if benchmarks:
        console.print("[bold]Available Benchmarks:[/bold]")
        for benchmark in benchmarks:
            info = runner.get_benchmark_info(benchmark)
            console.print(f"  - {benchmark}: {info.get('class', 'Unknown')}")
    else:
        console.print("[yellow]No benchmarks enabled in configuration[/yellow]")


@cli.command()
def parameters():
    """Show available inference parameters."""
    param_info = InferenceParameterManager.get_parameter_info()

    table = Table(title="Inference Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Range", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Description")

    for param, info in param_info.items():
        range_str = f"[{info['range'][0]}, {info['range'][1]}]"
        table.add_row(
            param,
            info['type'],
            range_str,
            str(info['default']),
            info['description']
        )

    console.print(table)


def display_results(results):
    """Display evaluation results in a formatted table with failure statistics."""
    table = Table(title="Evaluation Results")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Success/Total", style="green")
    table.add_column("Accuracy", style="blue")
    table.add_column("Failures", style="red")
    table.add_column("Details", style="yellow")

    for result in results:
        details = []

        # Get success/failure statistics
        successful = result.details.get('successful_samples', result.details.get('total', 0))
        failed = result.details.get('failed_samples', 0)
        total_samples = result.details.get('total_samples', result.num_samples)
        accuracy = result.details.get('accuracy', 0.0)

        # Handle multiple responses display
        original_samples = result.details.get('original_samples', total_samples)
        responses_per_sample = result.details.get('responses_per_sample', 1)

        # Success rate information
        success_info = f"{successful}/{total_samples}"
        if failed > 0:
            success_rate = result.details.get('success_rate', successful / total_samples if total_samples > 0 else 0.0)
            success_info += f" ({success_rate:.1%})"

        # Failure information
        failure_info = "None"
        if failed > 0:
            failure_rate = result.details.get('failure_rate', failed / total_samples if total_samples > 0 else 0.0)
            failure_info = f"{failed} ({failure_rate:.1%})"

            # Add failure types if available
            failure_types = result.details.get('failure_types', {})
            if failure_types:
                failure_breakdown = []
                for error_type, count in failure_types.items():
                    failure_breakdown.append(f"{error_type}: {count}")
                failure_info += f" [{', '.join(failure_breakdown)}]"

        # Additional details
        if 'correct' in result.details:
            details.append(f"Correct: {result.details['correct']}")

        # Multiple responses information
        if responses_per_sample > 1:
            details.append(f"Responses/sample: {responses_per_sample}")
            details.append(f"Original samples: {original_samples}")

        # Benchmark-specific details
        if result.benchmark_name == "mmlu_pro" and 'num_subjects' in result.details:
            details.append(f"Subjects: {result.details['num_subjects']}")
        elif result.benchmark_name == "aime25":
            details.append(f"Problems solved: {result.details.get('problems_solved', 0)}")
        elif result.benchmark_name == "ifeval":
            if 'instruction_accuracy' in result.details:
                details.append(f"Inst Acc: {result.details['instruction_accuracy']:.4f}")
            if 'total_instructions' in result.details:
                details.append(f"Instructions: {result.details.get('instructions_passed', 0)}/{result.details['total_instructions']}")

        table.add_row(
            result.benchmark_name,
            f"{result.score:.4f}",
            success_info,
            f"{accuracy:.4f}",
            failure_info,
            ", ".join(details) if details else "N/A"
        )

    console.print(table)

    # Print summary of failures if any
    total_failures = sum(result.details.get('failed_samples', 0) for result in results)
    if total_failures > 0:
        console.print(f"\n[red]⚠️  Total inference failures: {total_failures}[/red]")
        console.print("[red]Note: Accuracy calculations exclude failed samples[/red]")

        # Aggregate failure types across all benchmarks
        all_failure_types = {}
        for result in results:
            failure_types = result.details.get('failure_types', {})
            for error_type, count in failure_types.items():
                all_failure_types[error_type] = all_failure_types.get(error_type, 0) + count

        if all_failure_types:
            console.print(f"[red]Failure breakdown: {', '.join(f'{error_type}: {count}' for error_type, count in all_failure_types.items())}[/red]")
    else:
        console.print(f"\n[green]✅ All samples processed successfully[/green]")


def display_comparison_table(comparison):
    """Display model comparison table."""
    console.print("[bold]Model Comparison[/bold]\n")

    # Overall comparison
    if 'overall_comparison' in comparison:
        overall = comparison['overall_comparison']
        console.print(f"Best Overall: {overall.get('best_overall_model', 'N/A')}")
        console.print(f"Performance Gap: {overall.get('performance_gap', 0):.4f}\n")

    # Benchmark comparison table
    if 'benchmark_comparison' in comparison:
        table = Table(title="Benchmark Comparison")
        table.add_column("Benchmark", style="cyan")

        # Add model columns
        models = comparison.get('models_compared', [])
        for model in models:
            table.add_column(model, style="magenta")

        table.add_column("Best Model", style="green")

        for benchmark, data in comparison['benchmark_comparison'].items():
            row = [benchmark]

            for model in models:
                if model in data and data[model] is not None:
                    score = data[model]['score']
                    row.append(f"{score:.4f}")
                else:
                    row.append("N/A")

            row.append(data.get('best_model', 'N/A'))
            table.add_row(*row)

        console.print(table)


if __name__ == '__main__':
    cli()