#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.utils import setup_test_environment, create_test_prompts, save_prompts_to_file


def run_command(cmd: list, description: str):
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def run_full_benchmark_suite(model_name: str, output_dir: str = './benchmark_results',
                           prompts_file: str = None, engines: list = None):
    if engines is None:
        engines = ['vllm', 'sglang', 'mlc', 'comparison']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if prompts_file is None:
        prompts_file = str(output_path / 'test_prompts.txt')
        prompts = create_test_prompts(10)
        save_prompts_to_file(prompts, prompts_file)
        print(f"Created test prompts file: {prompts_file}")

    success_count = 0
    total_steps = len(engines)

    for engine in engines:
        if engine == 'comparison':
            cmd = [
                sys.executable, 'benchmark_runner.py',
                '--model', model_name,
                '--prompts', prompts_file,
                '--output', output_dir,
                '--engines', 'comparison'
            ]
        else:
            cmd = [
                sys.executable, 'benchmark_runner.py',
                '--model', model_name,
                '--prompts', prompts_file,
                '--output', output_dir,
                '--engines', engine
            ]

        if run_command(cmd, f"{engine.upper()} Benchmark"):
            success_count += 1

    if success_count == total_steps:
        print(f"\nAll benchmarks completed successfully!")
        print(f"Results saved to: {output_path}")

        next_steps = [
            f"python metrics_collector.py --input {output_dir} --output ./processed_results",
            f"python graph_generator.py --input ./processed_results/processed_metrics.csv --output ./plots"
        ]

        print(f"\nNext steps:")
        for step in next_steps:
            print(f"  {step}")

        return True
    else:
        print(f"\nWarning: {total_steps - success_count} benchmarks failed")
        return False


def run_data_processing(results_dir: str = './benchmark_results',
                       processed_dir: str = './processed_results'):
    # Use scripts from the src directory
    metrics_collector_path = Path(__file__).parent.parent / "src" / "core" / "metrics_collector.py"
    cmd = [
        sys.executable, str(metrics_collector_path),
        '--input', results_dir,
        '--output', processed_dir
    ]

    return run_command(cmd, "Data Processing")


def run_graph_generation(processed_file: str = './processed_results/processed_metrics.csv',
                        plots_dir: str = './plots'):
    if not Path(processed_file).exists():
        print(f"Warning: Processed data file not found: {processed_file}")
        print("Run data processing first")
        return False

    # Use scripts from the src directory
    graph_generator_path = Path(__file__).parent.parent / "src" / "core" / "graph_generator.py"
    cmd = [
        sys.executable, str(graph_generator_path),
        '--input', processed_file,
        '--output', plots_dir
    ]

    return run_command(cmd, "Graph Generation")


def main():
    parser = argparse.ArgumentParser(
        description='Complete LLM Benchmarking Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite (vLLM, SGLang, MLC, Comparison)
  python run_benchmarks.py --model microsoft/DialoGPT-small

  # Run only specific engines
  python run_benchmarks.py --model microsoft/DialoGPT-small --engines vllm mlc

  # Process existing results
  python run_benchmarks.py --process-only --results-dir ./my_results

  # Generate graphs from processed data
  python run_benchmarks.py --graphs-only
        """
    )

    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small',
                       help='Model name to benchmark (default: microsoft/DialoGPT-small)')

    parser.add_argument('--engines', nargs='+',
                       choices=['vllm', 'sglang', 'mlc', 'comparison'],
                       default=['vllm', 'sglang', 'mlc', 'comparison'],
                       help='Engines to benchmark')

    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for benchmark results')

    parser.add_argument('--prompts', type=str,
                       help='File containing prompts (one per line)')

    parser.add_argument('--process-only', action='store_true',
                       help='Only process existing results, do not run benchmarks')

    parser.add_argument('--graphs-only', action='store_true',
                       help='Only generate graphs from processed data')

    parser.add_argument('--results-dir', type=str, default='./benchmark_results',
                       help='Directory containing benchmark results (for --process-only)')

    parser.add_argument('--processed-dir', type=str, default='./processed_results',
                       help='Directory for processed data')

    parser.add_argument('--plots-dir', type=str, default='./plots',
                       help='Directory for generated plots')

    args = parser.parse_args()

    if not setup_test_environment():
        print("Environment setup failed. Please fix issues and try again.")
        sys.exit(1)

    if args.graphs_only:
        processed_file = f"{args.processed_dir}/processed_metrics.csv"
        success = run_graph_generation(processed_file, args.plots_dir)
        sys.exit(0 if success else 1)

    if args.process_only:
        success = run_data_processing(args.results_dir, args.processed_dir)
        if success:
            processed_file = f"{args.processed_dir}/processed_metrics.csv"
            run_graph_generation(processed_file, args.plots_dir)
        sys.exit(0 if success else 1)

    success = run_full_benchmark_suite(
        model_name=args.model,
        output_dir=args.output,
        prompts_file=args.prompts,
        engines=args.engines
    )

    if success:
        processed_file = f"{args.processed_dir}/processed_metrics.csv"
        run_data_processing(args.output, args.processed_dir)
        run_graph_generation(processed_file, args.plots_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
