#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time
import logging
import os
from pathlib import Path
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.utils import setup_test_environment

def setup_sm90_environment():
    """Setup environment for SM_90 CUDA backend support."""
    import sys
    
    # Set CUDA paths for SM_90 support
    os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
    
    # Add pip package CUDA libraries to LD_LIBRARY_PATH
    pip_lib_path = os.path.join(sys.prefix, 'lib/python3.12/site-packages/nvidia/cusparselt/lib')
    os.environ['LD_LIBRARY_PATH'] = f'/usr/local/cuda-12.8/lib64:{pip_lib_path}:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Set CUDA architecture for SM_90 (H100)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    os.environ['CUDA_ARCH'] = '90'
    os.environ['CUDAARCHS'] = '90'
    
    print("✓ Environment configured for SM_90 CUDA backend (H100)")


class ComprehensiveBenchmark:
    def __init__(self, model_name: str, base_dir: str = "./results"):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self.results_dir = self.run_dir / "results"
        self.processed_dir = self.run_dir / "processed"
        self.plots_dir = self.run_dir / "plots"
        self.reports_dir = self.run_dir / "reports"

        for dir_path in [self.run_dir, self.results_dir, self.processed_dir,
                        self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.run_dir}")
        print(f"Model: {self.model_name}")

    def run_command(self, cmd: list, description: str, cwd: Path = None) -> bool:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)

        try:
            working_dir = cwd or Path.cwd()
            result = subprocess.run(
                cmd,
                check=True,
                cwd=working_dir,
                capture_output=True,
                text=True
            )
            print(f"{description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{description} failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"{description} failed with error: {e}")
            return False

    def setup_prompts(self) -> str:
        print("Setting up test prompts...")

        prompts_file = self.run_dir / "test_prompts.txt"

        # Use longer, more complex prompts to better utilize GPU
        prompts = [
            "Explain the concept of machine learning in simple terms with examples from everyday life and discuss how it differs from traditional programming approaches.",
            "Write a detailed short story about artificial intelligence becoming self-aware and its implications for human society, including both positive and negative aspects.",
            "What are the benefits and challenges of renewable energy sources like solar, wind, and hydroelectric power? Provide specific examples and discuss current technological limitations and future potential.",
            "Describe how photosynthesis works in plants at a molecular level, including the role of chlorophyll, the light-dependent and light-independent reactions, and how this process contributes to the global carbon cycle.",
            "What is quantum computing and how does it differ from classical computing? Explain quantum bits, superposition, entanglement, and discuss current practical applications and challenges in building quantum computers.",
            "Explain the water cycle and its importance to life on Earth, including evaporation, condensation, precipitation, infiltration, and surface runoff. Discuss how human activities affect this cycle and its role in climate regulation.",
            "How do search engines like Google work? Describe the crawling, indexing, and ranking processes, including the role of algorithms like PageRank and modern machine learning approaches to search relevance.",
            "What are the main differences between Python and JavaScript programming languages? Compare their syntax, use cases, performance characteristics, ecosystem, and discuss when to choose one over the other for different types of projects.",
            "Describe the process of evolution by natural selection in detail, including genetic variation, natural selection, genetic drift, gene flow, and speciation. Provide examples from different species and discuss how this process has shaped life on Earth.",
            "What are the key principles of sustainable development? Discuss the triple bottom line approach, environmental protection, social equity, economic viability, and provide examples of sustainable development initiatives around the world."
        ]

        try:
            with open(prompts_file, 'w') as f:
                for prompt in prompts:
                    f.write(f"{prompt}\n")

            print(f"Created {len(prompts)} test prompts: {prompts_file}")
            return str(prompts_file)
        except Exception as e:
            print(f"Error creating prompts file: {e}")
            return None

    def run_individual_benchmarks(self, prompts_file: str) -> bool:
        print("Running individual engine benchmarks...")

        success_count = 0

        # Load prompts
        try:
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return False

        engines = [
            ('Transformers', 'src.engines.transformers.benchmark_transformers', 'TransformersBenchmark'),
            ('vLLM', 'src.engines.vllm.benchmark_vllm', 'VLLMBenchmark'),
            ('SGLang', 'src.engines.sglang.benchmark_sglang', 'SGLangBenchmark')
            # Note: SGLang currently falls back to mock mode due to CUDA ABI compatibility issues
            # but is included to show the benchmark framework can handle multiple engines
        ]

        for engine_name, module_path, class_name in engines:
            print(f"Benchmarking {engine_name}...")

            try:
                # Import engine class
                module_parts = module_path.split('.')
                module = __import__(module_path, fromlist=[class_name])
                engine_class = getattr(module, class_name)

                # Initialize engine
                if engine_name == 'Transformers':
                    engine = engine_class(
                        model_name=self.model_name,
                        device="auto",
                        max_tokens=100,
                        torch_dtype="auto"
                    )
                elif engine_name == 'vLLM':
                    engine = engine_class(
                        model_name=self.model_name,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=None,  # Use dynamic calculation
                        max_tokens=100
                    )
                elif engine_name == 'SGLang':
                    engine = engine_class(
                        model_name=self.model_name,
                        tp_size=1,
                        max_total_tokens=4096,
                        max_tokens=100
                    )

                # Run benchmark
                results = engine.benchmark(prompts, num_runs=3)

                # Save results
                import json
                result_file = self.results_dir / f"{engine_name.lower()}_{self.model_name.replace('/', '_')}_results.json"
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                print(f"{engine_name} benchmark completed")
                success_count += 1

                # Cleanup
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()

            except Exception as e:
                print(f"Warning: {engine_name} benchmark failed: {e}")

        print(f"\nIndividual benchmarks completed: {success_count}/{len(engines)} successful")
        return success_count > 0

    def run_comparison_benchmark(self, prompts_file: str) -> bool:
        print("Running comparison benchmark...")

        try:
            # Load prompts
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]

            # Run quick comparison test
            engines = [
                ('Transformers', 'src.engines.transformers.benchmark_transformers', 'TransformersBenchmark'),
                ('vLLM', 'src.engines.vllm.benchmark_vllm', 'VLLMBenchmark'),
                ('SGLang', 'src.engines.sglang.benchmark_sglang', 'SGLangBenchmark')
            ]

            comparison_results = {
                'model': self.model_name,
                'comparison': {},
                'system_info': {
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'cuda_available': True
                }
            }

            for engine_name, module_path, class_name in engines:
                try:
                    # Quick test with just first prompt
                    module = __import__(module_path, fromlist=[class_name])
                    engine_class = getattr(module, class_name)

                    if engine_name == 'Transformers':
                        engine = engine_class(model_name=self.model_name, device="auto", max_tokens=50, torch_dtype="auto")
                    elif engine_name == 'vLLM':
                        engine = engine_class(model_name=self.model_name, gpu_memory_utilization=None, max_tokens=50)
                    elif engine_name == 'SGLang':
                        engine = engine_class(model_name=self.model_name, tp_size=1, max_total_tokens=4096, max_tokens=50)

                    # Quick inference test
                    if prompts:
                        # Ensure engine is properly initialized
                        if hasattr(engine, 'run_warmup'):
                            try:
                                engine.run_warmup(prompts[0])  # Pass the first prompt to warmup
                            except Exception as warmup_error:
                                logger.warning(f"Warmup failed for {engine_name}: {warmup_error}")

                        output, latency = engine.run_inference(prompts[0])
                        comparison_results['comparison'][engine_name.lower()] = {
                            'latency': latency,
                            'output_length': len(output) if output else 0,
                            'status': 'success'
                        }
                    else:
                        comparison_results['comparison'][engine_name.lower()] = {'status': 'no_prompts'}

                    if hasattr(engine, 'cleanup'):
                        engine.cleanup()

                except Exception as e:
                    comparison_results['comparison'][engine_name.lower()] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # Save comparison results
            import json
            comparison_file = self.results_dir / f"comparison_{self.model_name.replace('/', '_')}_results.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)

            print("Comparison benchmark completed")
            return True

        except Exception as e:
            print(f"Comparison benchmark failed: {e}")
            return False

    def process_results(self) -> bool:
        print("Processing results...")

        metrics_collector_path = Path(__file__).parent.parent / "src" / "core" / "metrics_collector.py"

        if not metrics_collector_path.exists():
            print(f"Error: Metrics collector not found at {metrics_collector_path}")
            return False

        cmd = [
            sys.executable, str(metrics_collector_path),
            '--input', str(self.results_dir),
            '--output', str(self.processed_dir)
        ]

        success = self.run_command(cmd, "Results Processing")

        if success:
            # Check if processed files were created
            processed_csv = self.processed_dir / "processed_metrics.csv"
            summary_json = self.processed_dir / "summary_stats.json"

            if processed_csv.exists():
                print(f"Processed metrics CSV: {processed_csv}")
            else:
                print("Warning: Processed metrics CSV not found")

            if summary_json.exists():
                print(f"Summary statistics JSON: {summary_json}")
            else:
                print("Warning: Summary statistics JSON not found")

        return success

    def generate_visualizations(self) -> bool:
        print("Generating visualizations...")

        processed_csv = self.processed_dir / "processed_metrics.csv"

        if not processed_csv.exists():
            print(f"Error: Processed data file not found: {processed_csv}")
            return False

        graph_generator_path = Path(__file__).parent.parent / "src" / "core" / "graph_generator.py"

        if not graph_generator_path.exists():
            print(f"Error: Graph generator not found at {graph_generator_path}")
            return False

        cmd = [
            sys.executable, str(graph_generator_path),
            '--input', str(processed_csv),
            '--output', str(self.plots_dir)
        ]

        success = self.run_command(cmd, "Visualization Generation")

        if success:
            plot_files = list(self.plots_dir.glob("*.png"))
            if plot_files:
                print(f"Generated {len(plot_files)} plot files:")
                for plot_file in sorted(plot_files):
                    print(f"  - {plot_file.name}")
            else:
                print("Warning: No plot files found")

        return success

    def generate_final_report(self) -> bool:
        print("Generating final report...")

        try:
            # Create a simple summary report
            report_content = f"""# Benchmark Report

Model: {self.model_name}
Generated: {self.timestamp}

## Results Location
All results saved to: {self.run_dir}

## Generated Files
"""

            # List all generated files
            all_files = []
            for dir_path in [self.results_dir, self.processed_dir, self.plots_dir]:
                if dir_path.exists():
                    files = list(dir_path.glob("*"))
                    for file in files:
                        if file.is_file():
                            rel_path = file.relative_to(self.run_dir)
                            all_files.append(str(rel_path))

            if all_files:
                for file_path in sorted(all_files):
                    report_content += f"- {file_path}\n"
            else:
                report_content += "- No files generated\n"

            # Save report
            report_file = self.reports_dir / f"benchmark_report_{self.timestamp}.md"
            with open(report_file, 'w') as f:
                f.write(report_content)

            print(f"Final report generated: {report_file}")
            return True

        except Exception as e:
            print(f"Error generating final report: {e}")
            return False

    def run_complete_workflow(self) -> bool:
        """Execute the complete benchmarking workflow."""
        print("="*100)
        print("COMPREHENSIVE BENCHMARK WORKFLOW")
        print("="*100)
        print(f"Model: {self.model_name}")
        print(f"Output: {self.run_dir}")
        print("="*100)

        start_time = time.time()

        try:
            # Phase 1: Setup
            prompts_file = self.setup_prompts()
            if not prompts_file:
                return False

            # Phase 2: Individual Benchmarks
            if not self.run_individual_benchmarks(prompts_file):
                print("Warning: Some individual benchmarks failed, continuing...")

            # Phase 3: Comparison Benchmark
            self.run_comparison_benchmark(prompts_file)

            # Phase 4: Process Results
            if not self.process_results():
                print("Error: Results processing failed")
                return False

            # Phase 5: Generate Visualizations
            if not self.generate_visualizations():
                print("Error: Visualization generation failed")
                return False

            # Phase 6: Generate Final Report
            self.generate_final_report()

            end_time = time.time()
            duration = end_time - start_time

            print(f"\n{'='*100}")
            print("COMPREHENSIVE BENCHMARK COMPLETED SUCCESSFULLY!")
            print('='*100)
            print(f"Total duration: {duration:.2f} seconds")
            print(f"All results saved to: {self.run_dir}")
            print("\nDirectory contents:")
            print(f"  Raw Results: {self.results_dir}")
            print(f"  Processed Data: {self.processed_dir}")
            print(f"  Visualizations: {self.plots_dir}")
            print(f"  Reports: {self.reports_dir}")

            return True

        except Exception as e:
            print(f"Error in comprehensive benchmark workflow: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Complete end-to-end benchmarking workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive benchmark with default model
  python benchmark.py

  # Run with specific model
  python benchmark.py --model microsoft/DialoGPT-medium

  # Run with custom output directory
  python benchmark.py --output ./custom_results
        """
    )

    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small',
                       help='Model name to benchmark (default: microsoft/DialoGPT-small)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Base output directory for all results (default: ./results)')

    args = parser.parse_args()

    # Setup environment for SM_90 CUDA support
    setup_sm90_environment()
    
    if not setup_test_environment():
        print("Environment setup failed. Please fix issues and try again.")
        sys.exit(1)

    # Run comprehensive benchmark
    benchmark = ComprehensiveBenchmark(args.model, args.output)
    success = benchmark.run_complete_workflow()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
