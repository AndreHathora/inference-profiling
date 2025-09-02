import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.vllm.benchmark_vllm import VLLMBenchmark
from engines.sglang.benchmark_sglang import SGLangBenchmark
from engines.mlc.benchmark_mlc import MLCBenchmark
from core.benchmark_core import ResourceMonitor


def load_prompts(file_path: str) -> List[str]:
    if not Path(file_path).exists():
        return [
            "Write a short story about a robot learning to paint.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?"
        ]

    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return []


def save_results(results: Dict[str, Any], output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def run_vllm_benchmarks(model_name: str, prompts: List[str], output_dir: str):
    print(f"Running vLLM benchmarks for {model_name}")

    benchmark = VLLMBenchmark(
        model_name=model_name,
        tensor_parallel_size=1,
        max_tokens=100
    )

    results = {}

    print("Running standard benchmark...")
    results['standard'] = benchmark.benchmark(prompts, num_runs=3)

    print("Running concurrent benchmark...")
    results['concurrent'] = benchmark.benchmark_concurrent(prompts, batch_size=4, num_runs=3)

    print("Running length analysis...")
    base_prompt = "The future of artificial intelligence is "
    lengths = [128, 256, 512, 1024]
    results['length_analysis'] = benchmark.benchmark_different_lengths(
        base_prompt, lengths, num_runs=3
    )

    save_results(results, f"{output_dir}/vllm_{model_name.replace('/', '_')}_results.json")
    print(f"vLLM results saved to {output_dir}/vllm_{model_name.replace('/', '_')}_results.json")


def run_sglang_benchmarks(model_name: str, prompts: List[str], output_dir: str):
    print(f"Running SGLang benchmarks for {model_name}")

    benchmark = SGLangBenchmark(
        model_name=model_name,
        tp_size=1,
        max_total_tokens=4096,
        max_tokens=100
    )

    results = {}

    print("Running standard benchmark...")
    results['standard'] = benchmark.benchmark(prompts, num_runs=3)

    print("Running concurrent benchmark...")
    results['concurrent'] = benchmark.benchmark_concurrent(prompts, batch_size=4, num_runs=3)

    print("Running streaming analysis...")
    if prompts:
        results['streaming'] = benchmark.benchmark_streaming(prompts[0], num_runs=3)

    save_results(results, f"{output_dir}/sglang_{model_name.replace('/', '_')}_results.json")
    print(f"SGLang results saved to {output_dir}/sglang_{model_name.replace('/', '_')}_results.json")


def run_mlc_benchmarks(model_name: str, prompts: List[str], output_dir: str):
    print(f"Running MLC benchmarks for {model_name}")

    benchmark = MLCBenchmark(
        model_name=model_name,
        device="cuda",
        max_tokens=100,
        quantization="q4f16_1"
    )

    results = {}

    print("Running standard benchmark...")
    results['standard'] = benchmark.benchmark(prompts, num_runs=3)

    print("Running concurrent benchmark...")
    results['concurrent'] = benchmark.benchmark_concurrent(prompts, concurrency=4)

    print("Running length analysis...")
    base_prompt = "The future of artificial intelligence is "
    lengths = [128, 256, 512, 1024]
    results['length_analysis'] = benchmark.run_length_analysis(base_prompt, lengths)

    save_results(results, f"{output_dir}/mlc_{model_name.replace('/', '_')}_results.json")
    print(f"MLC results saved to {output_dir}/mlc_{model_name.replace('/', '_')}_results.json")


def run_comparison_benchmarks(model_name: str, prompts: List[str], output_dir: str):
    print(f"Running comparison benchmarks for {model_name}")

    monitor = ResourceMonitor()
    results = {
        'model': model_name,
        'comparison': {},
        'system_info': monitor.get_system_info()
    }

    # Test vLLM
    print("Benchmarking vLLM...")
    try:
        vllm_benchmark = VLLMBenchmark(model_name, max_tokens=50)
        vllm_result = vllm_benchmark.benchmark(prompts[:3], num_runs=2)
        results['comparison']['vllm'] = {
            'ttft': vllm_result.get('overall_metrics', {}).ttft if 'overall_metrics' in vllm_result else 0,
            'throughput': vllm_result.get('overall_metrics', {}).throughput if 'overall_metrics' in vllm_result else 0,
            'gpu_memory': vllm_result.get('overall_metrics', {}).gpu_memory_used if 'overall_metrics' in vllm_result else 0,
            'status': 'success'
        }
    except Exception as e:
        print(f"vLLM benchmark failed: {e}")
        results['comparison']['vllm'] = {'status': 'failed', 'error': str(e)}

    # Test SGLang
    print("Benchmarking SGLang...")
    try:
        sglang_benchmark = SGLangBenchmark(model_name, max_tokens=50)
        sglang_result = sglang_benchmark.benchmark(prompts[:3], num_runs=2)
        results['comparison']['sglang'] = {
            'ttft': sglang_result.get('overall_metrics', {}).ttft if 'overall_metrics' in sglang_result else 0,
            'throughput': sglang_result.get('overall_metrics', {}).throughput if 'overall_metrics' in sglang_result else 0,
            'gpu_memory': sglang_result.get('overall_metrics', {}).gpu_memory_used if 'overall_metrics' in sglang_result else 0,
            'status': 'success'
        }
    except Exception as e:
        print(f"SGLang benchmark failed: {e}")
        results['comparison']['sglang'] = {'status': 'failed', 'error': str(e)}

    # Test MLC
    print("Benchmarking MLC...")
    try:
        mlc_benchmark = MLCBenchmark(model_name, max_tokens=50)
        mlc_result = mlc_benchmark.benchmark(prompts[:3], num_runs=2)
        results['comparison']['mlc'] = {
            'ttft': mlc_result.get('overall_metrics', {}).ttft if 'overall_metrics' in mlc_result else 0,
            'throughput': mlc_result.get('overall_metrics', {}).throughput if 'overall_metrics' in mlc_result else 0,
            'gpu_memory': mlc_result.get('overall_metrics', {}).gpu_memory_used if 'overall_metrics' in mlc_result else 0,
            'status': 'success'
        }
    except Exception as e:
        print(f"MLC benchmark failed: {e}")
        results['comparison']['mlc'] = {'status': 'failed', 'error': str(e)}

    save_results(results, f"{output_dir}/comparison_{model_name.replace('/', '_')}_results.json")
    print(f"Comparison results saved to {output_dir}/comparison_{model_name.replace('/', '_')}_results.json")


def main():
    parser = argparse.ArgumentParser(description='Run LLM inference benchmarks')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small',
                       help='Model name to benchmark')
    parser.add_argument('--prompts', type=str, default='',
                       help='File containing prompts (one per line)')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--engines', nargs='+', default=['vllm', 'sglang', 'mlc'],
                       choices=['vllm', 'sglang', 'mlc', 'comparison'],
                       help='Engines to benchmark')

    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting benchmarks for model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Engines to test: {args.engines}")

    if 'vllm' in args.engines:
        try:
            run_vllm_benchmarks(args.model, prompts, str(output_dir))
        except Exception as e:
            print(f"vLLM benchmark failed: {e}")

    if 'sglang' in args.engines:
        try:
            run_sglang_benchmarks(args.model, prompts, str(output_dir))
        except Exception as e:
            print(f"SGLang benchmark failed: {e}")

    if 'mlc' in args.engines:
        try:
            run_mlc_benchmarks(args.model, prompts, str(output_dir))
        except Exception as e:
            print(f"MLC benchmark failed: {e}")

    if 'comparison' in args.engines:
        try:
            run_comparison_benchmarks(args.model, prompts, str(output_dir))
        except Exception as e:
            print(f"Comparison benchmark failed: {e}")

    print(f"All benchmarks completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
