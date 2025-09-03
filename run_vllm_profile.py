#!/usr/bin/env python3

import sys
import os

# Fix CUDA library path issues
cuda_paths = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda-12/lib64",
    "/usr/local/cuda-11/lib64"
]

current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
cuda_lib_paths = []
for path in cuda_paths:
    if os.path.exists(path):
        cuda_lib_paths.append(path)

if cuda_lib_paths:
    new_ld_path = ':'.join(cuda_lib_paths)
    if current_ld_path:
        new_ld_path = f"{new_ld_path}:{current_ld_path}"
    os.environ['LD_LIBRARY_PATH'] = new_ld_path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the profiler directly
try:
    # Import without going through the problematic module structure
    import importlib.util
    spec = importlib.util.spec_from_file_location("vllm_latency_profiler", "src/engines/vllm/vllm_latency_profiler.py")
    profiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler_module)
    VLLMLatencyProfiler = profiler_module.VLLMLatencyProfiler
    print("Successfully imported VLLMLatencyProfiler")
except Exception as e:
    print(f"Error importing profiler: {e}")
    sys.exit(1)

def main():
    """Run comprehensive vLLM latency analysis with specified model"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run comprehensive vLLM latency analysis')
    parser.add_argument('--model', '-m', type=str, default='distilgpt2',
                       help='Hugging Face model name (default: distilgpt2)')
    parser.add_argument('--prompt', '-p', type=str, default='Explain machine learning concepts',
                       help='Test prompt for analysis (default: "Explain machine learning concepts")')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all caches before running analysis')
    
    args = parser.parse_args()
    
    # Initialize profiler with specified model
    profiler = VLLMLatencyProfiler(args.model)
    
    # Clear caches if requested
    if args.clear_cache:
        profiler.clear_all_caches()
    
    print(f"=== vLLM Comprehensive Latency Analysis ({args.model}) ===\n")
    print(f"Model: {args.model}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}")
    print("Running comprehensive analysis across multiple prompt sizes...")
    print("This will generate 9 individual plots + CSV data in a timestamped run folder.")
    print()
    
    try:
        # Run the comprehensive analysis that generates the plots
        profiles, df = profiler.run_comprehensive_analysis(args.prompt)
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")
        print("9 plots + CSV data generated:")
        print("  01_component_distribution.png - Runtime component pie chart")
        print("  02_component_scaling_comparison.png - All components scaling together")
        print("  03_memory_requirements.png - Memory vs prompt size")
        print("  04_optimization_opportunities.png - Optimization targets")
        print("  05_attention_kernel_breakdown.png - Deep attention analysis")
        print("  06_decode_kernel_breakdown.png - Deep decode analysis")
        print("  07_batch_latency_throughput.png - Batch latency vs throughput")
        print("  08_batch_memory_scaling.png - Memory scaling with batch size")
        print("  09_batch_efficiency_analysis.png - Batch efficiency analysis")
        print("  batch_analysis_data.csv - Complete batch metrics dataset")
        print("  batch_analysis_summary.csv - Key insights summary")
        print()
        print("Model loading time excluded from plots (too dominant)")
        print("All files saved in timestamped run folder under latency_data/")
        
    except Exception as e:
        print(f"Error running comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
