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
    """Run comprehensive vLLM latency analysis with distilgpt2"""
    
    # Initialize profiler with distilgpt2 for fast profiling
    profiler = VLLMLatencyProfiler("distilgpt2")
    
    print("=== vLLM Comprehensive Latency Analysis (DistilGPT2) ===\n")
    print("Running comprehensive analysis across multiple prompt sizes...")
    print("This will generate 4 individual plots in a timestamped run folder.")
    print()
    
    try:
        # Run the comprehensive analysis that generates the 4 plots
        profiles, df = profiler.run_comprehensive_analysis("Explain machine learning concepts")
        
        print(f"\n{'='*60}")
        print("Analysis done.")
        print(f"{'='*60}")
        print("4 individual plots generated:")
        print("  01_component_distribution.png - Runtime component pie chart")
        print("  02_component_scaling_comparison.png - All components scaling together")
        print("  03_memory_requirements.png - Memory vs prompt size")
        print("  04_optimization_opportunities.png - Optimization targets")
        print()
        print("Model loading time excluded from plots (too dominant)")
        print("All plots saved in timestamped run folder under latency_data/")
        
    except Exception as e:
        print(f"Error running comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
