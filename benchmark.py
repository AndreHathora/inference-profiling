#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Suite")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                        help="Model name to benchmark")
    parser.add_argument("--output", type=str, default="./data",
                        help="Output directory for results (default: ./data)")

    args = parser.parse_args()

    # Import and create comprehensive benchmark
    from comprehensive_benchmark import ComprehensiveBenchmark

    print("=" * 80)
    print("LLM Inference Benchmark Suite")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("Running complete benchmarking workflow...")
    print("This will take several minutes and generate comprehensive results.")
    print("=" * 80)

    # Create and run benchmark
    benchmark = ComprehensiveBenchmark(args.model, args.output)
    success = benchmark.run_complete_workflow()

    if success:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"All results saved to: {benchmark.run_dir}")
    else:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETED WITH SOME ISSUES")
        print("=" * 80)
        print(f"Results saved to: {benchmark.run_dir}")

if __name__ == "__main__":
    main()
