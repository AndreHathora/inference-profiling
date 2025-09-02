#!/usr/bin/env python3
"""
Comprehensive Engine Test Suite

Tests all benchmarking engines (vLLM, SGLang, MLC) to ensure functionality
and performance validation before integration.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Starting comprehensive engine test suite...")
print(f"Python path: {sys.path[:3]}")
print(f"Current working directory: {os.getcwd()}")

# Import all engine classes
try:
    from src.engines.vllm.benchmark_vllm import VLLMBenchmark
    from src.engines.sglang.benchmark_sglang import SGLangBenchmark
    from src.engines.mlc.benchmark_mlc import MLCBenchmark
    print("Successfully imported all engine classes")
except ImportError as e:
    print(f"Failed to import engine classes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class EngineTester:
    """Test harness for individual engines."""

    def __init__(self, engine_name: str, engine_class, model_name: str = "microsoft/DialoGPT-small"):
        self.engine_name = engine_name
        self.engine_class = engine_class
        self.model_name = model_name
        self.benchmark = None

    def test_initialization(self) -> bool:
        """Test engine initialization."""
        print(f"\n--- Testing {self.engine_name} Initialization ---")
        try:
            if self.engine_name == "vLLM":
                self.benchmark = self.engine_class(
                    model_name=self.model_name,
                    tensor_parallel_size=1,
                    max_tokens=50
                )
            elif self.engine_name == "SGLang":
                self.benchmark = self.engine_class(
                    model_name=self.model_name,
                    tp_size=1,
                    max_tokens=50
                )
            elif self.engine_name == "MLC":
                self.benchmark = self.engine_class(
                    model_name=self.model_name,
                    device="cuda",
                    max_tokens=50,
                    quantization="q4f16_1"
                )

            print(f"{self.engine_name} benchmark created successfully")
            print(f"Model: {self.benchmark.model_name}")

            # Test engine info
            info = self.benchmark.get_engine_info()
            print(f"Engine info retrieved: {info['engine']}")

            return True

        except Exception as e:
            print(f"{self.engine_name} initialization failed: {e}")
            return False

    def test_warmup(self) -> bool:
        """Test engine warmup."""
        print(f"\n--- Testing {self.engine_name} Warmup ---")
        if self.benchmark is None:
            print(f"Skipping {self.engine_name} warmup - no benchmark instance")
            return False

        try:
            test_prompt = "Hello, how are you today?"
            print(f"Testing warmup with prompt: '{test_prompt}'")

            self.benchmark.run_warmup(test_prompt, num_warmup=2)
            print(f"{self.engine_name} warmup completed successfully")
            return True

        except Exception as e:
            print(f"{self.engine_name} warmup failed: {e}")
            return False

    def test_single_inference(self) -> bool:
        """Test single inference."""
        print(f"\n--- Testing {self.engine_name} Single Inference ---")
        if self.benchmark is None:
            print(f"Skipping {self.engine_name} inference - no benchmark instance")
            return False

        try:
            test_prompt = "What is artificial intelligence?"
            print(f"Testing inference with prompt: '{test_prompt}'")

            result = self.benchmark.run_inference(test_prompt)

            if isinstance(result, tuple) and len(result) == 2:
                output, latency = result
                print(f"Inference completed: {len(output)} chars, {latency:.4f}s")
            else:
                print(f"Unexpected result format: {type(result)}")

            return True

        except Exception as e:
            print(f"{self.engine_name} single inference failed: {e}")
            return False

    def test_concurrent_inference(self) -> bool:
        """Test concurrent inference."""
        print(f"\n--- Testing {self.engine_name} Concurrent Inference ---")
        if self.benchmark is None:
            print(f"Skipping {self.engine_name} concurrent - no benchmark instance")
            return False

        try:
            test_prompts = [
                "What is the weather?",
                "Tell me a joke",
                "How does AI work?"
            ]

            results = self.benchmark.run_concurrent_benchmark(test_prompts, concurrency=2)
            print(f"{self.engine_name} concurrent benchmark completed: {len(results)} results")
            return True

        except Exception as e:
            print(f"{self.engine_name} concurrent inference failed: {e}")
            return False


def run_engine_tests():
    """Run comprehensive tests for all engines."""
    engines = [
        ("vLLM", VLLMBenchmark),
        ("SGLang", SGLangBenchmark),
        ("MLC", MLCBenchmark)
    ]

    results = {}

    for engine_name, engine_class in engines:
        print(f"\n{'='*80}")
        print(f"TESTING ENGINE: {engine_name}")
        print('='*80)

        tester = EngineTester(engine_name, engine_class)

        # Run all tests for this engine
        test_results = {
            'initialization': tester.test_initialization(),
            'warmup': tester.test_warmup(),
            'single_inference': tester.test_single_inference(),
            'concurrent_inference': tester.test_concurrent_inference()
        }

        results[engine_name] = test_results

        # Summary for this engine
        passed = sum(test_results.values())
        total = len(test_results)
        print(f"\n{engine_name} Summary: {passed}/{total} tests passed")

    return results


def print_overall_summary(results: Dict[str, Dict[str, bool]]):
    """Print overall test summary."""
    print(f"\n{'='*80}")
    print("OVERALL TEST SUMMARY")
    print('='*80)

    total_passed = 0
    total_tests = 0

    for engine_name, test_results in results.items():
        print(f"\n{engine_name}:")
        for test_name, passed in test_results.items():
            status = "PASS" if passed else "FAIL"
            print("20")
            if passed:
                total_passed += 1
            total_tests += 1

    print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("All tests passed! All engines are ready for benchmarking.")
        return 0
    else:
        print("Some tests failed. Check output above for details.")
        return 1


def main():
    """Main test function."""
    try:
        results = run_engine_tests()
        return print_overall_summary(results)
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
