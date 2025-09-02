#!/usr/bin/env python3

import sys
import subprocess
from utils import setup_test_environment, create_test_prompts, save_prompts_to_file


def test_imports():
    print("Testing imports...")
    tests = [
        ("torch", "import torch; print(f'PyTorch: {torch.__version__}')"),
        ("vllm", "import vllm; print(f'vLLM: {vllm.__version__}')"),
        ("sglang", "import sglang; print('SGLang imported')"),
        ("flashinfer", "import flashinfer; print('FlashInfer imported')"),
        ("matplotlib", "import matplotlib; print('Matplotlib imported')"),
        ("seaborn", "import seaborn; print('Seaborn imported')"),
        ("pandas", "import pandas; print('Pandas imported')")
    ]

    for package, test_code in tests:
        try:
            subprocess.run([sys.executable, '-c', test_code],
                         check=True, capture_output=True, text=True)
            print(f"[OK] {package}")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {package}: {e.stderr.strip()}")
        except Exception as e:
            print(f"[FAIL] {package}: {e}")


def test_basic_functionality():
    print("\nTesting basic functionality...")

    try:
        result = subprocess.run([
            sys.executable, '-c',
            "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
        ], capture_output=True, text=True, check=True)
        print("[OK] CUDA/GPU detection")
        print(f"  {result.stdout.strip()}")
    except Exception as e:
        print(f"[FAIL] CUDA/GPU detection: {e}")

    try:
        prompts = create_test_prompts(3)
        save_prompts_to_file(prompts, './test_prompts.txt')
        print("[OK] Test prompts generation")
        print(f"  Created {len(prompts)} test prompts")
    except Exception as e:
        print(f"[FAIL] Test prompts generation: {e}")


def test_benchmark_scripts():
    print("\nTesting benchmark scripts...")

    scripts = [
        'benchmark_core.py',
        'benchmark_vllm.py',
        'benchmark_sglang.py',
        'benchmark_runner.py',
        'metrics_collector.py',
        'graph_generator.py',
        'utils.py',
        'run_benchmarks.py'
    ]

    for script in scripts:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', script
            ], check=True, capture_output=True, text=True)
            print(f"[OK] {script}")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {script}: {e.stderr.strip()}")
        except Exception as e:
            print(f"[FAIL] {script}: {e}")


def main():
    print("Testing LLM Benchmarking Suite Setup")
    print("=" * 50)

    if not setup_test_environment():
        print("Environment setup issues detected")
        return False

    test_imports()
    test_basic_functionality()
    test_benchmark_scripts()

    print("\n" + "=" * 50)
    print("Setup test completed!")
    print("\nQuick start:")
    print("  python run_benchmarks.py --model microsoft/DialoGPT-small")
    print("\nSee README.md for detailed usage instructions")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
