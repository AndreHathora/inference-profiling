import torch
import psutil
import GPUtil
from typing import Dict, Any
import subprocess
import sys


def check_system_requirements() -> Dict[str, Any]:
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'total_memory': psutil.virtual_memory().total / (1024**3),
        'python_version': sys.version.split()[0]
    }

    if info['cuda_available']:
        info['gpu_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['gpu_count'])]

        try:
            gpus = GPUtil.getGPUs()
            info['gpu_memory'] = [gpu.memoryTotal for gpu in gpus]
        except:
            info['gpu_memory'] = ['Unknown'] * info['gpu_count']

    return info


def get_installed_packages():
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[2:]  # Skip header
        packages = {}
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    packages[parts[0]] = parts[1]
        return packages
    except:
        return {}


def validate_environment():
    issues = []

    if not torch.cuda.is_available():
        issues.append("CUDA not available - GPU acceleration will not work")

    try:
        import vllm
    except ImportError:
        issues.append("vLLM not installed")

    try:
        import sglang
    except ImportError:
        issues.append("SGLang not installed")

    try:
        import flashinfer
    except ImportError:
        issues.append("FlashInfer not installed - performance may be degraded")

    if issues:
        print("Environment validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


def create_test_prompts(count: int = 10) -> list:
    base_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about artificial intelligence.",
        "What are the benefits of renewable energy sources?",
        "Describe how photosynthesis works in plants.",
        "What is quantum computing and how does it work?",
        "Explain the water cycle in detail.",
        "What are the main causes of climate change?",
        "Describe the process of evolution by natural selection.",
        "How do vaccines work to protect against diseases?",
        "What is the significance of the theory of relativity?"
    ]

    prompts = []
    for i in range(count):
        prompt = base_prompts[i % len(base_prompts)]
        if i >= len(base_prompts):
            prompt += f" Please provide additional details and examples."

        prompts.append(prompt)

    return prompts


def save_prompts_to_file(prompts: list, filename: str):
    with open(filename, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')


def estimate_memory_requirements(model_name: str, max_tokens: int = 4096) -> Dict[str, float]:
    model_size_estimates = {
        '7b': 14.0,
        '13b': 26.0,
        '30b': 60.0,
        '70b': 140.0
    }

    estimated_size = 14.0  # Default to 7B
    for size, mem in model_size_estimates.items():
        if size in model_name.lower():
            estimated_size = mem
            break

    kv_cache_memory = estimated_size * 0.3 * (max_tokens / 2048)
    total_memory = estimated_size + kv_cache_memory

    return {
        'model_size_gb': estimated_size,
        'kv_cache_gb': kv_cache_memory,
        'total_estimated_gb': total_memory,
        'recommended_gpu_memory_gb': total_memory * 1.2
    }


def print_system_info():
    info = check_system_requirements()

    print("=== System Information ===")
    print(f"Python Version: {info['python_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(info.get('gpu_names', []), info.get('gpu_memory', []))):
            print(f"  GPU {i}: {name} ({memory} MB)")
    print(f"Total System Memory: {info['total_memory']:.1f} GB")
    print()

    packages = get_installed_packages()
    key_packages = ['torch', 'vllm', 'sglang', 'flashinfer-python']

    print("=== Key Package Versions ===")
    for package in key_packages:
        version = packages.get(package, 'Not installed')
        print(f"{package}: {version}")
    print()

    return info


def setup_test_environment():
    print("Setting up test environment...")

    if not validate_environment():
        print("Environment validation failed. Please install missing packages.")
        return False

    print_system_info()
    return True


if __name__ == "__main__":
    setup_test_environment()
