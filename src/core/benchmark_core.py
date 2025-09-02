import time
import statistics
import torch
import psutil
import GPUtil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class InferenceMetrics:
    ttft: float
    tpot: float
    throughput: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    p50_latency: float
    p95_latency: float
    p99_latency: float
    gpu_memory_used: float
    gpu_utilization: float
    cpu_utilization: float


@dataclass
class PhaseMetrics:
    tokenization_time: float
    model_loading_time: float
    prefill_time: float
    decode_time: float
    output_processing_time: float
    memory_operations_time: float


class BenchmarkTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    def reset(self):
        self.start_time = None
        self.end_time = None


class ResourceMonitor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()

    def get_gpu_memory(self) -> float:
        if not self.gpu_available:
            return 0.0
        try:
            gpus = GPUtil.getGPUs()
            return gpus[0].memoryUsed if gpus else 0.0
        except:
            return 0.0

    def get_gpu_utilization(self) -> float:
        if not self.gpu_available:
            return 0.0
        try:
            gpus = GPUtil.getGPUs()
            return gpus[0].load * 100 if gpus else 0.0
        except:
            return 0.0

    def get_cpu_utilization(self) -> float:
        return psutil.cpu_percent(interval=0.1)

    def get_system_info(self) -> Dict[str, Any]:
        return {
            'gpu_memory_used': self.get_gpu_memory(),
            'gpu_utilization': self.get_gpu_utilization(),
            'cpu_utilization': self.get_cpu_utilization(),
            'total_memory': psutil.virtual_memory().total / (1024**3),
            'available_memory': psutil.virtual_memory().available / (1024**3)
        }


class BaseBenchmark:
    def __init__(self, model_name: str, max_tokens: int = 100):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timer = BenchmarkTimer()
        self.monitor = ResourceMonitor()
        self.phase_timer = BenchmarkTimer()

    @contextmanager
    def measure_phase(self, phase_name: str):
        self.phase_timer.start()
        try:
            yield
        finally:
            duration = self.phase_timer.stop()
            if hasattr(self, 'phase_metrics'):
                setattr(self.phase_metrics, f"{phase_name}_time", duration)

    def calculate_metrics(self, latencies: List[float], total_tokens: int,
                         input_tokens: int, output_tokens: int) -> InferenceMetrics:
        latencies_sorted = sorted(latencies)

        return InferenceMetrics(
            ttft=latencies[0] if latencies else 0,
            tpot=statistics.mean(latencies) if latencies else 0,
            throughput=total_tokens / sum(latencies) if latencies and sum(latencies) > 0 else 0,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            p50_latency=statistics.median(latencies_sorted) if latencies_sorted else 0,
            p95_latency=latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 19 else max(latencies_sorted) if latencies_sorted else 0,
            p99_latency=latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 99 else max(latencies_sorted) if latencies_sorted else 0,
            gpu_memory_used=self.monitor.get_gpu_memory(),
            gpu_utilization=self.monitor.get_gpu_utilization(),
            cpu_utilization=self.monitor.get_cpu_utilization()
        )

    def run_warmup(self, prompt: str, num_warmup: int = 3):
        raise NotImplementedError

    def run_inference(self, prompt: str) -> tuple[str, float]:
        raise NotImplementedError

    def benchmark(self, prompts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        results = {
            'model': self.model_name,
            'phases': {},
            'metrics': [],
            'system_info': self.monitor.get_system_info()
        }

        self.phase_metrics = PhaseMetrics(0, 0, 0, 0, 0, 0)

        if prompts:
            with self.measure_phase('model_loading'):
                self.run_warmup(prompts[0])

        all_latencies = []
        total_input_tokens = 0
        total_output_tokens = 0

        for prompt in prompts:
            prompt_latencies = []

            for _ in range(num_runs):
                with self.measure_phase('tokenization'):
                    pass

                with self.measure_phase('prefill'):
                    pass

                output, latency = self.run_inference(prompt)
                prompt_latencies.append(latency)

                output_tokens = len(output.split())
                total_input_tokens += len(prompt.split())
                total_output_tokens += output_tokens

            all_latencies.extend(prompt_latencies)

            metrics = self.calculate_metrics(
                prompt_latencies,
                len(prompt.split()) + len(output.split()),
                len(prompt.split()),
                len(output.split())
            )
            results['metrics'].append(metrics)

        if all_latencies:
            overall_metrics = self.calculate_metrics(
                all_latencies,
                total_input_tokens + total_output_tokens,
                total_input_tokens,
                total_output_tokens
            )
            results['overall_metrics'] = overall_metrics

        results['phase_breakdown'] = {
            'tokenization': self.phase_metrics.tokenization_time,
            'model_loading': self.phase_metrics.model_loading_time,
            'prefill': self.phase_metrics.prefill_time,
            'decode': self.phase_metrics.decode_time,
            'output_processing': self.phase_metrics.output_processing_time,
            'memory_operations': self.phase_metrics.memory_operations_time
        }

        return results
