import asyncio
import time
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from src.core.benchmark_core import BaseBenchmark, InferenceMetrics, PhaseMetrics


class VLLMBenchmark(BaseBenchmark):
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, max_model_len: int = None,
                 gpu_memory_utilization: float = 0.9, max_tokens: int = 100):
        super().__init__(model_name, max_tokens)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["\n\n", "###"]
        )

    def run_warmup(self, prompt: str, num_warmup: int = 3):
        if self.llm is None:
            with self.measure_phase('model_loading'):
                llm_kwargs = {
                    'model': self.model_name,
                    'tensor_parallel_size': self.tensor_parallel_size,
                    'gpu_memory_utilization': self.gpu_memory_utilization,
                    'trust_remote_code': True
                }
                if self.max_model_len is not None:
                    llm_kwargs['max_model_len'] = self.max_model_len

                self.llm = LLM(**llm_kwargs)

        for _ in range(num_warmup):
            self.run_inference(prompt)

    def run_inference(self, prompt: str) -> tuple[str, float]:
        start_time = time.perf_counter()

        outputs = self.llm.generate([prompt], self.sampling_params)

        end_time = time.perf_counter()
        latency = end_time - start_time

        output_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

        return output_text, latency

    def benchmark_concurrent(self, prompts: List[str], batch_size: int = 4,
                           num_runs: int = 3) -> Dict[str, Any]:
        results = {
            'model': self.model_name,
            'batch_size': batch_size,
            'concurrent_metrics': [],
            'system_info': self.monitor.get_system_info()
        }

        if self.llm is None:
            with self.measure_phase('model_loading'):
                llm_kwargs = {
                    'model': self.model_name,
                    'tensor_parallel_size': self.tensor_parallel_size,
                    'gpu_memory_utilization': self.gpu_memory_utilization,
                    'trust_remote_code': True
                }
                if self.max_model_len is not None:
                    llm_kwargs['max_model_len'] = self.max_model_len

                self.llm = LLM(**llm_kwargs)

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_latencies = []

            for _ in range(num_runs):
                start_time = time.perf_counter()
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
                end_time = time.perf_counter()

                batch_latency = end_time - start_time
                batch_latencies.append(batch_latency)

                total_tokens = sum(
                    len(prompt.split()) + len(output.outputs[0].text.split())
                    for prompt, output in zip(batch_prompts, outputs)
                )

            if batch_latencies:
                metrics = InferenceMetrics(
                    ttft=min(batch_latencies),
                    tpot=sum(batch_latencies) / len(batch_latencies),
                    throughput=total_tokens / sum(batch_latencies),
                    total_tokens=total_tokens,
                    input_tokens=sum(len(p.split()) for p in batch_prompts),
                    output_tokens=total_tokens - sum(len(p.split()) for p in batch_prompts),
                    p50_latency=sorted(batch_latencies)[len(batch_latencies)//2],
                    p95_latency=sorted(batch_latencies)[int(len(batch_latencies)*0.95)] if batch_latencies else max(batch_latencies),
                    p99_latency=sorted(batch_latencies)[int(len(batch_latencies)*0.99)] if batch_latencies else max(batch_latencies),
                    gpu_memory_used=self.monitor.get_gpu_memory(),
                    gpu_utilization=self.monitor.get_gpu_utilization(),
                    cpu_utilization=self.monitor.get_cpu_utilization()
                )
                results['concurrent_metrics'].append(metrics)

        return results

    def benchmark_different_lengths(self, base_prompt: str, lengths: List[int],
                                  num_runs: int = 3) -> Dict[str, Any]:
        results = {
            'model': self.model_name,
            'length_analysis': [],
            'system_info': self.monitor.get_system_info()
        }

        if self.llm is None:
            with self.measure_phase('model_loading'):
                llm_kwargs = {
                    'model': self.model_name,
                    'tensor_parallel_size': self.tensor_parallel_size,
                    'gpu_memory_utilization': self.gpu_memory_utilization,
                    'trust_remote_code': True
                }
                if self.max_model_len is not None:
                    llm_kwargs['max_model_len'] = self.max_model_len

                self.llm = LLM(**llm_kwargs)

        for length in lengths:
            prompt = base_prompt * (length // len(base_prompt) + 1)
            prompt = prompt[:length]

            latencies = []
            total_output_tokens = 0

            for _ in range(num_runs):
                output, latency = self.run_inference(prompt)
                latencies.append(latency)
                total_output_tokens += len(output.split())

            if latencies:
                metrics = InferenceMetrics(
                    ttft=min(latencies),
                    tpot=sum(latencies) / len(latencies),
                    throughput=(len(prompt.split()) + total_output_tokens) / sum(latencies),
                    total_tokens=len(prompt.split()) + total_output_tokens,
                    input_tokens=len(prompt.split()),
                    output_tokens=total_output_tokens,
                    p50_latency=sorted(latencies)[len(latencies)//2],
                    p95_latency=sorted(latencies)[int(len(latencies)*0.95)] if latencies else max(latencies),
                    p99_latency=sorted(latencies)[int(len(latencies)*0.99)] if latencies else max(latencies),
                    gpu_memory_used=self.monitor.get_gpu_memory(),
                    gpu_utilization=self.monitor.get_gpu_utilization(),
                    cpu_utilization=self.monitor.get_cpu_utilization()
                )

                results['length_analysis'].append({
                    'input_length': length,
                    'metrics': metrics
                })

        return results

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the vLLM engine configuration."""
        return {
            "engine": "vLLM",
            "version": "0.10.1.1",
            "device": "cuda",
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "features": [
                "PagedAttention",
                "Continuous Batching",
                "CUDA Graphs",
                "FlashAttention",
                "Quantization Support"
            ]
        }
