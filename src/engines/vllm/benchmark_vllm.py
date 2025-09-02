import asyncio
import time
import os
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from src.core.benchmark_core import BaseBenchmark, InferenceMetrics, PhaseMetrics
import torch
from transformers import AutoConfig


def calculate_optimal_gpu_memory_utilization(model_name: str) -> float:
    """
    Calculate optimal GPU memory utilization based on model size.
    Returns a fraction between 0.05 and 0.9 representing what portion of GPU memory to use.
    """
    try:
        # Get model configuration to determine size
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Estimate model size based on parameters
        if hasattr(config, 'n_parameters'):
            # Some configs have explicit parameter count
            num_params = config.n_parameters
        else:
            # Calculate based on common attributes
            vocab_size = getattr(config, 'vocab_size', 50257)
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 768))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
            
            # Rough estimation of parameters
            # Embedding layer: vocab_size * hidden_size
            # Each transformer layer: ~4 * hidden_size^2 (attention + feedforward)
            # Final layer norm and output head
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * 4 * hidden_size * hidden_size
            output_params = hidden_size * vocab_size
            
            num_params = embedding_params + layer_params + output_params
        
        # Convert parameters to approximate memory requirements
        # Assume fp16 (2 bytes per parameter) + some overhead for activations
        model_memory_gb = (num_params * 2) / (1024**3)  # Convert to GB
        
        # Get available GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = 80  # Default assumption for H100
        
        # Calculate memory utilization based on model size
        if model_memory_gb < 1:  # Very small models (< 1GB)
            memory_util = 0.05  # Use minimal memory for KV cache
        elif model_memory_gb < 3:  # Small models (1-3GB)
            memory_util = 0.1   # Use 10% for KV cache
        elif model_memory_gb < 7:  # Medium models (3-7GB)
            memory_util = 0.2   # Use 20% for KV cache
        elif model_memory_gb < 15:  # Large models (7-15GB)
            memory_util = 0.4   # Use 40% for KV cache
        elif model_memory_gb < 30:  # Very large models (15-30GB)
            memory_util = 0.6   # Use 60% for KV cache
        else:  # Huge models (>30GB)
            memory_util = 0.8   # Use 80% for KV cache
        
        print(f"📊 Model analysis for {model_name}:")
        print(f"   Estimated parameters: {num_params:,}")
        print(f"   Estimated model memory: {model_memory_gb:.2f} GB")
        print(f"   Available GPU memory: {gpu_memory_gb:.2f} GB")
        print(f"   Optimal GPU memory utilization: {memory_util:.1%}")
        
        return memory_util
        
    except Exception as e:
        print(f"⚠️ Could not analyze model {model_name}: {e}")
        print(f"   Using conservative default: 10%")
        return 0.1  # Conservative default


class VLLMBenchmark(BaseBenchmark):
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, max_model_len: int = None,
                 gpu_memory_utilization: float = None, max_tokens: int = 100):
        super().__init__(model_name, max_tokens)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        
        # Use dynamic memory allocation if not specified
        if gpu_memory_utilization is None:
            self.gpu_memory_utilization = calculate_optimal_gpu_memory_utilization(model_name)
        else:
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
                # Ensure SM_90 environment is set for vLLM
                os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
                os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
                os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
                
                llm_kwargs = {
                    'model': self.model_name,
                    'tensor_parallel_size': self.tensor_parallel_size,
                    'gpu_memory_utilization': self.gpu_memory_utilization,
                    'trust_remote_code': True,
                    'enforce_eager': True
                }
                if self.max_model_len is not None:
                    llm_kwargs['max_model_len'] = self.max_model_len

                self.llm = LLM(**llm_kwargs)

        for _ in range(num_warmup):
            self.run_inference(prompt)

    def run_inference(self, prompt: str) -> tuple[str, float]:
        start_time = time.perf_counter()

        # Ensure model is loaded
        if self.llm is None:
            print("Model not loaded, initializing...")
            self.run_warmup(prompt)

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

    def run_concurrent_benchmark(self, prompts: List[str], concurrency: int = 4) -> List[Dict[str, Any]]:
        """Run concurrent benchmark with vLLM."""
        print(f"Running vLLM concurrent benchmark with {len(prompts)} prompts, concurrency: {concurrency}")

        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            output, latency = self.run_inference(prompt)
            result = {
                "output": output,
                "latency": latency,
                "batch_index": i % concurrency,
                "prompt": prompt
            }
            results.append(result)
            print(f"Prompt {i+1} completed with latency {latency:.4f}s")

        print(f"vLLM concurrent benchmark completed with {len(results)} results")
        return results

    def cleanup(self):
        """Clean up vLLM resources."""
        try:
            if self.llm is not None:
                # Try to shutdown gracefully
                if hasattr(self.llm, 'shutdown'):
                    self.llm.shutdown()
                self.llm = None
        except Exception as e:
            print(f"Warning: Error during vLLM cleanup: {e}")

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
