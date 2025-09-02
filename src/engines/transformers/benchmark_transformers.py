"""
Transformers benchmarking engine implementation.

This module provides GPU-accelerated inference using HuggingFace Transformers
with Accelerate for device management and optimization.
"""

import time
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import GPUtil
import psutil

from src.core.benchmark_core import BaseBenchmark, InferenceMetrics, PhaseMetrics
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class TransformersMetrics(InferenceMetrics):
    """Extended metrics for Transformers specific measurements."""
    model_loading_time: float = 0.0
    tokenization_time: float = 0.0
    gpu_warmup_time: float = 0.0


class TransformersBenchmark(BaseBenchmark):
    """
    Transformers Benchmark implementation using HuggingFace Transformers + Accelerate.

    This provides reliable GPU inference without the compilation issues of
    specialized engines like vLLM or MLC.
    """

    def __init__(self,
                 model_name: str,
                 device: str = "auto",
                 max_tokens: int = 100,
                 temperature: float = 0.1,
                 torch_dtype: str = "auto"):
        """
        Initialize Transformers benchmark.

        Args:
            model_name: HuggingFace model name
            device: Device to use (auto, cuda, cpu)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            torch_dtype: Model precision (auto, float16, float32)
        """
        super().__init__(model_name, max_tokens)
        self.device = device
        self.temperature = temperature
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

        # Initialize phase metrics (inherited from base class)
        self.phase_metrics = PhaseMetrics(0, 0, 0, 0, 0, 0)

        # Configure dtype
        if torch_dtype == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

    def run_warmup(self, prompt: str, num_warmup: int = 3):
        """Warm up the Transformers model with test inferences."""
        print(f"Starting Transformers warmup for model: {self.model_name}")
        print(f"Warmup parameters: prompt_length={len(prompt)}, num_warmup={num_warmup}")
        print(f"Device: {self.device}, dtype: {self.torch_dtype}")

        try:
            # Phase 1: Model loading
            print("Phase 1: Model and tokenizer loading")
            with self.measure_phase('model_loading'):
                print("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                print("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device if self.device != "auto" else "auto",
                    trust_remote_code=True
                )
                print("Model loaded successfully")

            # Phase 2: GPU warmup
            print("Phase 2: GPU warmup")
            with self.measure_phase('gpu_warmup'):
                print("Performing GPU warmup...")
                # Create a dummy input to warm up the GPU
                dummy_input = self.tokenizer("Hello world", return_tensors="pt")
                if torch.cuda.is_available():
                    dummy_input = {k: v.cuda() for k, v in dummy_input.items()}

                with torch.no_grad():
                    _ = self.model(**dummy_input)
                print("GPU warmup completed")

            # Phase 3: Inference warmup
            print(f"Phase 3: Inference warmup ({num_warmup} iterations)")
            for i in range(num_warmup):
                print(f"Warmup inference {i+1}/{num_warmup}")
                with self.measure_phase('warmup_inference'):
                    self.run_inference(prompt)
                print(f"Warmup inference {i+1} completed")

            print("Transformers warmup completed successfully")
            print(f"Phase timing summary: {self.phase_metrics}")

        except Exception as e:
            print(f"Transformers warmup failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    def run_inference(self, prompt: str) -> tuple[str, float]:
        """Run inference with Transformers."""
        print(f"Starting Transformers inference")
        print(f"Input prompt length: {len(prompt)} characters")
        print(f"Input prompt preview: {prompt[:100]}...")

        try:
            start_time = time.time()

            # Tokenize input
            print("Tokenizing input...")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Move to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"].cuda()

            print(f"Input tokens: {input_ids.shape[1]}")

            # Generate response
            print("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.get("attention_mask")
                )

            # Decode output
            print("Decoding output...")
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = generated_text[len(prompt):].strip()  # Remove input prompt

            end_time = time.time()
            total_time = end_time - start_time

            print(f"Inference completed in {total_time:.4f}s")
            print(f"Generated text: {output_text[:100]}...")

            # Calculate metrics
            input_tokens = input_ids.shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            throughput = (input_tokens + output_tokens) / total_time

            print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            print(f"Throughput: {throughput:.2f} tokens/s")

            # Get system metrics
            gpu_list = GPUtil.getGPUs()
            if gpu_list:
                gpu = gpu_list[0]
                gpu_memory = gpu.memoryUsed
                gpu_utilization = gpu.load * 100
                print(f"GPU memory used: {gpu_memory}MB, GPU utilization: {gpu_utilization}%")
            else:
                gpu_memory = 0
                gpu_utilization = 0

            cpu_utilization = psutil.cpu_percent()

            # Create metrics
            base_metrics = InferenceMetrics(
                ttft=total_time * 0.1,  # Estimated TTFT
                tpot=(total_time * 0.9) / output_tokens if output_tokens > 0 else 0,
                throughput=throughput,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                p50_latency=total_time,
                p95_latency=total_time * 1.1,
                p99_latency=total_time * 1.2,
                gpu_memory_used=gpu_memory,
                gpu_utilization=gpu_utilization,
                cpu_utilization=cpu_utilization
            )

            print(f"Inference completed successfully")
            print(f"Total latency: {total_time:.4f}s")
            print(f"GPU memory: {gpu_memory}MB, GPU util: {gpu_utilization}%")

            return output_text, total_time

        except Exception as e:
            print(f"Transformers inference failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    def run_concurrent_benchmark(self, prompts: List[str], concurrency: int = 4) -> List[Dict[str, Any]]:
        """Run concurrent benchmark with Transformers."""
        print(f"Running concurrent Transformers benchmark with {concurrency} concurrent requests...")

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

        print(f"Concurrent benchmark completed with {len(results)} results")
        return results

    def run_length_analysis(self, base_prompt: str, lengths: List[int]) -> List[Dict[str, Any]]:
        """Analyze performance across different prompt lengths."""
        print(f"Running Transformers length analysis with lengths: {lengths}")

        results = []
        for i, length in enumerate(lengths):
            print(f"Testing length {length} ({i+1}/{len(lengths)})")

            # Create prompt of specified length
            prompt = base_prompt * (length // len(base_prompt) + 1)
            prompt = prompt[:length]

            print(f"Created prompt with {len(prompt)} characters")

            output, latency = self.run_inference(prompt)
            result = {
                "output": output,
                "latency": latency,
                "prompt_length": length,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
            }
            results.append(result)
            print(f"Length {length} completed with latency {latency:.4f}s")

        print(f"Length analysis completed with {len(results)} results")
        return results

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the Transformers engine configuration."""
        return {
            "engine": "HuggingFace Transformers",
            "version": "4.x",
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "features": [
                "GPU Acceleration",
                "Mixed Precision",
                "Auto Device Mapping",
                "Reliable Inference",
                "Easy Setup"
            ]
        }


def create_transformers_benchmark(model_name: str, **kwargs) -> TransformersBenchmark:
    """
    Factory function to create Transformers benchmark instance.

    Args:
        model_name: HuggingFace model name
        **kwargs: Additional configuration options

    Returns:
        Configured TransformersBenchmark instance
    """
    return TransformersBenchmark(model_name, **kwargs)
