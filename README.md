# Inference Latency Analysis Notes

## Core Metrics
- **TTFT** (Time to First Token) - prefill + tokenization
- **TPOT** (Time Per Output Token) - decode efficiency
- **Throughput** - tokens/sec under load
- **P50/P95/P99** latency percentiles

## Key Inference Phases to Analyze

### 1. **Input Processing & Tokenization**
   - Text -> token conversion time
   - Input validation overhead
   - Tensor preparation for GPU

### 2. **Model Loading & Initialization**
   - Cold start time (first request)
   - Warm start performance
   - Memory allocation overhead

### 3. **Prefill Phase**
   - Initial attention computation
   - KV cache building time
   - Memory bandwidth utilization

### 4. **Decode Phase**
   - Token-by-token generation latency
   - KV cache reuse efficiency
   - Sampling strategy overhead

### 5. **Output Processing**
   - Token -> text detokenization
   - Stop sequence detection
   - Output formatting/cleanup
   - Streaming buffer management

### 6. **Memory Operations**
   - GPU memory allocation/deallocation
   - PCIe transfer overhead
   - Cache eviction policies
   - Memory defragmentation

## Running the the profiler for vLLM

```bash
python3 run_vllm_profile.py --model "Qwen/Qwen2.5-7B-Instruct"
```