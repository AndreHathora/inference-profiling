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

### 7. **Scheduling & Orchestration**
   - Request queuing time
   - Batch formation overhead
   - GPU kernel launch latency
   - Context switching between requests

## Engine-Specific Focus
- **MLC**: TVM compilation time, memory layout
- **vLLM**: Continuous batching, PagedAttention
- **SGLang**: Radix attention, compression overhead

## Test Setup
- Models: 7B, 13B, 30B+
- Hardware: A100/H100 vs consumer GPUs
- Quantization: FP16, INT8, INT4
- Sequences: 128-4096 tokens
- Batch sizes: 1-16 concurrent requests

## Tools
- LM Evaluation Harness
- PyTorch Profiler
- Custom timing scripts
- Hardware monitoring
