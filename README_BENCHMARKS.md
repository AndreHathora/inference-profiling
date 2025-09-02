# LLM Inference Benchmarks

### Complete Workflow
```bash
cd /home/ubuntu/andre_projects
source myenv/bin/activate
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-12.8"
export TORCH_CUDA_ARCH_LIST="9.0" # for H100
python benchmark.py # default output dir is data/

# or you can do custom output dir
python benchmark.py --output ./dir_name
```

This runs the complete benchmarking pipeline and saves everything automatically.

### Generated Visualizations
- **Latency Comparison**: TTFT and TPOT across engines
- **Throughput Analysis**: Tokens/second performance
- **Memory Usage**: GPU/CPU resource consumption
- **Concurrent Performance**: Multi-request handling
- **Resource Utilization**: System resource tracking

### Comprehensive Reports
- **Performance Rankings**: Side-by-side engine comparison
- **Detailed Metrics**: Raw data and processed statistics
- **Summary Report**: Complete overview with next steps
