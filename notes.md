# Benchmark Implementation Notes

## SGLang Compatibility Issue

SGLang has a fundamental PyTorch version dependency conflict: `sgl-kernel` requires PyTorch 2.8.0+ while vLLM requires PyTorch 2.7.1 exactly. This creates an impossible dependency situation where SGLang's C++ ABI (specifically `_ZN3c104cuda9SetDeviceEab` symbol) is incompatible with the PyTorch version needed for vLLM. The issue is not CUDA version compatibility but PyTorch C++ API version mismatch at the binary level.

## Working Engines

- **Transformers**: Full CUDA 12.8 + SM_90 support
- **vLLM**: Full CUDA 12.8 + SM_90 support with dynamic memory allocation
