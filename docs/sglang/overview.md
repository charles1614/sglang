# Overview

**Part of**: [Architecture Documentation](index.md)
**Generated**: 2025-11-02
**Source commit**: 358ae35

---

## What is SGLang?

SGLang (Structured Generative Language) is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.

## Key Capabilities

### Core Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Multi-Modal Support** | Native support for text, images, audio, and video inputs | `python/sglang/srt/layers/` |
| **OpenAI Compatibility** | Drop-in replacement for OpenAI APIs | `python/sglang/srt/entrypoints/http_server.py` |
| **RadixAttention** | Advanced prefix caching for reduced computation | `python/sglang/srt/mem_cache/radix_cache.py` |
| **HiCache** | Hierarchical caching across GPU/CPU/Storage tiers | `python/sglang/srt/mem_cache/hicache.py` |
| **Quantization** | FP4/FP8/INT4/AWQ/GPTQ quantization support | `python/sglang/srt/layers/quantization/` |
| **Distributed Inference** | Tensor, pipeline, and expert parallelism | `python/sglang/srt/managers/data_parallel_controller.py` |

### Model Support

| Model Family | Supported Models | File Location |
|--------------|------------------|---------------|
| **Llama** | Llama 2/3, CodeLlama | `python/sglang/srt/models/llama.py` |
| **Qwen** | Qwen1.5/2.5, Qwen-VL | `python/sglang/srt/models/qwen.py` |
| **DeepSeek** | DeepSeek-V2/V3, DeepSeek-Coder | `python/sglang/srt/models/deepseek.py` |
| **GPT** | GPT-2, GPT-NeoX, GPT-OSS | `python/sglang/srt/models/gpt.py` |
| **Mistral** | Mistral 7B, Mixtral | `python/sglang/srt/models/mistral.py` |
| **Gemma** | Gemma 1/2 | `python/sglang/srt/models/gemma.py` |
| **Multimodal** | LLaVA, Qwen-VL, DeepSeek-VL | `python/sglang/srt/models/vlm/` |

### Hardware Platform Support

| Platform | Support Level | Key Components |
|----------|---------------|----------------|
| **NVIDIA GPUs** | Full support | CUDA kernels, TensorRT integration |
| **AMD GPUs** | Full support | ROCm/ HIP kernels |
| **Intel XPU** | Full support | OneAPI runtime, AMX instructions |
| **Google TPUs** | Native support | SGLang-Jax backend |
| **Ascend NPUs** | Experimental support | NPU-specific kernels |
| **CPUs** | Fallback support | PyTorch CPU inference |

## Primary Use Cases

### 1. Production LLM Serving
- Low-latency interactive applications
- High-throughput batch processing
- Real-time conversational AI
- Multi-tenant serving environments

### 2. Research and Development
- Model experimentation and evaluation
- Custom model integration
- Performance benchmarking
- Algorithm development

### 3. Enterprise Deployment
- On-premises deployment
- Multi-cloud infrastructure
- Compliance and data privacy
- Cost optimization

## Target Users

### Developers
- ML engineers building LLM applications
- Researchers experimenting with models
- Software engineers integrating LLMs
- System architects designing AI infrastructure

### Organizations
- Companies deploying AI services at scale
- Research institutions requiring model serving
- Cloud providers offering AI infrastructure
- Edge computing platforms

## Performance Characteristics

### Key Metrics
- **Latency**: Sub-100ms time-to-first-token (TTFT)
- **Throughput**: Up to 10x higher than baseline serving
- **Memory Efficiency**: 3-5x reduction via prefix caching
- **Scalability**: Linear scaling to 100+ GPUs

### Benchmarks
SGLang achieves significant performance improvements over alternative serving frameworks:

| Scenario | Improvement | Source |
|----------|-------------|---------|
| **Prefill Latency** | 3.8x faster | [GB200 DeepSeek Deployment](https://lmsys.org/blog/2025-09-25-gb200-part-2/) |
| **Decode Throughput** | 4.8x higher | [GB200 DeepSeek Deployment](https://lmsys.org/blog/2025-09-25-gb200-part-2/) |
| **Memory Usage** | 70% reduction | [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) |
| **JSON Decoding** | 3x faster | [Compressed FSM](https://lmsys.org/blog/2024-02-05-compressed-fsm/) |

## Architecture Philosophy

### Design Principles
1. **Performance First**: Optimized for production workloads
2. **Modular Design**: Clear separation of concerns
3. **Hardware Agnostic**: Support diverse compute platforms
4. **Developer Friendly**: Easy to use and extend
5. **Production Ready**: Reliability, monitoring, and observability

### Technical Approach
- **Zero-Copy Operations**: Minimize data movement overhead
- **Speculative Execution**: Overlap computation and I/O
- **Hierarchical Caching**: Multi-level memory optimization
- **Adaptive Batching**: Dynamic request grouping
- **Custom Kernels**: Hardware-specific optimizations

## Ecosystem Integration

### Standards and Protocols
- **OpenAI API**: Full compatibility with existing tooling
- **gRPC**: High-performance RPC interface
- **HTTP/REST**: Standard web protocols
- **Python**: Native Python ecosystem integration

### Tooling and Frameworks
- **PyTorch**: Core deep learning framework
- **Transformers**: Hugging Face model integration
- **FastAPI**: High-performance web framework
- **Docker**: Container deployment support
- **Kubernetes**: Orchestration and scaling

This overview provides a foundation for understanding SGLang's capabilities and use cases. The following sections will dive deep into the technical architecture and implementation details.

[← Back to Index](index.md)