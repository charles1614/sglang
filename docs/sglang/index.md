# SGLang Architecture Documentation

Last indexed: 2 November 2025 (358ae35)

---

## Overview
[Overview](overview.md)

## Installation and Setup
*Installation and Setup documentation coming soon*

## System Architecture
[System Architecture](system-architecture.md)
- Multi-Process Architecture and IPC
- Request Scheduling and Batching
- Memory Management and HiCache
- Component Interaction Patterns

## Request Processing Pipeline
[Request Processing Pipeline](request-processing-pipeline.md)
- Request Lifecycle
- Batching Strategies
- Scheduling Policies
- Response Generation

## Memory Management and Caching
[Memory Management and Caching](memory-management-caching.md)
- RadixAttention System
- HiCache Hierarchical Caching
- KV Cache Management
- Memory Pool Allocation

## Model Execution
[Model Execution](model-execution.md)
- Model Loading and Configuration
- Attention Mechanisms and Backends
- Quantization Support
- Multi-modal Processing

## Programming Interfaces
[Programming Interfaces](programming-interfaces.md)
- Python Engine API
- HTTP Server and OpenAI API
- gRPC Interface
- Language Frontend (SGLang Lang)

## Distributed Execution
[Distributed Execution](distributed-execution.md)
- Parallelism Strategies
- Tensor Parallelism
- Pipeline Parallelism
- Expert Parallelism
- Distributed Coordination

## Hardware and Platform Support
[Hardware and Platform Support](hardware-platform-support.md)
- NVIDIA GPU Support
- AMD GPU Support
- Intel XPU Support
- TPU Support (SGLang-Jax)
- CPU Fallback
- Platform Abstraction Layer

## Performance Optimizations
[Performance Optimizations](performance-optimizations.md)
- Zero-Overhead Scheduling
- Custom CUDA Kernels
- Speculative Decoding
- Kernel Fusion
- Prefill-Decode Disaggregation

## Configuration System
[Configuration System](configuration-system.md)
- Server Arguments
- Model Configuration
- Environment Variables
- Runtime Parameters

## Testing Infrastructure
[Testing Infrastructure](testing-infrastructure.md)
- Test Framework
- Benchmarking
- Performance Validation
- CI/CD Integration

## Deployment and Operations
[Deployment and Operations](deployment-operations.md)
- Production Deployment
- Docker Configuration
- Kubernetes Deployment
- Monitoring and Observability
- Health Checks and Metrics