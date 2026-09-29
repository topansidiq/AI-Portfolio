# Artificial Intelligence: My Own AI Model Portfolio

An end-to-end repository for fine-tuning open-source Large Language Models (LLMs) and vision-language architectures sourced from Hugging Face. This project covers the full pipeline—from local prototyping on consumer hardware to distributed training on high-performance compute clusters.

---

## Overview

This setup bridges the gap between hardware-constrained experimentations and enterprise-grade distributed fine-tuning. By utilizing **Parameter-Efficient Fine-Tuning (PEFT)** techniques locally and scaling to **Full Fine-Tuning (FFT)** or large-scale **LoRA** across multi-GPU setups, it provides a structured workflow for optimizing open-source models (e.g., Llama, Qwen, Mistral).

### Key Features

- **Hybrid Execution Model**: Local prototyping on constrained GPUs paired with multi-node cluster deployment.
- **Memory Optimization**: Integrated support for QLoRA (4-bit/8-bit quantization), Unsloth, FlashAttention-2, and DeepSpeed.
- **Distributed Training**: Multi-GPU pipeline configurations using Hugging Face `Accelerate`, `TRL`, and DeepSpeed ZeRO-3.
- **Standardized Pipeline**: Unified scripts for dataset preprocessing, training, evaluation, and GGUF/vLLM export.

---

## Hardware Architecture

| Tier                   | Hardware                              | Compute Capability   | Primary Use Case                                                                                                                     |
| :--------------------- | :------------------------------------ | :------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Local / Edge**       | NVIDIA RTX 4050 Laptop GPU (6GB VRAM) | Ada Lovelace (Sm 89) | Quantized fine-tuning (QLoRA 4-bit), prompt engineering validation, data pipeline testing, and lightweight debugging.                |
| **Enterprise Cluster** | NVIDIA DGX H100 (8x H100 80GB SXM5)   | Hopper (Sm 90)       | Full parameter fine-tuning (FFT), large-scale distributed training, high-batch alignment (RLHF/DPO), and FP8 precision optimization. |

---

## Technical Stack

- **Frameworks**: PyTorch, Hugging Face (`transformers`, `datasets`, `peft`, `trl`, `accelerate`)
- **Optimization & Acceleration**: FlashAttention-2, DeepSpeed, BitsAndBytes, Unsloth
- **Precision Modes**: FP16, BF16, FP8 (H100 native), INT4 (QLoRA)
- **Model Formats**: Hugging Face Safetensors, GGUF, AWQ, vLLM-compatible exports

---

## Methodology & Workflow

### 1. Local Development (NVIDIA RTX 4050)

- **Strategy**: PEFT / QLoRA with 4-bit quantization (`bitsandbytes`).
- **Target Models**: Small Language Models (SLMs) ranging from 1B to 7B parameters (e.g., Qwen-2.5-1.5B, Llama-3.2-3B, Phi-3.5-mini).
- **Focus**: Hyperparameter validation, dataset formatting (ChatML/ShareGPT), and memory-leak prevention.

### 2. High-Performance Compute Deployment (NVIDIA DGX H100)

- **Strategy**: Full Parameter Fine-Tuning or Distributed QLoRA via DeepSpeed (ZeRO Stage 2/3) and HF Accelerate.
- **Target Models**: Mid-to-Large scale LLMs (7B to 70B+ parameters).
- **Focus**: Native BF16/FP8 mixed-precision training, massive context window extension, high throughput utilization, and fast convergence.

---
