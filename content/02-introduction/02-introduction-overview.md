# Introduction

> **Module 02** — Your gateway to LLM fine-tuning. Setup, architecture, and your first fine-tune.

This module takes you from zero to your first fine-tuned model. We cover environment setup, transformer architecture refresher, decision frameworks, and a complete hands-on fine-tuning example.

---

## Table of Contents

### Environment & Foundations

- [**Setting Up Your Environment**](./00-setup.md)
  - Python virtual environment setup
  - Installing PyTorch with CUDA support
  - Hugging Face Hub authentication
  - Essential libraries: `transformers`, `peft`, `trl`, `datasets`
  - Docker alternative for reproducibility
  - Environment verification script

- [**Understanding LLM Architecture**](./01-llm-architecture.md)
  - Transformer architecture refresher
  - Attention mechanisms: Causal, Sliding Window, GQA
  - Tokenization and its impact on training
  - Base models vs. instruction-tuned models
  - Model families: Llama, Mistral, Qwen, Phi-3
  - Memory breakdown by component

### Concepts & Decisions

- [**What is Fine-Tuning?**](./02-what-is-fine-tuning.md)
  - Transfer learning fundamentals
  - The fine-tuning spectrum: Prompting → LoRA → Full FT
  - When to fine-tune vs. when to prompt
  - When to use RAG instead
  - Cost-benefit analysis with ROI calculator
  - Decision framework with case studies

- [**Fine-Tuning Workflows Overview**](./03-workflows.md)
  - Full fine-tuning: When and why
  - Parameter-Efficient Fine-Tuning (PEFT)
  - LoRA: Low-Rank Adaptation math and implementation
  - QLoRA: Quantized LoRA for consumer GPUs
  - Method comparison: Memory, speed, performance
  - Choosing the right workflow for your constraints

### Hands-On

- [**Your First Fine-Tune**](./04-first-fine-tune.md)
  - Complete end-to-end example with TinyLlama-1.1B
  - Casual → Formal text conversion
  - QLoRA configuration for 8GB GPUs
  - Training, evaluation, and inference
  - Save and deploy your adapter

---

## Module Roadmap

```mermaid
flowchart LR
    A[00-setup.md<br/>Environment] --> B[01-llm-architecture.md<br/>Theory]
    B --> C[02-what-is-fine-tuning.md<br/>Decisions]
    C --> D[03-workflows.md<br/>Methods]
    D --> E[04-first-fine-tune.md<br/>Hands-On]
    
    style A fill:#e3f2fd
    style B fill:#bbdefb
    style C fill:#fff9c4
    style D fill:#ffccbc
    style E fill:#c8e6c9
```

---

## Prerequisites

| Requirement | Status |
|-------------|--------|
| Python 3.10+ | Required |
| Basic Python | Required |
| Command line basics | Required |
| ML experience | Not required |
| GPU access | Helpful (cloud options covered) |

---

## What You'll Build

By the end of this module:

1. **A working environment** — All libraries installed and verified
2. **A decision framework** — Know when to fine-tune vs. prompt vs. RAG
3. **Your first fine-tuned model** — TinyLlama adapted for formal writing
4. **Understanding of trade-offs** — Full FT vs. LoRA vs. QLoRA

---

## Estimated Time

| Lesson | Time |
|--------|------|
| 00-setup | 30 min |
| 01-llm-architecture | 45 min |
| 02-what-is-fine-tuning | 30 min |
| 03-workflows | 30 min |
| 04-first-fine-tune | 60 min (includes training) |
| **Total** | **~3-4 hours** |

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Transfer Learning** | Pre-train once, fine-tune for many tasks |
| **LoRA** | Train 1% of parameters, get 95% of full FT performance |
| **QLoRA** | 4-bit quantization enables 7B fine-tuning on 12GB GPU |
| **Prompting vs. FT** | Prompting for knowledge, FT for behavior |
| **RAG** | Retrieval for facts, FT for style |

---

## Common Pitfalls

| Pitfall | How to Avoid |
|---------|--------------|
| CUDA OOM errors | Use QLoRA, reduce batch size, gradient checkpointing |
| Wrong method choice | Use decision framework in 02-what-is-fine-tuning.md |
| Data formatting issues | Follow templates in 04-first-fine-tune.md |
| Learning rate too high | Start with 2e-4 for LoRA, 1e-5 for full FT |
| Not validating | Always test on held-out examples |

---

## Next Module

Continue to [Module 03: Hardware Setup](../03-hardware-setup/) — GPU selection, cloud options, and cost optimization.

---

## Quick Links

- [Environment Setup Script](./00-setup.md#full-environment-check-script)
- [Decision Framework](./02-what-is-fine-tuning.md#the-decision-framework)
- [LoRA Implementation](./03-workflows.md#lora-low-rank-adaptation)
- [Complete Fine-Tuning Script](./04-first-fine-tune.md#complete-script)
