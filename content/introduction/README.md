# Introduction

Getting started with LLM fine-tuning.

## Contents

- [**Setting Up Your Environment**](./00-setup.md)  
  - Python virtual environment setup  
  - Installing PyTorch with CUDA support  
  - Hugging Face Hub authentication  
  - Essential libraries: `transformers`, `peft`, `trl`, `datasets`

- [**Understanding LLM Architecture**](./01-llm-architecture.md)  
  - Transformer architecture refresher  
  - Attention mechanisms and tokenization  
  - Base models vs. instruction-tuned models

- [**What is Fine-Tuning?**](./02-what-is-fine-tuning.md)  
  - Transfer learning concepts  
  - When to fine-tune vs. when to prompt  
  - Cost-benefit analysis

- [**Fine-Tuning Workflows Overview**](./03-workflows.md)  
  - Full fine-tuning  
  - Parameter-Efficient Fine-Tuning (PEFT)  
  - LoRA vs. QLoRA vs. Full FT comparison

## Hands-On

- [**Your First Fine-Tune**](./04-first-fine-tune.md)  
  A minimal end-to-end example using a small model (e.g., TinyLlama)

## Common Pitfalls

- GPU OOM errors
- Dataset formatting issues
- Learning rate selection
