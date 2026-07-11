# Setting Up Your Environment

> **Lesson 00** — Getting your development environment ready for LLM fine-tuning.

This guide walks you through setting up a production-ready environment for fine-tuning large language models. We'll cover Python environments, GPU drivers, essential libraries, and verification steps.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Virtual Environment Setup](#python-virtual-environment-setup)
3. [Installing PyTorch with CUDA Support](#installing-pytorch-with-cuda-support)
4. [Hugging Face Hub Authentication](#hugging-face-hub-authentication)
5. [Essential Libraries](#essential-libraries)
6. [Verification and Troubleshooting](#verification-and-troubleshooting)
7. [Docker Alternative](#docker-alternative)

---

## Prerequisites

Before you begin, ensure you have:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10 / macOS 12 / Linux | Linux (Ubuntu 24.04+) |
| **Python** | 3.10 | 3.11-3.12 |
| **GPU** | 8GB VRAM (RTX 4060) | 24GB+ VRAM (RTX 5090, H100) |
| **Disk Space** | 50GB free | 500GB+ NVMe SSD |
| **RAM** | 16GB | 64GB+ |

### macOS Note

Apple Silicon (M1/M2/M3) Macs can fine-tune smaller models (≤7B) using MPS (Metal Performance Shaders). However, training is 2-3x slower than NVIDIA GPUs. For serious work, consider cloud GPUs.

---

## Python Virtual Environment Setup

### Why Virtual Environments Matter

Virtual environments isolate dependencies per project. Without one, installing `torch==2.1.0` for fine-tuning could break `torch==2.0.0` required by another project.

### Step-by-Step Setup

```bash
# Create project directory
mkdir llm-finetune-project && cd llm-finetune-project

# Create virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# Upgrade pip
pip install --upgrade pip
```

### Verification

```bash
# Check Python version (should be 3.10+)
python --version

# Check pip is using virtual environment
which pip  # Should show path inside .venv/
```

---

## Installing PyTorch with CUDA Support

### Understanding CUDA Versions

CUDA is NVIDIA's parallel computing platform. PyTorch must be compiled against a specific CUDA version:

| GPU Architecture | Example Cards | CUDA Support |
|------------------|---------------|---------------|
| Ampere | RTX 3090, A100 | CUDA 11.0+ |
| Ada Lovelace | RTX 4090 | CUDA 12.4+ |
| Hopper | H100 | CUDA 12.0+ |
| Blackwell | B200, B100, RTX 5090 | CUDA 12.4+ |

### Installation Commands

**For CUDA 12.4 (Recommended for RTX 40-series & Blackwell):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 12.8 (Latest builds):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For CUDA 11.8 (Legacy GPUs):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only (No GPU - for testing):**

```bash
pip3 install torch torchvision torchaudio
```

**For macOS with MPS:**

```bash
pip3 install torch torchvision torchaudio
# MPS is enabled by default in PyTorch 2.0+
```

### Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output:
```
PyTorch version: 2.12.1+cu128
CUDA available: True
CUDA version: 12.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 5090
GPU memory: 32.0 GB
```

---

## Hugging Face Hub Authentication

### Why Authenticate?

Authentication is required for:
- Downloading gated models (Llama 3.1/3.2/3.3/4, Mistral, Gemma)
- Pushing your fine-tuned models to the Hub
- Accessing private repositories
- Using the Inference API

### Step 1: Create an Account

1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up" (free account works for most use cases)
3. Verify your email

### Step 2: Generate an Access Token

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "fine-tuning-workstation")
4. Select **Write** permission (needed for uploading models)
5. Copy the token (starts with `hf_`)

### Step 3: Authenticate Locally

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login via CLI
huggingface-cli login

# Paste your token when prompted
```

Alternatively, set an environment variable:

```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

### Verification

```python
from huggingface_hub import login, whoami

# Login programmatically (optional)
login(token="hf_your_token_here")

# Check authentication status
try:
    user = whoami()
    print(f"Logged in as: {user['name']}")
except:
    print("Not logged in")
```

### Accessing Gated Models

Some models require additional approval:

1. **Llama 3.2/3.3/4** (Meta):
   - Go to the model page (e.g., [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct))
   - Click "Agree and access repository"
   - Accept Meta's terms

2. **Gemma 3/4** (Google):
   - Go to [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)
   - Accept Google's Gemma license

3. **Mistral** (Mistral AI):
   - [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

---

## Essential Libraries

Install all core libraries in one command:

```bash
pip install \
    transformers \
    peft \
    trl \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    wandb \
    matplotlib
```

### Optional: Performance & Productivity Tools

```bash
# Liger Kernel — fused attention & RMSNorm kernels that reduce memory 20-30%
pip install liger-kernel

# Unsloth — 2x faster training with memory optimizations
pip install unsloth

# vLLM — high-throughput serving
pip install vllm

# Gradio — build demo UIs for your models
pip install gradio
```

### Library Breakdown

| Library | Version | Purpose | Required For |
|---------|---------|---------|-------------|
| `transformers` | 5.13+ | Model loading, tokenizers, training loops | Everything |
| `peft` | 0.19+ | LoRA, QLoRA, 40+ adapter methods incl. GraLoRA, TinyLoRA | Parameter-efficient fine-tuning |
| `trl` | 1.7+ | DPO, ORPO, GRPO, KTO, RLOO, GMPO alignment algorithms | RLHF and preference optimization |
| `datasets` | 5.x+ | Efficient dataset loading and preprocessing | Data pipelines |
| `accelerate` | 1.1+ | Multi-GPU, mixed precision training | Scaling training |
| `bitsandbytes` | 0.49+ | 4-bit and 8-bit quantization | QLoRA |
| `liger-kernel` | 0.8+ | Fused kernels (attention, RMSNorm) for 20-30% VRAM savings | Performance optimization |
| `unsloth` | 2026.6+ | 2x faster training, 70% less VRAM, RL support | Speed optimization |
| `scipy` | latest | Mathematical utilities | Evaluation metrics |
| `wandb` | latest | Experiment tracking | Monitoring training |
| `gradio` | 6+ | Build demo UIs for your models | Demo/exposure |
| `vllm` | 0.11+ | High-throughput model serving with continuous batching | Deployment |
| `matplotlib` | latest | Visualization | Plotting results |

### Optional: Development Tools

```bash
pip install \
    jupyter \
    ipykernel \
    black \
    ruff \
    pytest
```

---

## Verification and Troubleshooting

### Full Environment Check Script

Save this as `check_env.py`:

```python
#!/usr/bin/env python3
"""Verify your LLM fine-tuning environment is correctly configured."""

import sys
import subprocess

def check_package(name):
    """Check if a package is installed and return its version."""
    try:
        version = __import__(name).__version__
        return f"✓ {name}=={version}"
    except ImportError:
        return f"✗ {name} NOT INSTALLED"
    except AttributeError:
        return f"✓ {name} (installed, no version)"

def main():
    print("=" * 60)
    print("LLM Fine-Tuning Environment Check")
    print("=" * 60)
    
    # Python version
    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    # CUDA check
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slow).")
    except ImportError:
        print("✗ PyTorch NOT INSTALLED")
    
    # Core packages
    print("\n--- Core Packages ---")
    packages = [
        'transformers', 'peft', 'trl', 'datasets', 
        'accelerate', 'bitsandbytes', 'scipy'
    ]
    
    for pkg in packages:
        print(check_package(pkg))
    
    # Optional packages
    print("\n--- Optional Packages ---")
    optional = ['wandb', 'matplotlib', 'jupyter']
    
    for pkg in optional:
        print(check_package(pkg))
    
    # Hugging Face authentication
    print("\n--- Hugging Face Hub ---")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print(f"✗ Not logged in: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python check_env.py
```

### Common Issues and Fixes

| Issue | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large | Reduce `per_device_train_batch_size` |
| `ImportError: bitsandbytes` | CUDA mismatch | Reinstall with `pip install bitsandbytes --force-reinstall` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate firewall | Set `export CURL_CA_BUNDLE=""` |
| `ModuleNotFoundError: peft` | Package not installed | `pip install peft` |
| GPU not detected on macOS | MPS not enabled | Set `export PYTORCH_ENABLE_MPS_FALLBACK=1` |

---

## Docker Alternative

For reproducible environments, use Docker:

### Pre-built Images

**Official PyTorch Image (CUDA 12.8):**
```bash
docker run --gpus all -it \
    --shm-size=8g \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.12.1-cuda12.8-cudnn9-runtime
```

**Hugging Face Transformers Image:**
```bash
docker run --gpus all -it \
    --shm-size=8g \
    -v $(pwd):/workspace \
    huggingface/transformers
```

**vLLM Serving Image:**
```bash
docker run --gpus all -it \
    --shm-size=16g \
    -p 8000:8000 \
    -v $(pwd):/workspace \
    vllm/vllm-openai:latest
```

### Custom Dockerfile

```dockerfile
FROM pytorch/pytorch:2.12.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Install Hugging Face ecosystem + performance tools
RUN pip install --no-cache-dir \
    transformers==5.13.0 \
    peft==0.19.1 \
    trl==1.7.1 \
    datasets \
    accelerate \
    bitsandbytes \
    liger-kernel \
    unsloth \
    scipy \
    wandb

# Create non-root user
RUN useradd -m -u 1000 fine-tuner
USER fine-tuner

CMD ["/bin/bash"]
```

Build and run:
```bash
docker build -t llm-finetune .
docker run --gpus all -it -v $(pwd):/workspace llm-finetune
```

---

## Next Steps

Now that your environment is set up:

1. **Read [Understanding LLM Architecture](./01-llm-architecture.md)** — Transformer basics
2. **Clone the repository examples:**
   ```bash
   git clone https://github.com/Parv17k/llm-fine-tuning-guide.git
   cd llm-fine-tuning-guide/examples
   ```
3. **Try the quickstart notebook** — A 5-minute fine-tuning example

---

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [Unsloth](https://github.com/unslothai/unsloth)
- [vLLM Serving](https://github.com/vllm-project/vllm)
- [Docker for Machine Learning](https://docs.docker.com/guides/ai-and-ml/)
