# Your First LLM Fine-Tune

A Step-by-Step Guide for Technical People

---

## What This Guide Covers

**From zero LLM knowledge to production-ready fine-tuning** - no machine learning background required.

### What You'll Learn

| Stage | Topic | Outcome |
|-------|-------|---------|
| 00 | Neural Network Basics | Understand how LLMs actually work |
| 01 | Foundations | Prompt vs RAG vs Fine-tune decisions |
| 02 | Setup & Environment | Get your tools ready |
| 03 | Hardware Setup | Understand VRAM requirements, choose right GPU |
| 04 | Data Prep | Format data for LLM training (ChatML, JSON) |
| 05 | Training Dynamics | Run SFT on a model |
| 06 | Parameter Efficiency | Master LoRA/QLoRA/Unsloth for cost savings |
| 07 | Alignment | Steer model behavior with DPO/ORPO |
| 08 | Evaluation | Validate your model properly |
| 09 | Deployment | Quantize and serve your custom model |
| 10 | MLOps | Build automated training pipelines |

### Who This Is For

- **Developers** who can write Python but don't know ML
- **DevOps Engineers** who want to deploy custom models
- **Technical Founders** who need custom LLMs for their product
- **Curious Enthusiasts** with basic programming skills

### What You Need

- Python > 3.10
- A Hugging Face account (free)
- Basic Python knowledge (functions, loops, imports)
- Optional: Access to an NVIDIA GPU (can use cloud, or free Colab/Gradient tiers)

---

## Guide Architecture

```mermaid
graph TB
    subgraph "Foundation Layer"
        N[Module 00<br/>Neural Nets]
        F[Module 01<br/>Foundations]
    end

    subgraph "Infrastructure Layer"
        S[Module 02<br/>Setup]
        H[Module 03<br/>Hardware]
        D[Module 04<br/>Data]
    end

    subgraph "Training Layer"
        T[Module 05<br/>Training]
        PE[Module 06<br/>Param. Efficiency]
        A[Module 07<br/>Alignment]
    end

    subgraph "Production Layer"
        E[Module 08<br/>Evaluation]
        DP[Module 09<br/>Deployment]
        M[Module 10<br/>MLOps]
    end

    N --> F
    F --> S
    S --> H
    S --> D
    H --> T
    D --> T
    T --> PE
    T --> A
    PE --> E
    A --> E
    E --> DP
    DP --> M

    style N fill:#2196f3,stroke:#1565c0,color:#ffffff,stroke-width:2px
    style F fill:#2196f3,stroke:#1565c0,color:#ffffff,stroke-width:2px
    style S fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style H fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style D fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style T fill:#9c27b0,stroke:#7b1fa2,color:#ffffff,stroke-width:2px
    style PE fill:#9c27b0,stroke:#7b1fa2,color:#ffffff,stroke-width:2px
    style A fill:#9c27b0,stroke:#7b1fa2,color:#ffffff,stroke-width:2px
    style E fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style DP fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style M fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
```

### Learning Paths Overview

```mermaid
flowchart LR
    A[Start] --> B[Neural Nets + Foundations]
    
    B --> C1[Path 1<br/>Full Learning]
    B --> C2[Path 2<br/>Quick Start]
    B --> C3[Path 3<br/>Skip Ahead]
    
    C1 --> D1[Module 02]
    C1 --> E1[Module 03]
    C1 --> F1[Module 04]
    C1 --> G1[Module 05]
    C1 --> H1[Module 06]
    C1 --> I1[Module 07]
    C1 --> J1[Module 08]
    C1 --> K1[Module 09]
    C1 --> L1[Module 10]
    
    C2 --> D2[Module 02]
    C2 --> E2[Module 03]
    C2 --> F2[Module 04]
    C2 --> G2[Module 05]
    
    C3 --> G3[Your Level]
    
    style A fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style C1 fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style C2 fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style C3 fill:#f44336,stroke:#d32f2f,color:#ffffff,stroke-width:2px
```

---

## How to Use This Guide

### Path 1: Full Learning (Recommended)

Follow modules in order. Each builds on the previous.

```
Neural Nets → Foundations → Setup → Hardware → Data → 
Training → Parameter Efficiency → Alignment → Evaluation → 
Deployment → MLOps
```

### Path 2: Quick Start to Training

Skip theory and dive in quickly:

```
Neural Nets + Foundations (quick read) → Setup → Hardware → Data → Training
```

### Path 3: From Known to Advanced

| Know This? | Start Here |
|------------|------------|
| Basic LLM concepts | Module 02: Setup |
| Environment setup | Module 04: Data Engineering |
| Data engineering | Module 05: Training Dynamics |
| SFT basics | Module 06: Parameter Efficiency |
| LoRA/QLoRA | Module 07: Alignment |
| Fine-tuning | Module 09: Deployment |

---

## What's in Each Module

| Module | Title | Key Takeaway |
|--------|-------|--------------|
| 00 | Neural Networks | Core concepts that won't change |
| 01 | Foundations | Prompt vs RAG vs Fine-tune decisions |
| 02 | Setup & Environment | Tooling, libraries, environment |
| 03 | Hardware Matrix | VRAM math, GPU selection |
| 04 | Data Engineering | Tokenization, ChatML, curation |
| 05 | Training Dynamics | SFT, hyperparameters, multi-GPU |
| 06 | Parameter Efficiency | LoRA, QLoRA, Unsloth, adapters |
| 07 | Alignment | DPO, ORPO without RL |
| 08 | Evaluation | Avoid overfitting, custom evals |
| 09 | Model Deployment | GGUF, AWQ, FP8, vLLM, TGI |
| 10 | MLOps | CI/CD, monitoring, production |

### Module Progression Timeline

```mermaid
timeline
    title Guide Progression
    section Foundations (Mod 00-01)
        Module 00 : Neural Networks
        Module 01 : Foundations
    section Setup (Mod 02-04)
        Module 02 : Setup
        Module 03 : Hardware
        Module 04 : Data Engineering
    section Core Training (Mod 05-07)
        Module 05 : Training Dynamics
        Module 06 : Parameter Efficiency
        Module 07 : Alignment
    section Production (Mod 08-10)
        Module 08 : Evaluation
        Module 09 : Deployment
        Module 10 : MLOps
```

---

## Ready to Begin?

Head to **Module 00: Neural Networks** to understand how LLMs work (covering Llama 4, Qwen 3.6, Gemma 4, and more), or **Module 01: Foundations** for the big picture. If you already have the basics, jump straight to **Module 02: Setup** to get your environment ready.
