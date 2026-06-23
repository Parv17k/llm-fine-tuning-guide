# Your First LLM Fine-Tune

A Step-by-Step Guide for Technical People

---

## What This Guide Covers

**From zero LLM knowledge to production-ready fine-tuning** - no machine learning background required.

### What You'll Learn

| Stage | Topic | Outcome |
|-------|-------|---------|
| 1 | Hardware Setup | Understand VRAM requirements, choose right GPU |
| 2 | Data Prep | Format data for LLM training (ChatML, JSON) |
| 3 | First Fine-Tune | Run SFT on a small model |
| 4 | Parameter Efficiency | Master LoRA/QLoRA for cost savings |
| 5 | Alignment | Steer model behavior with DPO/ORPO |
| 6 | Evaluation | Validate your model properly |
| 7 | Deployment | Quantize and serve your custom model |
| 8 | MLOps | Build automated training pipelines |

### Who This Is For

- **Developers** who can write Python but don't know ML
- **DevOps Engineers** who want to deploy custom models
- **Technical Founders** who need custom LLMs for their product
- **Curious Enthusiasts** with basic programming skills

### What You Need

- Python > 3.10
- A Hugging Face account (free)
- Basic Python knowledge (functions, loops, imports)
- Optional: Access to an NVIDIA GPU (can use cloud)

---

## Guide Architecture

```mermaid
graph TB
    subgraph "Foundations Layer"
        F[Module 01<br/>Foundations]
    end

    subgraph "Infrastructure Layer"
        H[Module 02<br/>Hardware]
        D[Module 03<br/>Data]
    end

    subgraph "Training Layer"
        T[Module 04<br/>Training]
        PE[Module 05<br/>Param. Efficiency]
        A[Module 06<br/>Alignment]
    end

    subgraph "Production Layer"
        E[Module 07<br/>Evaluation]
        DP[Module 08<br/>Deployment]
        M[Module 09<br/>MLOps]
    end

    F --> H
    F --> D
    H --> T
    D --> T
    T --> PE
    T --> A
    PE --> E
    A --> E
    E --> DP
    DP --> M

    style F fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
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
    A[Start] --> B[Foundations]
    
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
    
    C2 --> D2[Module 02]
    C2 --> E2[Module 03]
    C2 --> F2[Module 04]
    
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
Foundations → Hardware → Data → Training → 
Parameter Efficiency → Alignment → Evaluation → 
Deployment → MLOps
```

### Path 2: Quick Start to Training

Skip theory and dive in quickly:

```
Hardware (quick read) → Data → Training
```

### Path 3: From Known to Advanced

| Know This? | Start Here |
|------------|------------|
| Hardware/GPU | Module 03: Data |
| SFT basics | Module 05: Parameter Efficiency |
| DPO/ORPO | Module 07: Evaluation |

---

## What's in Each Module

| Module | Title | Key Takeaway |
|--------|-------|--------------|
| 01 | Foundations | Core concepts that won't change |
| 02 | Hardware Matrix | VRAM math, GPU selection |
| 03 | Data Engine | Tokenization, ChatML, curation |
| 04 | Training Dynamics | SFT, hyperparameters, multi-GPU |
| 05 | Parameter Efficiency | LoRA, QLoRA, adapters |
| 06 | Alignment | DPO, ORPO without RL |
| 07 | Evaluation | Avoid overfitting, custom evals |
| 08 | Model Deployment | GGUF, AWQ, vLLM, TGI |
| 09 | MLOps | CI/CD, monitoring, production |

### Module Progression Timeline

```mermaid
timeline
    title Guide Progression
    section Foundational (Mod 01-03)
        Module 01 : Foundations
        Module 02 : Hardware Matrix
        Module 03 : Data Engineering
    section Core Training (Mod 04-06)
        Module 04 : Training Dynamics
        Module 05 : Parameter Efficiency
        Module 06 : Alignment
    section Production (Mod 07-09)
        Module 07 : Evaluation
        Module 08 : Deployment
        Module 09 : MLOps
```

---

## Ready to Begin?

Head to **Module 01: Foundations** to understand the big picture, or jump straight to **Hardware** if you're ready to set up your machine.
