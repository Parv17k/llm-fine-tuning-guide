# Neural Networks for LLMs: A Developer's Primer

No math, no algorithms - just the concepts you need to understand how LLMs work.

---

## Why This Matters

You don't need to be a machine learning expert to fine-tune LLMs. You do need to understand:

- What a neural network actually does (beyond "magic")
- Key terms like embedding, attention, transformer
- How LLMs process text
- Why data quality matters
- What's happening during training and inference

This chapter gives you that foundation.

---

## Analogy: LLMs as Pattern Completion

Think of an LLM as an extremely sophisticated autocomplete on steroids.

```mermaid
graph LR
    A[You type: The capital of France is] --> B[LLM sees trillions of patterns]
    B --> C[Paris appears 99.9% of time]
    C --> D[LLM predicts: Paris]
    
    style A fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style D fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Key insight**: LLMs don't "know" facts. They predict the most statistically likely continuation of any text pattern they've seen.

---

## How LLMs Process Text

### Step 1: Tokenization

LLMs don't read words - they read **tokens** (pieces of words).

```mermaid
flowchart LR
    Input[Input text: Fine-tuning] --> Tokenizer[Tokenizer]
    Tokenizer --> T1[Fine]
    Tokenizer --> T2[-]
    Tokenizer --> T3[tun]
    Tokenizer --> T4[ing]
    T1 --> Num[1234]
    T2 --> Num2[5]
    T3 --> Num3[67]
    T4 --> Num4[89]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Tokenizer fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style Num,Num2,Num3,Num4 fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Why**: Computers process numbers better than text. This creates a "vocabulary" of common word pieces.

### Step 2: Embedding

Numbers alone don't capture meaning. **Embeddings** convert tokens into vectors (lists of numbers) where similar meanings are close together.

```mermaid
graph LR
    king[Token: king] -.->|Similar| queen[Token: queen]
    king[Token: king] -- Different --> apple[Token: apple]
    queen[Token: queen] -- Different --> apple[Token: apple]
    
    style king fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style queen fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style apple fill:#f44336,stroke:#d32f2f,color:#ffffff,stroke-width:2px
```

**Simple visualization**: In vector space, "king" and "queen" are closer together (similar meaning), while "apple" is farther away (different concept).

### Step 3: Attention - The "Focus" Mechanism

**Attention** lets the model focus on relevant parts of input when generating output.

```mermaid
flowchart LR
    Input[Input: The cat sat on the mat] --> Embeddings[Token Embeddings]
    Embeddings --> Attention[Attention Layer]
    Attention --> Output[Output Embeddings]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Attention fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style Output fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Simple example**: When generating "mat", the model pays attention to what came before ("sat", "cat").

### Step 4: MLP (Multi-Layer Perceptron) - The Core Engine

Before transformers, **MLPs** were the fundamental building block of neural networks.

```mermaid
flowchart LR
    Input[Input: 5 values] --> Hidden1[Hidden Layer 1: 10 neurons]
    Hidden1 --> Hidden2[Hidden Layer 2: 10 neurons]
    Hidden2 --> Output[Output: 3 values]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Hidden1,Hidden2 fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style Output fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Key ideas**:
- **Layers**: Collections of neurons that process data
- **Weights**: Parameters that get adjusted during training
- **Activation functions** (ReLU, etc.): Add non-linearity so the network can learn complex patterns
- **Backpropagation**: How errors flow backward to adjust weights

**Why MLP matters in LLMs**: The "Feed Forward" block in the Transformer architecture is just an MLP - it takes embeddings, processes them through layers, and outputs transformed embeddings. The magic of Transformers isn't the MLP itself, but how attention works alongside it.

### Step 5: Transformers - The Architecture

Modern LLMs use the **Transformer** architecture (introduced in 2017).

```mermaid
flowchart LR
    Input[Input Embeddings] --> AttentionBlock[Attention Block]
    AttentionBlock --> AddNorm[Add & Norm]
    AddNorm --> MLP[MLP / Feed Forward]
    MLP --> AddNorm2[Add & Norm]
    AddNorm2 --> Output[Output Embeddings]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style AttentionBlock,MLP fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style Output fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Stacked transformers**: LLMs stack hundreds of these blocks, each learning different patterns.

---

## Key Concepts

### What is a "Large" Language Model?

| Model Size | Parameters | Rough Analogy |
|------------|------------|---------------|
| Small | 100M | Small vocabulary, simple patterns |
| Medium | 1B | Good grammar, basic facts |
| Large | 7B | Reasoning, multiple topics |
| Very Large | 70B | Close to human-level understanding |

**Parameters**: Think of these as the model's "knobs" it adjusts during training. More parameters = more capacity to learn complex patterns.

### Training vs. Inference

**Training (what happens during fine-tuning)**:
```mermaid
flowchart LR
    Examples[You provide examples] --> Predict[Model makes prediction]
    Predict --> Error[Calculate error]
    Error --> Adjust[Adjust parameters]
    Adjust --> Repeat[Repeat millions of times]
    
    style Examples fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Adjust fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
```

**Inference (what happens when you use the model)**:
```mermaid
flowchart LR
    Input[You type: What is Python?] --> Process[Model processes] --> Output[Python is a programming language...]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Output fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

### Prompt Engineering

A **prompt** is the input text you give to an LLM.

```mermaid
flowchart LR
    Poor[Poor Prompt: Write about AI] --> Vague[Too vague, model guesses]
    Better[Better Prompt: Write about AI for 10-year-old] --> Clear[Clear instructions, better output]
    
    style Poor fill:#f44336,stroke:#d32f2f,color:#ffffff,stroke-width:2px
    style Better fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

---

## How Fine-Tuning Works (High Level)

```mermaid
flowchart LR
    Start[Start with pre-trained model] --> Data[Add specialized data]
    Data --> Train[Train on your data]
    Train --> Result[Model that's both general + specialized]
    
    style Start fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Result fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

### Why Fine-Tune?

| Approach | Pros | Cons |
|----------|------|------|
| **Prompting** | Fast, no compute | Limited to what model already knows |
| **RAG** | Adds new data, no retraining | Can't change model behavior |
| **Fine-Tuning** | Custom behavior, better results | Requires data, compute, time |

---

## Memory Aid: The LLM Mental Model

```mermaid
flowchart LR
    Input[Input: Text] --> Tokenize[Tokenize] --> Embed[Embedding] --> Attn[Attention] --> Transformer[Transformer Blocks] --> Predict[Next Token Prediction] --> Output[Output]
    
    style Input fill:#4a90d9,stroke:#2c5f9d,color:#ffffff,stroke-width:2px
    style Transformer fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
    style Output fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
```

**Remember**: 
- LLMs predict the next token based on patterns
- They don't "know" anything - they predict what comes next
- Training adjusts the model to predict better on your data
- More data + more compute = better predictions (up to a point)

---

## Next Steps

Now you understand:
- How LLMs process text (tokenization → embeddings → attention)
- What an MLP is (the building block of neural networks)
- What training actually does (adjusting parameters to predict better)
- The difference between prompting, RAG, and fine-tuning

This foundation will help you make better decisions when:
- Choosing models
- Preparing data
- Designing prompts
- Evaluating results
