# Neural Networks for LLMs: A technical's Primer

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
    style Num fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style Num2 fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style Num3 fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style Num4 fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
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

**Embedding matrix**: Think of it as a giant lookup table. Each row is a token from the vocabulary, and each column is a feature dimension. For a model with a 50,000-word vocabulary and 4,092 dimensions, the matrix looks like:

```
          Dim 1   Dim 2   Dim 3   ...   Dim 4096
Token 1   0.23   -0.87    0.45   ...    0.12
Token 2  -0.56    0.34   -0.21   ...   -0.67
Token 3   0.19    0.91    0.78   ...    0.44
   ...
Token 50000
```

When the model sees the word "king" (say, token ID 1234), it looks up row 1234 and gets a 4,096-number vector.

> **Note**: "King" might be row 1234 and "queen" could be row 4892 — completely unrelated row numbers. The similarity isn't in *where* the rows sit, but in *what numbers are inside* those rows.

**Why similar words look alike**: During training, the model reads trillions of sentences. It notices that "king" and "queen" both appear near words like "royal," "crown," "throne," "ruler." Because they share similar surroundings, the model adjusts their rows to have similar numbers. Over time, words that appear in similar contexts naturally end up with similar vectors — like two people who hang out in the same circles probably share similar interests.

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

**What attention actually does**:

Imagine you're reading this sentence: *"The animal didn't cross the street because **it** was too tired."*

When you see "it", your brain instantly asks: *what does "it" refer to?* You look back and figure out — "it" = "the animal", not "the street". That's **attention** — the ability to connect related words across a sentence.

In an LLM, when generating each word, the model looks at *all* previous words and assigns a **weight** (importance score) to each one. For the word "it", the model would give a high weight to "animal" and a low weight to "street". This lets the model understand context no matter how far apart the related words are.

**Without attention**: The model would only see the immediately preceding word, like reading a sentence one word at a time with no memory.

**With attention**: The model can reach back across the entire sentence (or even the whole document) and pick out what matters.

### Step 4: MLP (Multi-Layer Perceptron) - The Core Engine

Before transformers, **MLPs** were the fundamental building block of neural networks.

```mermaid
graph TD
%% Layer Layout Setup
    subgraph Input Layer
        I1(("x1"))
        I2(("x2"))
        I3(("x3"))
    end

    subgraph Hidden Layer 1
        H11(("h11"))
        H12(("h12"))
        H13(("h13"))
    end

    subgraph Hidden Layer 2
        H21(("h21"))
        H22(("h22"))
        H23(("h23"))
    end

    subgraph Output Layer
        O1(("y1"))
        O2(("y2"))
    end

%% Sequential Connection Ordering
    I1 --> H11 & H12 & H13
    I2 --> H11 & H12 & H13
    I3 --> H11 & H12 & H13

    H11 --> H21 & H22 & H23
    H12 --> H21 & H22 & H23
    H13 --> H21 & H22 & H23

    H21 --> O1 & O2
    H22 --> O1 & O2
    H23 --> O1 & O2

%% High-Contrast Styling
    classDef ip fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1,stroke-width:2px,font-weight:bold
    classDef hl fill:#FFF3E0,stroke:#FB8C00,color:#E65100,stroke-width:2px,font-weight:bold
    classDef op fill:#E8F5E9,stroke:#43A047,color:#1B5E20,stroke-width:2px,font-weight:bold

    class I1,I2,I3 ip
    class H11,H12,H13,H21,H22,H23 hl
    class O1,O2 op

%% Box Borders
    style Input Layer fill:none,stroke:#1E88E5,stroke-dasharray: 5 5
    style Hidden Layer 1 fill:none,stroke:#FB8C00,stroke-dasharray: 5 5
    style Hidden Layer 2 fill:none,stroke:#FB8C00,stroke-dasharray: 5 5
    style Output Layer fill:none,stroke:#43A047,stroke-dasharray: 5 5

```

**How data flows through the network**:

1. **Input layer** — Your raw data enters (e.g., 3 numbers representing features)
2. **Hidden layers** — Each neuron multiplies its inputs by **weights** (the numbers on the arrows), adds them up, and passes the result through an **activation function** (like ReLU) to introduce non-linearity
3. **Output layer** — The final processed result

**What are weights?**
- Each connection between neurons has a **weight** — a number that determines how much influence one neuron has on another
- During training, the model adjusts these weights to minimize errors
- Think of weights as "knobs" that control signal strength: a weight of 0.9 means "this connection matters a lot", while 0.1 means "barely matters"

**What are neurons?**
- Each neuron is a tiny computation unit: it takes inputs, multiplies by weights, sums them up, applies an activation function, and passes the result forward
- Hidden layers are where the "thinking" happens — each layer learns progressively more complex patterns from the previous layer's output

### Step 5: Transformers - The Architecture

Modern LLMs use the **Transformer** architecture (introduced in 2017).

```mermaid
flowchart TD
%% Nodes with plain-English explanations
    Input["1. Input Embeddings
    Converts text words into lists of numbers
so the computer can read them."]

Attention["2. Attention Block
Looks at the whole sentence at once.
Figures out how words relate to each other."]

AddNorm1["3. Add & Norm
Combines new insights with original data
and keeps the math stable."]

MLP["4. Feed Forward / MLP
Processes each word individually.
Thinks deeply about what the word means here."]

AddNorm2["5. Add & Norm
Another backup and cleanup step
to finalize the data."]

Output["6. Output Embeddings
The finished, highly-accurate numerical
meaning, ready to predict the next word."]

%% Main Flow Path
Input --> Attention
Attention --> AddNorm1
AddNorm1 --> MLP
MLP --> AddNorm2
AddNorm2 --> Output

%% Visual Skip Connections (The 'Add' part)
Input -.->|"Skip Connection: Don't forget raw input"| AddNorm1
AddNorm1 -.->|"Skip Connection: Don't forget attention output"| AddNorm2

%% Color Coding Styles
classDef blue fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1,stroke-width:2px,font-weight:bold
classDef orange fill:#FFF3E0,stroke:#FB8C00,color:#E65100,stroke-width:2px,font-weight:bold
classDef gray fill:#F5F5F5,stroke:#757575,color:#333333,stroke-width:2px,font-weight:bold
classDef green fill:#E8F5E9,stroke:#43A047,color:#1B5E20,stroke-width:2px,font-weight:bold

class Input blue
class Attention,MLP orange
class AddNorm1,AddNorm2 gray
class Output green

```

**Stacked transformers**: LLMs stack hundreds of these blocks, each learning different patterns.
```mermaid
graph TD
    %% Global Layout Setup
    Start["Raw Text Input
    'The bank of the...'"] --> Embeddings["Input Embeddings + Positional Encoding
    Turns words into numerical vectors with position markers."]

    Embeddings --> Block1

    %% The Stack Structure
    subgraph LLM Brain: Stacked Transformer Blocks
        direction TB
        
        subgraph Block1 [Block 1: Surface Level]
            direction LR
            Att1[Attention] --> MLP1[MLP]
        end
        
        subgraph Block2 [Block 2: Low-Level Context]
            direction LR
            Att2[Attention] --> MLP2[MLP]
        end
        
        Dot[" . . . Hundreds of Repeating Blocks . . . "]
        
        subgraph BlockN [Block N: Deep Abstract Meaning]
            direction LR
            AttN[Attention] --> MLPN[MLP]
        end
    end

    %% Flow through the stack
    Block1 -->|Passes basic grammar & syntax details| Block2
    Block2 -->|Passes contextual relationships| Dot
    Dot -->|Passes highly complex concept understandings| BlockN

    %% Output Generation
    BlockN --> OutputHead["Prediction Head
    Converts numbers back into word probabilities."]
    
    OutputHead --> FinalWord["Final Predicted Word
    '...river'"]

    %% Styling and Aesthetics
    classDef startEnd fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1,stroke-width:2px,font-weight:bold
    classDef blockStyle fill:#FFF3E0,stroke:#FB8C00,color:#E65100,stroke-width:2px,font-weight:bold
    classDef dotStyle fill:#ECEFF1,stroke:#90A4AE,color:#37474F,stroke-width:1px,stroke-dasharray: 5 5,font-style:italic

    class Start,Embeddings,OutputHead,FinalWord startEnd
    class Block1,Block2,BlockN blockStyle
    class Dot dotStyle

    style LLM Brain: Stacked Transformer Blocks fill:#fafafa,stroke:#37474F,stroke-width:2px,stroke-dasharray: 5 5

```


---

## Key Concepts

### What is a "Large" Language Model?

| Model Size | Parameters | Example Models | Rough Analogy |
|------------|------------|----------------|---------------|
| Tiny | 100-500M | SmolLM2-135M | Simple patterns, educational |
| Small | 1B-5B | Gemma-4-E2B, Qwen3.6-4B | Good grammar, fast edge deployment |
| Medium | 7B-30B | Llama-4-Scout-17B, Qwen3.6-27B | Reasoning, multiple topics, sweet spot |
| Large | 30B-100B | Gemma-4-31B, Qwen3.6-35B-A3B | Near human-level understanding |
| Very Large | 100B-500B | DeepSeek-V4-Pro, GLM-5.2 | State-of-the-art reasoning & coding |
| Massive | 500B-1T+ | Kimi-K2.5 (1T) | Frontier capabilities, multi-GPU only |

**Parameters**: Think of these as the model's "knobs" it adjusts during training. More parameters = more capacity to learn complex patterns. Modern models also use **Mixture of Experts (MoE)** architecture where only a subset of parameters are active per token — giving large capacity with efficient inference (e.g., **Qwen3.6-35B-A3B** has 35B total params but only 3B active per token, while **Kimi-K2.5** has 1T total but only 32B active).

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

A **prompt** is the input text you give to an LLM. Modern prompts often include:

```mermaid
flowchart LR
    System[System Prompt: `You are a helpful coding assistant`] --> User[User Prompt: `Fix this Python bug`] --> Context[Optional: Examples / Context]
    
    style System fill:#2196f3,stroke:#1565c0,color:#ffffff,stroke-width:2px
    style User fill:#4caf50,stroke:#388e3c,color:#ffffff,stroke-width:2px
    style Context fill:#ff9800,stroke:#f57c00,color:#ffffff,stroke-width:2px
```

**Key prompt techniques**:
- **System prompts**: Set the model's role and behavior
- **Chain-of-thought**: Ask the model to "think step by step" for reasoning
- **Few-shot examples**: Provide examples in the prompt for better results
- **Role-playing**: Assign a specific persona or expertise area

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
    Input[Input: Text] --> Tokenize[1. Tokenize]
    Tokenize --> Embed[2. Embedding]
    Embed --> Transformer[3. Stacked Transformer Blocks\nAttention + MLP inside]
    Transformer --> Predict[4. Next Token Prediction]
    Predict --> Output[Output: Next Word]

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

---

## Related: LLM Architectures

For a deep dive into specific model architectures (Llama, Qwen, DeepSeek, Gemma, etc.), see:
- [LLM Architectures: From Transformer to 2026](./01-llm-architectures.md)
