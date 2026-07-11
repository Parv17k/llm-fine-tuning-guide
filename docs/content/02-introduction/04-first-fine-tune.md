# Your First Fine-Tune

> **Lesson 04** — A minimal end-to-end example using Qwen3-8B.

This guide walks you through fine-tuning a small model in under 30 minutes. We'll use Qwen3-8B because it fits on any GPU, trains quickly, and delivers strong performance for its size.

---

## Table of Contents

1. [What We'll Build](#what-well-build)
2. [Prerequisites](#prerequisites)
3. [Step 1: Setup](#step-1-setup)
4. [Step 2: Prepare the Dataset](#step-2-prepare-the-dataset)
5. [Step 3: Configure PEFT](#step-3-configure-peft)
6. [Step 4: Training](#step-4-training)
7. [Step 5: Evaluation](#step-5-evaluation)
8. [Step 6: Inference](#step-6-inference)
9. [Complete Script](#complete-script)
10. [Next Steps](#next-steps)

---

## What We'll Build

A model that converts casual text to formal business writing.

**Example:**
- Input: `"hey can u send me the report asap"`
- Output: `"Hello, could you please send me the report at your earliest convenience?"`

**Training data:** 500 examples of casual → formal conversions.

**Time:** 20-30 minutes on a consumer GPU.

**Hardware:** Any GPU with 8GB+ VRAM.

---

## Prerequisites

```bash
# Install required packages
pip install \
    transformers \
    peft \
    trl \
    datasets \
    accelerate \
    bitsandbytes

# Verify installation
python -c "import transformers, peft, trl; print('✓ All packages installed')"
```

---

## Step 1: Setup

### Import Libraries

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Model Selection

We'll use Qwen3-8B because:

| Model | Parameters | VRAM Required | Training Time |
|-------|------------|---------------|---------------|
| **Qwen3-8B** | 8B | 12 GB | 25-40 min |
| SmolLM2-1.7B-Instruct | 1.7B | 4 GB | 15 min |
| Gemma-4-12B-it | 12B | 16 GB | 30-45 min |
| Llama-4-Scout-17B-16E | 17B (MoE) | 24 GB | 60-90 min |
| Qwen3.5-35B-A3B | 35B (3B active) | 16 GB | 45-60 min |

```python
model_name = "Qwen/Qwen3-8B"  # or "google/gemma-4-12B-it", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

---

## Step 2: Prepare the Dataset

### Create Training Data

For this tutorial, we'll create a synthetic dataset:

```python
from datasets import Dataset

# Sample training data (casual → formal pairs)
training_data = [
    {"input": "hey can u send me the report asap", 
     "output": "Hello, could you please send me the report at your earliest convenience?"},
    {"input": "thx for the info", 
     "output": "Thank you for the information."},
    {"input": "im gonna be late to the meeting", 
     "output": "I apologize, but I will be arriving late to the meeting."},
    {"input": "whats the status on this", 
     "output": "Could you please provide an update on the current status?"},
    {"input": "need this done by eod", 
     "output": "Please complete this task by the end of the day."},
    # Add 495 more examples...
]

# For a real project, load from file or Hugging Face
# dataset = load_dataset("json", data_files="training_data.jsonl")

# Convert to Dataset
dataset = Dataset.from_list(training_data)
```

### Format for Instruction Tuning

```python
def format_instruction(example):
    """Format data as instruction-following examples."""
    prompt = f"""### Instruction:
Rewrite the following casual text in formal business language.

### Input:
{example['input']}

### Response:
{example['output']}"""
    return {"text": prompt}

# Apply formatting
dataset = dataset.map(format_instruction)
print(f"Dataset size: {len(dataset)}")
print(f"Example:\n{dataset[0]['text']}")
```

### Tokenization Check

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Check sequence lengths
lengths = [len(tokenizer.encode(text)) for text in dataset["text"]]
print(f"Average length: {sum(lengths)/len(lengths):.0f} tokens")
print(f"Max length: {max(lengths)} tokens")

# Set max_length based on your data
max_seq_length = 256
```

---

## Step 3: Configure LoRA

### Quantization Config (Optional)

Use QLoRA if you have <12GB VRAM:

```python
use_qlora = True  # Set to False if you have 16GB+ VRAM

if use_qlora:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 recommended over float16
        bnb_4bit_use_double_quant=True,
    )
else:
    bnb_config = None
```

### Load Model

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False  # Required for training
```

### Prepare for Training

```python
# Gradient checkpointing for memory efficiency
if use_qlora:
    model = prepare_model_for_kbit_training(model)
```

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=32,                # Alpha scaling
    target_modules=[              # Modules to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # Add MLP for full adaptation
    ],
    lora_dropout=0.05,
    bias="none",
    use_dora=True,  # Optional: Use DoRA for more stable convergence
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## Step 4: Training

### Training Arguments

```python
# In TRL v1, use SFTConfig instead of TrainingArguments
training_args = SFTConfig(
    output_dir="./qwen3-formal",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=not use_qlora,  # QLoRA handles its own precision
    bf16=use_qlora,  # Use bf16 for QLoRA compute
    report_to="none",     # Set to "wandb" for experiment tracking
    packing=True,  # Pack sequences for faster training
    max_seq_length=512,  # Max sequence length
    use_liger_kernel=False,  # Set True for 20-30% VRAM savings (requires liger-kernel)
    loss_type="chunked_nll",  # Default in TRL v1.7: ~30% less peak VRAM
    trust_remote_code=True,  # New in TRL v1.7: single flag for all loaders
)
```

### Initialize Trainer

```python
# In TRL v1.7+, pass model and tokenizer directly
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    training_args=training_args,
)
```

### Start Training

```python
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model
trainer.save_model("./qwen3-formal")
```

### Expected Training Output

```
Step  Training Loss
10    2.3456
20    1.8765
30    1.4321
40    1.1234
50    0.9876

Training complete!
```

---

## Step 5: Evaluation

### Qualitative Evaluation

Test on unseen examples:

```python
def generate_response(input_text):
    """Generate formal response for casual input."""
    prompt = f"""### Instruction:
Rewrite the following casual text in formal business language.

### Input:
{input_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = full_text.split("### Response:")[-1].strip()
    return response

# Test examples
test_inputs = [
    "hey whats up",
    "can u help me with this",
    "sorry im busy rn",
]

print("=" * 60)
for inp in test_inputs:
    output = generate_response(inp)
    print(f"Input:  {inp}")
    print(f"Output: {output}")
    print("-" * 60)
```

### Expected Results

```
============================================================
Input:  hey whats up
Output: Hello, how are you doing today?
------------------------------------------------------------
Input:  can u help me with this
Output: Could you please assist me with this matter?
------------------------------------------------------------
Input:  sorry im busy rn
Output: I apologize, but I am currently occupied at the moment.
------------------------------------------------------------
```

### Quantitative Evaluation (Optional)

If you have a validation set:

```python
from datasets import load_metric

# Load validation set
val_dataset = load_dataset("json", data_files="validation_data.jsonl")

# Compute BLEU score (optional)
def compute_bleu(predictions, references):
    bleu = load_metric("bleu")
    return bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# Generate predictions
predictions = []
references = []

for example in val_dataset["train"]:
    pred = generate_response(example["input"])
    predictions.append(pred)
    references.append(example["output"])

bleu_score = compute_bleu(predictions, references)
print(f"BLEU Score: {bleu_score['bleu']:.3f}")
```

---

## Step 6: Inference

### Save for Deployment

```python
# Save adapter weights
model.save_pretrained("./qwen3-formal-adapter")
tokenizer.save_pretrained("./qwen3-formal-adapter")

# Save base model info for later loading
import json
with open("./qwen3-formal-adapter/config.json", "w") as f:
    json.dump({
        "base_model": model_name,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),
        }
    }, f, indent=2)
```

### Load for Inference

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "./qwen3-formal-adapter",
)

# Now use for inference
response = generate_response("thx for ur help")
print(response)
```

### Push to Hugging Face Hub (Optional)

```python
# Push to Hub
model.push_to_hub("your-username/qwen3-formal")
tokenizer.push_to_hub("your-username/qwen3-formal")

# Now others can use:
# from peft import PeftModel
# model = PeftModel.from_pretrained(base_model, "your-username/qwen3-formal")
```

---

## Complete Script

Here's the complete script in one file:

```python
#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B for casual → formal text conversion.
Run time: ~15 minutes on RTX 3060 (12GB VRAM)

Requirements: transformers>=5.13, peft>=0.19, trl>=1.7
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# ============ Configuration ============
MODEL_NAME = "Qwen/Qwen3-8B"  # or "google/gemma-4-12B-it", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
OUTPUT_DIR = "./qwen3-formal"
USE_QLORA = True  # Set False for 16GB+ VRAM
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4

# ============ Training Data ============
training_data = [
    {"input": "hey can u send me the report asap", 
     "output": "Hello, could you please send me the report at your earliest convenience?"},
    {"input": "thx for the info", 
     "output": "Thank you for the information."},
    {"input": "im gonna be late to the meeting", 
     "output": "I apologize, but I will be arriving late to the meeting."},
    {"input": "whats the status on this", 
     "output": "Could you please provide an update on the current status?"},
    {"input": "need this done by eod", 
     "output": "Please complete this task by the end of the day."},
    # Add 495 more examples for best results
]

# ============ Setup ============
print(f"Using CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ============ Quantization ============
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:
    bnb_config = None

# ============ Load Model ============
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

# ============ LoRA Config ============
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_dora=True,  # Optional: enable DoRA for stable convergence
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ Prepare Dataset ============
def format_instruction(example):
    prompt = f"""### Instruction:
Rewrite the following casual text in formal business language.

### Input:
{example['input']}

### Response:
{example['output']}"""
    return {"text": prompt}

dataset = Dataset.from_list(training_data)
dataset = dataset.map(format_instruction)

# ============ Training ============
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=not USE_QLORA,
    bf16=USE_QLORA,
    report_to="none",
    packing=True,  # Pack sequences for faster training
    max_seq_length=512,
    use_liger_kernel=False,  # Set True if you have liger-kernel installed
)

# In TRL v1.7+, pass model and tokenizer directly
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    training_args=training_args,
)

print("Starting training...")
trainer.train()
print("Training complete!")

# ============ Save ============
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ============ Inference Test ============
def generate_response(input_text):
    prompt = f"""### Instruction:
Rewrite the following casual text in formal business language.

### Input:
{input_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

test_inputs = ["hey whats up", "can u help me with this", "sorry im busy rn"]
for inp in test_inputs:
    print(f"\nInput:  {inp}")
    print(f"Output: {generate_response(inp)}")
```

Run it:
```bash
python fine_tune_qwen3.py
```

---

## Next Steps

1. **Expand the dataset** — 500+ examples for production quality
2. **Try larger models** — Qwen3.5-35B-A3B (MoE) or Gemma-4-26B-A4B-it for better results
3. **Add alignment** — DPO, ORPO, KTO, GRPO, or GMPO for preference optimization
4. **Add evaluation metrics** — BLEU, ROUGE, human evaluation
5. **Deploy the model** — vLLM 0.24, TGI, or llama.cpp (GGUF)
6. **Read [Module 04: Data Engineering](../04-data-engineering/)** — Building datasets

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce batch_size to 1, increase gradient_accumulation_steps, enable QLoRA |
| **Loss not decreasing** | Increase learning rate, check data formatting, verify tokenizer |
| **Garbage outputs** | Verify tokenizer, check for special token issues, use correct chat template |
| **Slow training** | Enable `bf16` if GPU supports it, use `packing=True`, try `use_liger_kernel=True` |
| **Import errors** | Ensure transformers>=5.13, peft>=0.19, trl>=1.7 |

---

## References

- [Qwen3 Technical Report](https://github.com/QwenLM/Qwen3)
- [Qwen3.5 Technical Report](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.6 Technical Report](https://github.com/QwenLM/Qwen3.6)
- [Gemma 4 Technical Report](https://huggingface.co/google/gemma-4-12B-it)
- [TRL Documentation](https://huggingface.co/docs/trl) — TRL v1.7, GRPO, DPO, ORPO, KTO, GMPO
- [PEFT Quickstart](https://huggingface.co/docs/peft/quicktour) — 40+ adapter methods incl. GraLoRA, TinyLoRA
- [SFTTrainer API Reference](https://huggingface.co/docs/trl/sft_trainer)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — Memory-efficient fused kernels
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster training, 70% less VRAM
- [vLLM Serving](https://docs.vllm.ai) — Continuous batching, FP8 support
