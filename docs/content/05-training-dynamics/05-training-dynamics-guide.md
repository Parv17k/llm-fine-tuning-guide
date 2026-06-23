# Training Dynamics

Supervised Fine-Tuning (SFT) and hyperparameter optimization for stable, efficient training.

## Overview

Training dynamics govern how your model learns during fine-tuning. Poor choices lead to:
- **Catastrophic forgetting**: Model loses pre-trained knowledge
- **Overfitting**: Model memorizes training data, fails on new inputs
- **Underfitting**: Model doesn't learn the task adequately
- **Training instability**: Loss spikes, divergence, NaN errors

This module covers the complete training workflow from SFT fundamentals to multi-GPU scaling.

---

## Chapter 1: Supervised Fine-Tuning (SFT)

### What is SFT?

Supervised Fine-Tuning trains a pre-trained model on labeled input-output pairs to learn a specific task or behavior.

**Training Objective**: Minimize cross-entropy loss between predicted and target tokens:

```
L(θ) = -Σ log P(y_t | x, y_<t; θ)
```

Where:
- `x` = input prompt
- `y` = target output
- `θ` = model parameters

### Full Fine-Tuning Workflow

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train[:1000]")

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-3.2-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()

# Save
trainer.save_model()
trainer.save_state()
```

### Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all parameters—expensive and prone to forgetting. PEFT methods freeze most parameters and train small adapters.

#### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    r=16,                      # Rank of update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

# Apply LoRA to model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Output:
# trainable params: 8,388,608 || all params: 3,210,000,000 || trainable%: 0.26%
```

**Why LoRA works**: Instead of updating weight matrix `W ∈ ℝ^(d×d)`, LoRA learns low-rank decomposition:

```
W' = W + ΔW = W + BA
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×d), and r << d
```

For `d=4096, r=16`: Full FT = 16.7M params, LoRA = 131K params (127× reduction).

### Freezing vs. Unfreezing Layers

**When to freeze**:
- Small dataset (<10k samples)
- Task similar to pre-training
- Limited compute resources

**When to unfreeze**:
- Large dataset (>100k samples)
- Domain shift from pre-training
- Task requires deep adaptation

```python
# Freeze all layers, unfreeze last 4
def freeze_except_last_n_layers(model, n=4):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last n transformer blocks
    for i in range(len(model.model.layers) - n, len(model.model.layers)):
        for param in model.model.layers[i].parameters():
            param.requires_grad = True
    
    # Always keep lm_head trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

# Usage
model = freeze_except_last_n_layers(model, n=4)
```

---

## Chapter 2: Hyperparameter Tuning

### Learning Rate

The most critical hyperparameter. Too high → divergence. Too low → slow convergence or stuck in local minima.

**Recommended ranges by method**:

| Method | Learning Rate Range | Typical Value |
|--------|---------------------|---------------|
| Full Fine-Tuning | 1e-6 to 5e-5 | 2e-5 |
| LoRA | 1e-4 to 5e-4 | 2e-4 |
| QLoRA | 1e-4 to 1e-3 | 2e-4 |

**Learning Rate Finder**:

```python
from transformers import TrainerCallback
import matplotlib.pyplot as plt

class LRFinderCallback(TrainerCallback):
    def __init__(self, start_lr=1e-8, end_lr=1.0):
        self.lrs = []
        self.losses = []
        self.start_lr = start_lr
        self.end_lr = end_lr
    
    def on_step_end(self, args, state, control, **kwargs):
        # Log current learning rate
        self.lrs.append(state.learning_rate)
        
        # Get loss from logs
        if len(state.log_history) > 0:
            self.losses.append(state.log_history[-1].get('loss', float('inf')))
        
        # Linearly increase LR
        progress = state.global_step / args.max_steps
        new_lr = self.start_lr * (self.end_lr / self.start_lr) ** progress
        args.learning_rate = new_lr

# Run LR finder
training_args = TrainingArguments(
    output_dir="./lr-finder",
    per_device_train_batch_size=8,
    max_steps=500,  # Run for 500 steps
    logging_steps=10,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.add_callback(LRFinderCallback())
trainer.train()

# Plot
plt.figure(figsize=(10, 6))
plt.semilogx(lr_finder.lrs, lr_finder.losses)
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.show()

# Best LR is where loss decreases fastest (steepest negative slope)
```

### Batch Size and Gradient Accumulation

**Effective batch size** = `per_device_batch_size × gradient_accumulation_steps × num_gpus`

**Trade-offs**:
- **Large batch** (≥64): Stable gradients, faster training, may generalize worse
- **Small batch** (≤8): Noisy gradients (can help generalization), slower training

**Memory constraints**: If you hit OOM, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` to maintain effective batch size.

```yaml
# Scenario 1: Single GPU, 24GB VRAM
per_device_train_batch_size: 8
gradient_accumulation_steps: 8
effective_batch_size: 64

# Scenario 2: Single GPU, 12GB VRAM (OOM on batch=8)
per_device_train_batch_size: 2
gradient_accumulation_steps: 32
effective_batch_size: 64  # Same effective batch, longer training
```

### Epoch Count and Early Stopping

**Rule of thumb**:
- Small dataset (<10k): 3-5 epochs
- Medium dataset (10k-100k): 2-3 epochs
- Large dataset (>100k): 1-2 epochs

**Early stopping** prevents overfitting:

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./sft-early-stop",
    num_train_epochs=10,  # Set high, let early stopping decide
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
trainer.train()
```

### Optimizer Selection

| Optimizer | Use Case | Learning Rate |
|-----------|----------|---------------|
| **AdamW** | Default choice, works well for most tasks | 2e-5 (full FT), 2e-4 (LoRA) |
| **AdamW 8-bit** | Memory-constrained training | Same as AdamW |
| **SGD** | Rarely used for LLMs | 0.01-0.1 |
| **Adafactor** | Very large models, memory efficient | Relative scale |

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft",
    optim="adamw_torch",       # PyTorch AdamW
    # optim="adamw_bnb_8bit",  # 8-bit AdamW (requires bitsandbytes)
    # optim="adafactor",       # Adafactor
)
```

### Hyperparameter Search with Optuna

```python
import optuna
from optuna.integration import HuggingFacePruningCallback

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    
    # Create model with trial's LoRA config
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_r*2, ...)
    model = get_peft_model(base_model, lora_config)
    
    # Training args with trial's parameters
    training_args = TrainingArguments(
        output_dir=f"./trial-{trial.number}",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        load_best_model_at_end=True,
    )
    
    # Trainer with pruning callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[HuggingFacePruningCallback(trial, "eval_loss")],
    )
    
    trainer.train()
    return trainer.state.log_history[-1]["eval_loss"]

# Run study
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.number}")
print(f"Best params: {study.best_params}")
```

---

## Chapter 3: Training Loops

### Using Hugging Face Trainer

The `Trainer` class handles training loops, logging, checkpointing, and distributed training.

**Minimal example**:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()              # Start training
trainer.save_model()         # Save final model
trainer.save_state()         # Save training state (losses, etc.)
```

### Custom Training Loop with Accelerate

For more control, use `accelerate`:

```python
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Initialize accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="bf16",
)

# Prepare everything
model, optimizer, dataloader = accelerator.prepare(
    model,
    AdamW(model.parameters(), lr=2e-5),
    train_dataloader,
)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item()
            if step % 10 == 0:
                avg_loss = total_loss / (step + 1)
                accelerator.print(f"Epoch {epoch}, Step {step}, Loss {avg_loss:.4f}")
    
    # Save checkpoint
    accelerator.save_model(model, f"./checkpoint-epoch-{epoch}")
```

### Logging and Tracking

**TensorBoard**:

```python
from transformers import TrainingArguments
from torch.utils.tensorboard import SummaryWriter

training_args = TrainingArguments(
    output_dir="./output",
    logging_dir="./logs",
    logging_steps=10,
    report_to="tensorboard",
)

# Start tensorboard: tensorboard --logdir ./logs
# View at: http://localhost:6006
```

**Weights & Biases**:

```python
import wandb
from transformers import TrainerCallback

wandb.init(project="llm-fine-tuning", name="run-1")

class WandBCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)

training_args = TrainingArguments(
    output_dir="./output",
    logging_steps=10,
    report_to="wandb",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.add_callback(WandBCallback())
trainer.train()
```

### Gradient Checkpointing

Reduce memory usage by ~40% at cost of ~20% slower training:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

# Or via training args
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
)
```

---

## Chapter 4: Preventing Catastrophic Forgetting

Catastrophic forgetting occurs when fine-tuning overwrites pre-trained knowledge, degrading performance on unrelated tasks.

### Elastic Weight Consolidation (EWC)

EWC penalizes changes to important weights using Fisher Information Matrix.

```python
import torch
import torch.nn as nn
from torch.optim import AdamW

class EWC:
    def __init__(self, model: nn.Module, fisher_estimate: dict, lambda_ewc: float = 5000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = fisher_estimate  # Pre-computed Fisher Information
        self.star_params = {n: p.clone() for n, p in model.named_parameters()}
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = torch.tensor(0.0, device=self.model.device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name]
                old_param = self.star_params[name]
                loss += (fisher * (param - old_param).pow(2)).sum()
        
        return self.lambda_ewc * loss

# Usage in training loop
ewc = EWC(model, fisher_estimate, lambda_ewc=5000)

for batch in dataloader:
    outputs = model(**batch)
    ce_loss = outputs.loss
    ewc_loss = ewc.ewc_loss()
    total_loss = ce_loss + ewc_loss
    
    total_loss.backward()
    optimizer.step()
```

### Replay Methods

Keep model grounded by mixing pre-training data with fine-tuning data.

```python
from torch.utils.data import ConcatDataset, DataLoader

# Mix fine-tuning data with pre-training samples
replay_ratio = 0.2  # 20% replay data

replay_dataset = load_dataset("c4", "en", split="train[:10000]")
mixed_dataset = ConcatDataset([finetune_dataset, replay_dataset])

# Weighted sampling to control ratio
from torch.utils.data import WeightedRandomSampler

finetune_weights = [1.0] * len(finetune_dataset)
replay_weights = [replay_ratio] * len(replay_dataset)
sampler = WeightedRandomSampler(
    weights=finetune_weights + replay_weights,
    num_samples=len(finetune_dataset),
    replacement=True,
)

dataloader = DataLoader(mixed_dataset, batch_size=8, sampler=sampler)
```

### Regularization Techniques

**L2 Regularization (Weight Decay)**:

```python
training_args = TrainingArguments(
    ...
    weight_decay=0.01,  # Default L2 penalty
)
```

**Layer-wise Learning Rate Decay**:

Lower LR for earlier layers (preserve pre-trained features), higher LR for later layers.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layerwise_lr_decay=0.8,  # Each layer gets 0.8× the LR of the next
)
```

**AdapterFusion**:

Train multiple task adapters, then fuse them without forgetting:

```python
from peft import AdapterConfig, AdapterFusionConfig

# Train task-specific adapters
model.add_adapter("task_1", config=AdapterConfig())
model.train_adapter("task_1")

model.add_adapter("task_2", config=AdapterConfig())
model.train_adapter("task_2")

# Fuse adapters
model.add_adapter_fusion(["task_1", "task_2"])
model.train_adapter_fusion(["task_1", "task_2"])
```

---

## Chapter 5: Multi-GPU Training

### Distributed Data Parallel (DDP)

PyTorch DDP trains on multiple GPUs with synchronized gradients.

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (8 GPUs total, 2 nodes)
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --nproc_per_node=4 train.py
```

**Training script** (`train.py`):

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# Training
local_rank = setup_ddp()

# Model wrapped in DDP
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Distributed sampler ensures each GPU sees different data
sampler = DistributedSampler(dataset, rank=local_rank)
dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

# Training loop (same as single GPU)
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

cleanup_ddp()
```

### DeepSpeed

DeepSpeed extends DDP with ZeRO (Zero Redundancy Optimizer) for memory efficiency.

**ZeRO Stages**:

| Stage | What's Sharded | Memory Savings |
|-------|----------------|----------------|
| ZeRO-1 | Optimizer states | 8× |
| ZeRO-2 | Optimizer + gradients | 16× |
| ZeRO-3 | Optimizer + gradients + parameters | 32× |

**DeepSpeed config** (`ds_config.json`):

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 100
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

**Run with DeepSpeed**:

```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

**Hugging Face integration**:

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    deepspeed="./ds_config.json",
)
```

### Multi-GPU with Accelerate

Simpler alternative to raw DDP:

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Prepare model, optimizer, dataloader
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# Training loop works on any number of GPUs
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

```bash
# Configure accelerate
accelerate config

# Run
accelerate launch train.py
```

### Memory Optimization Comparison

| Configuration | Max Model Size (24GB GPU) | Training Speed |
|---------------|---------------------------|----------------|
| Single GPU | ~3B | Baseline |
| DDP (4× GPU) | ~3B | 3.5× faster |
| ZeRO-2 (4× GPU) | ~7B | 3× faster |
| ZeRO-3 Offload (4× GPU) | ~13B | 2.5× faster |

---

## Training Configuration Template

```yaml
# Full fine-tuning template
model_name: "meta-llama/Llama-3.2-3B"

# Data
dataset: "mlabonne/FineTome-100k"
max_length: 512

# Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 2e-5
weight_decay: 0.01

# Scheduler
warmup_steps: 100
lr_scheduler_type: "cosine"

# Optimization
optim: "adamw_torch"
max_grad_norm: 1.0

# Mixed precision
bf16: true
gradient_checkpointing: true

# Logging
logging_steps: 10
report_to: "wandb"

# Checkpointing
save_strategy: "epoch"
save_total_limit: 2

# LoRA (if using PEFT)
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

---

## Monitoring Training

### Loss Curves

```python
import matplotlib.pyplot as plt
import json

# Load training log
with open("./output/trainer_log.jsonl") as f:
    logs = [json.loads(line) for line in f]

# Plot
steps = [log["step"] for log in logs if "loss" in log]
losses = [log["loss"] for log in logs if "loss" in log]

plt.figure(figsize=(12, 5))
plt.plot(steps, losses, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**What to look for**:
- **Smooth decrease**: Training is stable
- **Spikes**: Learning rate too high, gradient clipping needed
- **Plateau**: May need more data or different LR
- **NaN/Inf**: Numerical instability, reduce LR or check data

### Perplexity Metrics

Perplexity = `exp(loss)`. Lower is better.

```python
def compute_perplexity(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Usage
train_ppl = compute_perplexity(model, train_dataloader)
val_ppl = compute_perplexity(model, val_dataloader)
print(f"Train PPL: {train_ppl:.2f}, Val PPL: {val_ppl:.2f}")
```

### Learning Rate Schedules

```python
from transformers import get_scheduler

# Available schedulers:
# - linear: Linear decay from initial LR to 0
# - cosine: Cosine decay from initial LR to 0
# - constant: Keep LR constant
# - constant_with_warmup: Constant after warmup

num_training_steps = len(dataloader) * num_epochs

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

# Plot schedule
lrs = []
for _ in range(num_training_steps):
    lrs.append(lr_scheduler.get_lr()[0])
    lr_scheduler.step()

plt.plot(lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| **OOM (Out of Memory)** | Batch size too large | Reduce batch size, enable gradient checkpointing, use ZeRO |
| **Loss spikes** | LR too high, bad data | Reduce LR, check for corrupted samples |
| **Loss not decreasing** | LR too low, model frozen | Increase LR, check requires_grad |
| **NaN in outputs** | Numerical instability | Use mixed precision, gradient clipping |
| **Overfitting** | Too many epochs, small dataset | Early stopping, more data, regularization |
| **Slow training** | I/O bottleneck, small batch | Use DataLoader with num_workers > 0, increase batch |

### Debug Checklist

```python
# 1. Check model is on correct device
print(next(model.parameters()).device)  # Should be cuda

# 2. Check gradients are flowing
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")

# 3. Check input shapes
print(f"Input IDs shape: {batch['input_ids'].shape}")

# 4. Check for NaN
if torch.isnan(loss):
    print("NaN loss detected!")

# 5. Check memory usage
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## Summary

**Key takeaways**:

1. **SFT** trains on labeled pairs; LoRA reduces trainable params by 100×
2. **Learning rate** is the most critical hyperparameter—use LR finder
3. **Trainer** handles most use cases; `accelerate` for custom loops
4. **Catastrophic forgetting** prevented via EWC, replay, regularization
5. **Multi-GPU** with DDP or DeepSpeed ZeRO for large models

**Next**: Module 06 covers Parameter-Efficient Fine-Tuning (PEFT) in depth.
