# Training Dynamics

Supervised Fine-Tuning (SFT) and hyperparameter optimization.

## Chapters

- [**Supervised Fine-Tuning (SFT)**](./01-sft.md)  
  - Full fine-tuning workflow  
  - Parameter-Efficient Fine-Tuning (PEFT)  
  - Freezing vs. unfreezing layers

- [**Hyperparameter Tuning**](./02-hyperparameters.md)  
  - Learning rate: Finding the sweet spot  
  - Batch size and gradient accumulation  
  - Epoch count and warmup steps  
  - Optimizer selection (AdamW, SGD)

- [**Training Loops**](./03-training-loops.md)  
  - Using `Trainer` from `transformers`  
  - Custom training loops with `accelerate`  
  - Logging and tracking

- [**Preventing Catastrophic Forgetting**](./04-forgetting.md)  
  - EWC (Elastic Weight Consolidation)  
  - Replay methods  
  - Regularization techniques

- [**Multi-GPU Training**](./05-multi-gpu.md)  
  - DDP (Distributed Data Parallel)  
  - DeepSpeed configurations  
  - ZeRO stages

## Training Configuration Template

```yaml
model_name: "meta-llama/Llama-3.2-3B"
learning_rate: 2e-5
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
warmup_steps: 100
lr_scheduler_type: "cosine"
```

## Monitoring Training

- Loss curves
- Perplexity metrics
- Learning rate schedules
