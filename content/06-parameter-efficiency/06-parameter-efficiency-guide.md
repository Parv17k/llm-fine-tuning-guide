# Parameter Efficiency

LoRA, QLoRA, and adapter-based fine-tuning.

## Chapters

- [**Understanding LoRA**](./01-lora.md)  
  - Low-Rank Adaptation theory  
  - Rank (r) selection  
  - Alpha (α) scaling  
  - What gets trained vs. frozen

- [**QLoRA: Quantized LoRA**](./02-qlora.md)  
  - 4-bit quantization basics  
  - NF4 (NormalFloat 4-bit)  
  - Double quantization  
  - Memory vs. quality tradeoffs

- [**Adapter Integration**](./03-adapters.md)  
  - Loading and merging adapters  
  - Adapter stacking  
  - Multi-adapter scenarios

- [**Rank Selection Guide**](./04-rank-selection.md)  
  - Small datasets: r=4 to r=16  
  - Medium datasets: r=32 to r=64  
  - Large datasets: r=128+  
  - Ablation study methodology

- [**Training Tips**](./05-tuning-tips.md)  
  - Optimizing for speed vs. quality  
  - Memory profiling  
  - Gradient checkpointing with LoRA

## LoRA Configuration Reference

| Parameter | Typical Values | Recommendation |
|-----------|---------------|----------------|
| r (rank) | 4, 8, 16, 32, 64 | Start at 8, increase if underfitting |
| alpha | 16, 32, 64 | alpha/r ≈ 1 for baseline |
| dropout | 0.0 to 0.5 | 0.1 for regularization |
| target_modules | See below | Common patterns provided |

## Common target_modules

```python
# For Llama models
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# For Mistral models
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```
