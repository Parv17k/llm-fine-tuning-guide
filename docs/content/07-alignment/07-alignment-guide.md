# Alignment & Optimization

DPO and ORPO for steering model behavior without RL.

## Chapters

- [**Why Alignment Matters**](./01-alignment-intro.md)  
  - Reward modeling vs. alignment  
  - The RLHF complexity problem  
  - When you need alignment

- [**Direct Preference Optimization (DPO)**](./02-dpo.md)  
  - Theory and derivation from RLHF  
  - Preference datasets structure  
  - Loss function and training  
  - Hyperparameter differences from SFT

- [**Odds Ratio Preference Optimization (ORPO)**](./03-orpo.md)  
  - Simplified DPO formulation  
  - SFT + preference in one pass  
  - Advantages over DPO

- [**Preference Dataset Creation**](./04-preference-data.md)  
  - Generating win/lose pairs  
  - Self-instruct for preferences  
  - Human annotation strategies

- [**Common Pitfalls**](./05-pitfalls.md)  
  - Over-alignment  
  - Preference noise handling  
  - Evaluation challenges

## Alignment vs. SFT Comparison

| Aspect | SFT | DPO | ORPO |
|--------|-----|-----|------|
| Data format | Instructions → Responses | Pairs of responses | Pairs of responses |
| Training complexity | Simple | Moderate | Moderate |
| RL components | None | None | None |
| Best for | Factual accuracy | Behavior control | Tone/ethics |

## DPO Configuration Template

```yaml
beta: 0.1  # KL regularization strength
loss_type: "sigmoid"  # Also: "hinge", "cosine", "reference_free"
```
