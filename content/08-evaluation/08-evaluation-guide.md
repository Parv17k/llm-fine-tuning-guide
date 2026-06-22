# Evaluation

Benchmarking fine-tuned models and avoiding overfitting.

## Chapters

- [**Evaluation Fundamentals**](./01-fundamentals.md)  
  - Perplexity vs. task accuracy  
  - Holding out validation data  
  - Avoiding data leakage

- [**Benchmarking Frameworks**](./02-benchmarks.md)  
  - Human evaluation vs. automated  
  - Common benchmarks: MMLU, TruthfulQA, HellaSwag  
  - Domain-specific evals

- [**Building Custom Evals**](./03-custom-evals.md)  
  - Exact match vs. soft metrics  
  - Multiple choice evaluation  
  - Generation length analysis

- [**Overfitting Detection**](./04-overfitting.md)  
  - Gap between train and eval loss  
  - Early stopping strategies  
  - Regularization for eval performance

- [**Quantitative vs. Qualitative**](./05-qual-quant.md)  
  - Automated metrics limitations  
  - Manual review checklists  
  - Case studies

## Evaluation Checklist

- [ ] Eval on held-out data (not used in training)
- [ ] Multiple metrics (not just one)
- [ ] Compare against baseline model
- [ ] Test on out-of-distribution data
- [ ] Human review for critical applications

## Tools

- `evaluate` library
- `lm-evaluation-harness`
- Custom test scripts
