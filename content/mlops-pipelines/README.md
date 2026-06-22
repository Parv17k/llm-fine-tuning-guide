# MLOps & DevOps

CI/CD for LLMs and automated pipelines.

## Chapters

- [**CI/CD for Model Training**](./01-ci-cd.md)  
  - Automated training triggers  
  - GPU job orchestration  
  - Buildkite, GitHub Actions, Jenkins

- [**Automated Model Packaging**](./02-packaging.md)  
  - Model cards and documentation  
  - Docker containerization  
  - Hugging Face model uploads

- [**Continuous Training (CT)**](./03-continuous-training.md)  
  - Triggering on new data  
  - Model registry integration  
  - Rollback strategies

- [**Automated Evaluation**](./04-automated-eval.md)  
  - Running evals on commit  
  - Quality gates  
  - Performance thresholds

- [**Monitoring in Production**](./05-monitoring.md)  
  - Latency tracking  
  - Token consumption  
  - Drift detection

## GitHub Actions Template

```yaml
name: Fine-Tune Model

on:
  push:
    branches: [main]
    paths: ['data/**']

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Fine-tune
        run: python train.py
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
      - name: Upload model
        run: huggingface-cli upload ...
```

## Model Registry

- Hugging Face Hub (public/private)
- MLflow
- Weights & Biases
- Custom artifact storage
