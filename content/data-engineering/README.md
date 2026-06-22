# Data Engineering

Dataset curation, tokenization, and formatting for fine-tuning.

## Chapters

- [**Dataset Curation**](./01-curation.md)  
  - Finding quality datasets  
  - Custom dataset creation  
  - Size recommendations (1k-100k samples)

- [**Tokenization Mechanics**](./02-tokenization.md)  
  - BPE, WordPiece, SentencePiece  
  - Special tokens and vocab expansion  
  - Token counting and costs

- [**ChatML Format**](./03-chatml.md)  
  - Structure and syntax  
  - System messages and user/assistant turns  
  - End-to-end examples

- [**JSON/Alpaca Format**](./04-alpaca.md)  
  - Instruction-dataset format  
  - Multi-turn conversations  
  - Field validation

- [**Data Quality & Filtering**](./05-quality.md)  
  - Removing duplicate samples  
  - Toxicity and safety filters  
  - Length filtering (too short/too long)

- [**Deduplication Pipelines**](./06-dedup.md)  
  - MinHash for similarity detection  
  - Exact string dedup  
  - Cross-dataset deduplication

## Data Checklist

- [ ] ChatML formatted correctly
- [ ] No duplicates
- [ ] Appropriate length distribution
- [ ] Safe content (no toxic labels)
- [ ] Validation/train/test split (80/10/10)

## Tools

- `datasets` library for preprocessing
- `tokenizers` for custom tokenization
- `text-dedup` for deduplication pipelines
