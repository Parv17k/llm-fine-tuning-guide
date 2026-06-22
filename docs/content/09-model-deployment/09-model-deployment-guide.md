# Model Deployment

Quantization, serving, and production readiness.

## Chapters

- [**Checkpoint Merging**](./01-merging.md)  
  - Merging LoRA adapters into base model  
  - Weight decomposition  
  - Saving merged models

- [**Quantization Methods**](./02-quantization.md)  
  - GGUF: llama.cpp format  
  - AWQ: Activation-aware Weight Quantization  
  - EXL2, EXL3: Efficient LLM EXtensions  
  - 4-bit vs. 6-bit vs. 8-bit tradeoffs

- [**Serving Fine-Tuned Models**](./03-serving.md)  
  - vLLM: High-throughput serving  
  - TGI (Text Generation Inference)  
  - FastAPI wrappers  
  - Streaming responses

- [**Production Considerations**](./04-production.md)  
  - Model versioning  
  - A/B testing  
  - Rate limiting and scaling  
  - Monitoring and alerting

- [**Edge Deployment**](./05-edge.md)  
  - ONNX export  
  - Mobile and WASM targets  
  - Model size optimization

## Serving Comparison

| Server | Max Throughput | Memory | Best For |
|--------|---------------|--------|----------|
| vLLM | Very High | Moderate | API services |
| TGI | High | Low | Production |
| llama.cpp | Low | Minimal | Local/edge |
| Transformers | Low | High | Development |

## GGUF Quantization Guide

```bash
# Convert to GGUF
python convert_llama_weights_to_gguf.py \
    --model /path/to/model \
    --outfile /path/to/model-Q4_K_M.gguf \
    --quantize Q4_K_M
```
