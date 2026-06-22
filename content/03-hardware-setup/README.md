# Hardware & Infrastructure

Calculating requirements and provisioning compute for LLM training.

## Chapters

- [**VRAM Mathematics**](./01-vram-math.md)  
  - Weight storage: `parameters × bytes_per_parameter`  
  - Gradient storage: Same as weights  
  - Optimizer states: Adam = 2× weights  
  - Calculating total VRAM needs

- [**GPU Selection Guide**](./02-gpu-selection.md)  
  - Consumer cards (RTX 4090, 5090)  
  - Data center GPUs (A100, H100, B200)  
  - Multi-GPU configurations

- [**Model Size vs. Hardware**](./03-sizing.md)  
  - 7B, 8B, 13B, 70B models: What fits where  
  - Quantization impact on memory

- [**Cluster Provisioning**](./04-cluster.md)  
  - Cloud providers (AWS, GCP, Azure, Lambda Labs)  
  - On-premise setup  
  - Network optimization

- [**CUDA Optimization**](./05-cuda.md)  
  - cuDNN settings  
  - Memory efficient backends  
  - Gradient checkpointing

## Tools & Scripts

- [**Hardware Calculator**](./calculator/)  
  Python script to determine VRAM requirements

## Example Configurations

| Model Size | Full Precision | Q4 quantized | Q8 quantized |
|------------|---------------|--------------|--------------|
| 7B         | 2× A100 40GB  | RTX 4090     | RTX 3090     |
| 13B        | 2× A100 80GB  | A10 48GB     | 2× RTX 4090  |
| 70B        | 8× H100       | 4× A100      | 2× A100      |
