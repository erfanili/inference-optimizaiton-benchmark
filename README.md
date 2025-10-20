# Inference Optimization Benchmarking Suite

A reproducible framework for evaluating **LLM inference engines** under varying configurations of batch size, precision, quantization, and GPU memory. The suite benchmarks multiple backends side-by-side, providing insights into **throughput, latency, and memory efficiency** — critical metrics for deploying LLMs at scale.  

## ✨ Features

### Supported Engines
- **vLLM** — paged attention, continuous batching  
- **Hugging Face Transformers** — PyTorch baseline  
- **DeepSpeed-MII** — DeepSpeed inference runtime  
- **TensorRT-LLM** — NVIDIA kernel-optimized backend  
- **Text Generation Inference (TGI)** — Hugging Face production server  
- *(Planned)* llama.cpp  

### Collected Metrics
- **Latency** (s per request)  
- **Throughput** (tokens/sec)  
- **Total tokens generated**  
- **Peak GPU memory usage** (GB)  
- **Automatic OOM detection** (batch size limits)  

### Outputs
- Structured results in **JSON / CSV**  
- **Markdown summary tables**  
- **Latency vs. batch size** plots  
- **Throughput vs. batch size** plots  
- *(Optional)* efficiency curves (tokens/sec per GB)  

---


## 🚀 Quick Start

### 1. Install dependencies
```bash
git clone https://github.com/erfanili/inference-benchmark.git
cd inference-benchmark
pip install -r requirements.txt
```

### 2. Prepare a model

Download or symlink models into the models/ directory, e.g.:
```bash
mkdir -p models
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b
```

### 3. Run vLLM benchmark
```bash
python benchmark/benchmark_vllm.py \
  --model models/llama-2-7b \
  --dtype float16 \
  --out-tag fp16
  ```

### 4. Run TGI (Text Generation Inference) with Docker
```bash
docker run --gpus all --shm-size 1g -p 8003:80 \
  -v $PWD/models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/llama-2-7b
```

### 5. Run TensorRT-LLM with Docker
```bash
docker run --gpus all -it --rm \
  -v $PWD/models:/models \
  nvcr.io/nvidia/tritonserver:23.08-py3 \
  bash -c "tritonserver --model-repository=/models"
```

### 6. Run DeepSpeed-MII benchmark
```bash
python benchmark/benchmark_mii.py --model microsoft/phi-2
```


### 7. Compare results
```bash
python benchmark/merge_and_plot.py
```