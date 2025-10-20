#!/bin/bash
# prepare_trtllm.sh - Setup TensorRT-LLM and convert model

set -e

# ----- CONFIG -----
MODEL_DIR="$PWD/models/llama-3-8b"       # Hugging Face model path
ENGINE_DIR="$PWD/trt_llama3_engine"      # Where to store TRT-LLM engine
DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm:latest"
CONTAINER_NAME="trtllm_build"

# ----- STEP 1: Pull image -----
echo ">>> Pulling TensorRT-LLM Docker image..."
docker pull $DOCKER_IMAGE

# ----- STEP 2: Run container and mount volumes -----
echo ">>> Launching container $CONTAINER_NAME..."
docker run --rm -it \
  --gpus all \
  --name $CONTAINER_NAME \
  -v $MODEL_DIR:/workspace/model \
  -v $ENGINE_DIR:/workspace/engine \
  $DOCKER_IMAGE \
  bash -c "
    echo '>>> Converting HF model to TensorRT-LLM engine...'
    python3 tensorrt_llm/examples/llama/convert_checkpoint.py \
      --model_dir /workspace/model \
      --output_dir /workspace/engine \
      --dtype float16
  "

echo "✅ Conversion complete. Engine saved at $ENGINE_DIR"
