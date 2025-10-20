#!/bin/bash
# run_tgi.sh - Launch Hugging Face TGI with local models

docker run --gpus all --shm-size 1g -p 8003:80 \
  -v "$PWD/models:/data" \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/llama-3-8b
