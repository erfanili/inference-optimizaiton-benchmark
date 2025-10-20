# vllm_server.py
import argparse
import os
import sys
from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backends'))

# ----------------------------
# CLI Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="vLLM FastAPI Server")

parser.add_argument("--model", type=str, default="models/llama-3-8b",
                    help="Model path or HuggingFace repo id")
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "auto"],
                    help="Data type precision")
parser.add_argument("--max-model-len", type=int, default=2048,
                    help="Maximum sequence length")
parser.add_argument("--quantization", type=str, default=None,
                    choices=[None, "gptq", "awq", "bitsandbytes"],
                    help="Quantization method (if supported)")
parser.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="Default sampling temperature")
parser.add_argument("--max-tokens", type=int, default=200,
                    help="Default max new tokens to generate")
parser.add_argument("--port", type=int, default=8001,
                    help="Port to expose FastAPI app")

args = parser.parse_args()

# ----------------------------
# Initialize LLM
# ----------------------------
llm = LLM(
    model=args.model,
    dtype=args.dtype,
    max_model_len=args.max_model_len,
    tensor_parallel_size=args.tensor_parallel_size,
    quantization=args.quantization,
)

sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])
    # allow overrides from request body if provided
    temperature = data.get("temperature", args.temperature)
    max_tokens = data.get("max_tokens", args.max_tokens)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    outputs = llm.generate(prompts, sp)
    responses = [out.outputs[0].text for out in outputs]
    return {"responses": responses}

# ----------------------------
# Main entrypoint
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("vllm_server:app", host="0.0.0.0", port=args.port, workers=1)
