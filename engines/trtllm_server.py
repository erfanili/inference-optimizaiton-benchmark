from fastapi import FastAPI, Request
from tensorrt_llm.runtime import GenerationSession
import torch

app = FastAPI()

# Load prebuilt TensorRT engine
engine_dir = "trt_llama3_engine"
session = GenerationSession.from_prebuilt(engine_dir, device="cuda")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])
    max_new_tokens = data.get("max_new_tokens", 100)

    responses = []
    for p in prompts:
        tokens = session.generate(p, max_new_tokens=max_new_tokens)
        responses.append(tokens)
    return {"responses": responses}
