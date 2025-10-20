#transformers_baseline.py

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import torch

app = FastAPI()

model_id ="models/llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16)
generator = pipeline("text-generation",model= model, tokenizer = tokenizer)


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts",[])

    max_new_tokens = data.get("max_new_tokens",100)
    outputs = generator(prompts, max_new_tokens = max_new_tokens, do_sample =False)
    responses = [out[0]["generated_text"] for out in outputs]
    return {"responses": responses}