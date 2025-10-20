from fastapi import FastAPI, Request
import requests

app = FastAPI()
TGI_URL = "http://localhost:8003/generate"

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])
    max_new_tokens = data.get("max_new_tokens", 100)

    responses = []
    for p in prompts:
        resp = requests.post(TGI_URL, json={
            "inputs": p,
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7}
        })
        responses.append(resp.json()["generated_text"])
    return {"responses": responses}
