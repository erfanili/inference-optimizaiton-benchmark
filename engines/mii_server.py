from fastapi import FastAPI, Request
import mii

app = FastAPI()

# Load model via MII (this will spin up a local inference service)
model_id = "meta-llama/Llama-2-7b-chat-hf"   # swap with your local model if needed
deployment_name = "llama_mii"

mii.deploy(deployment_name, model=model_id, dtype="fp16")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])
    max_new_tokens = data.get("max_new_tokens", 100)

    # Query MII service
    result = mii.query(deployment_name, prompts, max_new_tokens=max_new_tokens)
    responses = [out["generated_text"] for out in result]
    return {"responses": responses}
