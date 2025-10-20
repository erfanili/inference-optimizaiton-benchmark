import requests, json, time

ENGINES = {"transformers": "http://localhost:8000/generate",
           "vllm": "http://localhost:8001/generate"}

prompt = "Explain General Relativty in 100 characters."

payload = {"prompts": [prompt], "max_new_tokens": 100}

results = []

for engine, url in ENGINES.items():
    print(f"Querying {engine} ...")
    start = time.time()
    
    response = requests.post(url, json=payload)
    latency = time.time()-start
    
    output = response.json()["responses"]
    print(f"{engine} responses:\n{'\n'.join(output)}\n")

    
    results.append({"engine": engine,
                    "latency": latency,
                    "output_lens": [len(out.split()) for out in output]})
    
with open("benchmark/results/single_prompt_results.json", "w") as f:
    json.dump(results, f, indent=2)