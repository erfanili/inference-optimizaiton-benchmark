# run_benchmark.py
import requests, json, time, yaml, argparse
import torch
import pandas as pd
from pathlib import Path

# ----------------------------
# CLI Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--out-tag",
    type=str,
    required=True,
    help="Tag to add to result filenames (e.g., fp16, gptq, awq)"
)
args = parser.parse_args()

# ----------------------------
# Setup
# ----------------------------
VLLM_URL = "http://localhost:8001/generate"

RESULTS_DIR = Path("benchmark/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open("benchmark/prompts.json", "r") as f:
    prompts = json.load(f)

with open("benchmark/config_grid.yaml", "r") as f:
    config = yaml.safe_load(f)

results = []

# ----------------------------
# Benchmark loop
# ----------------------------
for bs in config["batch_sizes"]:
    payload = {
        "prompts": [p["text"] for p in prompts[:bs]],
        "max_new_tokens": 100,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"🔹 Querying vLLM ({args.out_tag}) with batch size {bs}...")

    try:
        start = time.time()
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        latency = time.time() - start

        output = response.json()["responses"]
        resp_text = "\n".join(output)
        print(f"vLLM ({args.out_tag}) responses:\n{resp_text}\n")

        total_tokens = sum(len(out.split()) for out in output)
        throughput = total_tokens / latency

        peak_mem = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else None
        )

        results.append({
            "engine": "vllm",
            "config": args.out_tag,
            "batch_size": bs,
            "latency_s": latency,
            "total_tokens": total_tokens,
            "throughput_tok_per_s": throughput,
            "peak_mem_GB": peak_mem,
            "status": "OK"
        })

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"❌ OOM at batch {bs} on vLLM ({args.out_tag})")
            results.append({
                "engine": "vllm",
                "config": args.out_tag,
                "batch_size": bs,
                "latency_s": None,
                "total_tokens": None,
                "throughput_tok_per_s": None,
                "peak_mem_GB": None,
                "status": "OOM"
            })
        else:
            raise

# ----------------------------
# Save results
# ----------------------------
json_path = RESULTS_DIR / f"vllm_{args.out_tag}.json"
csv_path = RESULTS_DIR / f"vllm_{args.out_tag}.csv"
md_path = RESULTS_DIR / f"vllm_{args.out_tag}.md"

with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
with open(md_path, "w") as f:
    f.write(df.to_markdown(index=False))

print(f"\n📊 Results saved to:\n- {json_path}\n- {csv_path}\n- {md_path}")