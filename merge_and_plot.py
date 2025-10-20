# merge_and_plot.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("benchmark/results")

# ----------------------------
# Merge results
# ----------------------------
all_results = []
for f in RESULTS_DIR.glob("vllm_*.json"):
    with open(f) as infile:
        all_results.extend(json.load(infile))

df = pd.DataFrame(all_results)
df = df.sort_values(by=["config", "batch_size"])
df.to_csv(RESULTS_DIR / "vllm_merged.csv", index=False)

with open(RESULTS_DIR / "vllm_merged.json", "w") as f:
    json.dump(all_results, f, indent=2)

with open(RESULTS_DIR / "vllm_merged.md", "w") as f:
    f.write(df.to_markdown(index=False))

print(f"✅ Merged {len(df)} results into vllm_merged.json/csv/md")

# ----------------------------
# Plot Latency vs Batch Size
# ----------------------------
plt.figure(figsize=(8, 5))
for config, group in df.groupby("config"):
    plt.plot(group["batch_size"], group["latency_s"], marker="o", label=config)

plt.xlabel("Batch Size")
plt.ylabel("Latency (s)")
plt.title("Latency vs Batch Size (vLLM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vllm_latency_vs_batch.png")
plt.close()

# ----------------------------
# Plot Throughput vs Batch Size
# ----------------------------
plt.figure(figsize=(8, 5))
for config, group in df.groupby("config"):
    plt.plot(group["batch_size"], group["throughput_tok_per_s"], marker="o", label=config)

plt.xlabel("Batch Size")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs Batch Size (vLLM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vllm_throughput_vs_batch.png")
plt.close()

# ----------------------------
# Plot Memory vs Batch Size
# ----------------------------
if "peak_mem_GB" in df.columns and df["peak_mem_GB"].notna().any():
    plt.figure(figsize=(8, 5))
    for config, group in df.groupby("config"):
        plt.plot(group["batch_size"], group["peak_mem_GB"], marker="o", label=config)

    plt.xlabel("Batch Size")
    plt.ylabel("Peak GPU Memory (GB)")
    plt.title("Peak Memory vs Batch Size (vLLM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vllm_memory_vs_batch.png")
    plt.close()

print("📊 Plots saved to benchmark/results/")
