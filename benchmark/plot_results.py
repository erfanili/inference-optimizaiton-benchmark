#plot results
import json
import matplotlib.pyplot as plt

# Load results
with open("benchmark/results/grid_results.json") as f:
    data = json.load(f)

# Sort results by engine and batch_size
data.sort(key=lambda x: (x["engine"], x["batch_size"]))

# Grouped values
batch_sizes = sorted(set(d["batch_size"] for d in data))
engines = sorted(set(d["engine"] for d in data))

latency_data = {engine: [] for engine in engines}
throughput_data = {engine: [] for engine in engines}

for bs in batch_sizes:
    for engine in engines:
        match = next((d for d in data if d["engine"] == engine and d["batch_size"] == bs), None)
        if match:
            latency_data[engine].append(match["latency"])
            throughput_data[engine].append(match["throughput"])
        else:
            latency_data[engine].append(None)
            throughput_data[engine].append(None)

# Plot latency
plt.figure(figsize=(8, 5))
for engine in engines:
    plt.plot(batch_sizes, latency_data[engine], marker='o', label=engine)
plt.xlabel("Batch Size")
plt.ylabel("Latency (s)")
plt.title("Latency vs Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("benchmark/results/latency_vs_batch.png")
plt.close()

# Plot throughput
plt.figure(figsize=(8, 5))
for engine in engines:
    plt.plot(batch_sizes, throughput_data[engine], marker='o', label=engine)
plt.xlabel("Batch Size")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("benchmark/results/throughput_vs_batch.png")
plt.close()

print("✅ Plots saved to benchmark/results/")
