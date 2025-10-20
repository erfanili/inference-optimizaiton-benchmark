from datasets import load_dataset
import json

# Load the Alpaca dataset
ds = load_dataset("yahma/alpaca-cleaned")

# Hugging Face returns a DatasetDict -> access 'train' split
train_ds = ds["train"]

# Select first 256 rows
subset = train_ds.select(range(4096))

# Build prompt list
prompts = [{"id": f"p{i+1}", "text": row["instruction"]} for i, row in enumerate(subset)]

# Save to file
with open("benchmark/prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"✅ Saved {len(prompts)} prompts to benchmark/prompts.json")

