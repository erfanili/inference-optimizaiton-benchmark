# benchmark_vllm.py
import argparse, time, json, yaml, subprocess
from pathlib import Path
import torch
import pandas as pd
from vllm import LLM, SamplingParams

def get_gpu_memory():
    """Return GPU memory usage in GB (as float)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        mem_values = [int(x) for x in output.strip().split("\n")]
        return [round(m / 1024, 2) for m in mem_values]  # GB
    except Exception as e:
        print(f"⚠️ Failed to get GPU memory: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM under one config")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "gptq", "awq", "bitsandbytes"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--out-tag", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    RESULTS_DIR = Path("benchmark/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open("benchmark/prompts.json", "r") as f:
        prompts = json.load(f)

    with open("benchmark/config_grid.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"🚀 Loading vLLM {args.model} | dtype={args.dtype} | quant={args.quantization}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        disable_paged_attention=True,         # <-----
        # disable_cuda_graph=True,              # <-----
        # max_num_batched_tokens=8192,          # <-----
    )

    results = []
    for bs in config["batch_sizes"]:
        payload = [p["text"] for p in prompts[:bs]]
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"🔹 Running batch size {bs}...")
        start = time.time()
        outputs = llm.generate(payload, sp)
        latency = time.time() - start

        responses = [out.outputs[0].text for out in outputs]
        total_tokens = sum(len(r.split()) for r in responses)
        throughput = total_tokens / latency

        # Use nvidia-smi for memory tracking
        mem_used = get_gpu_memory()
        peak_mem = max(mem_used) if mem_used else None

        results.append({
            "engine": "vllm",
            "config": args.out_tag,
            "batch_size": bs,
            "latency_s": latency,
            "total_tokens": total_tokens,
            "throughput_tok_per_s": throughput,
            "peak_mem_GB": peak_mem,
        })

    json_path = RESULTS_DIR / f"vllm_{args.out_tag}.json"
    csv_path = RESULTS_DIR / f"vllm_{args.out_tag}.csv"
    md_path = RESULTS_DIR / f"vllm_{args.out_tag}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))

    print(f"✅ Saved results to:\n- {json_path}\n- {csv_path}\n- {md_path}")

if __name__ == "__main__":
    main()
