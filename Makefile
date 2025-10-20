.PHONY: serve-vlllm bench

serve-vllm:
	python3 serve/api/server.py --backend vllm

bench:
	echo "Benchmarks will run here"

clean:
	find . -name '.pyc' -delete