vllm serve QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 --tensor-parallel-size 4 --max-model-len 262144 --enable-expert-parallel
# vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size 4 --max-model-len 262144 --enable-expert-parallel
