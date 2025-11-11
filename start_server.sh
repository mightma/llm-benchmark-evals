# vllm serve QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 --tensor-parallel-size 4 --max-model-len 262144 --enable-expert-parallel
# vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size 4 --max-model-len 262144 --enable-expert-parallel

python3 -m vllm.entrypoints.openai.api_server --model Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8 --host "0.0.0.0" --port 8000 --tensor-parallel-size 4 --max-model-len 262144 --enable-expert-parallel --gpu-memory-utilization 0.9
