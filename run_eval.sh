#!/bin/bash

python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507 --benchmark ifeval --num-responses 3
python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507 --benchmark aime25 --num-responses 50 --presence-penalty 1.5 --max-concurrent 1
python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507 --benchmark mmlu_pro

python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --benchmark ifeval --num-responses 3
python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --benchmark aime25 --num-responses 50 --presence-penalty 1.5 --max-concurrent 1
python3 main.py evaluate --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --benchmark mmlu_pro

python3 main.py evaluate --model QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 --benchmark ifeval --num-responses 3
python3 main.py evaluate --model QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 --benchmark aime25 --num-responses 50 --presence-penalty 1.5 --max-concurrent 1
python3 main.py evaluate --model QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 --benchmark mmlu_pro

python3 main.py evaluate --model Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8 --benchmark aime25 --num-responses 50 --presence-penalty 1.5 --max-concurrent 1
python3 main.py evaluate --model Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8 --benchmark ifeval --num-responses 3
python3 main.py evaluate --model Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8 --benchmark mmlu_pro
