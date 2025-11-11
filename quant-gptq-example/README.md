An example to quanitize mdoel with GPTQModel

* Install Dependencies:
```bash
uv add -r requirements.txt
```

* Quantize Model
```bash
uv run quant_qwen.py
```
The quanized model will be saved at "Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8"

