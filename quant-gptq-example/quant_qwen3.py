from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# 模型ID
model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
quant_path = "Qwen3-30B-A3B-Instruct-2507-gptqmodel-int8"

# 准备校准数据集
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]

# 配置INT8量化 (8bits)
quant_config = QuantizeConfig(bits=8, group_size=128)

# 加载模型
model = GPTQModel.load(model_id, quant_config)

# 执行量化 (根据GPU显存调整batch_size)
model.quantize(calibration_dataset, batch_size=1)

# 保存量化后的模型
model.save(quant_path)

