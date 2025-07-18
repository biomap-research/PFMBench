import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


tokenizer = AutoTokenizer.from_pretrained("model_zoom/esm2_3b")
pretrain_model = AutoModelForMaskedLM.from_pretrained("model_zoom/esm2_3b")

toy_data = [
    "MKTLLLTLVVVTIVCLDLGYT",  # 序列1
    "GAVLFGYTPGGLAAGALYGVK"   # 序列2
]

# 将蛋白质序列进行分词
inputs = tokenizer(toy_data, return_tensors="pt", padding=True, truncation=True)

# 进行推理并获取模型输出
with torch.no_grad():
    outputs = pretrain_model.esm(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_dict=True,
                )
    embeddings = outputs.last_hidden_state

# 打印 embedding 形状
print(f"Embedding shape for sequence 1: {embeddings[0].shape}")
print(f"Embedding shape for sequence 2: {embeddings[1].shape}")