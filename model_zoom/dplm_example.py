import torch
from byprot.models.lm.dplm import DiffusionProteinLanguageModel


model_name = "YOUR WEIGHTS PATH"
dplm = DiffusionProteinLanguageModel.from_pretrained(model_name)
params = torch.load(f'{model_name}/pytorch_model.bin')
dplm.net.load_state_dict(params)

toy_data = [
    "MKTLLLTLVVVTIVCLDLGYT",  # 序列1
    "GAVLFGYTPGGLAAGALYGVK"   # 序列2
]

inputs = dplm.net.tokenizer(toy_data, return_tensors="pt", padding=True, truncation=True)

outputs = dplm.net.esm(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True,
        )

embeddings = outputs.last_hidden_state

# 打印 embedding 形状
print(f"Embedding shape for sequence 1: {embeddings[0].shape}")
print(f"Embedding shape for sequence 2: {embeddings[1].shape}")
