
# Obtain residue embeddings
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer  = AutoTokenizer.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/proteinglm-1b-mlm", trust_remote_code=True, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/proteinglm-1b-mlm",  trust_remote_code=True, torch_dtype=torch.bfloat16)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

seq = 'MILMCQHFSGQFSKYFLAVSSDFCHFVFPIILVSHVNFKQMKRKGFALWNDRAVPFTQGIFTTVMILLQYLHGTG'
output = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
with torch.inference_mode():
    inputs = {"input_ids": output["input_ids"].cuda(), "attention_mask": output["attention_mask"].cuda()}
    output_embeddings = model(**inputs, output_hidden_states=True, return_last_hidden_state=True).hidden_states[:-1, 0] # get rid of the <eos> token

print(output_embeddings.shape) # (1, 2048)

