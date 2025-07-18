import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence='MGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGYR\nVNVEGVAQLLELYARDILAEGRLVQLLPEWAD'

#Convert the sequence to a string like this
#(note we have to introduce new line characters every 60 amino acids,
#following the FASTA file format).

sequence = "<|endoftext|>\nMGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGY\nRVNVEGVAQLLELYARDILAEGRLVQLLPEWAD\n<|endoftext|>"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProtGPT2")
model = AutoModelForCausalLM.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProtGPT2")
model = model.to(device)

input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
input_ids = input_ids.to(device)
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]

print(logits.shape)

