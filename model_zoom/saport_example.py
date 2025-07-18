from transformers import EsmTokenizer, EsmForMaskedLM
from src.data.protein import Protein
import mini3di
encoder = mini3di.Encoder()

model_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/SaPort/ckpt" # Note this is the directory path of SaProt, not the ".pt" file
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)

#################### Example ####################
device = "cuda"
model.to(device)

data = Protein.from_PDB('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/N128/8rk2A.pdb')
X, C, S = data.to_XCS(all_atom=True)
N,  CA, C,  O,  CB = X[0,:, 0], X[0,:, 1], X[0,:, 2], X[0,:, 3], X[0,:, 4]
states = encoder.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
struct_sequence = encoder.build_sequence(states)
AA_sequence = data.sequence()

merged_seq = ''.join(a + b.lower() for a, b in zip(AA_sequence, struct_sequence))

# seq = "M#EvVpQpL#VyQdYaKv" # Here "#" represents lower plDDT regions (plddt < 70)
tokens = tokenizer.tokenize(merged_seq)
print(tokens)

inputs = tokenizer(merged_seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)
print(outputs.logits.shape)

"""
['M#', 'Ev', 'Vp', 'Qp', 'L#', 'Vy', 'Qd', 'Ya', 'Kv']
torch.Size([1, 11, 446])
"""