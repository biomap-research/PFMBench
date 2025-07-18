from transformers import AutoTokenizer, AutoModelForMaskedLM
from ProSST.prosst.structure.quantizer import PdbQuantizer
import torch


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )


prosst_2048_weight_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProSST/prosst_2048_weight"
device = "cuda:1"

model = AutoModelForMaskedLM.from_pretrained(prosst_2048_weight_path, trust_remote_code=True).to(device)

aa_seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGLDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"
pdb_path = "/nfs_beijing/onesystem/share_data/pdbs/glmfold/2d1b76cead45d9d7783547e6ce02d918/ranked_unrelax_0.pdb"

tokenizer = AutoTokenizer.from_pretrained(prosst_2048_weight_path, trust_remote_code=True)
tokenized_results = tokenizer([aa_seq], return_tensors="pt")
input_ids = tokenized_results["input_ids"].to(device)
attention_mask = tokenized_results["attention_mask"].to(device)

processor = PdbQuantizer(structure_vocab_size=2048)
ss_seq = processor(pdb_path, return_residue_seq=False)['2048']['ranked_unrelax_0.pdb']["struct"]
ss_input_ids = tokenize_structure_sequence(ss_seq).to(device)



outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    ss_input_ids=ss_input_ids,
    output_hidden_states=True
)

print(outputs.hidden_states[-1].shape) # [1, L, 768]
