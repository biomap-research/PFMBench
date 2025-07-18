import torch
from src.data.protein import Protein
from model_zoom.ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from model_zoom.ProTrek.utils.foldseek_util import get_struc_seq
import mini3di
encoder = mini3di.Encoder()

# Load model
config = {
    "protein_config": "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProTrek/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
    "text_config": "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProTrek/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProTrek/ProTrek_650M_UniRef50/foldseek_t30_150M",
    "load_protein_pretrained": False,
    "load_text_pretrained": False,
    "from_checkpoint": "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/ProTrek/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
}

device = "cuda"
model = ProTrekTrimodalModel(**config).eval().to(device)

# Load protein and text
pdb_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/N128/8rk2A.pdb"
# seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"])["A"]
# aa_seq = seqs[0]
# foldseek_seq = seqs[1].lower()

data = Protein.from_PDB(pdb_path)
X, C, S = data.to_XCS(all_atom=True)
N,  CA, C,  O,  CB = X[0,:, 0], X[0,:, 1], X[0,:, 2], X[0,:, 3], X[0,:, 4]
states = encoder.encode_atoms(ca = CA.numpy(), cb = CB.numpy(), n = N.numpy(), c = C.numpy())
foldseek_seq = encoder.build_sequence(states).lower()
aa_seq = data.sequence()
text = "Replication initiator in the monomeric form, and autogenous repressor in the dimeric form."

with torch.no_grad():
    # Obtain protein sequence embedding
    seq_embedding = model.get_protein_repr([aa_seq])
    print("Protein sequence embedding shape:", seq_embedding.shape)
    
    # Obtain protein structure embedding
    struc_embedding = model.get_structure_repr([foldseek_seq])
    print("Protein structure embedding shape:", struc_embedding.shape)
    
    # Obtain text embedding
    text_embedding = model.get_text_repr([text])
    print("Text embedding shape:", text_embedding.shape)
    
    # Calculate similarity score between protein sequence and structure
    seq_struc_score = seq_embedding @ struc_embedding.T / model.temperature
    print("Similarity score between protein sequence and structure:", seq_struc_score.item())

    # Calculate similarity score between protein sequence and text
    seq_text_score = seq_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein sequence and text:", seq_text_score.item())
    
    # Calculate similarity score between protein structure and text
    struc_text_score = struc_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein structure and text:", struc_text_score.item())
   

"""
Protein sequence embedding shape: torch.Size([1, 1024])
Protein structure embedding shape: torch.Size([1, 1024])
Text embedding shape: torch.Size([1, 1024])
Similarity score between protein sequence and structure: 28.506675720214844
Similarity score between protein sequence and text: 17.842409133911133
Similarity score between protein structure and text: 11.866174697875977
"""