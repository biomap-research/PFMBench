import os
os.environ["HF_MODULES_CACHE"] = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom"
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor
from graphein.protein.features.nodes.dssp import  phi, psi, asa, rsa, secondary_structure
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_distance_threshold,
)
from model_zoom.prot2text.pdb2graph import PDB2Graph, download_alphafold_structure



prot2text_weight = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/prot2text"
tokenizer = AutoTokenizer.from_pretrained(prot2text_weight, 
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(prot2text_weight, 
                                             trust_remote_code=True)

config = {
    "node_metadata_functions": [
        amino_acid_one_hot, 
        expasy_protein_scale,
        meiler_embedding,
        hydrogen_bond_acceptor, 
        hydrogen_bond_donor
    ],
    "edge_construction_functions": [
        add_peptide_bonds,
        add_hydrogen_bond_interactions,
        partial(add_distance_threshold, long_interaction_threshold=3, threshold=10.)
    ],
    "graph_metadata_functions":[
        asa,
        phi, 
        psi, 
        secondary_structure, 
        rsa
    ],
    "dssp_config": DSSPConfig()
}
config = ProteinGraphConfig(**config)

function = model.generate_protein_description(protein_pdbID='Q10MK9', 
                                              tokenizer=tokenizer, 
                                              device='cuda' # replace with 'mps' to run on a Mac device
                                              )

print(function)

gpdb = PDB2Graph(root = PATH_TO_DATA, output_folder = OUTPUT_FOLDER, config=config, n_processors=1).create_pyg_graph(structure_filename)
                seq = esmtokenizer(gpdb.sequence, add_special_tokens=True, truncation=True, max_length=1021, padding='max_length',return_tensors="pt") #
                torch.save(gpdb, graph_filename)
                gpdb.edge_type = [np.array(gpdb.edge_type.transpose(0,1))]
                gpdb.encoder_input_ids = seq['input_ids']
                gpdb.attention_mask = seq['attention_mask']



