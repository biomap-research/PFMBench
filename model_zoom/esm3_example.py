from model_zoom.esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.sdk.api import (
    ESMCInferenceClient,
    ESMProtein,
    ESMProteinTensor,
    ForwardTrackData,
    LogitsConfig,
    LogitsOutput,
)
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.misc import stack_variable_length_tensors
from esm.utils import encoding

model = ESM3.from_pretrained("esm3-sm-open-v1", ).to("cuda")

protein1 = ESMProtein.from_pdb('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/N128/8rk2A.pdb')
protein2 = ESMProtein.from_pdb('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/N128/8rkfA.pdb')
pad = model.tokenizers.sequence.pad_token_id
sequence_list = [protein1.sequence, protein2.sequence]
coordinates_list = [protein1.coordinates, protein2.coordinates]

seq_tokenizer = model.tokenizers.sequence
struct_tokenizer = model.tokenizers.structure

sequence_tokens = stack_variable_length_tensors(
            [
                encoding.tokenize_sequence(x, seq_tokenizer, add_special_tokens=True)
                for x in sequence_list
            ],
            constant_value=pad,
        ).to(next(model.parameters()).device)


structure_tokens_batch = []
coordinates_batch = []
for coordinate in coordinates_list:
    coordinates, plddt, structure_token = encoding.tokenize_structure(coordinate, model.get_structure_encoder(), struct_tokenizer, add_special_tokens=True)
    structure_tokens_batch.append(structure_token)
    coordinates_batch.append(coordinates)
    
structure_tokens_batch = stack_variable_length_tensors(
            structure_tokens_batch,
            constant_value=pad,
        ).to(next(model.parameters()).device)
                
coordinates_batch = stack_variable_length_tensors(
            coordinates_batch,
            constant_value=pad,
        ).to(next(model.parameters()).device)


protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens_batch, coordinates=coordinates_batch).to(
            next(model.parameters()).device
        )

output = model.logits(
        protein_tensor,
        LogitsConfig(
            sequence=True,
            structure=True,
            secondary_structure=True,
            sasa=True,
            function=True,
            residue_annotations=True,
            return_embeddings=True,
        ),
    )

print(output.embeddings.shape)