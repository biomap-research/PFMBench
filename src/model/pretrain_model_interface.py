import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.protein_dataset import dynamic_pad
from src.data.esm.sdk.api import LogitsConfig
import sys; sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')

MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'

class PretrainModelInterface(nn.Module):
    """
    Interface for pre-trained models.
    """

    def __init__(self, pretrain_model_name, batch_size = 64, max_length = 1022, device = 'cuda'):
        super(PretrainModelInterface, self).__init__()
        self.pretrain_model_name = pretrain_model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """
        Setup the pre-trained model based on the specified name.
        """
        if self.pretrain_model_name == 'esm2_650m':
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self.pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m")
        elif self.pretrain_model_name == 'esm3_1.4b':
            from model_zoom.esm.models.esm3 import ESM3
            self.pretrain_model = ESM3.from_pretrained("esm3-sm-open-v1", ).to("cuda")
            
        elif self.pretrain_model_name == 'esmc_600m':
            pass
        elif self.pretrain_model_name == 'procyon':
            pass
        elif self.pretrain_model_name == 'progen2':
            pass
        elif self.pretrain_model_name == 'prostt5':
            pass
        elif self.pretrain_model_name == 'protgpt2':
            pass
        elif self.pretrain_model_name == 'protrek':
            pass
        elif self.pretrain_model_name == 'saport':
            pass
        else:
            raise ValueError(f"Unknown pretrain model name: {self.pretrain_model_name}")
        

    def forward(self, x) -> torch.Tensor:
        
        if self.pretrain_model_name == 'esm2_650m':
            # Forward pass through the pre-trained model
            outputs = self.pretrain_model.esm(
                        x['seq'],
                        attention_mask=x['attention_mask'],
                        return_dict=True,
                    )
            embeddings = outputs.last_hidden_state
            
        if self.pretrain_model_name == 'esm3_1.4b':
            output = self.pretrain_model.logits(
                x['protein_tensor'],
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
            embeddings = output.embeddings
            embeddings = F.pad(embeddings, (0, 0, 0, self.max_length-embeddings.shape[1], 0, 0), value=0)
        
        if self.pretrain_model_name == 'esmc_600m':
            pass
        
        if self.pretrain_model_name == 'procyon':
            pass
        
        if self.pretrain_model_name == 'progen2':
            pass
        
        if self.pretrain_model_name == 'prostt5':
            pass
        
        if self.pretrain_model_name == 'protgpt2':
            pass
        
        if self.pretrain_model_name == 'protrek':
            pass
        
        if self.pretrain_model_name == 'saport':
            pass
        
        
       
        return embeddings
    
    def pad_data(self, data, dim=0, pad_value=0):
        if data.shape[0] < self.max_length:
            data = dynamic_pad(data, [0, self.max_length-data.shape[0]], dim=dim, pad_value=pad_value)
        else:
            start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            data = data[start:start+self.max_length]
        return data
            
    def construct_batch(self, data, batch_size):
        """
        Constructs batches of data.

        Args:
            data (list): List of data samples.
            batch_size (int): Size of each batch.

        Yields:
            dict: A batch of data.
        """
        for i in range(0, len(data), batch_size):
            if self.pretrain_model_name == 'esm2_650m':
                name_batch, X_batch, S_batch, mask_batch, label_batch = [], [], [], [], []
                for sample in data[i:i + batch_size]:
                    seq = sample['seq']
                    label = sample['label']
                    X = sample['X']
                    attention_mask = torch.zeros(self.max_length)
                    
                    seq_token = torch.tensor(self.tokenizer.encode(seq))
                    seq_token = self.pad_data(seq_token, dim=0)
                    
                    X = dynamic_pad(X, [1, 1], dim=0, pad_value=0)
                    X = self.pad_data(X, dim=0)
                    
                    attention_mask[:len(seq_token)] = 1
                    
                    S_batch.append(seq_token)
                    X_batch.append(X)
                    mask_batch.append(attention_mask)
                    label_batch.append(torch.tensor(label))
                    name_batch.append(sample['name'])
        
                yield {
                    'name': name_batch,
                    'seq': torch.stack(S_batch).to(self.device),
                    'X': torch.stack(X_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                    'label': torch.stack(label_batch).to(self.device),
                }
            
            if self.pretrain_model_name == 'esm3_1.4b':
                from model_zoom.esm.utils.misc import stack_variable_length_tensors
                from model_zoom.esm.utils.sampling import _BatchedESMProteinTensor
                from model_zoom.esm.utils import encoding
                model = self.pretrain_model
                name_batch, sequence_list, coordinates_list, label_batch = [], [], [], []
                seq_tokenizer = model.tokenizers.sequence
                struct_tokenizer = model.tokenizers.structure
                pad = model.tokenizers.sequence.pad_token_id
                for sample in data[i:i + batch_size]:
                    sequence_list.append(sample['seq'])
                    coordinates_list.append(sample['X'])
                    name_batch.append(sample['name'])
                    label_batch.append(torch.tensor(sample['label']))
                
                sequence_tokens = stack_variable_length_tensors(
                    [
                        encoding.tokenize_sequence(x, seq_tokenizer, add_special_tokens=True)
                        for x in sequence_list
                    ],
                    constant_value=pad,
                ).to(self.device)
                
                structure_tokens_batch = []
                coordinates_batch = []
                mask_batch = []
                for coordinate in coordinates_list:
                    mask = torch.zeros(self.max_length)
                    coordinates, plddt, structure_token = encoding.tokenize_structure(coordinate, model.get_structure_encoder(), struct_tokenizer, add_special_tokens=True)
                    mask[:coordinates.shape[0]] = 1
                    structure_tokens_batch.append(structure_token)
                    coordinates_batch.append(coordinates)
                    mask_batch.append(mask)
                
                structure_tokens_batch = stack_variable_length_tensors(
                            structure_tokens_batch,
                            constant_value=pad,
                        ).to(self.device)
                                
                coordinates_batch = stack_variable_length_tensors(
                            coordinates_batch,
                            constant_value=pad,
                        ).to(self.device)
                
                protein_tensor = _BatchedESMProteinTensor(sequence=sequence_tokens,
                                                          structure=structure_tokens_batch, coordinates=coordinates_batch).to(self.device)

                yield {
                    'name': name_batch,
                    'protein_tensor': protein_tensor,
                    'label': torch.stack(label_batch).to(self.device),
                    'attention_mask': torch.stack(mask_batch).to(self.device)==1,
                }
                    
            
            if self.pretrain_model_name == 'esmc_600m':
                pass
            
            if self.pretrain_model_name == 'procyon':
                pass
            
            if self.pretrain_model_name == 'progen2':
                pass
            
            if self.pretrain_model_name == 'prostt5':
                pass
            
            if self.pretrain_model_name == 'protgpt2':
                pass
            
            if self.pretrain_model_name == 'protrek':
                pass
            
            if self.pretrain_model_name == 'saport':
                pass
            
            
    
    def inference_datasets(self, data):
        self.pretrain_model.eval()
        with torch.no_grad():
            proccessed_data = []
            for batch in self.construct_batch(data, self.batch_size):
                embeddings = self.forward(batch)
                for i in range(len(batch['name'])):
                    proccessed_data.append({
                        'name': batch['name'][i],
                        'attention_mask': batch['attention_mask'][i].cpu(),
                        'label': batch['label'][i].cpu(),
                        'embedding': embeddings[i].cpu()
                    })
        return proccessed_data
            


