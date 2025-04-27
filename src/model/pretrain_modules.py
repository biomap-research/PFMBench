import torch.nn as nn
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'

class BaseProteinModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
    
    def construct_batch(self, data, batch_size, task_name=None):
        raise NotImplementedError
    
    def forward(self, batch):
        raise NotImplementedError

class UtilsModel:
    def __init__(self):
        super().__init__()

    def post_process_cpu(self, batch, embeddings, attention_masks, start, ends, task_type='binary_classification'):

        # sparse return
        results = []
        for i, end in enumerate(ends):
            end = int(end.item())
            embedding = embeddings[i][start:end].cpu()
            name = batch['name'][i]
            attention_mask = attention_masks[i][start:end].cpu()
            label = torch.tensor(batch['label'][i])
            
            results.append({'name': name,
                            'embedding': embedding,
                            'attention_mask': attention_mask.bool(),
                            'label': label}  )
        return results

    def pad_data(self, data, dim=0, pad_value=0, max_length=1022):
        if data.shape[dim] < max_length:
            data = self.dynamic_pad(data, [0, max_length-data.shape[dim]], dim=dim, pad_value=pad_value)
        else:
            # start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            start = 0
            data = data[start:start+max_length]
        return data

    def dynamic_pad(self, tensor, pad_size, dim=0, pad_value=0):
        # 获取原始形状
        shape = list(tensor.shape)
        num_dims = len(shape)
        
        # 生成 padding 参数
        pad = [0] * (2 * num_dims)
        prev_pad_size, post_pad_size = pad_size
        pad_index = 2 * (num_dims - dim - 1)
        pad[pad_index] = prev_pad_size  # 前面 padding
        pad[pad_index + 1] = post_pad_size  # 后面 padding

        # 应用 padding
        padded_tensor = F.pad(tensor, pad, mode="constant", value=pad_value)
        return padded_tensor
    
class ESM2Model(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.pretrain_model = AutoModelForMaskedLM.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ZOOM_PATH}/esm2_650m")
        self.max_length = max_length

    def construct_batch(self, batch):
        MAXLEN = self.max_length
        max_length_batch = max([len(sample['seq']) for sample in batch]) + 2 # +2 for <s> and </s>
        result = {
                'name': [],
                'seq': [],
                'attention_mask': [],
                'label': []}
        for sample in batch:
            seq_token = torch.tensor(self.tokenizer.encode(sample['seq']))[:MAXLEN]
            attention_mask = torch.zeros(max_length_batch)
            attention_mask[:len(seq_token)] = 1
            seq_token = self.pad_data(seq_token, dim=0, max_length=max_length_batch)
            result['name'].append(sample['name'])
            result['seq'].append(seq_token)
            result['attention_mask'].append(attention_mask)
            result['label'].append(sample['label'])

        result['seq'] = torch.stack(result['seq'], dim=0).to(self.device)
        result['attention_mask'] = torch.stack(result['attention_mask'], dim=0).to(self.device)
            
        return result

    def forward(self, batch, post_process=True, task_type='binary_classification', return_prob=False):
        attention_mask = batch['attention_mask']
        outputs = self.pretrain_model.esm(
                        batch['seq'],
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
        if return_prob:
            logits = self.pretrain_model.lm_head(outputs.last_hidden_state)
            probs = F.softmax(logits, dim=-1)
            return probs
        
        embeddings = outputs.last_hidden_state
        ends = attention_mask.sum(dim=-1)-1
        start = 1
        if post_process:
            result = self.post_process_cpu(batch, embeddings, attention_mask, start, ends, task_type=task_type)
        else:
            result = embeddings
        return result

class SmilesModel(BaseProteinModel, UtilsModel):
    def __init__(self, device, max_length=1022, **kwargs):
        super().__init__(device)


    def construct_batch(self, batch):
        result = {'smiles': []}
        for sample in batch:
            mol = Chem.MolFromSmiles(sample['smiles'])
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            smiles = torch.tensor([int(ele) for ele in list(fp.ToBitString())]).float()
            result['smiles'].append(smiles)
        return result
    
    def forward(self, batch, post_process=True, task_type='binary_classification'):
        return batch
        