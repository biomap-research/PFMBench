import torch
from torch.utils.data import Dataset
import pandas as pd
from src.data.protein import Protein
from transformers import AutoTokenizer
import torch.nn.functional as F
from src.utils.utils import pmap_multi
from src.data.esm.sdk.api import ESMProtein
from sklearn.preprocessing import MultiLabelBinarizer


def read_data(aa_seq, pdb_path, label, unique_id, task_type, num_classes, smiles):
    try:
        if unique_id is None:
            unique_id = str(hash(aa_seq))
            
        if task_type == "multi_labels_classification":
            mlb = MultiLabelBinarizer(classes=range(int(num_classes)))
            label = str(label)
            label = mlb.fit_transform([[int(ele) for ele in label.split(",")]]).flatten().tolist()
        
        if task_type == "contact":
            label = torch.load(label, weights_only=True) # diable warning
        
        if task_type == "residual_classification":
            label = torch.tensor(list(map(int, label.strip('[]').replace('\n', ' ').split())))
            
        name = str(hash(pdb_path))
        if pdb_path is not None:
            # 解析 pdb 文件，unique_id 作为结构的 id
            if "|" not in pdb_path:
                structure = ESMProtein.from_pdb(pdb_path)
                return {
                    'name':name, 
                    'seq': structure.sequence, 
                    'X': structure.coordinates, 
                    'label': label, 
                    'unique_id': unique_id, 
                    'pdb_path': pdb_path,
                    'smiles': smiles
                }
            # X, C, S = structure.to_XCS(all_atom=True)
            # X, C, S = X[0], C[0], S[0]
            else: # PPI
                structures = []
                for _pdb_path in pdb_path.split("|"):
                    structure = ESMProtein.from_pdb(_pdb_path)
                    structures.append(structure.coordinates)
                return {
                    'name':name, 
                    'seq': aa_seq, 
                    'X': structures, # coords is organized as a list here
                    'label': label, 
                    'unique_id': unique_id, 
                    'pdb_path': pdb_path,
                    'smiles': smiles
                }
        else:
            return {
                'name':name, 
                'seq': aa_seq, 
                'X': None, 
                'label': label, 
                'unique_id': unique_id, 
                'pdb_path': pdb_path,
                'smiles': smiles
            }
    except:
        return None


class ProteinDataset(Dataset):
    def __init__(self, csv_file, pretrain_model_name='esm2_650m', max_length=1022, pretrain_model_interface=None, task_name='pretrain', task_type='classification', num_classes=None):
        """
        Args:
            csv_file (str): CSV 文件路径，文件中包含蛋白质序列和结构等信息。
        """
        self.max_length = max_length
        self.pretrain_model_name = pretrain_model_name
        self.task_name = task_name
        self.task_type = task_type
        self.num_classes = num_classes
        
        # 读取 CSV 数据
        csv_data = pd.read_csv(csv_file)

        path_list = []
        for i in range(len(csv_data)):
            path_list.append((csv_data.iloc[i].get('aa_seq'), csv_data.iloc[i].get('pdb_path'), csv_data.iloc[i]['label'], csv_data.iloc[i].get('unique_id'), task_type, num_classes, csv_data.iloc[i].get('smiles'))) #列表里面必须是元组，不然debug模式下并行加载数据会报错
        
        # path_list = path_list[:10] # this is for fast debug, please comment it in production
        self.data = pmap_multi(read_data, path_list, n_jobs=-1)
        self.data = [d for d in self.data if d is not None]
        self.max_length = min(self.max_length, max([len(d['seq']) for d in self.data])+2)
        self.pretrain_model_interface = pretrain_model_interface

        if pretrain_model_interface is not None:
            self.data = pretrain_model_interface.inference_datasets(self.data, task_name=self.task_name)

        print(f"ProteinDataset: {len(self.data)} samples loaded.")
        
    def __len__(self):
        return len(self.data)
    
    def pad_data(self, data, dim=0, pad_value=0, max_length=1022):
        if data.shape[dim] < max_length:
            data = dynamic_pad(data, [0, max_length-data.shape[dim]], dim=dim, pad_value=pad_value)
        else:
            start = 0
            data = data[start:start+max_length]
        return data
    
    def __getitem__(self, idx):
        if self.pretrain_model_interface is not None:
            max_length_batch = self.max_length
            name = self.data[idx]['name']
            embedding = self.pad_data(self.data[idx]['embedding'], dim=0, pad_value=0, max_length=max_length_batch)
            attention_mask = self.pad_data(self.data[idx]['attention_mask'], dim=0, pad_value=0, max_length=max_length_batch)
            label = self.data[idx]['label']
            

            if self.task_type == 'binary_classification':
                label = label[None].float()
            if self.task_type == 'contact':
                label = (label == 0).int()
                label = F.pad(label, [0, max_length_batch-label.shape[0], 0, max_length_batch-label.shape[0]])
            if self.task_type == 'residual_classification': 
                label = F.pad(label, [0, max_length_batch-label.shape[0]])
            
            result = {
                'name': name,
                'embedding': embedding,
                'attention_mask': attention_mask,
                'label': label,
            }
            
            if self.data[idx].get('smiles') is not None:
                smiles = self.data[idx]['smiles']
                result['smiles'] = smiles
                
            return result
        else:
            seq = self.data[idx]['seq']
            label = self.data[idx]['label']
            X = self.data[idx]['X']
            # X = dynamic_pad(X, [1, 1], dim=0, pad_value=0) ## todo, 这里需要根据实际情况进行修改, 如果sequence tokenizer对数据加上了EOS, BOS，则需要这一行
            X = self.pad_data(X, dim=0)
            # if self.pretrain_model_name == 'prollama':
            mask = torch.ones(self.max_length)
            seq_token = torch.tensor(self.tokenizer.encode(seq))
            mask[:len(seq_token)] = 0
            seq_token = self.pad_data(seq_token, dim=0)
            smiles = self.data[idx].get('smiles')
            sample = {
                'seq': seq_token,
                'label': torch.tensor(label),
                'mask': mask,
                'coords': X,
                'smiles': smiles
            }

            return sample



def dynamic_pad(tensor, pad_size, dim=0, pad_value=0):
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

# 示例用法
if __name__ == "__main__":
    dataset = ProteinDataset("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/fold_prediction/fold_prediction_with_glmfold_structure_test.csv")
    sample = dataset[0]
    print(sample['coords'].shape)
    print(sample['chain'])
    print(sample['sequence'])

