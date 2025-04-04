import torch
from torch.utils.data import Dataset
import pandas as pd
from src.data.protein import Protein
from transformers import AutoTokenizer
import torch.nn.functional as F
from src.utils.utils import pmap_multi


class ProteinDataset(Dataset):
    def __init__(self, csv_file, pretrain_model_name='esm2_650m', max_length=1022):
        """
        Args:
            csv_file (str): CSV 文件路径，文件中包含蛋白质序列和结构等信息。
        """
        self.max_length = max_length
        self.pretrain_model_name = pretrain_model_name
        
        # 读取 CSV 数据
        csv_data = pd.read_csv(csv_file)
        
        def read_data(pdb_path, label):
            try:
                # 解析 pdb 文件，unique_id 作为结构的 id
                structure = Protein.from_PDB(pdb_path)
                X, C, S = structure.to_XCS(all_atom=True)
                X, C, S = X[0], C[0], S[0]
                seq = structure.sequence()
                return {'seq': seq, 'X': X, 'C': C, 'S': S, 'label': label}
            except:
                return None
        
        path_list = []
        for i in range(len(csv_data)):
            path_list.append([csv_data.iloc[i]['pdb_path'], csv_data.iloc[i]['label']])
        
        self.data = pmap_multi(read_data, path_list)
        self.data = [d for d in self.data if d is not None]
        
        if pretrain_model_name == 'esm2_650m':
            self.tokenizer = AutoTokenizer.from_pretrained("model_zoom/esm2_650m")
        
    def __len__(self):
        return len(self.data)
    
    def pad_data(self, data, dim=0, pad_value=0):
        if data.shape[0] < self.max_length:
            data = dynamic_pad(data, [0, self.max_length-data.shape[0]], dim=dim, pad_value=pad_value)
        else:
            start = torch.randint(0, data.shape[0]-self.max_length+1, (1,)).item()
            data = data[start:start+self.max_length]
        return data
    
    def __getitem__(self, idx):
        seq = self.data[idx]['seq']
        label = self.data[idx]['label']
        X = self.data[idx]['X']
        C = self.data[idx]['C']
        S = self.data[idx]['S']
        
        
        if self.pretrain_model_name == 'esm2_650m':
            mask = torch.ones(self.max_length)
            seq_token = torch.tensor(self.tokenizer.encode(seq))
            mask[:len(seq_token)] = 0
            # BOS = seq_token[0]
            # EOS = seq_token[-1]
            # seq_token = seq_token[1:-1]
            
            seq_token = self.pad_data(seq_token, dim=0)
            X = dynamic_pad(X, [1, 1], dim=0, pad_value=0)
            C = dynamic_pad(C, [1, 1], dim=0, pad_value=-1)
            
            X = self.pad_data(X, dim=0)
            C = self.pad_data(C, dim=0, pad_value=-1)

        
        sample = {
            'seq': seq_token,
            'coords': X,
            'chain': C,
            'label': torch.tensor(label),
            'mask': mask,
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
