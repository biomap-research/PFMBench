import torch
import torch.nn as nn
import os
from tqdm import tqdm
from src.model.pretrain_modules import ESM2Model, SmilesModel
import sys; sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')
MODEL_ZOOM_PATH = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom'
os.environ["TOKENIZERS_PARALLELISM"] = "true"



class PretrainModelInterface(nn.Module):
    """
    setup_model: 注册预训练模型
    construct_batch: 构建模型的输入batch, 需要进行padding。如果序列加上EOS, BOS, 结构也需要在这里进行padding
    forward: 调用不同模型提取蛋白质氨基酸级别的embedding, 在后处理阶段去除EOS, BOS embedding，只返回氨基酸embedding
    """

    def __init__(self, pretrain_model_name, batch_size = 64, max_length = 1022, device = 'cuda', sequence_only=False, task_type=None):
        super(PretrainModelInterface, self).__init__()
        self.pretrain_model_name = pretrain_model_name
        self.sequence_only = sequence_only
        self.batch_size = batch_size
        self.task_type = task_type
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """
        Setup the pre-trained model based on the specified name.
        """
        device = 'cuda'
        self.smiles_model = SmilesModel(device)
        if self.pretrain_model_name == 'esm2_650m':
            self.pretrain_model = ESM2Model(device)
        
    @torch.no_grad()
    def inference_datasets(self, data, task_name=None):
        self.pretrain_model.eval()
        proccessed_data = []
        for i in tqdm(range(0, len(data), self.batch_size), desc='Extracting embeddings'):
            if "|" in data[0]['seq']: # PPI case
                samples_A, samples_B = [], []
                for sample in data[i:i + self.batch_size]:
                    sample_A = {key: value for key, value in sample.items() if key != 'seq'}
                    sample_B = {key: value for key, value in sample.items() if key != 'seq'}
                    sample_A['seq'] = sample['seq'].split('|')[0]
                    sample_B['seq'] = sample['seq'].split('|')[1]
                    samples_A.append(sample_A)
                    samples_B.append(sample_B)

                batch_A = self.pretrain_model.construct_batch(samples_A)
                batch_B = self.pretrain_model.construct_batch(samples_B)
                results_A = self.pretrain_model(batch_A, task_type=self.task_type)
                results_B = self.pretrain_model(batch_B, task_type=self.task_type)
                results = []
                for idx in range(len(results_A)):
                    result = {}
                    for key in results_A[0].keys():
                        PAD = torch.ones_like(results_A[idx]['embedding'])[:1]
                        if key == 'embedding':
                            result[key] = torch.cat([results_A[idx]['embedding'], PAD, results_B[idx]['embedding']], dim=0)
                        elif key == 'atttention_mask':
                            result[key] = torch.cat([results_A[idx]['atttention_mask'], PAD, results_B[idx]['atttention_mask']], dim=0)==1
                        else:
                            result[key] = results_A[idx][key] 
                    results.append(result)
            else: # sinlge protein case
                samples = data[i:i + self.batch_size]
                batch = self.pretrain_model.construct_batch(samples)
                results = self.pretrain_model(batch, task_type=self.task_type)
            
                if samples[0].get('smiles') is not None:
                    batch_smi = self.smiles_model.construct_batch(samples)
                    for idx in range(len(samples)):
                        results[idx]['smiles'] = batch_smi['smiles'][idx]
            
            proccessed_data.extend(results)
            
        return proccessed_data
            


