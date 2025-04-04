from torch.utils.data import DataLoader
import torch
import random
import lmdb
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from src.interface.data_interface import DInterface_base
from src.data.protein_dataset import ProteinDataset

class DInterface(DInterface_base):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass
    
    def data_setup(self):
        self.train_set = ProteinDataset(self.hparams.train_data_path, self.hparams.pretrain_model_name, self.hparams.seq_len)
        self.val_set = ProteinDataset(self.hparams.val_data_path, self.hparams.pretrain_model_name, self.hparams.seq_len)
        self.test_set = ProteinDataset(self.hparams.test_data_path, self.hparams.pretrain_model_name, self.hparams.seq_len)
        
        self.train_set.data = pretrained_model.extract_features(self.train_set, self.hparams.pretrain_model_name, self.hparams.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)



    def data_process_fn(self, data_list):
        seq_list = []
        coords_list = []
        chain_list = []
        label_list = []
        mask_list = []
        for data in data_list:
            seq_list.append(data['seq'])
            coords_list.append(data['coords'])
            chain_list.append(data['chain'])
            label_list.append(data['label'])
            mask_list.append(data['mask'])
        
        seq = torch.stack(seq_list, dim=0)
        coords = torch.stack(coords_list, dim=0)
        chain = torch.stack(chain_list, dim=0)
        label = torch.stack(label_list, dim=0)
        mask = torch.stack(mask_list, dim=0)
        
        
        return {'seq': seq,
                'coords': coords,
                'chain': chain,
                'label': label,
                'mask': mask}
        