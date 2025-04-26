from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, DistributedSampler
from src.interface.data_interface import DInterface_base
from src.model.pretrain_model_interface import PretrainModelInterface
from src.data.proteingym_dataset import ProteinGYMDataset

class DInterface(DInterface_base):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass
    
    def data_setup(self, data_path):
        self.validation_set = ProteinGYMDataset(data_path)

    def train_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)

    def test_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.data_process_fn)

    