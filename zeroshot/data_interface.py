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
    
    def data_setup(self):
        self.mut_dataset = ProteinGYMDataset(
            dms_csv_dir = self.hparams.dms_csv_dir,
            dms_pdb_dir = self.hparams.dms_pdb_dir,
            dms_reference_csv_path = self.hparams.dms_reference_csv_path,
        )

    def train_dataloader(self):
        return DataLoader(self.mut_dataset, batch_size=1, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mut_dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mut_dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
