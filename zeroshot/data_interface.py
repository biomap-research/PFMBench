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
    
    def data_setup(self, data_path, tokenizer):
        self.validation_set = ProteinGYMDataset(data_path, tokenizer, model_name=self.hparams.pretrain_model_name)

    def train_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)


        # merge_mutation_probs_partial = partial(merge_mutation_probs, prob_dict=predict_dms)
        
        # data_df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
        # data_df = data_df.drop_duplicates(subset='mutant', keep='first')
        # data_df["predict_DMS_score"] = data_df["mutant"].apply(merge_mutation_probs_partial)
        # print_rank_0(f"Saving result file to {c_save_path}...")
        # os.makedirs(os.path.dirname(c_save_path), exist_ok=True)
        # data_df.to_csv(c_save_path, index=False)
        # score, _ = spearmanr(data_df["predict_DMS_score"].tolist(), data_df["DMS_score"].tolist())