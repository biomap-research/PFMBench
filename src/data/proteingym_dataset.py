import torch
import os, glob
import pandas as pd
import numpy as np


class ProteinGYMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dms_csv_dir: str = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions",
        dms_pdb_dir: str = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions/ProteinGym_AF2_structures",
        dms_reference_csv_path: str = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions/ProteinGym_AF2_structures/DMS_substitutions.csv",
    ):
        self.dms_csv_dir = dms_csv_dir
        self.pdb_dir = dms_pdb_dir
        self.dms_reference_csv_path = dms_reference_csv_path
        self.dms_reference_df = pd.read_csv(self.dms_reference_csv_path)
        self.dms_ids = self.dms_reference_df["DMS_id"].tolist()
        self.target_sequences = self.dms_reference_df["target_seq"].tolist()
        self.dms_csv_path = [os.path.join(self.dms_csv_dir, ele) for ele in self.dms_reference_df["DMS_filename"].tolist()]
        self.pdb_file_path = [os.path.join(self.pdb_dir, ele) for ele in self.dms_reference_df["pdb_file"].tolist()]
        self.pdb_file_ranges = [[int(ele.split("-")[0])-1, int(ele.split("-")[-1])] for ele in self.dms_reference_df["pdb_range"].tolist()] # 0-index

    def __len__(self):
        return len(self.dms_reference_df)
    
    def __getitem__(
        self, 
        idx
    ):
        dms_id = self.dms_ids[idx]
        dms_csv_path = self.dms_csv_path[idx]
        target_sequence = self.target_sequences[idx]
        pdb_file_path = self.pdb_file_path[idx]
        pdb_range = self.pdb_file_ranges[idx]
        assert len(pdb_range) == 2, f"invalid pdb range: {pdb_range}"
        # target_sequence = target_sequence[pdb_range[0]:pdb_range[1]]

        return {
            "dms_id": dms_id,
            "dms_csv_path": dms_csv_path,
            "target_sequence": target_sequence,
            "pdb_file_path": pdb_file_path,
            "pdb_range": pdb_range,
            "max_length": pdb_range[1] - pdb_range[0]
        }
    

if __name__ == "__main__":
    proteingym = ProteinGYMDataset()
    print(f"length of proteingym dataset: {len(proteingym)}...")
