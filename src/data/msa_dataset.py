import torch
import os, glob
import pandas as pd
import numpy as np
from src.data.esm.sdk.api import ESMProtein
from src.utils.utils import pmap_multi


def read_data(aa_seq, pdb_path, target, unique_id, position=None, msa_aas=None):
    try:
        if unique_id is None:
            unique_id = str(hash(aa_seq))
        
        if position:
            position = torch.tensor([int(ele) for ele in position.split(",")])
        # name = target
        if pdb_path:
            structure = ESMProtein.from_pdb(pdb_path)
            coordinates = structure.coordinates
        else:
            coordinates = None

        return {
            'name': target, 
            'seq': aa_seq, 
            'X': coordinates,
            'label': torch.tensor([0]), 
            'unique_id': unique_id, 
            'pdb_path': pdb_path,
            'position': position,
            'msa_aas': msa_aas
        }
    except:
        return None
    

class MSADataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        msa_csv_path: str,
        type: str = "center"
    ):
        self.msa_csv_path = msa_csv_path
        msa_df = pd.read_csv(self.msa_csv_path)
        # msa_df = msa_df.drop_duplicates(subset="aa_seq", keep="first")
        if type == "center":
            msa_df = msa_df[msa_df["type"] == type]
        else:
            center_df = msa_df[msa_df["type"] == "center"]
            id_seq_dict = dict(zip(center_df['unique_id'], center_df['aa_seq_ori']))
            msa_df = msa_df[msa_df["type"] == type]

            def compute_position_diff(x):
                seq_with_gap = x["aa_seq_ori"]
                center_seq = id_seq_dict[x["target"]]
                
                pos_diff, msa_pos_aa = [], []
                center_idx = 0

                for i, aa in enumerate(seq_with_gap):
                    if aa == "-":
                        continue

                    if center_idx >= len(center_seq):
                        break

                    if aa.upper() != center_seq[center_idx].upper():
                        pos_diff.append(str(i)) 
                        msa_pos_aa.append(aa.upper())
                    center_idx += 1

                return ",".join(pos_diff)
            
            def compute_aa_diff(x):
                seq_with_gap = x["aa_seq_ori"]
                center_seq = id_seq_dict[x["target"]]
                
                pos_diff, msa_pos_aa = [], []
                center_idx = 0

                for i, aa in enumerate(seq_with_gap):
                    if aa == "-":
                        continue

                    if center_idx >= len(center_seq):
                        break

                    if aa.upper() != center_seq[center_idx].upper():
                        pos_diff.append(str(i)) 
                        msa_pos_aa.append(aa.upper())
                    center_idx += 1

                return "".join(msa_pos_aa)
            
            def compute_aa_diff(x):
                seq_with_gap = x["aa_seq_ori"]
                center_seq = id_seq_dict[x["target"]]
                
                pos_diff, msa_pos_aa = [], []
                center_idx = 0

                for i, aa in enumerate(seq_with_gap):
                    if aa == "-":
                        continue

                    if center_idx >= len(center_seq):
                        break

                    if aa.upper() != center_seq[center_idx].upper():
                        pos_diff.append(str(i)) 
                        msa_pos_aa.append(aa.upper())
                    center_idx += 1

                return "".join(msa_pos_aa)

            msa_df["position"] = msa_df.apply(compute_position_diff, axis=1)
            msa_df["msa_aas"] = msa_df.apply(compute_aa_diff, axis=1)
        path_list = []
        for i in range(len(msa_df)):
            path_list.append(
                (
                    msa_df.iloc[i].get('aa_seq'), 
                    msa_df.iloc[i].get('pdb_path'), 
                    msa_df.iloc[i].get('target'), 
                    msa_df.iloc[i].get('unique_id'), 
                    msa_df.iloc[i].get('position'),
                    msa_df.iloc[i].get('msa_aas'),
                )
            )
        
        self.data = pmap_multi(read_data, path_list, n_jobs=8)
        self.data = [d for d in self.data if d is not None]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(
        self, 
        idx
    ):
        return self.data[idx]
    

if __name__ == "__main__":
    msa_data_center = MSADataset(
        msa_csv_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/zeroshot/msa/msa_samples_zeroshot_w_pdb.csv",
        type="center"
    )
    print(f"length of msa dataset: {len(msa_data_center)}...")

    msa_data_msa = MSADataset(
        msa_csv_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/zeroshot/msa/msa_samples_zeroshot_w_pdb.csv",
        type="msa"
    )
    print(f"length of msa dataset: {len(msa_data_msa)}...")
