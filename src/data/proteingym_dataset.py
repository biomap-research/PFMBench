import torch
import os
import pandas as pd
import numpy as np


class ProteinGYMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"data path: {data_path} not found")
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
        df = df.drop_duplicates(subset='mutant', keep='first')
        if len(df) > 0:
            f_mutants = df["mutant"][0]
            f_seq = list(df["mutated_sequence"][0])
            f_mutants = f_mutants.split(":")
            for f_mutant in f_mutants:
                wt, pos, mt = f_mutant[0], int(f_mutant[1:-1])-1, f_mutant[-1]
                f_seq[pos] = wt
            self.wt_seq = "".join(f_seq)
        else:
            raise RuntimeError(f"empty data: {data_path}")
        # find all single mutations
        self.mutations = {}
        for mutants in df["mutant"].tolist():
            for mutant in mutants.split(":"):
                wt, position, mt = mutant[0], int(mutant[1:-1])-1, mutant[-1]
                if position not in self.mutations:
                    self.mutations[position] = [f"{wt}{mt}"]
                else:
                    if f"{wt}{mt}" not in self.mutations[position]:
                        self.mutations[position].append(f"{wt}{mt}")
        
        self.mutations_positions = list(self.mutations.keys())

        self.true_length = [len(self.mutations[mutations_position]) for mutations_position in self.mutations_positions]
        self.max_mutation_number = max(self.true_length)


        self.desc = "Single Position Mutation Dataset"
        self.t = self.sequence_tokenizer.special_tokens['tMASK']
        self.eos = self.sequence_tokenizer.special_tokens['eos']
        self.label = torch.tensor(self.sequence_tokenizer.tokenize(self.wt_seq)+[self.eos]).long()

    def __len__(self):
        return len(self.mutations_positions)
    
    def __getitem__(
        self, 
        idx
    ):
        sequence_token = self.label.clone()
        mutation_position = self.mutations_positions[idx]
        true_length = [self.true_length[idx]]
        mutation_aas = self.mutations[mutation_position]
        wt_aa = [self.sequence_tokenizer.convert_tokens_to_ids(mutation_aas[0][0])[0]]
        mutation_aas = [self.sequence_tokenizer.convert_tokens_to_ids(ele[-1])[0] for ele in mutation_aas]

        additional_elements = self.max_mutation_number - len(mutation_aas)
        mutation_aas.extend([0] * additional_elements)

        true_length = torch.tensor(true_length, dtype=sequence_token.dtype).flatten()
        wt_aa = torch.tensor(wt_aa, dtype=sequence_token.dtype).flatten()
        mutation_aas = torch.tensor(mutation_aas, dtype=sequence_token.dtype).flatten()
        mutation_position = torch.tensor(mutation_position, dtype=sequence_token.dtype).flatten()
        
        sequence_token[mutation_position] = self.t
        sequence_mask = torch.zeros_like(self.label).bool()
        sequence_mask[mutation_position] = True
        sequence_length=len(sequence_token)

        
        uni_key_position_ids = torch.arange(sequence_token.shape[0], device=sequence_token.device)
        uni_query_position_ids = uni_key_position_ids
        return {
            "uni_source_tokens": sequence_token,
            "uni_target_tokens": sequence_token,
            "uni_query_position_ids": uni_query_position_ids,
            "uni_key_position_ids": uni_key_position_ids,
            "sequence_token": sequence_token,
            "sequence_mask": sequence_mask,
            "sequence_label": self.label,
            "sequence_mt_aas": mutation_aas,
            "sequence_wt_aa": wt_aa,
            "sequence_true_length": true_length,
            "sequence_mt_pos": mutation_position,
            "seequence_length": sequence_length
        }
