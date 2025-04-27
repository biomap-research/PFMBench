import torch
import os
import pandas as pd
import numpy as np


class ProteinGYMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer,
        model_name
    ):
        if model_name == "esm2_650m":
            add_BOS = 1
            
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"data path: {data_path} not found")
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
        df = df.drop_duplicates(subset='mutant', keep='first')
        if len(df) > 0:
            f_mutants = df["mutant"][0]
            f_seq = list(df["mutated_sequence"][0])
            f_mutants = f_mutants.split(":")
            for f_mutant in f_mutants:
                wt, pos, mt = f_mutant[0], int(f_mutant[1:-1])-1+add_BOS, f_mutant[-1]
                f_seq[pos] = wt
            self.wt_seq = "".join(f_seq)
        else:
            raise RuntimeError(f"empty data: {data_path}")
        # find all single mutations
        self.mutations = {}
        for idx, row in df.iterrows():
            mutants = row["mutant"]
            DMS_score = row["DMS_score"]
            for mutant in mutants.split(":"):
                wt, position, mt = mutant[0], int(mutant[1:-1])-1+add_BOS, mutant[-1]
                if position not in self.mutations:
                    self.mutations[position] = [[f"{wt}{mt}"], [DMS_score]]
                else:
                    if f"{wt}{mt}" not in self.mutations[position]:
                        self.mutations[position][0].append(f"{wt}{mt}")
                        self.mutations[position][1].append(DMS_score)
        
        self.mutations_positions = list(self.mutations.keys())

        self.true_length = [len(self.mutations[mutations_position][0]) for mutations_position in self.mutations_positions]
        self.max_mutation_number = max(self.true_length)
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.get_vocab()['<mask>']
        self.eos = self.tokenizer.get_vocab()['<eos>']
        self.bos = self.tokenizer.get_vocab()['<cls>']
        self.label = torch.tensor(self.tokenizer.encode(self.wt_seq)).long()

        self.desc = "Single Position Mutation Dataset"


    def __len__(self):
        return len(self.mutations_positions)
    
    def __getitem__(
        self, 
        idx
    ):
        sequence_token = torch.tensor(self.tokenizer.encode(self.wt_seq))
        mutation_position = self.mutations_positions[idx]
        true_length = [self.true_length[idx]]
        mutation_aas = self.mutations[mutation_position][0]
        dms_scores = self.mutations[mutation_position][1]
        wt_aa = [self.tokenizer.convert_tokens_to_ids(mutation_aas[0][0])]
        mutation_aas = [self.tokenizer.convert_tokens_to_ids(ele[-1]) for ele in mutation_aas]

        additional_elements = self.max_mutation_number - len(mutation_aas)
        mutation_aas.extend([0] * additional_elements)
        dms_scores.extend([np.nan] * additional_elements)

        true_length = torch.tensor(true_length, dtype=sequence_token.dtype).flatten()
        wt_aa = torch.tensor(wt_aa, dtype=sequence_token.dtype).flatten()
        mutation_aas = torch.tensor(mutation_aas, dtype=sequence_token.dtype).flatten()
        dms_scores = torch.tensor(dms_scores).flatten()
        mutation_position = torch.tensor(mutation_position, dtype=sequence_token.dtype).flatten()
        
        sequence_token[mutation_position] = self.mask_token
        mutation_mask = torch.zeros_like(self.label).bool()
        mutation_mask[mutation_position] = True
        sequence_length=len(sequence_token)

        return {
            "attention_mask": torch.ones_like(sequence_token).bool(),
            "mutation_mask": mutation_mask,
            "sequence_label": self.label,
            "sequence_mt_aas": mutation_aas,
            "sequence_wt_aa": wt_aa,
            "sequence_true_length": true_length,
            "sequence_mt_pos": mutation_position,
            "sequence_length": sequence_length,
            "dms_scores": dms_scores,
        }
