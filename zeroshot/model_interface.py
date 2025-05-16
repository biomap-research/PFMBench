import os
import numpy as np
import pandas as pd
import torch
import re
import math
import torch.nn.functional as F
from tqdm import tqdm
from src.interface.model_interface import MInterface_base
from src.model.pretrain_model_interface import PretrainModelInterface
from scipy.stats import spearmanr
from src.data.esm.sdk.api import ESMProtein
from src.model.pretrain_modules import (
    ESM2Model, ESMC600MModel, ESM3Model, VenusPLMModel,
    ProSTModel, ProstT5Model, ProTrekModel, SaPortModel,
    ProtT5, DPLMModel, PGLMModel, ANKHBase, ProtGPT2Model
)

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = PretrainModelInterface(
            self.hparams.pretrain_model_name, 
            task_type=self.hparams.task_type
        )
        self.tokenizer = self.model.pretrain_model.get_tokenizer()
        self.multimodal_models = [
            ProTrekModel, 
            SaPortModel
        ]
        self.mlm_models = [
            ESM2Model,
            ESMC600MModel,
            ESM3Model,
            PGLMModel,
            VenusPLMModel,
            ProTrekModel,
            DPLMModel,
        ]
        self.start, self.end = 1, -1
        if type(self.model) == ProtGPT2Model:
            self.start -= 1
        self._context = {
            "test": {
                "spearmans": []
            },
        }
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def test_step(self, batch, batch_idx):
        dms_id = batch["dms_id"][0]
        dms_csv_path = batch["dms_csv_path"][0]
        pdb_range = batch["pdb_range"]
        target_sequence = batch["target_sequence"][0]
        pdb_file_path = batch["pdb_file_path"][0]
        dms_df = pd.read_csv(dms_csv_path)
        true_dms_scores = dms_df["DMS_score"].tolist()
        predict_dms = []

        if type(self.model) in self.multimodal_models:
            # truncated by pdb
            target_sequence = target_sequence[pdb_range[0]:pdb_range[1]]
            offset = pdb_range[0].item() if isinstance(pdb_range[0], torch.Tensor) else pdb_range[0]
        else:
            offset = 0

        if pdb_file_path is not None:
            structure = ESMProtein.from_pdb(pdb_file_path)
            coordinates = structure.coordinates
        else:
            coordinates = None

        if type(self.model.pretrain_model) in self.mlm_models:
            sequence = target_sequence
            mutations = dms_df["mutant"].tolist()
            model = self.model.pretrain_model
            tokenizer = self.tokenizer
            batch_size = 8
            window_size = 1022

            if len(sequence) == 0:
                raise ValueError("Empty wildtype sequence provided")
            print(f"Working with sequence of length {len(sequence)} using optimized MLM approach")

            parsed_mutations, unique_positions = [], set()
            for mutation in mutations:
                if ":" in mutation:
                    sub_mutations = mutation.split(":")
                    multi_wt, multi_mt = "", ""
                    multi_pos, multi_seq_pos, valid_multi = [], [], True
                    for sub_mut in sub_mutations:
                        match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mut)
                        if not match:
                            print(f"Warning: Could not parse mutation {sub_mut}, skipping")
                            valid_multi = False
                            break
                        wt, pos_str, mt = match.groups()
                        pos = int(pos_str)
                        seq_pos = pos - (1 + offset)

                        if seq_pos < 0 or seq_pos >= len(sequence):
                            print(f"Warning: Position {pos} out of range, skipping")
                            valid_multi = False
                            break
                        if sequence[seq_pos] != wt:
                            print(f"Warning: Wild-type {wt} at pos {pos} doesn't match sequence {sequence[seq_pos]}, skipping")
                            valid_multi = False
                            break

                        multi_wt += wt
                        multi_mt += mt
                        multi_pos.append(pos)
                        multi_seq_pos.append(seq_pos)
                        unique_positions.add(pos)

                    if valid_multi:
                        parsed_mutations.append((multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))
                else:
                    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
                    if not match:
                        print(f"Warning: Could not parse mutation {mutation}, skipping")
                        continue

                    wt, pos_str, mt = match.groups()
                    pos = int(pos_str)
                    seq_pos = pos - (1 + offset)

                    if seq_pos < 0 or seq_pos >= len(sequence):
                        print(f"Warning: Position {pos} out of range, skipping")
                        continue

                    if sequence[seq_pos] != wt:
                        print(f"Warning: Wild-type {wt} at pos {pos} doesn't match sequence {sequence[seq_pos]}, skipping")
                        continue

                    parsed_mutations.append((wt, [pos], mt, [seq_pos], mutation))
                    unique_positions.add(pos)
            if not parsed_mutations:
                print("No valid mutations to score")
                predict_dms = [0.0] * len(mutations)
            else:
                unique_positions = sorted(list(unique_positions))
                print(f"Found {len(unique_positions)} unique mutation positions to pre-compute")

                aa_to_token = {}
                amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                for aa in amino_acids:
                    tokens = tokenizer.encode(aa, add_special_tokens=False)
                    aa_to_token[aa] = tokens[0]
                mask_token_id = tokenizer.mask_token_id

                position_aa_scores = {}
                num_batches = math.ceil(len(unique_positions) / batch_size)
                progress_bar = tqdm(total=num_batches, desc="Pre-computing position scores")
                for batch_idx in range(0, len(unique_positions), batch_size):
                    batch_positions = unique_positions[batch_idx:batch_idx + batch_size]
                    window_groups = {}

                    for pos in batch_positions:
                        seq_pos = pos - (1 + offset)
                        if len(sequence) > window_size:
                            window_half = (window_size) // 2
                            start_pos = max(0, seq_pos - window_half)
                            end_pos = min(len(sequence), start_pos + window_size)
                            if end_pos == len(sequence):
                                start_pos = max(0, len(sequence) - (window_size))
                            seq_window = sequence[start_pos:end_pos]
                            rel_pos = seq_pos - start_pos
                        else:
                            seq_window = sequence
                            rel_pos = seq_pos

                        window_key = (seq_window, start_pos if len(sequence) > window_size else 0)
                        if window_key not in window_groups:
                            window_groups[window_key] = []
                        window_groups[window_key].append((pos, seq_pos, rel_pos))

                    for (seq_window, window_start), positions_in_window in window_groups.items():
                        unique_rel_positions = set(info[2] for info in positions_in_window)
                        input_items = []
                        rel_pos_map = {}

                        for rel_pos in unique_rel_positions:
                            masked_seq = list(seq_window)
                            masked_seq[rel_pos] = tokenizer.mask_token
                            input_items.append({ # 保留原 construct_batch 用法
                                "seq": ''.join(masked_seq),
                                "X": coordinates,  
                                "name": f"masked_pos_{rel_pos}",
                                "label": 1.0
                            })
                            rel_pos_map[len(input_items) - 1] = rel_pos

                        with torch.no_grad():
                            batch = model.construct_batch(input_items)
                            outputs = model.forward(batch=batch, return_logits=True)
                            batch_logits = outputs[:, self.start:self.end, :]  # [batch, seq_len, vocab_size]

                        for idx, rel_pos in rel_pos_map.items():
                            logits = batch_logits[idx, rel_pos, :]
                            log_probs = torch.log_softmax(logits, dim=-1)
                            for pos, seq_pos, pos_rel_pos in positions_in_window:
                                if pos_rel_pos == rel_pos:
                                    if pos not in position_aa_scores:
                                        position_aa_scores[pos] = {}
                                    for aa in amino_acids:
                                        token_id = aa_to_token[aa]
                                        position_aa_scores[pos][aa] = log_probs[token_id].item()

                    if progress_bar is not None:
                        progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.close()

                mutation_scores = {}
                print("Calculating scores for all mutations using pre-computed values")

                for wt, pos_list, mt, seq_pos_list, mutation_name in tqdm(parsed_mutations, desc="Scoring mutations"):
                    cumulative_score = 0.0
                    for i, (pos, aa_mt) in enumerate(zip(pos_list, mt)):
                        aa_wt = wt[i] if i < len(wt) else wt
                        if pos in position_aa_scores:
                            wt_score = position_aa_scores[pos][aa_wt]
                            mt_score = position_aa_scores[pos][aa_mt]
                            cumulative_score += (mt_score - wt_score)
                        else:
                            print(f"Warning: Position {pos} not found in pre-computed scores, mutation {mutation_name} may be incomplete")
                    mutation_scores[mutation_name] = cumulative_score

                predict_dms = [mutation_scores.get(mut, 0.0) for mut in mutations]
        else:
            # this is similarity logic
            batch["max_length"][0] = len(target_sequence)
            target_sequence = target_sequence[pdb_range[0]:pdb_range[1]]
            offset = pdb_range[0].item() if isinstance(pdb_range[0], torch.Tensor) else pdb_range[0]
            if isinstance(self.model.pretrain_model, ProSTModel):
                target_sequence = target_sequence[:1022]
            self.model.pretrain_model.max_length = len(target_sequence)
            mutants = dms_df["mutant"].tolist()
            coordinates = ESMProtein.from_pdb(pdb_file_path).coordinates
            # Step 1: Get wildtype embedding
            wt_input = [{
                "seq": target_sequence,
                "X": coordinates,  # 保留原 construct_batch 的接口
                "name": "wildtype",
                "label": 1.0
            }]
            with torch.no_grad():
                try:
                    wt_batch = self.model.pretrain_model.construct_batch(wt_input)
                    wt_logits = self.model.pretrain_model.forward(batch=wt_batch, return_logits=True).squeeze(0)[self.start:self.end,:]
                except Exception:
                    return None
            # Step 2: Prepare mutant inputs
            mutant_inputs = []
            selected_true_dms_scores = []
            for j, mutant in enumerate(mutants):
                mutated_seq = list(target_sequence)
                mut_positions = []
                for mut in mutant.split(":"):
                    wt_res, pos_str, mut_res = mut[0], mut[1:-1], mut[-1]
                    pos = int(pos_str) - (1 + offset)
                    if isinstance(self.model.pretrain_model, ProSTModel):
                        if pos > 1020:
                            continue
                    mutated_seq[pos] = mut_res
                    mut_positions.append(pos)
                mutant_inputs.append({
                    "seq": ''.join(mutated_seq),
                    "X": coordinates,
                    "name": f"mutant",
                    "label": 1.0,
                    "mut_positions": mut_positions  # 记录突变位置
                })
                selected_true_dms_scores.append(true_dms_scores[j])
            # Step 3: Batch inference and compute similarity scores
            predict_dms = []
            batch_size = 8  # 根据显存调整
            for i in tqdm(range(0, len(mutant_inputs), batch_size), desc=f"Processing {dms_id} (Similarity)..."):
                batch_mutants = mutant_inputs[i:i + batch_size]
                batch_scores = selected_true_dms_scores[i:i + batch_size]
                mut_positions_batch = [x["mut_positions"] for x in batch_mutants]
                with torch.no_grad():
                    try:
                        mut_batch = self.model.pretrain_model.construct_batch(batch_mutants)
                        mut_logits = self.model.pretrain_model.forward(batch=mut_batch, return_logits=True)[:, self.start:self.end, :]
                    except Exception as e:
                        # 如果失败，把对应的 true scores 删掉
                        print(f"Batch {i}-{i+batch_size} failed with error: {e}")
                        selected_true_dms_scores = selected_true_dms_scores[:i] + selected_true_dms_scores[i + len(batch_mutants):]
                        continue

                for j in range(mut_logits.size(0)):
                    mut_emb = mut_logits[j]  # Flatten embedding
                    if isinstance(self.model.pretrain_model, ProtGPT2Model):
                        wt_emb_mean = wt_logits.mean(0)
                        mut_emb_mean = mut_emb.mean(0)
                        similarity = -1.0 * local_l2_difference_single(wt_emb_mean, mut_emb_mean, [0], window_size=0)
                    else:
                        # 对于 residue-level embedding，取突变位置的 embedding
                        mut_pos = mut_positions_batch[j]
                        # wt_emb_mut = wt_logits[mut_pos, :].mean(0)  # 突变位点平均
                        # mut_emb_mut = mut_emb[mut_pos, :].mean(0)
                        similarity = -1.0 * local_l2_difference_single(wt_logits, mut_emb, mut_pos, window_size=5)
                    score = similarity
                    predict_dms.append(score)

        assert len(predict_dms) == len(true_dms_scores)
        # true_dms_scores = true_dms_scores
        spearman = spearmanr(np.array(predict_dms), np.array(true_dms_scores)).statistic
        log_dict = {
            "test_spearman": spearman
        }
        if not np.isnan(spearman):
            self._context['test']['spearmans'].append(spearman)
            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True)
    
    def on_test_epoch_end(self):
        spearmans = np.array(self._context['test']['spearmans']).mean()
        metric = {
            "avg_spearman": spearmans
        }
        self.log_dict(metric, prog_bar=True, logger=True, on_epoch=True)


def local_l2_difference_single(wt_emb, mut_emb, mut_pos_list, window_size=5):
    """
    Args:
        wt_emb: (L, d) wild-type embedding
        mut_emb: (L, d) mutant embedding
        mut_pos_list: list of int, 突变位点列表
        window_size: int

    Returns:
        score: float (单个样本的局部L2差异)
    """
    L, d = wt_emb.shape
    window_indices = set()
    for pos in mut_pos_list:
        start = max(pos - window_size, 0)
        end = min(pos + window_size + 1, L)
        window_indices.update(range(start, end))
    window_indices = sorted(window_indices)

    wt_local = wt_emb[window_indices, :]  # (window_len, d)
    mut_local = mut_emb[window_indices, :]  # (window_len, d)
    delta = mut_local - wt_local
    score = torch.norm(delta)  # Frobenius norm
    return score.item()
