import os
import numpy as np
import torch
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.pretrain_model_interface import PretrainModelInterface
from scipy.stats import spearmanr

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = PretrainModelInterface(self.hparams.pretrain_model_name, task_type=self.hparams.task_type)
        
        self._context = {
            "validation": {
                "predict_dms": [],
                "true_dms": []
            },
        }
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
    
    def forward(self, batch):
        batch_custom = {
            'seq':batch['sequence_label'],
            'attention_mask': batch['attention_mask'],
            
        }
        probs = self.model.pretrain_model(batch_custom, return_prob=True)
        return probs

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        predict_dms = {}
        true_dms = {}

        # unpack labels
        sequence_mut_aas = batch['sequence_mt_aas']
        sequence_mut_pos = batch['sequence_mt_pos']
        sequence_wt_aa = batch['sequence_wt_aa']
        sequence_true_length = batch['sequence_true_length']

        # 1. 提取突变位置的 logits
        probs = logits  # (seq_len, batch, vocab_size)

        # sequence_mut_pos_expanded = sequence_mut_pos.unsqueeze(-1).expand(-1, -1, probs.size(-1))
        # probs = torch.gather(probs, 1, sequence_mut_pos_expanded)

        # 2. 计算 DMS 分数：突变氨基酸分数 - 野生型氨基酸分数
        dms = torch.gather(probs, 2, sequence_mut_aas.unsqueeze(1)) - torch.gather(probs, 2, sequence_wt_aa.unsqueeze(1))
        dms_true = batch['dms_scores'][:,None]

        # 3. 转成 list，方便后面处理
        dms = dms.tolist()
        dms_true = dms_true.tolist()
        sequence_wt_aa = sequence_wt_aa.squeeze().tolist()
        sequence_mut_aas = sequence_mut_aas.squeeze().tolist()
        sequence_mut_pos = sequence_mut_pos.squeeze().tolist()
        sequence_true_length = sequence_true_length.squeeze().tolist()

        # 4. 去重（基于突变位置）
        unique_indices = [sequence_mut_pos.index(key) for key in dict.fromkeys(sequence_mut_pos)]
        sequence_mut_pos = [sequence_mut_pos[k] for k in unique_indices]
        sequence_true_length = [sequence_true_length[k] for k in unique_indices]
        sequence_mut_aas = [sequence_mut_aas[k] for k in unique_indices]
        sequence_wt_aa = [sequence_wt_aa[k] for k in unique_indices]

        # 5. 整理到 predict_dms
        for _dms, _dms_true, _wt, _mt, _pos, _tl in zip(dms, dms_true, sequence_wt_aa, sequence_mut_aas, sequence_mut_pos, sequence_true_length):
            if not isinstance(_mt, list): 
                _mt = [_mt]
            _wt = self.model.pretrain_model.tokenizer.convert_ids_to_tokens(_wt)
            _mt = [self.model.pretrain_model.tokenizer.convert_ids_to_tokens(ele) for ele in _mt]
            for j, __mt in enumerate(_mt[:_tl]): 
                _mut_info = f"{_wt}{_pos+1}{__mt}"
                predict_dms[_mut_info] = _dms[0][j]
                true_dms[_mut_info] = _dms_true[0][j]

        self._context['validation']['predict_dms'].append(torch.tensor(list(predict_dms.values())))
        self._context['validation']['true_dms'].append(torch.tensor(list(true_dms.values())))
    
    
    def on_validation_epoch_end(self, ):
        predict_dms = torch.cat(self._context['validation']['predict_dms'], dim=0).numpy()
        true_dms = torch.cat(self._context['validation']['true_dms'], dim=0).numpy()
        score, _ = spearmanr(predict_dms, true_dms)
        self.val_spearman = score
        self.log("val_spearman", score, prog_bar=True, logger=True)
        return score


        