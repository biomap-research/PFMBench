import os
import numpy as np
import torch
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.pretrain_model_interface import PretrainModelInterface


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = PretrainModelInterface(self.hparams.pretrain_model_name, batch_size=self.hparams.pretrain_batch_size, max_length=self.hparams.seq_len, sequence_only=self.hparams.sequence_only, task_type=self.hparams.task_type)
        
        self._context = {
            "validation": {
                "predict_dms": [],
                "true_dms": []
            },
        }
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def validation_step(self, batch, batch_idx):
        logits, labels, tokenizer = self(batch)
        predict_dms = {}

        # unpack labels
        _, _, sequence_mut_aas, sequence_mut_pos, sequence_wt_aa, sequence_true_length = labels

        # 1. 提取突变位置的 logits
        probs = logits.permute(1, 0, 2)  # (seq_len, batch, vocab_size)

        sequence_mut_pos_expanded = sequence_mut_pos.unsqueeze(-1).expand(-1, -1, probs.size(-1))
        probs = torch.gather(probs, 1, sequence_mut_pos_expanded)

        # 2. 计算 DMS 分数：突变氨基酸分数 - 野生型氨基酸分数
        dms = torch.gather(probs, 2, sequence_mut_aas.unsqueeze(1)) - torch.gather(probs, 2, sequence_wt_aa.unsqueeze(1))

        # 3. 转成 list，方便后面处理
        dms = dms.tolist()
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
        for _dms, _wt, _mt, _pos, _tl in zip(dms, sequence_wt_aa, sequence_mut_aas, sequence_mut_pos, sequence_true_length):
            if not isinstance(_mt, list): 
                _mt = [_mt]
            _wt = tokenizer.IdToToken(_wt)
            _mt = [tokenizer.IdToToken(ele) for ele in _mt]
            for j, __mt in enumerate(_mt[:_tl]): 
                _mut_info = f"{_wt}{_pos+1}{__mt}"
                predict_dms[_mut_info] = _dms[0][j]

        self._context['validation']['predict_dms'].append(predict_dms)
        self._context['validation']['true_dms'].append(batch['dms'][0])
