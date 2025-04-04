import inspect
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.finetune_model import UniModel
from src.utils.metrics import spearman_correlation

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def forward(self, batch, batch_idx):
        ret = self.model(batch)
        return ret


    def training_step(self, batch, batch_idx, **kwargs):
        ret = self(batch, batch_idx) 
        loss = ret['loss']
        self.log("train_loss", ret['loss'], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret = self(batch, batch_idx)
        loss = ret['loss']
        metric = self.metrics(ret['logits'], batch['label'])
        log_dict = {'val_loss': loss}
        log_dict.update(metric)
        self.log_dict(log_dict)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        ret = self(batch, batch_idx)
        loss = ret['loss']
        metric = self.metrics(ret['logits'], batch['label'])
        log_dict = {'test_loss': loss}
        log_dict.update(metric)
        self.log_dict(log_dict)
        return self.log_dict

    def load_model(self):
        self.model = UniModel(self.hparams.pretrain_model_name, self.hparams.task_type,  self.hparams.finetune_type, self.hparams.num_classes, self.hparams.lora_r, self.hparams.lora_alpha, self.hparams.lora_dropout)
    
    def metrics(self, preds, target):
        if self.hparams.task_name == 'fold_prediction':
            preds = torch.argmax(preds, dim=-1)
            acc = (preds == target).float().mean()
            return {'acc': acc}

        if self.hparams.task_name == 'fitness_prediction':
            return {'spearman': spearman_correlation(target, preds)}
