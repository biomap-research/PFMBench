import inspect
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.finetune_model import UniModel
from scipy.stats import spearmanr

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self._context = {
            "validation": {
                "logits": [],
                "labels": []
            },
            "test": {
                "logits": [],
                "labels": []
            }
        }
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
        log_dict = {'val_loss': loss}
        self._context["validation"]["logits"].append(ret['logits'].float().cpu().numpy())
        self._context["validation"]["labels"].append(batch['label'].float().cpu().numpy())

        self.log_dict(log_dict)
        return self.log_dict

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # compute metrics and reset
        metric = self.metrics(
            self._context["validation"]["logits"], 
            self._context["validation"]["labels"],
            name="valid"
        )
        self._context["validation"]["logits"] = []
        self._context["validation"]["labels"] = []
        self.log_dict(metric)
        return self.log_dict
   
    def test_step(self, batch, batch_idx):
        ret = self(batch, batch_idx)
        loss = ret['loss']
        log_dict = {'test_loss': loss}
        self._context["test"]["logits"].append(ret['logits'].float().cpu().numpy())
        self._context["test"]["labels"].append(batch['label'].float().cpu().numpy())
        # metric = self.metrics(ret['logits'], batch['label'])
        # log_dict.update(metric)
        # self.log_dict({"test_acc": metric})
        return self.log_dict

    def on_test_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # compute metrics and reset
        metric = self.metrics(
            self._context["test"]["logits"], 
            self._context["test"]["labels"],
            name="test"
        )
        self._context["test"]["logits"] = []
        self._context["test"]["labels"] = []
        self.log_dict(metric)
        return self.log_dict

    def load_model(self):
        self.model = UniModel(self.hparams.pretrain_model_name, self.hparams.task_type,  self.hparams.finetune_type, self.hparams.num_classes, self.hparams.lora_r, self.hparams.lora_alpha, self.hparams.lora_dropout)
    
    def metrics(self, preds, target, name):
        if self.hparams.task_type == "classification":
            preds, target = np.vstack(preds), np.hstack(target) 
            if self.hparams.task_name == 'fold_prediction':
                preds = np.argmax(preds, axis=-1)
                acc = (preds == target).mean()
                return {f"{name}_acc": acc}
        elif self.hparams.task_type == "regression":
            preds, target = np.hstack(preds), np.hstack(target) 
            if self.hparams.task_name == 'fitness_prediction':
                return {f"{name}_spearman": spearmanr(target, preds).statistic}

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        if self.hparams.finetune_type == "lora":
            for name in list(state.keys()):
                if "lora" not in name:
                    state.pop(name)
        return state
