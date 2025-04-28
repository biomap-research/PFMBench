import os
import numpy as np
import torch
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.finetune_model import UniModel
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import binarize
from src.utils.metrics import f1_score_max

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self._context = {
            "validation": {
                "logits": [],
                "labels": [],
                "metric": []
            },
            "test": {
                "logits": [],
                "labels": [],
                "metric": []
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
        
        if self.hparams.task_type in ["contact"]:
            self._context["validation"]["metric"].append(
                self.metrics(
                ret['logits'][...,0].float().cpu(),
                batch['label'].float().cpu(),
                batch['attention_mask'].float().cpu(),
                name="valid"
            ))
        else:
            self._context["validation"]["logits"].append(ret['logits'].float().cpu().numpy())
            self._context["validation"]["labels"].append(batch['label'].float().cpu().numpy())

        self.log_dict(log_dict)
        return self.log_dict

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # compute metrics and reset
        if self.hparams.task_type in ["contact"]:
            value = torch.tensor(self._context["validation"]["metric"]).mean()
            metric = {f"valid_Top(L/5)": value}
            self._context["validation"]["metric"] = []
        else:
            metric = self.metrics(
                self._context["validation"]["logits"], 
                self._context["validation"]["labels"],
                # self._context["validation"]["attn_mask"],
                name="valid"
            )
        self._context["validation"]["logits"] = []
        self._context["validation"]["labels"] = []
        self._context["validation"]["metric"] = []
        self.log_dict(metric)
        return self.log_dict
   
    def test_step(self, batch, batch_idx):
        ret = self(batch, batch_idx)
        loss = ret['loss']
        if self.hparams.task_type in ["contact"]:
            self._context["test"]["metric"].append(self.metrics(
                ret['logits'][...,0].float().cpu(),
                batch['label'].float().cpu(),
                batch['attention_mask'].float().cpu(),
                name="test"
            ))
        else:
            self._context["test"]["logits"].append(ret['logits'].float().cpu().numpy())
            self._context["test"]["labels"].append(batch['label'].float().cpu().numpy())
        return self.log_dict

    def on_test_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        # compute metrics and reset
        if self.hparams.task_type in ["contact"]:
            value = torch.tensor(self._context["test"]["metric"]).mean()
            metric = {f"test_Top(L/5)": value}
            self._context["test"]["metric"] = []
        else:
            metric = self.metrics(
                self._context["test"]["logits"], 
                self._context["test"]["labels"],
                # self._context["test"]["attn_mask"],
                name="test"
            )
        self._context["test"]["logits"] = []
        self._context["test"]["labels"] = []
        self._context["test"]["attn_mask"] = []
        self.log_dict(metric)
        return self.log_dict

    def load_model(self):
        self.model = UniModel(self.hparams.pretrain_model_name, self.hparams.task_type, self.hparams.finetune_type, self.hparams.num_classes, self.hparams.lora_r, self.hparams.lora_alpha, self.hparams.lora_dropout)
    
    def metrics(self, preds, target, attn_mask=None, name='default'):
        if self.hparams.task_type == "classification":
            preds, target = np.vstack(preds), np.hstack(target) 
            preds = np.argmax(preds, axis=-1)
            acc = (preds == target).mean()
            return {f"{name}_acc": acc}
        elif self.hparams.task_type == "residual_classification":
            target_valid = []
            for i in range(len(target)):
                target_valid.append(target[i][attn_mask[i].astype(bool)])
                
            preds, target = np.vstack(preds), np.hstack(target_valid) 
            preds = np.argmax(preds, axis=-1)
            acc = (preds == target).mean()
            return {f"{name}_acc": acc}
        elif self.hparams.task_type in ["binary_classification", "pair_binary_classification"]:
            preds, target = np.vstack(preds), np.concatenate(target, axis=0) 
            auroc = roc_auc_score(target, preds)
            return {f"{name}_auroc": auroc}
        elif self.hparams.task_type in ["regression", "pair_regression"]:
            preds, target = np.hstack(preds), np.hstack(target) 
            return {f"{name}_spearman": spearmanr(target, preds).statistic}
        elif self.hparams.task_type == "multi_labels_classification":
            preds, target = np.vstack(preds), np.vstack(target) 
            f1_max = f1_score_max(torch.tensor(preds), torch.tensor(target)).item()
            return {f"{name}_f1_max": f1_max}
        elif self.hparams.task_type == "contact":
            from src.model.finetune_model import top_L_div_5_precision
            # preds = torch.cat([torch.from_numpy(one) for one in preds], dim=0)
            # target = torch.cat([torch.from_numpy(one) for one in target], dim=0)
            # attn_mask = torch.cat([torch.from_numpy(one) for one in attn_mask], dim=0)
            metrics = top_L_div_5_precision(preds, target, attn_mask)
            return metrics['Top(L/5)']

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        if self.hparams.finetune_type == "lora":
            for name in list(state.keys()):
                if "lora" not in name:
                    state.pop(name)
        return state
