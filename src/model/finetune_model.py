import torch
import torch.nn as nn
from src.model.pretrain_model_interface import PretrainModelInterface

class UniModel(nn.Module):
    def __init__(
        self,
        pretrain_model_name: str,
        task_type: str,
        finetune_type: str,
        num_classes: int,
        peft_type: str = "lora",
        **kwargs
    ):
        super().__init__()
        self.pretrain_model_name = pretrain_model_name
        self.task_type = task_type
        self.finetune_type = finetune_type
        hid_dim = 480
        if pretrain_model_name == 'esm2_35m':
            self.input_dim = 480
        
        if pretrain_model_name == 'esm2_150m':
            self.input_dim = 640
            
        if pretrain_model_name == 'esm2_650m':
            self.input_dim = 1280
        
        if pretrain_model_name == 'esm2_3b':
            self.input_dim = 2560
        
        if pretrain_model_name == 'esm2_15b':
            self.input_dim = 5120
        
        if pretrain_model_name == 'esm3_1.4b':
            self.input_dim = 1536
        
        if pretrain_model_name == 'esmc_600m':
            self.input_dim = 1152
        
        if pretrain_model_name == 'progen2':
            self.input_dim = 1536
        
        if pretrain_model_name == 'prostt5':
            self.input_dim = 2048
        
        if pretrain_model_name == 'protgpt2':
            self.input_dim = 1280
            
        if pretrain_model_name == 'protrek_35m':
            self.input_dim = 480*2
            
        if pretrain_model_name == 'protrek':
            self.input_dim = 1920
        
        if pretrain_model_name == 'saport':
            self.input_dim = 1280

        if pretrain_model_name == 'saport_35m':
            self.input_dim = 480
        
        if pretrain_model_name == 'saport_1.3b':
            self.input_dim = 1280

        if pretrain_model_name == 'procyon':
            self.input_dim = 4096

        if pretrain_model_name == 'prollama':
            self.input_dim = 4096
        
        if pretrain_model_name == 'prost':
            self.input_dim = 512
        
        if pretrain_model_name == 'gearnet':
            self.input_dim = 3072
        
        if pretrain_model_name == 'venusplm':
            self.input_dim = 1024

        if pretrain_model_name == 'prosst2048':
            self.input_dim = 768

        if pretrain_model_name == 'prott5':
            self.input_dim = 1024
        
        if pretrain_model_name == 'dplm':
            self.input_dim = 1280

        if pretrain_model_name == 'dplm_150m':
            self.input_dim = 640

        if pretrain_model_name == 'dplm_3b':
            self.input_dim = 2560
        
        if pretrain_model_name == 'ontoprotein':
            self.input_dim = 1024

        if pretrain_model_name == 'ankh_base':
            self.input_dim = 768
            
        if pretrain_model_name == 'pglm':
            self.input_dim = 2048
        
        if pretrain_model_name == "pglm-3b":
            self.input_dim = 2560
            

        self.smiles_proj = nn.Sequential(
            nn.Linear(2048, hid_dim),
            # nn.GELU()
        )
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, hid_dim),
            # nn.LayerNorm(hid_dim)
        )
        self.layernorm = nn.LayerNorm(hid_dim)
        if finetune_type == 'adapter':
            self.adapter = TransformerAdapter(
                input_dim=hid_dim,               # 输入维度
                hidden_dim=hid_dim,          # 隐藏层维度
                num_layers=6,                # Transformer 层数
                num_heads=20,                 # 多头注意力头数
            )
        elif finetune_type == 'peft': 
            self.pretrain_model_interface = PretrainModelInterface(
                pretrain_model_name,
                task_type=self.task_type
            )
            self.pretrain_model_interface.setup_peft(
                peft_type=peft_type,
                **kwargs
            )
            self.pretrain_model = self.pretrain_model_interface.pretrain_model.model
        
        if task_type in ['classification', 'residual_classification']:
            self.task_head = nn.Linear(hid_dim, num_classes)
            self.loss = nn.CrossEntropyLoss()
        
        if task_type in [
            "regression",
            "pair_regression"
        ]:
            self.task_head = nn.Sequential(nn.Linear(hid_dim, 1),
                                           nn.Flatten(start_dim=0, end_dim=1))
            self.loss = nn.MSELoss()
        
        
        if task_type == 'contact':
            self.task_head = ContactPredictionHead(hid_dim)
            self.loss = ContatcLoss()
        
        if task_type in [
            'binary_classification', 
            'pair_binary_classification',
            'multi_labels_classification',
        ]:
            self.task_head = nn.Linear(hid_dim, num_classes)
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, batch):
        if self.finetune_type == 'adapter':
            labels = batch['label']
            attention_mask = batch['attention_mask']
            embeddings = batch['embedding']
            proj_output = self.proj(embeddings)
            proj_output = self.adapter(proj_output, mask=attention_mask)
            if batch['smiles'] is not None:
                smiles =  batch['smiles']
                smiles_proj_output = self.smiles_proj(smiles).unsqueeze(1)
                smiles_attention_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device).bool()
                proj_output = torch.cat((smiles_proj_output, proj_output), dim=1).contiguous()
                attention_mask = torch.cat((smiles_attention_mask, attention_mask), dim=-1).contiguous()

        elif self.finetune_type == "peft":
            out = self.pretrain_model_interface(batch)
            embeddings, labels, attention_mask, smiles = out
            proj_output = self.proj(embeddings.to(self.proj[0].weight.dtype))
            if smiles is not None:
                smiles_proj_output = self.smiles_proj(smiles).unsqueeze(1)
                smiles_attention_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device).bool()
                proj_output = torch.cat((smiles_proj_output, proj_output), dim=1).contiguous()
                attention_mask = torch.cat((smiles_attention_mask, attention_mask), dim=-1).contiguous()
                
        proj_output = self.layernorm(proj_output)
            
        if self.task_type == 'contact': # residue-level
            logits = self.task_head(proj_output)
            loss = self.loss(logits, labels.float(), attention_mask)
            return {'loss': loss, 'logits': logits, 'label': labels, 'attention_mask': attention_mask}
        elif self.task_type == 'residual_classification': # resideu-level
            logits = self.task_head(proj_output)
            logits = logits[attention_mask]
            labels = labels[attention_mask]
            loss = self.loss(logits, labels.long())
            return {'loss': loss, 'logits': logits, 'label': labels, 'attention_mask': attention_mask}
        else: # sequence-level
            pooled_output = torch.mean(proj_output, dim=1)
            logits = self.task_head(pooled_output)
            if isinstance(self.loss, nn.BCEWithLogitsLoss):
                labels = labels.float()
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
            elif isinstance(self.loss, nn.CrossEntropyLoss):
                # logits = logits.float()
                labels = labels.long()
            else:
                # MSELoss, L1Loss 等
                labels = labels.to(logits.dtype)
            loss = self.loss(logits, labels)
            return {'loss': loss, 'logits': logits, 'label': labels, 'attention_mask': attention_mask}
        
# Transformer Adapter 模块
class TransformerAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerAdapter, self).__init__()

        # 定义 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,        # 输入维度 (embedding size)
            nhead=num_heads,          # 多头注意力
            dim_feedforward=hidden_dim*4,  # FFN 中间维度
            activation='gelu',
            batch_first=True          # 使用 batch_first 使 (batch, seq_len, dim) 格式
        )
        
        # Transformer Encoder 堆叠 num_layers 层
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x, mask=None):
        """
        x: 输入嵌入, 形状 (batch_size, seq_len, dim)
        mask: 注意力掩码, 形状 (batch_size, seq_len)
        """
        
        # 通过 Transformer Adapter 处理
        output = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        return output

class ContactPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        hidden_size *= 2
        self.activation_func = nn.functional.relu 
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        self.final_activation = torch.nn.Sigmoid()
        for sz in [128, 1]:
            this_layer = torch.nn.Linear(last_size, sz, bias=True)
            last_size = sz
            torch.nn.init.kaiming_uniform_(this_layer.weight, nonlinearity='relu')
            torch.nn.init.zeros_(this_layer.bias)
            self.layers.append(this_layer)


    def forward(self, embeddings, **kwargs):
        logits = torch.cat([(embeddings[:,:,None]+embeddings[:,None,:]), torch.max(embeddings[:,:,None], embeddings[:,None,:])], dim=-1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits


class ContatcLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, attn_masks):
        """
        logits: logits Tensor of shape (batch_size, L, L)
        labels: Tensor of shape (batch_size, L, L)
        attn_masks: Tensor of shape (batch_size, L)
        """
        logits = logits.squeeze(-1).float()
        batch_size, L, _ = logits.shape

        # Create pairwise mask from 1D attention mask
        pairwise_mask = (attn_masks.unsqueeze(2) * attn_masks.unsqueeze(1)).bool()

        # Exclude positions where |i - j| < 6
        idxs = torch.arange(L, device=logits.device)
        distance_mask = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs() > 6

        # Only consider upper triangle
        upper_triangle_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=logits.device), diagonal=1)

        # Combine masks
        final_mask = pairwise_mask & distance_mask.unsqueeze(0) & upper_triangle_mask.unsqueeze(0)

        # Mask out invalid positions
        logits = logits[final_mask]
        labels = labels[final_mask]

        # Flatten and compute BCEWithLogits loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return loss
    

# def metric_eval(pred_y, y, inds, ls, lens):
#     tests = []
#     t_y = []
#     rs = []
#     for idx in inds:
#         row = idx // lens
#         col = idx % lens
#         if row >= col:
#             continue
#         if abs(row - col) <= 6:
#             continue
#         p = pred_y[idx]
#         gt = y[idx]
#         tests.append((p,gt))
#         if len(tests)>=ls:
#             break
#     cnt = 0
#     for p, gt in tests:
#         if gt == 1:
#             cnt += 1
#     return cnt, ls, cnt/ls


# def contact_metrics(preds, labels, attn_masks):
#     '''
#     pred, label: [B, L, L]
#     '''
#     total_acc = 0
#     valid_samples = 0
#     for b in range(preds.shape[0]):
#         pred = preds[b]
#         label = labels[b]
#         mask = attn_masks[b]==1
#         pred = pred[:mask.sum(), :mask.sum()]
#         label = label[:mask.sum(), :mask.sum()]
        
#         label[label>0] = -1
#         label[label==0] = 1
#         label[label==-1] = 0
#         pred = pred.reshape(-1)
#         label = label.reshape(-1)
#         indices = torch.argsort(-pred)
#         l = label.shape[-1]
#         _,_, acc = metric_eval(pred, label, indices, l//5, l)
#         total_acc += acc
#         valid_samples += 1
#     return {"Top(L/5)": total_acc / valid_samples if valid_samples > 0 else 0.0}


def top_L_div_5_precision(preds, labels, attn_masks):
    """
    preds: logits Tensor of shape (batch_size, L, L)
    labels: Tensor of shape (batch_size, L, L)
    attn_masks: Tensor of shape (batch_size, L)
    """
    batch_size, L, _ = preds.shape
    precisions = []

    # Precompute static masks
    idxs = torch.arange(L, device=preds.device)
    distance_mask = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs() >= 6
    upper_triangle_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=preds.device), diagonal=1)
    combined_static_mask = distance_mask & upper_triangle_mask

    for b in range(batch_size):
        pred = preds[b]  # (L, L)
        label = labels[b]  # (L, L)
        mask = attn_masks[b]  # (L,)

        # Only consider valid positions
        valid_mask = (mask.unsqueeze(0) * mask.unsqueeze(1)).bool()

        combined_mask = valid_mask & combined_static_mask

        pred_scores = pred[combined_mask].flatten()
        true_labels = label[combined_mask].flatten()

        # Apply sigmoid to logits to get probabilities
        pred_probs = torch.sigmoid(pred_scores)

        # Top L/5
        num_top = max(1, L // 5)
        if pred_probs.numel() < num_top:
            num_top = pred_probs.numel()
        topk = torch.topk(pred_probs, k=num_top)
        top_indices = topk.indices

        top_true = true_labels[top_indices]
        precision = top_true.sum().float() / num_top
        precisions.append(precision)

    return {'Top(L/5)': torch.stack(precisions).mean()}

