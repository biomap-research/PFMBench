import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
# from peft import LoraConfig, TaskType, get_peft_model


class UniModel(nn.Module):
    def __init__(
        self,
        pretrain_model_name: str,
        task_type: str,
        finetune_type: str,
        num_classes: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        self.pretrain_model_name = pretrain_model_name
        self.task_type = task_type
        self.finetune_type = finetune_type
        hid_dim = 480
        if finetune_type == 'adapter':
            if pretrain_model_name == 'esm2_650m':
                self.input_dim = 1280
            
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
            
            if pretrain_model_name == 'protrek':
                self.input_dim = 1920
            
            if pretrain_model_name == 'saport':
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

            self.smiles_proj = nn.Sequential(nn.Linear(2048, hid_dim),
                                      nn.GELU()
            )
            self.proj = nn.Sequential(nn.Linear(self.input_dim, hid_dim),
                                      nn.LayerNorm(hid_dim))
            
            self.adapter = TransformerAdapter(
                    input_dim=hid_dim,               # 输入维度
                    hidden_dim=hid_dim,          # 隐藏层维度
                    num_layers=6,                # Transformer 层数
                    num_heads=20,                 # 多头注意力头数
                )
        
        if pretrain_model_name == 'prollama':
            from transformers import LlamaForCausalLM, LlamaTokenizer
            llama_path = "/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_project/data/ProLLaMA"
            pretrain_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.bfloat16,
                # low_cpu_mem_usage=True,
                # device_map='auto',
                quantization_config=None
            )
            if finetune_type == 'lora':
                lora_config = LoraConfig(
                                        inference_mode=False,        # 训练模式
                                        r=lora_r,                         # 低秩矩阵的秩
                                        lora_alpha=lora_alpha,               # LoRA 的 alpha 参数
                                        lora_dropout=lora_dropout,            # Dropout 防止过拟合
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 仅调整 Attention 的 query 和 value
                                        )
                self.pretrain_model = get_peft_model(pretrain_model, lora_config)
                self.proj = nn.Linear(4096, hid_dim)

        # if pretrain_model_name == 'esm2_650m':
        #     self.tokenizer = AutoTokenizer.from_pretrained("model_zoom/esm2_650m")
        #     pretrain_model = AutoModelForMaskedLM.from_pretrained("model_zoom/esm2_650m")
        #     if finetune_type == 'lora':
        #         lora_config = LoraConfig(
        #                                 inference_mode=False,        # 训练模式
        #                                 r=lora_r,                         # 低秩矩阵的秩
        #                                 lora_alpha=lora_alpha,               # LoRA 的 alpha 参数
        #                                 lora_dropout=lora_dropout,            # Dropout 防止过拟合
        #                                 target_modules=["query", "value"],  # 仅调整 Attention 的 query 和 value
        #                                 )
        #         self.pretrain_model = get_peft_model(pretrain_model, lora_config)
        #         self.proj = nn.Linear(1280, hid_dim)
            
        #     if finetune_type == 'adapter':
        #         for param in pretrain_model.parameters():
        #             param.requires_grad = False
        #         self.pretrain_model = pretrain_model
        #         self.proj = nn.Linear(1280, hid_dim)
        #         self.adapter = TransformerAdapter(
        #             input_dim=hid_dim,               # 输入维度
        #             hidden_dim=hid_dim,          # 隐藏层维度
        #             num_layers=6,                # Transformer 层数
        #             num_heads=20,                 # 多头注意力头数
        #         )
                
            
            
        
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
            if batch['smiles'] is not None:
                smiles =  batch['smiles']
                smiles_proj_output = self.smiles_proj(smiles).unsqueeze(1)
                smiles_attention_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device).bool()
                proj_output = torch.cat((smiles_proj_output, proj_output), dim=1).contiguous()
                attention_mask = torch.cat((smiles_attention_mask, attention_mask), dim=-1).contiguous()
            proj_output = self.adapter(proj_output, mask=attention_mask)
            
            if self.task_type == 'contact': # resideu-level
                logits = self.task_head(proj_output)
                loss = self.loss(logits, labels, batch['attention_mask'])
                return {'loss': loss, 'logits': logits}
            elif self.task_type == 'residual_classification': # resideu-level
                logits = self.task_head(proj_output)
                logits = logits[attention_mask]
                labels = labels[attention_mask]
                loss = self.loss(logits, labels)
                return {'loss': loss, 'logits': logits}
            else: # sequence-level
                pooled_output = torch.mean(proj_output, dim=1)
                logits = self.task_head(pooled_output)
                if isinstance(self.loss, nn.BCEWithLogitsLoss):
                    labels = labels.float()
                    if labels.ndim == 1:
                        labels = labels.unsqueeze(1)
                loss = self.loss(logits, labels)
                return {'loss': loss, 'logits': logits}

        
        # seqs = batch['seq']
        # attention_mask = batch['mask']==0
        # labels = batch['label']
        # if self.pretrain_model_name == 'esm2_650m':
        #     if self.finetune_type == 'lora':
        #         outputs = self.pretrain_model.esm(
        #             seqs,
        #             attention_mask=attention_mask,
        #             return_dict=True,
        #         )
        #         hidden_states = outputs.last_hidden_state
        #         proj_output = self.proj(hidden_states)
        
        # if self.pretrain_model_name == 'prollama':                
        #     if self.finetune_type == 'lora':
        #         outputs = self.pretrain_model(
        #             input_ids = seqs,
        #             attention_mask=attention_mask,
        #             output_hidden_states=True
        #         )
        #         hidden_states = outputs.hidden_states[-1]
        #         proj_output = self.proj(hidden_states)
                
        # #     if self.finetune_type == 'adapter':
        # #         with torch.no_grad():
        # #             outputs = self.pretrain_model.esm(
        # #                 seqs,
        # #                 attention_mask=attention_mask,
        # #                 return_dict=True,
        # #             )
        # #         hidden_states = outputs.last_hidden_state
        # #         proj_output = self.proj(hidden_states)
        # #         # 通过 Transformer Adapter 处理
        # #         proj_output = self.adapter(proj_output, mask=attention_mask)
            
            
        
        #     if labels is not None:
        #         loss = self.loss(logits, labels)
        #         return {'loss': loss, 'logits': logits}
        #     else:
        #         return {'logits': logits}
        
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

