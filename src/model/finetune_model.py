import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, TaskType, get_peft_model


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
        if pretrain_model_name == 'esm2_650m':
            self.tokenizer = AutoTokenizer.from_pretrained("model_zoom/esm2_650m")
            pretrain_model = AutoModelForMaskedLM.from_pretrained("model_zoom/esm2_650m")
            if finetune_type == 'lora':
                lora_config = LoraConfig(
                                        inference_mode=False,        # 训练模式
                                        r=lora_r,                         # 低秩矩阵的秩
                                        lora_alpha=lora_alpha,               # LoRA 的 alpha 参数
                                        lora_dropout=lora_dropout,            # Dropout 防止过拟合
                                        target_modules=["query", "value"],  # 仅调整 Attention 的 query 和 value
                                        )
                self.pretrain_model = get_peft_model(pretrain_model, lora_config)
                self.proj = nn.Linear(1280, hid_dim)
            
            if finetune_type == 'adapter':
                for param in pretrain_model.parameters():
                    param.requires_grad = False
                self.pretrain_model = pretrain_model
                self.proj = nn.Linear(1280, hid_dim)
                self.adapter = TransformerAdapter(
                    input_dim=hid_dim,               # 输入维度
                    hidden_dim=hid_dim,          # 隐藏层维度
                    num_layers=6,                # Transformer 层数
                    num_heads=20,                 # 多头注意力头数
                )
                
            
            
        
        if task_type == 'classification':
            self.task_head = nn.Linear(hid_dim, num_classes)
            self.loss = nn.CrossEntropyLoss()
        
        if task_type == 'regression':
            self.task_head = nn.Sequential(nn.Linear(hid_dim, 1),
                                           nn.Flatten(start_dim=0, end_dim=1))
            
            self.loss = nn.MSELoss()
    
    def forward(self, batch):
        # coords, seqs, labels=None
        seqs = batch['seq']
        attention_mask = batch['mask']==0
        labels = batch['label']
        if self.pretrain_model_name == 'esm2_650m':
            if self.finetune_type == 'lora':
                outputs = self.pretrain_model.esm(
                    seqs,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state
                proj_output = self.proj(hidden_states)
                
            if self.finetune_type == 'adapter':
                with torch.no_grad():
                    outputs = self.pretrain_model.esm(
                        seqs,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                hidden_states = outputs.last_hidden_state
                proj_output = self.proj(hidden_states)
                # 通过 Transformer Adapter 处理
                proj_output = self.adapter(proj_output, mask=attention_mask)
            
            
            pooled_output = torch.mean(proj_output, dim=1)
            logits = self.task_head(pooled_output)
        
            if labels is not None:
                loss = self.loss(logits, labels)
                return {'loss': loss, 'logits': logits}
            else:
                return {'logits': logits}
        
# Transformer Adapter 模块
class TransformerAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerAdapter, self).__init__()

        # 定义 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,        # 输入维度 (embedding size)
            nhead=num_heads,          # 多头注意力
            dim_feedforward=hidden_dim,  # FFN 中间维度
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

