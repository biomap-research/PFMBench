import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class BaseProteinModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
    
    def construct_batch(self, data, batch_size, task_name=None):
        raise NotImplementedError
    
    def forward(self, batch):
        raise NotImplementedError


class ESM2Model(BaseProteinModel):
    def __init__(self, device, model_path, max_length=1022, **kwargs):
        super().__init__(device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length

    def construct_batch(self, data, batch_size, task_name=None):
        for i in range(0, len(data), batch_size):
            # 构造seq, attention_mask, label...
            yield {
                'seq': ...,
                'attention_mask': ...,
                'label': ...,
                'name': ...
            }

    def forward(self, batch):
        seq, attention_mask = batch['seq'], batch['attention_mask']
        output = self.model(
            input_ids=seq,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return output.last_hidden_state