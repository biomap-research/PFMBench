from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
import torch

venusplm_weight_path = "YOUR WEIGHTS PATH"
config = TransformerConfig.from_pretrained(venusplm_weight_path, attn_impl="sdpa") # or "flash_attn" if you have installed flash-attn
model = TransformerForMaskedLM.from_pretrained(venusplm_weight_path, config=config)
tokenizer = VPLMTokenizer.from_pretrained(venusplm_weight_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
sequence = "MALWMRLLPLLALLALWGPDPAAA"
encoded_sequence = tokenizer(sequence, return_tensors="pt").to(device)

input_ids = encoded_sequence["input_ids"]
attention_mask = encoded_sequence["attention_mask"]

with torch.no_grad():
   outputs = model(
      input_ids=input_ids, 
      attention_mask=attention_mask,
      output_hidden_states=True
   )

hidden_states = outputs.hidden_states[-1]
print(hidden_states.shape) # [1, L, 1024]
