# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/OntoProtein")
model = AutoModelForMaskedLM.from_pretrained("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/OntoProtein")

import re
sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model.bert(
    input_ids=encoded_input['input_ids'],
    attention_mask=encoded_input['attention_mask'],
    token_type_ids=encoded_input['token_type_ids'],
)

print(output.last_hidden_state)