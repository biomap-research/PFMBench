import torch
from progen2.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/progen2'
model = ProGenForCausalLM.from_pretrained(ckpt).to(device)

tokens = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'


tokenizer = create_tokenizer_custom(file='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/progen2/tokenizer.json')

target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
logits = model(target, labels=target).logits
print(logits.shape)