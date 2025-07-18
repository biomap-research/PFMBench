import logging
import functools
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


def tokenize_protein(example, protein_tokenizer=None, padding=None):
    protein_seqs = example["prot_seq"]
    
    protein_inputs = protein_tokenizer(protein_seqs, padding=padding, add_special_tokens=True)
    example["protein_input_ids"] = protein_inputs.input_ids
    example["protein_attention_mask"] = protein_inputs.attention_mask
    
    return example


def label_embedding(labels, text_tokenizer, text_model, device):
    # embed label descriptions
    label_feature = []
    with torch.inference_mode():
        for label in labels:
            label_input_ids = text_tokenizer.encode(label, max_length=128,
                                                    truncation=True, add_special_tokens=False)
            label_input_ids = [text_tokenizer.cls_token_id] + label_input_ids
            label_input_ids = torch.tensor(label_input_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = label_input_ids != text_tokenizer.pad_token_id
            attention_mask = attention_mask.to(device)

            text_outputs = text_model(label_input_ids, attention_mask=attention_mask)

            label_feature.append(text_outputs["text_feature"])
    label_feature = torch.cat(label_feature, dim=0)
    label_feature = label_feature / label_feature.norm(dim=-1, keepdim=True)

    return label_feature

def zero_shot_eval(logger, device,                                          
                   test_dataset, target_field, protein_model, logit_scale, label_feature):

    # get prediction and target
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    preds, targets = [], []
    with torch.inference_mode():
        for data in tqdm(test_dataloader):
            target = data[target_field]
            targets.append(target)

            protein_input_ids = torch.tensor(data["protein_input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(data["protein_attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            
            protein_outputs = protein_model(protein_input_ids, attention_mask=attention_mask)

            protein_feature = protein_outputs["protein_feature"]
            protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
            pred = logit_scale * protein_feature @ label_feature.t()
            preds.append(pred)

    preds = torch.cat(preds, dim=0)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    accuracy = (preds.argmax(dim=-1) == targets).float().mean().item()
    logger.warning("Zero-shot accuracy: %.6f" % accuracy)


if __name__ == "__main__":
    # get datasets
    raw_datasets = load_dataset("Jiqing/ProtST-SubcellularLocalization", cache_dir="~/.cache/huggingface/datasets", split='test') # cache_dir defaults to "~/.cache/huggingface/datasets"
    
    #device = torch.device("cuda:0")
    device = torch.device("cpu")
    
    protst_model = AutoModel.from_pretrained("Jiqing/ProtST-esm1b", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    protein_model = protst_model.protein_model
    # + import intel_extension_for_pytorch as ipex
    # + from optimum.intel.generation.modeling import jit_trace
    # + protein_model = ipex.optimize(protein_model, dtype=torch.bfloat16, inplace=True)
    # + protein_model = jit_trace(protein_model, "sequence-classification")
    text_model = protst_model.text_model
    logit_scale = protst_model.logit_scale
    logit_scale.requires_grad = False
    logit_scale = logit_scale.to(device)
    logit_scale = logit_scale.exp()

    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    func_tokenize_protein = functools.partial(tokenize_protein, protein_tokenizer=protein_tokenizer, padding=False)
    test_dataset = raw_datasets.map(
            func_tokenize_protein, batched=False,
            remove_columns=["prot_seq"
            desc="Running tokenize_proteins on dataset",
        )

    labels = load_dataset("Jiqing/subloc_template", cache_dir="~/.cache/huggingface/datasets")["train"]["name"]

    text_tokenizer.encode(labels[0], max_length=128, truncation=True, add_special_tokens=False)
    label_feature = label_embedding(labels, text_tokenizer, text_model, device)
    zero_shot_eval(logger, device, test_dataset, "localization",
                   protein_model, logit_scale, label_feature)
