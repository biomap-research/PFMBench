import torch
import os
from procyon.model.model_unified import UnifiedProCyon
from procyon.data.inference_utils import (
    create_caption_input_simple,
    uniprot_id_to_index,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Initialize tokenizer and model
CKPT_NAME = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom/procyon/model_weights/ProCyon-Full'
os.environ["LLAMA3_PATH"] = "/nfs_beijing/wanghao/2025-onesystem/vllm/Meta-Llama-3-8B"
model, _ = UnifiedProCyon.from_pretrained(pretrained_weights_dir=CKPT_NAME, checkpoint_dir=CKPT_NAME)
model = model.to(device)
data_args = torch.load(os.path.join(CKPT_NAME, "data_args.pt"))

# Internally, ProCyon uses integer IDs that have been assigned to UniProt proteins in ProCyon-Instruct.
want_proteins = ["Q5T1N1"]
protein_ids = [uniprot_id_to_index(x) for x in want_proteins]

input_simple = create_caption_input_simple(
    input_aaseq_ids=protein_ids,
    data_args=data_args,
    # The `instruction_source_dataset` and `instruction_source_relation` here control the style
    # of pre-templated instruction used in these queries. In particular, here we query for UniProt-style
    # functional descriptions.
    instruction_source_dataset="uniprot",
    instruction_source_relation="all",
    aaseq_type="protein",
    task_type="caption",
    icl_example_number=1,
    device=device,
)

text_gen_args = {
    "method": "beam",
    # Maximum length of generated text.
    "max_len": 200,
    # Total number of beams maintained per input. `beam_size` / `beam_group_size` = number of phenotypes returned per input.
    "beam_size": 20,
    # Size of the individual beam groups in DBS.
    "beam_group_size": 2,
    # Penalty applied to repetition within a beam group.
    "diversity_penalty": 0.8,
}

out_tokens, log_probs, output_logits, out_text = model.generate(
    inputs=input_simple,
    aaseq_type="protein",
    **text_gen_args
)

breakpoint()
print(":)")
