import argparse
import json, os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import GenerationConfig
from tqdm import tqdm

# basic configuration from HF: https://huggingface.co/GreatCaptainNemo/ProLLaMA
generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=400,
    output_hidden_states=True,
    output_logits=True,
    return_dict_in_generate=True
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str,help="The local path of the model. If None, the model will be downloaded from HuggingFace")
parser.add_argument('--interactive', action='store_true', help="If True, you can input instructions interactively. If False, the input instructions should be in the input_file.")
parser.add_argument('--input_file', default=None, help="You can put all your input instructions in this file (one instruction per line).")
parser.add_argument('--output_file', default=None, help="All the outputs will be saved in this file.")
args = parser.parse_args()

if __name__ == '__main__':
    # if args.interactive and args.input_file:
    #     raise ValueError("interactive is True, but input_file is not None.")
    # if (not args.interactive) and (args.input_file is None):
    #     raise ValueError("interactive is False, but input_file is None.")
    # if args.input_file and (args.output_file is None):
    #     raise ValueError("input_file is not None, but output_file is None.")

    load_type = torch.bfloat16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        raise ValueError("No GPU available.")

    args.model = "/nfs_beijing/kubeflow-user/wanghao/workspace/ai4sci/protein_benchmark_project/data/ProLLaMA"
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=None
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    model.eval()
    with torch.no_grad():
        if args.interactive:
            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                input_text = raw_input_text
                input_text = tokenizer(input_text,return_tensors="pt")  

                generation_output = model.generate(
                                input_ids = input_text["input_ids"].to(device),
                                attention_mask = input_text['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                generation_config = generation_config,
                                output_attentions=False
                            )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                print("Output:",output)
                print("\n")
        else:
            examples, logits_outpus, hidden_state_outputs = [], [], []
            # case 1: Generation task by provideing the superfamily name and specify the first few 
            # amino acids of the protein sequence: 
            # [Generate by superfamily] Superfamily=<Ankyrin repeat-containing domain superfamily>
            # examples.append("[Generate by superfamily] Superfamily=<Ankyrin repeat-containing domain superfamily> Seq=<MKRVL")

            # case 2: Determine the superfamily of the given sequence
            # Here we use a instance from our sota dataset -- `enzyme_catalytic_efficiency`` 
            examples.append("[Determine superfamily] Seq=<MKLNFSGLRALVTGAGKGIGRDTVKALHASGAKVVAVDRTNSDLVSLAKECPGIEPVCVDLGDWDATEKALGGIGPVDLLVNNAALVIMQPFLEVTKEAFDRSFSVNLRSVFQVSQMVARDMINRGVPGSIVNVSSMVAHVTFPNLITYSSTKGAMTMLTKAMAMELGPHKIRVNSVNPTVVLTDMGKKVSADPEFARKLKERHPLRKFAEVEDVVNSILFLLSDRSASTSGGGILVDAGYLAS>")
            print("Start generating...")
            for index, example in tqdm(enumerate(examples), total=len(examples)):
                input_text = tokenizer(example, return_tensors="pt")  #add_special_tokens=False ?

                # generation_output = model.generate(
                #     input_ids = input_text["input_ids"].to(device),
                #     attention_mask = input_text['attention_mask'].to(device),
                #     eos_token_id=tokenizer.eos_token_id,
                #     pad_token_id=tokenizer.pad_token_id,
                #     generation_config = generation_config
                # )
                generation_output = model(
                    input_ids = input_text["input_ids"].to(device),
                    attention_mask = input_text['attention_mask'].to(device),
                    output_hidden_states=True
                )
                breakpoint()
                logits, hidden_states = generation_output.logits, generation_output.hidden_states
                logits_outpus.append(logits)
                hidden_state_outputs.append(hidden_states)
                breakpoint()