import os
import sys
import sys; sys.path.append("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark")
sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
import warnings
warnings.filterwarnings("ignore")
# 打印当前工作目录
# print("Current working directory:", os.getcwd())
import argparse
import resource
import uuid
# 获取当前进程的内存使用情况（单位：KB）
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
memory_usage_mb = memory_usage / 1024

print(f"Memory Usage: {memory_usage_mb:.2f} MB")
import torch
import pandas as pd
import itertools
from src.data.msa_dataset import MSADataset
from src.utils.utils import process_args
import pytorch_lightning.loggers as plog
from tqdm import tqdm
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from src.utils.logger import SetupCallback
from src.data.esm.sdk.api import ESMProtein
from src.model.pretrain_model_interface import PretrainModelInterface
import torch.nn.functional as F

model_name_mapping = {
    'esm2_35m': "ESM-2(35M)", 
    'esm2_150m': "ESM-2(150M)",  
    "esm2_650m": "ESM-2(650M)", 
    'esm2_3b': "ESM-2(3B)", 
    'esm2_15b': "ESM-2(15B)", 
    "esmc_600m": "ESM-C(600M)", 
    "esm3_1.4b": "ESM-3(1.4B)", 
    "protrek_35m": "ProTrek(35M)", 
    "protrek": "ProTrek(650M)", 
    "saport_35m": "SaProt(35M)", 
    "saport": "SaProt(650M)", 
    "saport_1.3b": "SaProt(1.3B)", 
    "venusplm": "VenusPLM(300M)", 
    "dplm": "DPLM(650M)", 
    "pglm": "xTrimoPGLM(1B)", 
}
def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--msa_csv_path', default=None, type=str)
    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=['esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm', 'prott5', 'dplm', 'ontoprotein', 'ankh_base', 'pglm', 'pglm-3b'])
    parser.add_argument("--config_name", type=str, default='fitness_prediction', help="Name of the Hydra config to use")
    args = process_args(parser, config_path='../../tasks/configs')
    print(args)
    return args



def reinit_weights(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def main():
    args = create_parser()
    pl.seed_everything(args.seed)
    
    # dataset
    msa_center = MSADataset(
        msa_csv_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/zeroshot/msa/msa_samples_zeroshot_w_pdb.csv",
        type="center"
    )
    msa_member = MSADataset(
        msa_csv_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/zeroshot/msa/msa_samples_zeroshot_w_pdb.csv",
        type="msa"
    )

    model_msa_scores = {}
    for model_name in [
        'esm2_35m', 'esm2_150m', "esm2_650m", 'esm2_3b', 'esm2_15b', "esmc_600m", "esm3_1.4b", "protrek_35m", "protrek", "venusplm",  "dplm", "pglm"
        # 'saport_35m', "saport", 'saport_1.3b'
    ]:
        if model_name == "pglm":
            start, end = 0, -1
        else:
            start, end = 1, -1
        pretrained_model = PretrainModelInterface(
            model_name
        )
        model = pretrained_model.pretrain_model
        
        tokenizer = model.get_tokenizer()
        # batch_size = args.batch_size
        msa_kl_scores = []
        for ii in tqdm(range(0, len(msa_center)), desc=f"Processing msa center (length={len(msa_center)})..."):
            center = msa_center[ii]
            model_input = [
                {
                    "seq": center["seq"],
                    "X": center.get("X"),
                    "name": center["name"],
                    "label": 1.0,
                }
            ]
            # with torch.no_grad():
            if 'saport' in model_name:
                batch = model.construct_batch(model_input)
                center_seq_token = batch["seq"][0][start:end].to(model.device)
            # elif 'protrek' in model_name:
            #     center_seq_token = tokenizer.batch_encode_plus(center["seq"], return_tensors="pt", padding=True)['input_ids'][0][1:-1].to(model.device)
            else:
                center_seq_token = tokenizer(center["seq"], return_tensors="pt", padding=True)['input_ids'][0][start:end].to(model.device)
            corres_msas = [ele for ele in msa_member.data if ele["name"] == center["unique_id"]]
            if len(corres_msas) == 0:
                continue
            model_msa_input = [
                {
                    "seq": ele["seq"],
                    "X": ele.get("X"),
                    "name": ele["name"],
                    "label": 1.0,
                } for ele in corres_msas
            ]
            positions = [ele["position"] for ele in corres_msas] # alist of tensors
            # TODO
            with torch.no_grad():
                batch = model.construct_batch(model_msa_input)
                outputs = model.forward(batch=batch, return_logits=True)
                logits_msa = outputs[:, start:end, :]
                probs_msa = torch.log_softmax(logits_msa, dim=-1)
            
            information_list = []
            for i, pos in enumerate(positions):
                msa_probs_i = probs_msa[i]  # [L_msa_i, C]
                if isinstance(pos, str): 
                    information_list.append(0.0)
                    continue
                L_i = pos.shape[0]

                if L_i > msa_probs_i.shape[0]:
                    information_list.append(0.0)
                    continue  # 防止越界

                msa_probs_aligned = msa_probs_i[:L_i]  # [L_i, C]

                valid_mask = (pos >= 0) & (pos < len(center_seq_token))
                if valid_mask.sum() == 0:
                    information_list.append(0.0)
                    continue

                valid_pos = pos[valid_mask]                      # [L_valid]
                valid_center_aas = center_seq_token[valid_pos]     # [L_valid]
                valid_msa_probs = msa_probs_aligned[valid_mask]  # [L_valid, C]
                valid_msa_probs = torch.gather(valid_msa_probs, dim=1, index=valid_center_aas.unsqueeze(1))
                information_list.append(valid_msa_probs.mean().item())
            if information_list:
                msa_kl_scores.append(information_list)
        msa_kl_scores = list(itertools.chain.from_iterable(msa_kl_scores))
        print(f"{model_name_mapping[model_name]} length: {len(msa_kl_scores)}")
        model_msa_scores[f"{model_name_mapping[model_name]}"] = msa_kl_scores
    model_msa_scores = pd.DataFrame(model_msa_scores)
    model_msa_scores["name"] = [uuid.uuid4().hex for _ in range(len(model_msa_scores))]
    cols = model_msa_scores.columns.tolist()
    cols.insert(0, cols.pop(cols.index('name')))
    model_msa_scores = model_msa_scores[cols]

    reference_col = 'ESM-2(35M)'
    # 提取 name 列
    name_col = model_msa_scores[['name']]
    # 对除 'name' 外的所有列作差（减去 'ESM-2(35M)' 列）
    diff_df = model_msa_scores.drop(columns=['name']).subtract(model_msa_scores[reference_col], axis=0)
    # 将 name 列加回来
    result_df = pd.concat([name_col, diff_df], axis=1)
    result_df.to_csv("./model_msa_mi_mutation_relative.csv", index=False)

    # reference_col = 'ESM-2(35M)'
    # name_col = model_msa_scores[['name']]
    # cols_to_diff = [col for col in model_msa_scores.columns if col != 'name']
    # diff_df = model_msa_scores[cols_to_diff].apply(lambda col: model_msa_scores[reference_col] - col)
    # result_df = pd.concat([name_col, diff_df], axis=1)
    # result_df.to_csv("./model_msa_mi_mutation_relative.csv", index=False)
    model_msa_scores.to_csv("./model_msa_mi_mutation_absolute.csv", index=False)
    

if __name__ == "__main__":
    main()
    