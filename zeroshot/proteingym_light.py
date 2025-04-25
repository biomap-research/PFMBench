import datetime
import os
import sys
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
import warnings
warnings.filterwarnings("ignore")
# 打印当前工作目录
# print("Current working directory:", os.getcwd())
import argparse
import resource

# 获取当前进程的内存使用情况（单位：KB）
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# 转换为 MB
memory_usage_mb = memory_usage / 1024

print(f"Memory Usage: {memory_usage_mb:.2f} MB")
import torch
from model_interface import MInterface
from data_interface import DInterface
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from src.megatron import set_args, flatten_dict, get_args
from src.interface.trainer import Trainer
from src.megatron.arguments import parse_args
from src.data.tokenizer.esm3.omni_utils.function.general import merge_mutation_probs
from functools import partial
from megatron import print_rank_0
import pandas as pd
import glob
from megatron.core import mpu
from scipy.stats import spearmanr
from src.data.omni_dataset import ProteinGYMDataset

def eval_resolver(expr: str):
    return eval(expr, {}, {})

OmegaConf.register_new_resolver("eval", eval_resolver, use_cache=False)

CONFIG_NAME = os.getenv('CONFIG_NAME')
@hydra.main(version_base=None, config_path="configs", config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Set-up parameters
    # 将 Hydra 配置转换为字典
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    args = parse_args(None, False, hydra_config=argparse.Namespace(**flatten_dict(config_dict)))
    set_args(args.__dict__)
    args = get_args()
    


    import wandb
    os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.login()
    
    data_module = DInterface()
    data_module.initialize()
    data_module.setup()
    model_module = MInterface(**vars(args))
    model_module.load_checkpoint(load_dir=args.load)
    trainer = Trainer(model_module, data_module)
    
    
    database_path = args.infer_data_path
    save_path = args.infer_save_path
    dms_files = list(glob.iglob(os.path.join(database_path, "*.csv"), recursive=True))
    if mpu.get_data_parallel_rank() == 0:
        summary_path, summary = os.path.join(save_path, "summary.csv"), []
    print_rank_0(f"There are {len(dms_files)} to be evaluated")
    dms_files.sort()
    for i, data_path in enumerate(dms_files):
        model_module.predict_dms = {}
        ## ================= change dataset =================
        def train_valid_test_datasets_provider(train_val_test_num_samples=None):
            test_ds = ProteinGYMDataset(
                data_path,
                micro_batch_size=args.micro_batch_size,
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
            return None, None, test_ds
        data_module.setup(train_valid_test_datasets_provider)
        
        
        if mpu.get_data_parallel_rank() == 0:
            filename = os.path.basename(data_path)
            c_save_path = os.path.join(save_path, "predicted_" + filename)
        
        ## ================= inference results =================
        trainer.test(model_module, data_module)
        
        ## ================= save results =================
        predict_dms = model_module.predict_dms
        merge_mutation_probs_partial = partial(merge_mutation_probs, prob_dict=predict_dms)
        
        data_df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
        data_df = data_df.drop_duplicates(subset='mutant', keep='first')
        data_df["predict_DMS_score"] = data_df["mutant"].apply(merge_mutation_probs_partial)
        print_rank_0(f"Saving result file to {c_save_path}...")
        os.makedirs(os.path.dirname(c_save_path), exist_ok=True)
        data_df.to_csv(c_save_path, index=False)
        score, _ = spearmanr(data_df["predict_DMS_score"].tolist(), data_df["DMS_score"].tolist())
        print_rank_0(f"File:{data_path}, Spearman: {score}")
        summary.append({"file": filename, "spearman": score})
        if len(summary) > 0:
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
    