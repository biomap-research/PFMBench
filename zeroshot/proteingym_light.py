import datetime
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
# 获取当前进程的内存使用情况（单位：KB）
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
memory_usage_mb = memory_usage / 1024

print(f"Memory Usage: {memory_usage_mb:.2f} MB")
import torch
from model_interface import MInterface
from data_interface import DInterface
from src.utils.utils import process_args
import pytorch_lightning.loggers as plog
import math
import wandb
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from src.utils.logger import SetupCallback
from pytorch_lightning.trainer import Trainer


def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--offline', default=0, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gpus_per_node', default=1, type=int)
    
    parser.add_argument('--dms_csv_dir', default='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions', type=str)
    parser.add_argument('--dms_pdb_dir', default='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions/ProteinGym_AF2_structures', type=str)
    parser.add_argument('--dms_reference_csv_path', default='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions/ProteinGym_AF2_structures/DMS_substitutions.csv', type=str)

    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=['esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm', 'prott5', 'dplm', 'ontoprotein', 'ankh_base', 'pglm', 'pglm-3b'])
    parser.add_argument("--config_name", type=str, default='fitness_prediction', help="Name of the Hydra config to use")
   
    args = process_args(parser, config_path='../../tasks/configs')
    print(args)
    return args

def automl_setup(args, logger):
    args.res_dir = os.path.join(args.res_dir, args.ex_name)
    print(wandb.run)
    args.ex_name = wandb.run.id
    wandb.run.name = wandb.run.id
    logger._save_dir = str(args.res_dir)
    os.makedirs(logger._save_dir, exist_ok=True)
    logger._name = wandb.run.name
    logger._id = wandb.run.id
    return args, logger

def main():
    args = create_parser()
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='protein_benchmark', entity='biomap_ai', dir=str(os.path.join(args.res_dir, args.ex_name)))
    logger = plog.WandbLogger(
                    project='protein_benchmark',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    dir = str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    entity = "biomap_ai")
    
    #================ for wandb sweep ==================
    args, logger = automl_setup(args, logger)
    #====================================================
    
    # generated a random seed
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    data_module.data_setup()
    gpu_count = torch.cuda.device_count()
    # steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)
    # args.lr_decay_steps =  steps_per_epoch*args.epoch
    
    model = MInterface(**vars(args))

    data_module.MInterface = model
    trainer_config = {
        "accelerator": "gpu",
        "strategy": 'ddp', # 'ddp', 'deepspeed_stage_2
        'devices': gpu_count,
        "precision": 'bf16', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'logger': logger,
        'gradient_clip_val':1.0,
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    
    trainer = Trainer(**vars(trainer_opt))

    # 进行测试
    trainer.test(model, datamodule=data_module)



if __name__ == "__main__":
    main()
    