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
memory_usage_mb = memory_usage / 1024

print(f"Memory Usage: {memory_usage_mb:.2f} MB")
import torch
from model_interface import MInterface
from data_interface import DInterface
import math
import hydra
from model_interface import MInterface
from data_interface import DInterface
from functools import partial
from megatron import print_rank_0
import pandas as pd
import glob
from megatron.core import mpu
from scipy.stats import spearmanr
from src.utils.utils import process_args
import pytorch_lightning.loggers as plog
from src.utils.utils import process_args
import math
import wandb
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from src.utils.logger import SetupCallback
from pytorch_lightning.trainer import Trainer


def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    

    metric = "val_loss"
    sv_filename = 'best-{epoch:02d}-{val_loss:.4f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=3,
        mode='min',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        # every_n_train_steps=args.check_val_every_n_step
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks


def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gpus_per_node', default=1, type=int)
    
    parser.add_argument('--database_path', default='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/DMS_ProteinGym_substitutions', type=str)
    

    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=['esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm'])
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
    wandb.init(project='protein_benchmark', entity='biomap_ai')
    logger = plog.WandbLogger(
                    project = 'protein_benchmark',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    entity = "biomap_ai")
    
    #================ for wandb sweep ==================
    args, logger = automl_setup(args, logger)
    #====================================================
    
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    steps_per_epoch = 10000
    args.lr_decay_steps =  steps_per_epoch*args.epoch
    
    model_module = MInterface(**vars(args))
    
    callbacks = load_callbacks(args)
    trainer_config = {
        "accelerator": "gpu",
        'devices': args.gpus_per_node,  # Use all available GPUs
        "strategy": 'ddp', # 'ddp', 'deepspeed_stage_2
        "precision": 'bf16', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val':1.0,
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    
    trainer = Trainer(**vars(trainer_opt))
    
    dms_files = list(glob.iglob(os.path.join(args.database_path, "*.csv"), recursive=True))

    print_rank_0(f"There are {len(dms_files)} to be evaluated")
    dms_files.sort()
    for i, data_path in enumerate(dms_files):
        ## ================= change dataset =================
        data_module.data_setup(data_path, model_module.model.pretrain_model.tokenizer)
        
        ## ================= inference results =================
        trainer.validate(model_module, data_module)
        val_spearman = model_module.val_spearman
        
       


if __name__ == "__main__":
    main()
    