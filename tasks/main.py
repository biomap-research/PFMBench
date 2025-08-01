import datetime
import os
import random
import sys; sys.path.append("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark")
sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/model_zoom')
# sys.path.append(os.getcwd())
# os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # gzy
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # wh
import warnings
warnings.filterwarnings("ignore")
import argparse
import pandas as pd
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from model_interface import MInterface
from data_interface import DInterface
import pytorch_lightning.loggers as plog
from src.utils.logger import SetupCallback
from pytorch_lightning.callbacks import EarlyStopping
from src.utils.utils import process_args
import math
import wandb


def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    # parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pretrain_batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=1022, type=int)
    parser.add_argument('--gpus_per_node', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    
    # Training parameters
    parser.add_argument('--epoch', default=50, type=int, help='end epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='cosine')
    
    # Model parameters
    parser.add_argument('--sequence_only', default=0, type=int)
    parser.add_argument('--finetune_type', default='adapter', type=str, choices=['adapter', 'peft'])
    parser.add_argument('--peft_type', default='adalora', type=str, choices=['lora', 'adalora', 'ia3', 'dora', 'freeze'])
    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=[
        'esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 
        'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm', 
        'prott5', 'dplm', 'ontoprotein', 'ankh_base', 'pglm', 'esm2_35m', 'esm2_150m', 
        'esm2_3b',  'esm2_15b', 'protrek_35m', 'saport_35m', 'saport_1.3b', 'dplm_150m', 'dplm_3b', 'pglm-3b'
    ])
    parser.add_argument("--config_name", type=str, default='fitness_prediction', help="Name of the Hydra config to use")
    parser.add_argument("--metric", type=str, default='val_loss', help="metric for early stop")
    parser.add_argument("--direction", type=str, default='min', help="metric direction")
    parser.add_argument("--enable_es", type=int, default=1, help="enable early stopping")
    parser.add_argument("--feature_extraction", type=int, default=0, help="perform feature extraction(paper used only)")
    parser.add_argument("--feature_save_dir", type=str, default=None, help="feature saved dir(paper used only)")
   
    args = process_args(parser, config_path='../../tasks/configs')
    print(args)
    return args


def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")

    metric = "val_loss" if args.metric is None else args.metric
    direction = "min" if args.direction is None else args.direction
    # metric = "val_loss"
    print(f"metric: {metric}, direction: {direction}")
    sv_filename = 'best-{epoch:02d}-{val_loss:.4f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=3,
        mode=direction,
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
    
    if args.enable_es:
        early_stop_callback = EarlyStopping(
            monitor=metric,
            patience=5,
            mode=direction,
            strict=True,
        )
        callbacks.append(early_stop_callback)
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks


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
                    project = 'protein_benchmark',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    dir = str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    entity = "biomap_ai")
    
    #================ for wandb sweep ==================
    args, logger = automl_setup(args, logger)
    #====================================================
    
    # generated a random seed
    args.seed = random.randint(1, 9999)
    print(f"Generated random seed: {args.seed}")
    # seed everything
    pl.seed_everything(args.seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data_module = DInterface(**vars(args))

    # here we perform feature extraction
    if args.feature_extraction and args.finetune_type == "adapter":
        data_module.data_setup(target="test")
        feature_save_dir = "./feature_extraction" if not args.feature_save_dir else args.feature_save_dir
        os.makedirs(feature_save_dir, exist_ok=True)
        config_res_dir = os.path.join(feature_save_dir, args.config_name)
        os.makedirs(config_res_dir, exist_ok=True)
        result_parquet = f"{args.pretrain_model_name}-{args.config_name}.parquet"
        result_parquet = os.path.join(config_res_dir, result_parquet)
        test_dataset = data_module.test_set
        results = []
        for sdata in test_dataset.data:
            embedding = sdata['embedding'].mean(0).flatten().float().numpy()
            label = sdata['label'].numpy().item() # only for multi-label classification tasks
            results.append(
                {
                    "embedding": embedding,
                    "label": label,
                    "model": f"{args.pretrain_model_name}"
                }
            )
        results = pd.DataFrame(results)
        results.to_parquet(result_parquet, index=False, engine="pyarrow")
        print(f"[Feature extraction] Result parquet: {result_parquet}")
        return 
    
    data_module.data_setup()
    gpu_count = torch.cuda.device_count()
    steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)
    args.lr_decay_steps =  steps_per_epoch*args.epoch
    
    model = MInterface(**vars(args))

    data_module.MInterface = model
    callbacks = load_callbacks(args)
    trainer_config = {
        "accelerator": "gpu",
        'devices': gpu_count,  # Use gpu count
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
        "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
        "precision": 'bf16', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val':1.0,
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    
    trainer = Trainer(**vars(trainer_opt))

    trainer.fit(model, data_module)

    # ============================
    # 4. 评估最佳模型
    # ============================
    checkpoint_callback = callbacks[0]
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    # 载入最佳模型
    model_state_path = os.path.join(checkpoint_callback.best_model_path, "checkpoint", "mp_rank_00_model_states.pt")
    state = torch.load(model_state_path, map_location="cuda:0")
    model.load_state_dict(state['module'])

    # 进行测试
    results = trainer.test(model, datamodule=data_module)
    # 打印测试结果
    print(f"Test Results: {results}")


if __name__ == "__main__":
    main()
    
