import datetime
import os
import sys; sys.path.append("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark")
# sys.path.append(os.getcwd())
# os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # gzy
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # wh
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from model_interface import MInterface
from data_interface import DInterface
import pytorch_lightning.loggers as plog
from omegaconf import DictConfig, OmegaConf
import hydra
from src.utils.logger import SetupCallback
from src.utils.utils import flatten_dict
import math
import wandb
from hydra import initialize, compose

def eval_resolver(expr: str):
    return eval(expr, {}, {})

OmegaConf.register_new_resolver("eval", eval_resolver, use_cache=False)

# CONFIG_NAME = os.getenv('CONFIG_NAME')
# GPUS_PER_NODE = int(os.getenv('GPUS_PER_NODE', '1'))
# NNODES = int(os.getenv('NNODES', '1'))

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
    parser.add_argument('--finetune_type', default='adapter', type=str)
    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=['esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm'])
    parser.add_argument("--config_name", type=str, default='fitness_prediction', help="Name of the Hydra config to use")
    # ----------------------------------------------------------------------------
    # 1) 先不看命令行，拿到纯“parser 默认值”：
    defaults = parser.parse_args([])
    defaults_dict = vars(defaults)

    # 2) 真正去解析一次命令行（CLI + 默认）：
    args = parser.parse_args()
    args_dict = vars(args)

    # 3) 再读你的 Hydra config，平展开成普通 dict：
    with initialize(config_path="configs"):
        cfg: DictConfig = compose(config_name=args.config_name)
    config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # ----------------------------------------------------------------------------
    # 4) 挖出哪些 key 的值是“真由用户在命令行里指定”的：
    passed = set()
    for tok in sys.argv[1:]:
        if not tok.startswith('--'):
            continue
        # 支持 --foo=bar 和 --foo bar 两种写法
        key = tok.lstrip('-').split('=')[0].replace('-', '_')
        passed.add(key)

    # 5) 最终合并：CLI > config_file > parser_default
    merged = {}
    for key in set(list(defaults_dict.keys())+list(config_dict.keys())):
        if key in passed:
            # 用户显式传进来的
            merged[key] = args_dict[key]
        elif key in config_dict:
            # config 文件里有，且用户没在 CLI 指定，就用它
            merged[key] = config_dict[key]
        else:
            # 都没有指定，就退回 parser 默认
            merged[key] = defaults_dict[key]

    # 用合并后的结果更新 args Namespace
    args.__dict__.update(merged)
    
    print(args)
    return args


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
    

# @hydra.main(version_base=None, config_path="configs", config_name=CONFIG_NAME)
def main():
    args = create_parser()
    
    # config_dict = OmegaConf.to_container(cfg, resolve=True)
    # args = create_parser(hydra_config=argparse.Namespace(**flatten_dict(config_dict)))
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
    data_module.data_setup()
    gpu_count = torch.cuda.device_count()
    steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)
    args.lr_decay_steps =  steps_per_epoch*args.epoch
    
    model = MInterface(**vars(args))

    data_module.MInterface = model
    callbacks = load_callbacks(args)
    trainer_config = {
        "accelerator": "gpu",
        'devices': args.gpus_per_node,  # Use all available GPUs
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
    
