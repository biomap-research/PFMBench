import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
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

def eval_resolver(expr: str):
    return eval(expr, {}, {})

OmegaConf.register_new_resolver("eval", eval_resolver, use_cache=False)

CONFIG_NAME = os.getenv('CONFIG_NAME')
GPUS_PER_NODE = int(os.getenv('GPUS_PER_NODE', '1'))
NNODES = int(os.getenv('NNODES', '1'))

def create_parser(hydra_config=None):
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--offline', default=0, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=1022, type=int)
    
    # Training parameters
    parser.add_argument('--epoch', default=5, type=int, help='end epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='cosine')
    
    # Model parameters
    parser.add_argument('--finetune_type', default='adpter', type=str)

    # 使用 sys.argv 解析命令行参数，优先级高于 hydra 配置
    args, unknown = parser.parse_known_args(args=sys.argv[1:], namespace=hydra_config)
    
    # 重新解析一次以确保控制台参数生效
    args, unknown = parser.parse_known_args(namespace=args)

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
        save_top_k=15,
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

@hydra.main(version_base=None, config_path="configs", config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    args = create_parser(hydra_config=argparse.Namespace(**flatten_dict(config_dict)))
    
    
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
        'devices': int(GPUS_PER_NODE),  # Use all available GPUs
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': int(NNODES),  # Number of nodes to use for distributed training
        "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
        "precision": 'bf16', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': callbacks,
        'logger': plog.WandbLogger(
                    project = 'protein_benchmark',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                     entity = "gaozhangyang"),
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
    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    # 进行测试
    results = trainer.test(best_model, datamodule=data_module)

    # 打印测试结果
    print(f"Test Results: {results}")


if __name__ == "__main__":
    main()
    
