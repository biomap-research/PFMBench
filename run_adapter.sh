export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
CUDA_VISIBLE_DEVICES=4 python tasks/main.py --config_name enzyme_commission_number --pretrain_model_name esm3_1.4b --offline 0


# wandb agent biomap_ai/protein_benchmark/sle0eqeh


## ============== sweep 1 ==============
export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/n0r2hjd8 &
wandb agent biomap_ai/protein_benchmark/n0r2hjd8


## ============== sweep EC&CONTACTMAP ESM-2 ==============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/vpzg822x &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/vpzg822x &
wandb agent biomap_ai/protein_benchmark/io3xdph3
