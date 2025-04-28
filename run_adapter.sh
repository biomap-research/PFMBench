export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
python tasks/main.py


# wandb agent biomap_ai/protein_benchmark/sle0eqeh


## ============== sweep 1 ==============
export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/rsowxdxx &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/1y4wa75d &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/q73afwmk &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/f42mugbe &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/q73afwmk &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/q73afwmk &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/q73afwmk &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/q73afwmk &
wandb agent biomap_ai/protein_benchmark/f42mugbe

## ============== sweep EC&CONTACTMAP ESM-2 ==============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/b9j5kgpo &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/b9j5kgpo &
wandb agent biomap_ai/protein_benchmark/io3xdph3
