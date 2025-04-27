export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
python tasks/main.py


# wandb agent biomap_ai/protein_benchmark/sle0eqeh


## ============== sweep 1 ==============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/e3o436rb &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/e3o436rb &

wandb agent biomap_ai/protein_benchmark/e3o436rb