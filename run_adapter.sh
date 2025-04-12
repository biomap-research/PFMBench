export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
python tasks/main.py


# wandb agent biomap_ai/protein_benchmark/sle0eqeh


## ============== sweep 1 ==============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/s3k36twd &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/s3k36twd 

wandb agent biomap_ai/protein_benchmark/s3k36twd


# CUDA_VISIBLE_DEVICES=0 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=1 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=2 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16

# CUDA_VISIBLE_DEVICES=3 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16
