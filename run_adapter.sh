export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
python tasks/main.py


# wandb agent biomap_ai/protein_benchmark/sle0eqeh


## ============== sweep 1 ==============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/45hvbpwo &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/45hvbpwo &

# =============== sweep yeast ppi ===============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/kzduxv67 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/kzduxv67 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/kzduxv67 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/kzduxv67 &

# =============== EC ===============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/pomoov69 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/pomoov69 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/pomoov69 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/pomoov69 &

# =============== human ppi ===============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/f6tvsb74 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/f6tvsb74 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/f6tvsb74 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/f6tvsb74 &


# =============== GO CC ===============
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/97zkwpi2 &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/97zkwpi2 &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/97zkwpi2 &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/97zkwpi2 &

# =============== GO BP ===============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/4e0r6pcm &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/4e0r6pcm &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/sq4otzmx &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/sq4otzmx &

# =============== GO MF ===============
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/u9x8o6d0 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/u9x8o6d0 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/u9x8o6d0 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/u9x8o6d0 &
# CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/pomoov69

# CUDA_VISIBLE_DEVICES=0 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=1 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=2 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16

# CUDA_VISIBLE_DEVICES=3 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16
