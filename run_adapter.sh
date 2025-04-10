export PYTHONPATH=/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH
python tasks/main.py 

# CUDA_VISIBLE_DEVICES=0 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=1 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_adapter' OFFLINE='0' python tasks/main.py 

# CUDA_VISIBLE_DEVICES=2 CONFIG_NAME='fitness_prediction' EXP_NAME='fitness_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16

# CUDA_VISIBLE_DEVICES=3 CONFIG_NAME='fold_prediction' EXP_NAME='fold_prediction_lora' OFFLINE='0' python tasks/main.py Model.finetune_type=lora Training.batch_size=16
