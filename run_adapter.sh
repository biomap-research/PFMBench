export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
CUDA_VISIBLE_DEVICES=0 python tasks/main.py --config_name binding_db --pretrain_model_name esm2_35m --offline 0 &
CUDA_VISIBLE_DEVICES=1 python tasks/main.py --config_name binding_db --pretrain_model_name esm2_150m --offline 0 &
CUDA_VISIBLE_DEVICES=2 python tasks/main.py --config_name binding_db --pretrain_model_name protrek_35m --offline 0 


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

CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/4m9wj48s &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/4m9wj48s &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/4m9wj48s &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/4m9wj48s &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/4m9wj48s &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/a2ftj08u &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/pm5qxhzz &

CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/a2ftj08u &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/a2ftj08u &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/a2ftj08u &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/a2ftj08u &

CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/pm5qxhzz &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/pm5qxhzz &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/pm5qxhzz &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/pm5qxhzz &

wandb agent biomap_ai/protein_benchmark/0i48r0yz


## ============== sweep scaling ==============
export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark:$PYTHONPATH" 
CUDA_VISIBLE_DEVICES=0 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=1 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=2 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=3 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=4 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=5 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=6 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
CUDA_VISIBLE_DEVICES=7 wandb agent biomap_ai/protein_benchmark/yps8okh5 &
wandb agent biomap_ai/protein_benchmark/yps8okh5


## ============== sweep peft ==============
CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}-4_wandb_peft_sweep_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}-4_wandb_peft_sweep_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}-4_wandb_peft_sweep_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}-4_wandb_peft_sweep_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}_wandb_peft_sweep_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}_wandb_peft_sweep_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}_wandb_peft_sweep_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup wandb agent biomap_ai/protein_benchmark/90fygdhp >./.${HOSTNAME}_wandb_peft_sweep_7.log 2>&1 &
wandb agent biomap_ai/protein_benchmark/kms5qkdq


CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/8j872non >./.${HOSTNAME}_wandb_protrek_sweep_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/8j872non >./.${HOSTNAME}_wandb_protrek_sweep_5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/0rmyk46z >./.${HOSTNAME}_wandb_protrek_remains_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/m143xdba >./.${HOSTNAME}_wandb_protrek_remains_1.log 2>&1 &

# ============= pdbbind-binding-stability-rebuild =============
CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup wandb agent biomap_ai/protein_benchmark/kikjx1xd >./.${HOSTNAME}_wandb_pdbbindingstab_7.log 2>&1 &

wandb agent biomap_ai/protein_benchmark/kikjx1xd

# ===================== pglm-3b ======================
CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_7.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_aav_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/i4a0vmxe >./.${HOSTNAME}_pglm3b_aav_1.log 2>&1 &
wandb agent biomap_ai/protein_benchmark/wultpsw8
wandb agent biomap_ai/protein_benchmark/i4a0vmxe

CUDA_VISIBLE_DEVICES=0 nohup wandb agent biomap_ai/protein_benchmark/dv5orrrm >./.${HOSTNAME}_twom_check_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent biomap_ai/protein_benchmark/dv5orrrm >./.${HOSTNAME}_twom_check_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup wandb agent biomap_ai/protein_benchmark/01hcv1s8 >./.${HOSTNAME}_threetasks_check_7.log 2>&1 &

wandb agent biomap_ai/protein_benchmark/v3zc1qcg
