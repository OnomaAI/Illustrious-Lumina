#!/usr/bin/env sh

train_data_path='./configs/data.yaml'
model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=checkpoint
batch_size=16
snr_type=lognorm
lr=2e-4
precision=bf16
size=1024

exp_name="${model}_bs${batch_size}_lr${lr}_${precision}"
mkdir -p results/"$exp_name"

# Set environment variables
export NNODES=1
export NPROC_PER_NODE=8
export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=18182
export NODE_RANK=0

# (Optional) Activate conda environment or other environment setup
# source activate your-env

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  finetune.py \
    --master_port 18182 \
    --global_bsz_1024 768 \
    --micro_bsz_1024 6 \
    --model ${model} \
    --lr ${lr} \
    --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 100 \
    --log_every 10 \
    --precision ${precision} \
    --grad_precision fp32 \
    --qk_norm \
    --global_seed 20241207 \
    --num_workers 1 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    2>&1 | tee -a results/"$exp_name"/output.log
