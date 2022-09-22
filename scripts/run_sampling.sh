#/bin/bash

# $1: task
# $2: gpu number

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/$1_config.yaml \
    --gpu=$2 \
    --save_dir=$3;
