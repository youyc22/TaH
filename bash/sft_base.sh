#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m accelerate.commands.launch \
    --config_file ./script/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/train/SFT_TaH.py \
    --config ./script/recipes/qwen3_1.7/sft_base.yaml