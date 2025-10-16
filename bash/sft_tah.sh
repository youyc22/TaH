#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

### step 1
python -m accelerate.commands.launch \
    --config_file ./script/recipes/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    ./script/train/SFT_TaH.py \
    --config ./script/recipes/qwen3_1.7/sft_tah_step1.yaml

### step 2
# python -m accelerate.commands.launch \
#     --config_file ./script/recipes/accelerate_configs/zero2.yaml \
#     --num_processes 8 \
#     ./script/train/SFT_TaH.py \
#     --config ./script/recipes/qwen3_1.7/sft_tah_step2.yaml