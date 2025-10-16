export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python script/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/train.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/train \
    --max_input_length 10000

python script/preparation/label.py \
    --num_gpu 8 \
    --dataset_path ./data/initial_data/openr1-math/eval.jsonl \
    --test_model_list Qwen/Qwen3-1.7B \
    --output_path ./data/processed_data/openr1-math/1_7/eval \
    --max_input_length 10000 \