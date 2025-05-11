export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=7

accelerate launch --num_processes 1 --main_process_port 29510 train.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --batch_size 2 \
    --lr 0.0002 \
    --epochs 10 \
    --distill_lambda 0.5 \
    --use_distill \
    --log_path './log' \
    --output_path './checkpoints/qwen2.5' \