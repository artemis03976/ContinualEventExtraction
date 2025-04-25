export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --num_processes 2 --main_process_port 29501 train.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --lr 0.0005 \
    --epochs 10 \
    --kl_lambda 4.0 \
    --log_path './log' \