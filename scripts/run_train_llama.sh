export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --num_processes 2 --main_process_port 29501 train.py \
    --base_model_name 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --lr 0.0002 \
    --epochs 10 \
    --kl_lambda 2.0 \
    --log_path './log' \