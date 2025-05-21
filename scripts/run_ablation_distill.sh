export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=6,7

accelerate launch --num_processes 2 --main_process_port 29510 train.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --batch_size 4 \
    --lr 0.0004 \
    --epochs 10 \
    --log_path './log' \
    --output_path './checkpoints/ablation_distill/none'