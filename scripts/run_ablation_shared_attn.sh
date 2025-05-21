export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=5

accelerate launch --num_processes 1 --main_process_port 29520 train_seqlora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --batch_size 4 \
    --lr 0.0002 \
    --epochs 10 \
    --replay_lambda 0.8 \
    --distill_lambda 10 \
    --use_distill \
    --disable_shared_attn \
    --log_path './log' \
    --output_path './checkpoints/ablation_shared_attn/seqlora_distill' \