export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=6,7

accelerate launch --num_processes 2 --main_process_port 29516 train_olora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'ace2005' \
    --data_root './data/ACE2005-en' \
    --batch_size 4 \
    --lr 0.0004 \
    --epochs 10 \
    --disable_shared_attn \
    --log_path './log' \
    --output_path './checkpoints/olora/ace'