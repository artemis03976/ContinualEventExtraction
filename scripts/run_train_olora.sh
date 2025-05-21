export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=2,3

# accelerate launch --num_processes 2 --main_process_port 29515 train_olora.py \
#     --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
#     --dataset 'maven' \
#     --data_root './data/MAVEN' \
#     --batch_size 4 \
#     --lr 0.0003 \
#     --epochs 10 \
#     --disable_shared_attn \
#     --log_path './log' \
#     --output_path './checkpoints/olora/maven'

accelerate launch --num_processes 2 --main_process_port 29516 train_olora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'ace2005' \
    --data_root './data/ACE2005-en' \
    --strategy 'local' \
    --batch_size 4 \
    --lr 0.0004 \
    --epochs 10 \
    --disable_shared_attn \
    --log_path './log' \
    --output_path './checkpoints/olora/ace'

accelerate launch --num_processes 2 --main_process_port 29517 train_olora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'ere' \
    --data_root './data/ERE' \
    --batch_size 4 \
    --lr 0.0004 \
    --epochs 10 \
    --disable_shared_attn \
    --log_path './log' \
    --output_path './checkpoints/olora/ere'