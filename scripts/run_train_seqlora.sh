export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=0,2

# accelerate launch --num_processes 2 --main_process_port 29520 train_seqlora.py \
#     --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
#     --dataset 'maven' \
#     --data_root './data/MAVEN' \
#     --batch_size 4 \
#     --lr 0.0004 \
#     --epochs 10 \
#     --disable_shared_attn \
#     --log_path './log' \
#     --output_path './checkpoints/seqlora/maven' \

accelerate launch --num_processes 2 --main_process_port 29520 train_seqlora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'ace2005' \
    --data_root './data/ACE2005-en' \
    --strategy 'local' \
    --batch_size 4 \
    --lr 0.0004 \
    --epochs 10 \
    --disable_shared_attn \
    --log_path './log' \
    --output_path './checkpoints/seqlora/ace' \

# accelerate launch --num_processes 2 --main_process_port 29520 train_seqlora.py \
#     --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
#     --dataset 'ere' \
#     --data_root './data/ERE' \
#     --batch_size 4 \
#     --lr 0.0004 \
#     --epochs 10 \
#     --disable_shared_attn \
#     --log_path './log' \
#     --output_path './checkpoints/seqlora/ere' \
