export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --num_processes 2 --main_process_port 29510 train_seqlora.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --batch_size 2 \
    --lr 0.0002 \
    --epochs 10 \
    --disable_shared_attn \
    --n_tasks 1 \
    --log_path './log' \
    --output_path './checkpoints/joint/maven'

# accelerate launch --num_processes 2 --main_process_port 29511 train_seqlora.py \
#     --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
#     --dataset 'ace2005' \
#     --data_root './data/ACE2005-en' \
#     --batch_size 4 \
#     --lr 0.0002 \
#     --epochs 10 \
#     --disable_shared_attn \
#     --n_tasks 1 \
#     --log_path './log' \
#     --output_path './checkpoints/joint/ace'

# accelerate launch --num_processes 2 --main_process_port 29512 train_seqlora.py \
#     --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
#     --dataset 'ere' \
#     --data_root './data/ERE' \
#     --batch_size 4 \
#     --lr 0.0002 \
#     --epochs 10 \
#     --disable_shared_attn \
#     --n_tasks 1 \
#     --log_path './log' \
#     --output_path './checkpoints/joint/ere'