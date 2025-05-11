export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes 1 --main_process_port 29505 test.py \
    --checkpoint_path './checkpoints/qwen2.5' \