export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=7

accelerate launch --num_processes 1 --main_process_port 29508 test.py \
    --checkpoint_path './checkpoints/seqlora/ere' \