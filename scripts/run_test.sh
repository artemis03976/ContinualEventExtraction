export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=1

accelerate launch --num_processes 1 --main_process_port 29506 test.py \
    --checkpoint_path './checkpoints/qwen2.5/ace' \


# accelerate launch --num_processes 1 --main_process_port 29509 test.py \
#     --checkpoint_path './checkpoints/olora/maven' \

# accelerate launch --num_processes 1 --main_process_port 29508 test_bwt.py \
#     --checkpoint_path './checkpoints/qwen2.5/ace' \