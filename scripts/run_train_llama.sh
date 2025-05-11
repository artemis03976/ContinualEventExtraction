export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=6

accelerate launch --num_processes 1 --main_process_port 29502 train.py \
    --base_model_name '../hf_cache/hub/Llama-3.1-8B-Instruct' \
    --dataset 'maven' \
    --data_root './data/MAVEN' \
    --batch_size 2 \
    --lr 0.0002 \
    --epochs 10 \
    --distill_lambda 0.5 \
    --use_distill \
    --log_path './log' \
    --output_path './checkpoints/llama' \