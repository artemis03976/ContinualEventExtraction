export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=5

accelerate launch --num_processes 1 --main_process_port 29512 train.py \
    --base_model_name 'Qwen/Qwen2.5-7B-Instruct' \
    --dataset 'ere' \
    --data_root './data/ERE' \
    --batch_size 4 \
    --lr 0.00002 \
    --epochs 30 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --replay_lambda 1.0 \
    --distill_lambda 100 \
    --use_distill \
    --log_path './log' \
    --output_path './checkpoints/qwen2.5/ere'