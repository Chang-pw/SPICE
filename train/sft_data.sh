#!/bin/bash

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd ./LLaMA-Factory-main

methods=("full_data" "random_data" "fisher_data" "less_data" "selecIT_data" "ifd_data" "spice")

MODEL_NAME="Qwen/Qwen2-0.5B"

for method in "${methods[@]}"; do
    echo "Running method: ${method}"

    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path $MODEL_NAME \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template default \
        --flash_attn auto \
        --dataset_dir data \
        --dataset ${method} \
        --cutoff_len 4096 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 1000000 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 1000 \
        --warmup_steps 0 \
        --packing False \
        --enable_thinking True \
        --report_to none \
        --use_swanlab True \
        --output_dir saves/${MODEL_NAME}/lora/${method} \
        --bf16 True \
        --plot_loss True \
        --trust_remote_code True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --optim adamw_torch \
        --lora_rank 16 \
        --lora_alpha 32 \
        --lora_dropout 0 \
        --lora_target all \
        --swanlab_project spice_train_llama \
        --swanlab_run_name ${method} \
        --swanlab_mode cloud
    
    echo "Method ${method} finished"
    sleep 60
    echo "-------------------------------------"
done
