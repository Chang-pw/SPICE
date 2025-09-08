#!/bin/bash

# SPICE Training Script
# Example usage for multi-GPU training

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=36000
export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export TORCH_NCCL_BLOCKING_WAIT=1

accelerate launch --mixed_precision fp16 --multi_gpu --num_processes 4 train.py \
  --config configs/default.yaml \
  --model_name Qwen/Qwen2-0.5B \
  --project spice-experiment --run_name qwen2-0.5b-lora-selection \
  --pool_size 124 --select_k 64 \
  --max_length 2048 \
  --per_device_train_batch_size 4 \
  --selection_method top_k \
  --use_batch_gradient_optimization true --batch_gradient_size 4 \
  --use_gradient_projection false \
  --lora_r 8 --lora_alpha 16 \
  --logging_step_freq 10 \
  --save_checkpoint_freq 100 \
  --save_total_limit 3 \
  --output_dir ./outputs \
  --num_epochs 1 \
  --checkpoint_dir ./checkpoints \
  --update_frequency 10 \
  --conflict_penalty 0.1