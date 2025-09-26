#!/usr/bin/env bash
export MASTER_PORT=29501
export OUTPUT_DIR='./models/gen_cls_14b'
export EPOCHS=10

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model './pretrained_models/qwen2.5-14b/' \
    --train_type lora \
    --dataset './datasets/gen_cls/train_split_message.jsonl' \
    --val_dataset './datasets/gen_cls/valid_split_message.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --gradient_accumulation_steps 1 \
    --eval_strategy 'epoch' \
    --save_strategy 'epoch' \
    --logging_steps 50 \
    --max_length 1024 \
    --output_dir ${OUTPUT_DIR} \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
