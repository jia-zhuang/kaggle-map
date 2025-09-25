#!/usr/bin/env bash

DATA_DIR=./input/
IS_SMALL=False
MODEL_PATH='./pretrained_models/Qwen2.5-0.5B/'
OUTPUT_DIR='./output/qwen25_p5b_cls/'
MAX_SEQ_LEN=192
NUM_EPOCHS=10
ZERO_STAGE=2
per_device_batch_size=32
gradient_accumulation_steps=1
SEED=42
LR=5e-5
NUM_LABELS=65


deepspeed  --master_port 9527  --include localhost:0,1 deepspeed_qwen25_qlora_train.py \
       --data_dir ${DATA_DIR} \
       --is_small ${IS_SMALL} \
       --model_name_or_path ${MODEL_PATH} \
       --num_labels ${NUM_LABELS} \
       --output_dir ${OUTPUT_DIR} \
       --max_seq_len ${MAX_SEQ_LEN} \
       --num_train_epochs $NUM_EPOCHS \
       --warmup_ratio 0.1 \
       --learning_rate $LR \
       --per_device_train_batch_size ${per_device_batch_size} \
       --per_device_eval_batch_size ${per_device_batch_size} \
       --gradient_accumulation_steps ${gradient_accumulation_steps} \
       --eval_strategy epoch \
       --logging_strategy epoch \
       --save_strategy epoch \
       --report_to tensorboard \
       --dataloader_num_workers 0 \
       --bf16 True \
       --fp16 False \
       --seed $SEED \
       --do_train \
       --do_eval \
       --overwrite_output_dir
