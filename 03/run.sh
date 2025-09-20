DATA_DIR=''
IS_SMALL=True
MODEL_PATH=/root/storage/jiazhuang.jz/pretrained_models/qwen2.5-14b/
MAX_SEQ_LEN=192
ZERO_STAGE=2
per_device_batch_size=12
num_warmup_steps=100
gradient_accumulation_steps=1



deepspeed  --master_port 9527  --include localhost:0 debug.py \
       --data_dir ${DATA_DIR} \
       --is_small ${IS_SMALL} \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size ${per_device_batch_size} \
       --per_device_eval_batch_size ${per_device_batch_size} \
       --gradient_accumulation_steps ${gradient_accumulation_steps} \
       --max_seq_len ${MAX_SEQ_LEN} \
       --earlystop 0 \ 
       --save_batch_steps 100000000000 \
       --early_stop_epoch 5 \ 
       --save_per_epoch 1 \ 
       --num_train_epochs 20  \
       --learning_rate 5e-5 \
       --num_warmup_steps ${num_warmup_steps} \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing \
       --overwrite_cache False
