import argparse
import json
import time
import os
import math
import sys
import random
from dataclasses import dataclass
from typing import List, Dict

from filelock import FileLock

import pandas as pd
from torch import nn, Tensor
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModel, MistralPreTrainedModel, MistralConfig, DynamicCache, Cache
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig
)
from transformers import optimization

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from utils.sft_dataset import is_rank_0
import torch.distributed as dist
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

import logging
logger = logging.getLogger(__name__)


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: int


class MapDataset(Dataset):
    def __init__(self, data_dir, split, is_small, tokenizer, max_seq_len, overwrite_cache=False):
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(split, tokenizer.__class__.__name__, str(max_seq_len)),
        )
        if is_small: cached_features_file += '_small'

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)

            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                # read examples
                data_file = f'{split}_small.jsonl' if is_small else f'{split}.jsonl'
                examples = pd.read_json(os.path.join(data_dir, data_file), lines=True)
                # convert examples to features
                self.features = []
                for ex_idx, example in enumerate(examples.itertuples()):
                    tokend = tokenizer(example.text, padding="max_length", truncation=True, max_length=max_seq_len)
                    self.features.append(InputFeature(
                        input_ids=tokend.input_ids,
                        attention_mask=tokend.attention_mask,
                        label_ids=example.label,
                    ))

                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)   
 
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class Qwen2ForClassification(nn.Module):
    def __init__(self, args):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModel.from_pretrained(args.model_name_or_path, quantization_config=bnb_config)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="FEATURE_EXTRACTION",
        )

        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        self.dropout = nn.Dropout(args.cls_dropout)
        self.classifier = nn.Linear(model.config.hidden_size, args.num_labels)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state   # (batch_size, sequence_length, hidden_size)
        pooled_output = self.dropout(hidden_states[:, -1])   # use last token as pooled output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}
    

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    set_random_seed(args.seed)

    torch.distributed.barrier()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2ForClassification.from_pretrained(args)

    train_dataset = MapDataset(
        data_dir=args.data_dir,
        split='train',
        is_small=args.is_small,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        overwrite_cache=args.overwrite_cache,
    )

    def data_collator(features) -> Dict[str, torch.Tensor]:
        batch = {}
        for k in ('input_ids', 'attention_mask'):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        
        batch['labels'] = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        return batch

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,  # custom sampler
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
        num_workers=4,
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * 0.03) if args.num_warmup_steps == 0 else args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    total_steps = len(train_loader) * args.num_train_epochs
    total_loss = 0
    best_val_loss = 1000.
    no_improve_epoch = 0.
    global_step = -1
    time_start = time.time()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )

        model.train()
        for step, batch in enumerate(train_loader):
            global_step += 1
            batch = to_device(batch, device)
            loss = model(**batch)['loss']
            model.backward(loss)
            model.step()
            total_loss += loss.item()

            if global_step % 10 == 0:
                time_end = time.time()
                time_elapse = time_end - time_start
                time_start = time_end
                print_rank_0(
                    f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, curr_step:{global_step}/{total_steps} curr_loss {loss.item()} lr:{lr_scheduler.get_last_lr()[0]} use time:{time_elapse}s",
                    args.global_rank,
                )
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                total_loss = 0.
            
        if args.save_per_epoch == 1:
            save_model(args, model, tokenizer, f'epoch_{epoch}_model')
        
        # 保存最后一轮
        if epoch == args.num_train_epochs - 1:
            save_model(args, model, tokenizer, f'epoch_{epoch}_model')

        model.tput_timer.update_epoch_count()    


def save_model(args, model, tokenizer, sub_fold=None):
    if sub_fold is not None:
        output_dir = os.path.join(args.output_dir, sub_fold)
        print_rank_0('saving model ...', args.global_rank)
        tokenizer.save_pretrained(output_dir)
        # model = convert_lora_to_linear_layer(model)
        if args.global_rank == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save = model_to_save.model
            # model_to_save.save_pretrained(output_dir)

            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "adapter.bin"
            os.makedirs(output_dir, exist_ok=True)
            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            save_dict = model_to_save.state_dict()
            final_d = {}
            for k, v in save_dict.items():
                if "lora" in k:
                    final_d[k] = v
            torch.save(final_d, output_model_file)

            # state_dict = self.model.state_dict()
            # state_dict = type(state_dict)(
            #     {k: v.clone().cpu()
            #      for k,
            #          v in state_dict.items()})
            # self.model.save_pretrained(output_dir, state_dict=state_dict)
        print_rank_0('saving success ...', args.global_rank)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")

    parser.add_argument('--overwrite_cache',
                        action='store_true',
                        help='overwrite data prepare cache')

    parser.add_argument('--save_batch_steps', type=int, default=1000)
    parser.add_argument('--earlystop', type=bool, default=False)
    parser.add_argument('--early_stop_epoch', type=int, default=2)
    parser.add_argument('--save_per_epoch', type=int, default=-1)
    parser.add_argument('--data_dir', type=str, default=None, help="train and val data path")
    parser.add_argument('--max_seq_len', type=int, default=192, help="input text sequence length")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_path",
                        type=str,
                        default='none',
                        help="lora path")

    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_config",
                        type=str,
                        default="./configs/lora_config_llama.json",
                        help="If > 0, use LoRA for efficient training.")

    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

if __name__ == "__main__":
    main()
