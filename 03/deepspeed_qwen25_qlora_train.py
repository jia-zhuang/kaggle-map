import argparse
import json
import time
import os
import math
import sys
import random
from dataclasses import dataclass, field
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
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
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
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from utils import compute_map3

import logging
logger = logging.getLogger(__name__)


############# Dataset ###############

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
                self.features = torch.load(cached_features_file, weights_only=False)

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


############# Model ###############

class Qwen2ForClassification(nn.Module):
    def __init__(self, model_name_or_path, num_labels, cls_dropout=0.1):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModel.from_pretrained(model_name_or_path, quantization_config=bnb_config, use_cache=False)
        model = prepare_model_for_kbit_training(model)

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

        self.dropout = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state   # (batch_size, sequence_length, hidden_size)
        pooled_output = self.dropout(hidden_states[:, -1])   # use last token as pooled output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    

############# Train ###############

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained'})
    num_labels: int = field(default=2, metadata={'help': 'num of labels'})
    cls_dropout: float = field(default=0.1, metadata={'help': 'cls dropout'})


@dataclass
class DataArguments:
    data_dir: str = field(metadata={'help': 'train / dev data dir'})
    is_small: bool = field(metadata={'help': 'Use small dataset for debug'})
    max_seq_len: int = field(default=192, metadata={'help': 'The maximum total input sequence length after tokenization'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # gradient_checkpointing 
    training_args.gradient_checkpointing=True
    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}

    # Set deepspeed config
    with open('ds_config.json') as f:
        training_args.deepspeed = json.load(f)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = Qwen2ForClassification(
        model_name_or_path=model_args.model_name_or_path,
        num_labels=model_args.num_labels,
        cls_dropout=model_args.cls_dropout,
    )

    # Get dataset
    train_dataset = (
        MapDataset(
            data_dir=data_args.data_dir,
            split='train',
            is_small=data_args.is_small,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        MapDataset(
            data_dir=data_args.data_dir,
            split='valid',
            is_small=data_args.is_small,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_eval
        else None
    )

    def data_collator(features) -> Dict[str, torch.Tensor]:
        batch = {}
        for k in ('input_ids', 'attention_mask'):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        
        batch['labels'] = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        return batch

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_map3,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        save_model(model, training_args.output_dir, training_args.local_rank)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def save_model(model, output_dir, local_rank):
    if local_rank != 0: return
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # save Lora adapter
    model_to_save.model.save_pretrained(output_dir)

    # save classifier
    torch.save(model.classifier.state_dict(), os.path.join(output_dir, 'classifier.pt'))


if __name__ == "__main__":
    main()
