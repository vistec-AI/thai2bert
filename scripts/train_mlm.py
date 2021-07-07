from argparse import ArgumentError
import torch
import os
import logging


import glob
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CamembertTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import data

#thai2transformers
from thai2transformers.datasets import MLMDataset
from thai2transformers.conf import DATA_COLLATOR_CLASS_MAPPING, MLMObjective

logger = logging.getLogger(__name__)

@dataclass
class ArchitectureArguments:
    architecture: str = field(
        default='roberta-base',
        metadata={'help': 'Name of architecture to be pre-trained.'}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    use_fast_tokenizer: bool = field(
        default = True,
        metadata = {
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        }
    )
    do_lower_case: bool = field(
        default = False,
        metadata = {
            "help": "Whether to lower case the text during pretraining"
        }
    )
    space_token: str = field(
        default='▁'
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dir: str = field(
        metadata={"help": "The input training data dir (dir that contain text files)."}
    )
    eval_dir: str = field(
        metadata={"help": "The input evaluation data dir (dir that contain text files)."},
    )
    binarized_path_train: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the cached binrauzed train dataset."}
    )
    binarized_path_eval: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the cached binrauzed eval dataset"},
    ) 
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to the model checkpoint (for resume pretraining)."}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    train_max_length: int = field(
         default=510,
    )
    eval_max_length: int = field(
         default=510,
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mlm_strategy: str = field(
        default='subword-level',
        metadata={"help": "Strategy to mask tokens in sequence"}
    )
    span_mlm_max_gram: int = field(
        default=3,
        metadata={"help": "Maximum tokens in a span to be masked (for `span-level` masking strategy)"}
    )
    pad_to_multiple_of: int = field(
        default=None,
        metadata={'help': 'Add padding tokens to the multiple of an integer specified (e.g. 8).'}
    )

    def __post_init__(self):
        if self.train_dir is None and self.eval_dir is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def is_main_process(rank):
    return rank in [-1, 0]


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               ArchitectureArguments))

    (model_args, data_args, training_args, arch_args) = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG
    )

     # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # set seed
    set_seed(training_args.seed)

    tokenizer = CamembertTokenizer.from_pretrained(model_args.tokenizer_name_or_path,
                                              use_fast=model_args.use_fast_tokenizer,
                                              do_lower_case=model_args.do_lower_case,
                                              additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', model_args.space_token])
    
    logger.debug(f'\n\ntokenizer: {tokenizer}')
    logger.debug(f'\ntokenizer vocab size: {tokenizer.vocab_size}')

    config = AutoConfig.from_pretrained(arch_args.architecture, vocab_size=tokenizer.vocab_size)
    
    logger.debug(f'\n\nconfig: {config}')


    if data_args.checkpoint_dir != None:
        if is_main_process(training_args.local_rank):
            print(f'\n[INFO] Load pretrianed model (state_dict) from checkpoint: {data_args.checkpoint_dir}')
        model = AutoModelForMaskedLM.from_pretrained(data_args.checkpoint_dir)
    else:
        if is_main_process(training_args.local_rank):
            print(f'\nInitailize weights')
        model = AutoModelForMaskedLM.from_config(config=config)

    # Load dataset
    train_dataset = MLMDataset(tokenizer, data_args.train_dir, data_args.train_max_length, binarized_path=data_args.binarized_path_train)
    eval_dataset = MLMDataset(tokenizer, data_args.eval_dir, data_args.eval_max_length, binarized_path=data_args.binarized_path_eval)
    
    #data collator
    data_collator_class = DATA_COLLATOR_CLASS_MAPPING[data_args.mlm_strategy]
    if data_args.mlm_strategy == MLMObjective.SUBWORD_LEVEL.value:
        data_collator = data_collator_class(tokenizer=tokenizer,
                            mlm=True,
                            mlm_probability=data_args.mlm_probability,
                            pad_to_multiple_of=data_args.pad_to_multiple_of
                        )
    elif data_args.mlm_strategy == MLMObjective.SPAN_LEVEL.value:
        data_collator = data_collator_class(tokenizer=tokenizer,
                            mlm=True,
                            mlm_probability=data_args.mlm_probability,
                            pad_to_multiple_of=data_args.pad_to_multiple_of,
                            max_gram=data_args.span_mlm_max_gram,
                            max_seq_len=data_args.train_max_length
                        )
    else:
        raise ArgumentError('data_args.mlm_strategy specified is not in avaialble stratefy (%s))', list(DATA_COLLATOR_CLASS_MAPPING.keys()))

            
    

    logger.debug(f'\n\ntraining_args: {training_args}')


    logging.info(" Number of devices: %d", training_args.n_gpu)
    logging.info(" Device: %s", training_args.device)
    logging.info(" Local rank: %s", training_args.local_rank)
    logging.info(" FP16 Training: %s", training_args.fp16)
    logging.info(" Run name: %s", training_args.run_name)
    logging.info(" deepspeed: %s", training_args.deepspeed)

    if is_main_process(training_args.local_rank):
        print('\nmodel:', model)
        print('\ntraining_args:', training_args)

    # Initiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


    if data_args.checkpoint_dir != None:
        logger.info(f' \nTrainer resume from the checkpoint specified: {data_args.checkpoint_dir}.')
        trainer.train(resume_from_checkpoint=data_args.checkpoint_dir)
    else:
        trainer.train()

    if trainer.is_world_process_zero():
            
        logger.info('\nSave tokenizer and model state')
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'final_checkpoint-tokenizer'))
        trainer.save_model(os.path.join(training_args.output_dir, 'final_checkpoint-model'))

        logger.info('\n\nEvaluate final model checkpoint')
        eval_results = trainer.evaluate()
        if training_args.local_rank == 0:
            logger.info(f'\nEvaluation result: {eval_results}')

        return eval_results


if __name__ == "__main__":
    main()
