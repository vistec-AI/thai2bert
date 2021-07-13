# distributed data parallel
import torch
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from transformers import (
    CamembertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)

#thai2transformers
from thai2transformers.datasets import MLMDataset

#argparse
import argparse

def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_mlm_roberthai.py",
        description="train mlm for roberta with huggingface Trainer",
    )
    
    #required
    parser.add_argument("--tokenizer_name_or_path", type=str,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--eval_dir", type=str,)
    parser.add_argument("--num_train_epochs", type=int,)
    parser.add_argument("--max_steps", type=int,)
    
    # DeepSpeed feature
    # ref: https://huggingface.co/transformers/main_classes/trainer.html?highlight=deepspeed
    # (str or dict, optional) – Use Deepspeed. This is an experimental feature and its API may evolve in the future.
    # The value is either the location of DeepSpeed json config file (e.g., ds_config.json) or an already loaded json file as a dict”
    parser.add_argument("--deepspeed", type=str, default=None, help='Path to the DeepSpeed json config') 

    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--overwrite_output_dir', default=True, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    
    #logs
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=200)
    
    #eval
    parser.add_argument('--evaluation_strategy',type=str, default='epoch')
    parser.add_argument("--eval_steps", type=int, default=500)
    
    #train hyperparameters
    parser.add_argument("--train_max_length", type=int, default=512)
    parser.add_argument("--eval_max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument('--dataloader_drop_last', default=False, type=lambda x: (str(x).lower() in ['true','True','T']))
    
    #model architecture
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--num_attention_head", type=int, default=12)
    
    #others
    parser.add_argument("--ext", type=str, default=".txt")
    parser.add_argument("--seed", type=int, default=1412)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--model_path", type=str, default=None) # for resume training
    parser.add_argument("--model_dir", type=str, default=None) # for resume training

    parser.add_argument("--add_space_token", action='store_true', default=False)
    
    parser.add_argument("--binarized_path_train",  type=str, default=None)
    parser.add_argument("--binarized_path_val",  type=str, default=None)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #initialize tokenizer
   
    tokenizer = CamembertTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args.add_space_token:
        logging.info('Special token `<th_roberta_space_token>` will be added to the CamembertTokenizer instance.')
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', '<th_roberta_space_token>']


    #initialize models
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        type_vocab_size=1,
        #roberta base as default
        num_hidden_layers=args.num_hidden_layers, # L
        hidden_size=args.hidden_size,  # H
        intermediate_size=args.intermediate_size, 
        num_attention_head=args.num_attention_head, # A
    #     #roberta large
    #     num_hidden_layers=24,
    #     hidden_size=1024, 
    #     intermediate_size=4096,
    #     num_attention_head=16
    )
    
    if args.model_dir != None:
        print(f'[INFO] Load pretrianed model (state_dict) from {args.model_dir}')
        model = RobertaForMaskedLM.from_pretrained(args.model_dir)
    else:
        model = RobertaForMaskedLM(config=config)

    #datasets
    train_dataset = MLMDataset(tokenizer, args.train_dir,
                                binarized_path=args.binarized_path_train,
                                max_length=args.train_max_length)
    eval_dataset = MLMDataset(tokenizer, args.eval_dir,
                                binarized_path=args.binarized_path_val,
                                max_length=args.eval_max_length)
    
    #data collator
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    
    #training args
    training_args = TrainingArguments(        
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        #checkpoint
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        #logs
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        #eval
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        #others
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        dataloader_drop_last=args.dataloader_drop_last,
        local_rank=args.local_rank,
        run_name=args.run_name,
        #deepspeed
        deepspeed=args.deepspeed,
        prediction_loss_only=True,
    )

    logging.info(" Number of devices: %d", training_args.n_gpu)
    logging.info(" Device: %s", training_args.device)
    logging.info(" Local rank: %s", training_args.local_rank)
    logging.info(" FP16 Training: %s", training_args.fp16)

    #initiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    #train
    
    if args.model_dir != None:
        print(f'[INFO] Trainer resume from checkpoint {args.model_dir}')
        trainer.train(resume_from_checkpoint=args.model_dir)
    else:
        trainer.train()


    if training_args.local_rank == 0:
        print('main process')
        print('save tokenizer and model')
        tokenizer.save_pretrained(training_args.output_dir)
        #save
        trainer.save_model(os.path.join(training_args.output_dir, 'roberta_thai'))

        #evaluate
    print('evaluate')
    eval_results = trainer.evaluate()
    if training_args.local_rank == 0:
        print('eval_results', eval_results)

    return eval_results


if __name__ == "__main__":
    main()
