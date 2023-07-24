from config import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluator
import pandas as pd
import numpy as np

config = ConfigurationManager.get_instance()


def finetune(model_dict, tokenized_data):

    training_args = TrainingArguments(
        output_dir=config.ft_output_dir,
        learning_rate=config.ft_lr,
        num_train_epochs=config.ft_train_epochs,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=config.ft_max_steps
    )

    trainer = Trainer(
        model=model_dict['model'],
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['validation']
    )

    print(f'Starting training now: \n')

    trainer.train()
    
    print(f'Training Completed. \n')


