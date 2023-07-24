from peft import LoraConfig, get_peft_model, TaskType
from config import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer

config = ConfigurationManager.get_instance()


def _load_config():

    task_type = TaskType.SEQ_2_SEQ_LM if config.peft_task == 'summarization' else None

    lora_config = LoraConfig(r = config.peft_rank,
                             lora_alpha = config.peft_lora_alpha,
                             target_modules = config.peft_target_modules,
                             lora_dropout=0.1,
                             bias= "none",
                             task_type=task_type)
    return lora_config


def train_peft(model_dict, tokenized_data):

    peft_model = get_peft_model(model_dict['model'], _load_config())
    peft_model_path = config.peft_output_dir + '/training_ckpt'
    
    peft_training_args = TrainingArguments(
        output_dir= config.peft_output_dir,
        auto_find_batch_size=True,
        learning_rate=config.peft_lr,
        num_train_epochs=config.peft_train_epochs,
        logging_steps=1,
        max_steps=config.peft_max_steps
    )

    peft_trainer = Trainer(
        model = peft_model,
        args = peft_training_args, 
        train_dataset=tokenized_data['train'],

    )
    print(f'Starting PEFT training using LORA')
    peft_trainer.train()
    print('Saving PEFT Model ...')
    peft_trainer.model.save_pretrained(peft_model_path)
    model_dict['tokenizer'].save_pretrained(peft_model_path)
    config.peft_latest_model_path = peft_model_path



    