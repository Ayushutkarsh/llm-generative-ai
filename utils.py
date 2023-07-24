from config import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

config = ConfigurationManager.get_instance()

def fine_tune_tokenization(data, task='summarization'):
    if task == 'summarization':
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        prompt = [config.ft_start_prompt + text + config.ft_end_prompt for text in data['dialogue']]
        data['input_ids'] = tokenizer(prompt, padding = config.ft_padding, truncation = config.ft_truncation, return_tensors = "pt").input_ids
        data['labels'] = tokenizer(data["summary"], padding = config.ft_padding, truncation = config.ft_truncation, return_tensors="pt").input_ids
    return data

def get_tokenized_dataset(dataset):
    tokenized_datasets = dataset.map(fine_tune_tokenization, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
    return tokenized_datasets

def load_model():
    model_dict = {}
    model_dict['model'] = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype = config.quant)
    model_dict['tokenizer'] = AutoTokenizer.from_pretrained(config.model_name)
    return model_dict

