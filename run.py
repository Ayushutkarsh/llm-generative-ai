import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset

from config import ConfigurationManager



def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process configuration file')

    # Add an argument for the config file
    parser.add_argument('--config', help='Path to the config file')
    parser.add_argument('--zeroshot', help='Zero shot inference', default= True)
    # Parse the command-line arguments
    args = parser.parse_args()
    
    if args.config:
        config = ConfigurationManager.get_instance(args.config)
    else:
        config = ConfigurationManager.get_instance()

    
    model_dict = defaultdict(None)
    if args.zeroshot:
        from zeroshot import ZeroShotInference
        model_dict['model'] = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype = config.quant)
        model_dict['tokenizer'] = AutoTokenizer.from_pretrained(config.model_name)
        dataset = load_dataset(config.dataset_name)
        zs = ZeroShotInference(dataset, model_dict)
        zs.run()

if __name__ == '__main__':
    main()
        

