import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset
from utils import load_model, get_tokenized_dataset
from config import ConfigurationManager
from copy import deepcopy


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process configuration file')

    # Add an argument for the config file
    parser.add_argument('--config', help='Path to the config file')
    parser.add_argument('--zeroshot', help='Zero shot inference', default= False)
    parser.add_argument('--finetune', help='Fine Tune Instruct model', default= False)
    parser.add_argument('--peft', help='Fine Tune Instruct model', default= False)
    parser.add_argument('--evaluate', help='Evaluate your model', default= False)
    # Parse the command-line arguments
    args = parser.parse_args()
    
    if args.config:
        config = ConfigurationManager.get_instance(args.config)
    else:
        config = ConfigurationManager.get_instance()

    
    model_dict = load_model()
    peft_model_dict = {}
    dataset = load_dataset(config.dataset_name)


    if args.finetune:
        from finetune import finetune
        fine_tune_dataset = get_tokenized_dataset(dataset)
        finetune(model_dict, fine_tune_dataset)
    
    if args.peft:
        from lora import train_peft
        lora_dataset = get_tokenized_dataset(dataset)
        train_peft(model_dict, lora_dataset)
        

        

    if args.zeroshot:
        from zeroshot import ZeroShotInference
        
        
        zs = ZeroShotInference(dataset, model_dict)
        zs.run()
    
    if args.evaluate:
        from evaluator import eval_report
        if args.peft:
            ### TODO: Fix this function
            peft_model_dict['model'] = _load_peft_model()
            evaluations = eval_report(evals=config.evals, model_dict=peft_model_dict, dataset=dataset)
        else:
            evaluations = eval_report(evals=config.evals, model_dict=model_dict, dataset=dataset)
        
            
    


    

if __name__ == '__main__':
    main()
        

