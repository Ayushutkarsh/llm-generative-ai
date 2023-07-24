import evaluate
from config import ConfigurationManager
from transformers import GenerationConfig
import pandas as pd

config = ConfigurationManager.get_instance()

def _create_prompt(text):
    return config.ft_start_prompt + text + config.ft_end_prompt

def _create_results(eval, model_dict, dataset, save_csv = False):
    dialogues = dataset['test']['dialogue']
    ground_truths = dataset['test']['summary']
    predictions = []

    for _, dialogue in enumerate(dialogues):
        prompt = _create_prompt(dialogue)
        
        input_ids = model_dict['tokenizer'](prompt, return_tensors = 'pt').input_ids
        outputs = model_dict['model'].generate(input_ids=input_ids, generation_config= GenerationConfig(max_new_tokens = config.max_new_tokens, num_beams = config.num_beams))
        text_outputs = model_dict['tokenizer'].decode(outputs[0], skip_special_tokens = True)
        predictions.append(text_outputs)
    
    return_result = {}
    return_result['Baseline'] = ground_truths
    return_result['Generation'] = predictions
    
    if save_csv:
        results = list(zip(ground_truths, predictions))
        df = pd.DataFrame(results, columns = ['Baseline', 'Generation'])
        save_path = f'{config.ft_output_dir}/{eval}-summary-results.csv'
        df.to_csv(save_path,index=False)
        config.evaluation_results[f'{config.model_name}-{config.TASK}-{eval}'] = save_path
    
    return return_result

def eval_report(evals = ['rouge'], model_dict = None, dataset = None, save_csv = False):

    evaluations = {}
    result_dict = {}
    for eval in evals:
        evaluations[eval] = evaluate.load(eval)
    
    assert model_dict is not None
    assert dataset is not None

    for eval in evals:
        print(f'Running evaluation on {eval}')
        results = _create_results(eval, model_dict, dataset, save_csv = False)
        model_results = evaluations[eval].compute( predictions = results['Generation'],
                                                references = results['Baseline'], 
                                                use_aggregator = True,
                                                use_stemmer = True,
                                                )

        result_dict[eval] = model_results
        print(f'Evaluation results for {eval} scores: {model_results}')
    return result_dict



