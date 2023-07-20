import random

class ZeroShotInference:
    def __init__(self, dataset, model_dict, task = "summarize"):
        self.dataset = dataset
        #print(self.dataset)
        self.model_dict = model_dict
        self.test_set = self.dataset['test']
        self.task = task
        self.text = None
        self.prompt = None
        
    
    def __len__(self):
        return len(self.dataset['test'])
    
    
    def _display(self, output, summary):
        print(f'PROMPT:\n {self.prompt} \n\n\n')
        print(f'HUMAN:\n {summary} \n\n\n')
        print(f'GENERATED:\n {output} \n\n\n')
    
    def _create_prompt(self, text):

        if self.task == 'summarize':
            
            
            self.prompt = f"""
            Summarize the following conversation.

            {text}

            Summary:
            """
    
    def run(self, dialogue=None, summary = None, verbose = True):
        if not dialogue and not summary: 
            random_index = random.choice(range(self.__len__()))
            dialogue = self.dataset['test'][random_index]['dialogue']
            summary = self.dataset['test'][random_index]['summary']

        self._create_prompt(dialogue)
        model = self.model_dict['model']
        tokenizer = self.model_dict['tokenizer']
        
        #print("####\n\n",self.prompt)

        inputs = tokenizer(self.prompt, return_tensors='pt')
        output = tokenizer.decode(model.generate(inputs["input_ids"],\
                                                max_new_tokens=250,)[0],\
                                                skip_special_tokens=True)

        if verbose:
            self._display(output, summary)
    



        






