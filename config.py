import yaml, torch, time

class Config:
    def __init__(self, config_file_path=None):
        config_data = {}
        if config_file_path:
            with open(config_file_path, "r") as config_file:
                config_data = yaml.safe_load(config_file)
        self.TASK = config_data.get("TASK",'summarization')
        self.DEBUG = config_data.get("DEBUG", True)
        self.dataset_name = config_data.get("dataset_name", "knkarthick/dialogsum")
        self.model_name = config_data.get("model_name", "google/flan-t5-base")
        self.quant = config_data.get("quant", torch.bfloat16)


        self.run_zero_shot = config_data.get("run_zero_shot", True)


        ### FULL FINE TUNING ###
        self.ft_start_prompt = 'Provide a summary of the conversation below: \n\n'
        self.ft_end_prompt = '\n\n Summary:'
        self.ft_padding = config_data.get("ft_padding", "max_length")
        self.ft_truncation = config_data.get("ft_truncation", True)
        self.ft_output_dir = f'./output-{self.TASK}-{str(int(time.time()))}'
        self.ft_lr = config_data.get("ft_lr", 1e-5)
        self.ft_train_epochs = config_data.get("ft_train_epochs", 1)
        self.ft_max_steps = config_data.get("ft_max_steps", 1)


class ConfigurationManager:
    _instance = None

    @staticmethod
    def get_instance(config_file=None):
        if ConfigurationManager._instance is None:
            ConfigurationManager._instance = Config(config_file)
        return ConfigurationManager._instance

    
