from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

class ModelLoader:
    def __init__(self, base_model, bnb_config):
        self.base_model = base_model
        self.bnb_config = bnb_config

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            padding_side="right",
            add_eos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        config = AutoConfig.from_pretrained(
            self.base_model,
            load_in_8bit=self.bnb_config.load_in_8bit if hasattr(self.bnb_config, 'load_in_8bit') else False,
            torch_dtype=self.bnb_config.torch_dtype if hasattr(self.bnb_config, 'torch_dtype') else torch.float32,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            config=config,
        )
        return model
