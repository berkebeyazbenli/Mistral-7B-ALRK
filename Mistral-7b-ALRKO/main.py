from huggingface_hub import login
from models.model_loader import ModelLoader
from models.trainer import ModelTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import pandas as pd

def main():
    login(token='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')

    base_model = "mistralai/Mistral-7B-v0.3"
    
    bnb_config = AutoConfig.from_pretrained(base_model)
    bnb_config.load_in_8bit = False
    bnb_config.torch_dtype = torch.bfloat16

    model_loader = ModelLoader(base_model, bnb_config)
    tokenizer = model_loader.load_tokenizer()
    model = model_loader.load_model()

  
    dataset_path = "/Users/berkebeyazbenli/Desktop/Mistral-7b-ALRKO/dataset1.csv"
    dataset = pd.read_csv(dataset_path)

    
    model_trainer = ModelTrainer(model, tokenizer)
    model_trainer.configure_model()
    lora_config = model_trainer.apply_lora()
    model_trainer.train(dataset, None, lora_config)  

if __name__ == "__main__":
    main()