from huggingface_hub import login
from models.model_loader import ModelLoader
from models.trainer import ModelTrainer
from transformers import BitsAndBytesConfig
import torch  

def main():
    login(token='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')

    base_model = "mistralai/Mistral-7B-v0.3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_loader = ModelLoader(base_model, bnb_config)
    tokenizer = model_loader.load_tokenizer()
    model = model_loader.load_model()

    model_trainer = ModelTrainer(model, tokenizer)
    train_dataset, eval_dataset = model_trainer.prepare_datasets("BeyazB/componySomeInfo")
    model_trainer.configure_model()
    lora_config = model_trainer.apply_lora()
    model_trainer.train(train_dataset, eval_dataset, lora_config)

if __name__ == "__main__":
    main()
