from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_datasets(self, dataset_name):
        train_dataset = load_dataset(dataset_name, split="train[0:2]")
        eval_dataset = load_dataset(dataset_name, split="train[2:4]")
        train_dataset = train_dataset.map(self.generate_prompt, remove_columns=list(train_dataset.features))
        eval_dataset = eval_dataset.map(self.generate_prompt, remove_columns=list(eval_dataset.features))
        return train_dataset, eval_dataset

    def generate_prompt(self, sample):
        ful_prompt = f"""<s>[INST] {sample['kategori']}

{sample['metin']}

[/INST]</s>"""
        return {"prompt": ful_prompt}

    def configure_model(self):
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def apply_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            bias="none",
            lora_dropout=0.5,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        return lora_config

    def train(self, train_dataset, eval_dataset, lora_config):
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_strategy="steps",
            save_steps=25,
            logging_steps=10,
            learning_rate=2e-4,
            weight_decay=0.001,
            max_steps=20,
            evaluation_strategy="steps",
            eval_steps=25,
            do_eval=True,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.model.config.use_cache = False
        trainer.train()

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
