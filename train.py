from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Load your chat data as JSONL
dataset = load_dataset("json", data_files="your_chats.jsonl", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16
)

# Add LoRA adapters (fine-tune 1% of params)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Train (2-4 hours on T4)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Your data size
        output_dir="fine_tuned_model"
    )
)
trainer.train()
model.save_pretrained("your_finance_bot")