from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Check if dataset file exists
dataset_path = "data/reports.txt"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please create it with sample documents.")

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load dataset using datasets library
dataset = load_dataset("text", data_files={"train": dataset_path})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, clean_up_tokenization_spaces=True)

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")