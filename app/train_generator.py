from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Check if dataset file exists
dataset_path = "data/reports.txt"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please create it with sample documents.")

# Print dataset content for debugging
with open(dataset_path, "r", encoding="utf-8") as f:
    content = f.read()
    print(f"Dataset content:\n{content}")

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("text", data_files={"train": dataset_path})
print(f"Raw dataset: {dataset}")
print(f"First example: {dataset['train'][0]}")

# Filter out empty or invalid lines
dataset = dataset.filter(lambda example: example["text"].strip())

# Tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=64, clean_up_tokenization_spaces=True)
    return tokenized

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])

# Filter out examples with empty input_ids
tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) > 0)

# Check if dataset is empty
if len(tokenized_dataset) == 0:
    raise ValueError("Tokenized dataset is empty after filtering. Check data/reports.txt for valid content.")

# Print diagnostics
print(f"Tokenized dataset: {tokenized_dataset}")
print(f"Number of training examples: {len(tokenized_dataset)}")
for i in range(min(3, len(tokenized_dataset))):
    print(f"Example {i}: input_ids length = {len(tokenized_dataset[i]['input_ids'])}, input_ids = {tokenized_dataset[i]['input_ids']}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for Codespaces
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