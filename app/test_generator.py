from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Check if model directory exists
model_path = "./fine_tuned_gpt2"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found at {model_path}. Please run train_generator.py first.")

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Test prompts
prompts = [
    "Write a short business report for Q3 2025.",
    "Generate an article about AI trends."
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", clean_up_tokenization_spaces=True)
    outputs = model.generate(**inputs, max_length=200)
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")