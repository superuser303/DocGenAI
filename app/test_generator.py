from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Test prompts
prompts = [
    "Write a short business report for Q3 2025.",
    "Generate an article about AI trends."
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")