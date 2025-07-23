from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Load model and processor
model_name = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Load dataset (example: CORD dataset)
dataset = load_dataset("naver-clova-ix/cord-v2")

# Preprocess dataset (simplified)
def preprocess_data(examples):
    images = examples["image"]
    texts = examples["ground_truth"]
    encodings = processor(images, return_tensors="pt", padding=True)
    encodings["labels"] = processor(texts, return_tensors="pt")["input_ids"]
    return encodings

train_dataset = dataset["train"].map(preprocess_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_donut",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_donut")
processor.save_pretrained("./fine_tuned_donut")
