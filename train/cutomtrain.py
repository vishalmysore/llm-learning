from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load your custom dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path/to/custom_dataset.txt",  # Path to your custom dataset file
    block_size=128  # Adjust the block size according to your dataset and GPU memory constraints
)

# Define the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True if your dataset includes masked language modeling
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="path/to/output_dir",  # Directory to save the trained model and training logs
    overwrite_output_dir=True,
    num_train_epochs=5,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size per GPU
    save_total_limit=2,
    save_steps=1000,  # Save model checkpoint every 1000 steps
    learning_rate=1e-4,  # Learning rate for the optimizer
    warmup_steps=500,  # Number of warmup steps for the learning rate scheduler
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Start training
trainer.train()