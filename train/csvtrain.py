import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load and preprocess multiple CSV files
csv_files = ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"]
texts = []

for file in csv_files:
    df = pd.read_csv(file)
    # Assuming your CSV files have a column named "text" containing the text data
    texts.extend(df["text"].tolist())

# Tokenize the texts
tokenized_texts = tokenizer.batch_encode_plus(
    texts,
    truncation=True,
    padding=True
)

# Create the TextDataset from the tokenized texts
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=None,  # Pass None since we're providing the tokenized texts directly
    tokenized_datasets=tokenized_texts,
    block_size=128
)

# Define the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="path/to/output_dir",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_total_limit=2,
    save_steps=1000,
    learning_rate=1e-4,
    warmup_steps=500
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