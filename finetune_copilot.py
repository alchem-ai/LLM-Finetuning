import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TextDataset, DataCollatorForLanguageModeling

# Install required libraries
# !pip install transformers datasets peft


# 1. Load the smallest gemma model 
model_name = "google/gemma-3-1b-it"  # Smallest gemma model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')


# 2. Prepare dataset from code files in current folder
def get_code_files(folder):
    code_files = []
    for root, _, files in os.walk(folder):
        for file in files:  
            if file.endswith(('.py', '.ipynb', '.java', '.cpp', '.c', '.ts', '.go', '.rb', '.rs', '.php', '.cs')):
                code_files.append(os.path.join(root, file))
    return code_files

def merge_files_to_txt(files, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for fname in files:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as infile:
                outfile.write(infile.read() + "\n\n")

code_files = get_code_files(".")
dataset_path = "code_dataset.txt"
merge_files_to_txt(code_files, dataset_path)

# 3. Create a dataset for fine-tuning
def load_dataset(file_path, tokenizer, block_size=512):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset(dataset_path, tokenizer)

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./model-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 7. Fine-tune
trainer.train()

# 8. Save the fine-tuned model
trainer.save_model("./llama-finetuned")
tokenizer.save_pretrained("./llama-finetuned")

