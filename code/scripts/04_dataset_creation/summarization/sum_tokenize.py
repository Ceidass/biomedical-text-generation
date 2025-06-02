from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json
import random
import os

# Define input and output paths
INPUT_PATH = "../../../../data/final_datasets/summarization_ready.jsonl"
OUTPUT_DIR = "../../../../data/tokenized"
UNSEEN_PATH = "../../../../data/unseen"

#Create tokenized directory if does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Create unseen directory if does not exist
os.makedirs(UNSEEN_PATH, exist_ok=True)

# Tokenizer models
BIOBART_MODEL = "GanjinZero/biobart-base"
BIOT5_MODEL = "QizhiPei/biot5-base"
BIOBART_V2_MODEL = "GanjinZero/biobart-v2-base"

# Load the JSONL dataset
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Shuffle the dataset
random.shuffle(data)

# Split into train and validation
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# We will store the validation data in json form file so we can later test our models with unseen data
with open(os.path.join(UNSEEN_PATH, "sum_unseen.json"), 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')


# Helper function to tokenize and save dataset
def tokenize_and_save(dataset, tokenizer_name, model_name, split):
    print(f"Tokenizing for {model_name} ({split})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    #Add the "summarize:" prompt in the beginning of the input in T5 model case
    if tokenizer_name == "QizhiPei/biot5-base":
        for item in dataset:
            item['input'] = "summarize: " + item['input']

    def tokenize_fn(example):
        return tokenizer(
            example['input'],
            text_target=example['target'],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    ds = Dataset.from_list(dataset)
    tokenized = ds.map(tokenize_fn, batched=False)

    out_dir = os.path.join(OUTPUT_DIR, model_name, split)
    os.makedirs(out_dir, exist_ok=True)

    tokenized.save_to_disk(out_dir)
    print(f"Saved to {out_dir}\n")

# Tokenize and save for both models
for model_name, tokenizer in [("biobart_sum", BIOBART_MODEL), ("biov2bart_sum", BIOBART_V2_MODEL) ]:
    tokenize_and_save(train_data, tokenizer, model_name, "train")
    tokenize_and_save(val_data, tokenizer, model_name, "val")
