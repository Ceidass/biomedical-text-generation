import json
import pandas as pd

# Load enriched biomedical abstracts (with extracted entities)
with open("../../data/enriched/abstracts_with_entities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame for easier handling
df = pd.DataFrame(data)
df[["title", "entities"]].head()

# Flatten all entities from all abstracts into a single list
all_entities = [entity for entry in df["entities"] for entity in entry]
print(f"Total entities collected: {len(all_entities)}")

# Keep only entities with 2 or more words
multi_word_entities = [ent for ent in all_entities if len(ent.split()) >= 2]
print(f"Multi-word entities (2+ words): {len(multi_word_entities)}")

from collections import Counter

# Count frequency of multi-word entities
entity_counter = Counter(multi_word_entities)

# Get top 300 (Computed to have over 100 appearances)
top_entities = entity_counter.most_common(350)

# Preview top 20
print("Top 20 multi-word entities:")
for entity, count in top_entities[:20]:
    print(f"{entity}: {count}")

import os

# Define output path
output_path = "../../data/processed/top_multiword_entities.json"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(top_entities, f, ensure_ascii=False, indent=2)

print(f"\n Saved top multi-word entities to: {output_path}")


# # ---- Clean + Stable Environment Setup ----
# !pip -q uninstall -y sentence-transformers
# !pip -q install "jedi>=0.16"

# # Upgrade build tools (prevents wheel build issues)
# !pip -q install --upgrade pip setuptools wheel

# # Ensure compatible pandas version for Colab
# !pip -q install pandas==2.2.2

# # Install HuggingFace + evaluation stack with safe pinned versions
# !pip -q install \
#     tokenizers==0.15.2 \
#     transformers==4.39.3 \
#     accelerate \
#     datasets \
#     bert-score \
#     evaluate \
#     sentencepiece \
#     tqdm

# # ---- Verify versions ----
# import torch, pandas, transformers, tokenizers
# print("torch:", torch.__version__)
# print("pandas:", pandas.__version__)
# print("transformers:", transformers.__version__)
# print("tokenizers:", tokenizers.__version__)
# print("CUDA available:", torch.cuda.is_available())


import os

unseen_dir= "/content/drive/MyDrive/biomedical_text_generation/data/unseen/sum_unseen.jsonl"

# Folder containing: state.json, dataset_info.json, and the arrow shards
#TEST_DATASET_DIR = "/content/drive/MyDrive/biomedical_text_generation/data/tokenized/biov2bart_sum/test"
TEST_DATASET_DIR = "/content/drive/MyDrive/biomedical_text_generation/data/tokenized/biot5_sum/test"

# Folder containing your fine-tuned checkpoint (config.json, model weights, etc.)
#CHECKPOINT_DIR = "/content/drive/MyDrive/biomedical_text_generation/models/biov2bart_sum_final"
CHECKPOINT_DIR = "/content/drive/MyDrive/biomedical_text_generation/models/biot5_sum_final"

#OUTPUT_DIR = "/content/drive/MyDrive/biomedical_text_generation/data/plots/summarization/bertscore/biov2bart"
OUTPUT_DIR = "/content/drive/MyDrive/biomedical_text_generation/data/plots/summarization/bertscore/biot5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generation / eval params
MAX_INPUT_LEN  = 512
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 8

# BERTScore model (biomedical-friendly)
BERTSCORE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
LANG = "en"

import json
from datasets import load_from_disk
ds = load_from_disk(TEST_DATASET_DIR)
print("Loaded:", ds)

#
jsonl_path = unseen_dir

with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Total rows:", len(lines))

# Load one example (e.g., index 100)
ex = json.loads(lines[100])

print("Columns:", ex.keys())
print("\nExample input:", str(ex.get("input"))[:500])
print("\nExample target:", str(ex.get("target"))[:1500])

import json
import numpy as np

jsonl_path = unseen_dir


input_lengths = []
target_lengths = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        # Safely get fields (in case some rows are missing them)
        inp = example.get("input", "")
        tgt = example.get("target", "")

        input_lengths.append(len(inp.split()))
        target_lengths.append(len(tgt.split()))

print("Total examples:", len(input_lengths))
print("Mean input length:", np.mean(input_lengths))
print("Mean target length:", np.mean(target_lengths))

import json
import matplotlib.pyplot as plt
import numpy as np

jsonl_path = unseen_dir

input_lengths = []
target_lengths = []

# Read JSONL file
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        inp = example.get("input", "")
        tgt = example.get("target", "")

        input_lengths.append(len(inp.split()))
        target_lengths.append(len(tgt.split()))

print("Total examples:", len(input_lengths))

# ---- Input Histogram ----
plt.figure()
plt.hist(input_lengths, bins=50)
plt.title("Distribution of Input Word Counts")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# ---- Target Histogram ----
plt.figure()
plt.hist(target_lengths, bins=50)
plt.title("Distribution of Target Word Counts")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# ---- Descriptive Stats ----
print("Input Mean:", np.mean(input_lengths))
print("Input Median:", np.median(input_lengths))

print("Target Mean:", np.mean(target_lengths))
print("Target Median:", np.median(target_lengths))

# ---- Input Boxplot ----
plt.figure()
plt.boxplot(input_lengths)
plt.title("Boxplot of Input Word Counts")
plt.ylabel("Number of Words")
plt.show()

# ---- Target Boxplot ----
plt.figure()
plt.boxplot(target_lengths)
plt.title("Boxplot of Target Word Counts")
plt.ylabel("Number of Words")
plt.show()
