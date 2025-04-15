# All imports
import json
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os

# Data loading and first check

# Loading the cleaned dataset
with open("/content/drive/MyDrive/biomedical_text_generation/data/cleaned/all_abstracts_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame for preview
df = pd.DataFrame(data)
df.sample(5)


# Select tokenizer - for example the T5 base
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Calaculating title lengths and abstracts in tokens
df["title_tokens"] = df["title"].apply(lambda x: len(tokenizer.tokenize(x)))
df["abstract_tokens"] = df["abstract"].apply(lambda x: len(tokenizer.tokenize(x)))

# Statistics
print("Title Token")
print(df["title_tokens"].describe())
print("\nAbstract Token")
print(df["abstract_tokens"].describe())


# Visualization

plt.hist(df["abstract_tokens"], bins=40, color="skyblue", edgecolor="black")
plt.title("Token Length Distribution of Abstracts (T5 tokenizer)")
plt.xlabel("Number of tokens")
plt.ylabel("Number of examples")
plt.grid(True)
plt.show()

# Creating directory if does not exist
os.makedirs("/content/drive/MyDrive/biomedical_text_generation/data/processed", exist_ok=True)

# Saving a copy of enriched dataset (with token counts)
df.to_json("/content/drive/MyDrive/biomedical_text_generation/data/processed/abstracts_with_tokens.json", orient="records", force_ascii=False, indent=2)
