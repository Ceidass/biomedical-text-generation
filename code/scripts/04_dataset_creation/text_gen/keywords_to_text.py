import os
import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict


# Define the raw folder path
raw_folder_path = "../../../../data/raw/"

# List all JSON files in the raw folder
raw_files = [f for f in os.listdir(raw_folder_path) if f.endswith(".json")]

# Dictionary to hold data per keyword
keyword_to_abstracts = {}

# Load each keyword file
for filename in tqdm(raw_files):
    keyword = filename.replace(".json", "")
    full_path = os.path.join(raw_folder_path, filename)
    with open(full_path, "r", encoding="utf-8") as f:
        keyword_to_abstracts[keyword] = json.load(f)

print(f"Loaded {len(keyword_to_abstracts)} keyword files.")

# Loading cleaned dataset (where we have the clean titles/abstracts)
cleaned_path = "../../../../data/cleaned/all_abstracts_cleaned.json"

with open(cleaned_path, "r", encoding="utf-8") as f:
    cleaned_abstracts = json.load(f)

# Map from pmid to abstract
pmid_to_cleaned = {entry["pmid"]: entry for entry in cleaned_abstracts}

# Final collection with combined data
keyword_generation_dataset = []

# For every keyword and the list with its abstracts
for keyword, entries in keyword_to_abstracts.items():
    for entry in entries:
        pmid = entry.get("pmid")
        if not pmid:
            continue

        cleaned_entry = pmid_to_cleaned.get(pmid)
        if not cleaned_entry:
            continue

        keyword_generation_dataset.append({
            "pmid": pmid,
            "keyword": keyword,
            "title": cleaned_entry["title"],
            "abstract": cleaned_entry["abstract"]
        })

print(f"Final dataset size: {len(keyword_generation_dataset)}")


# Prompt creation: Ask from the model to write abstract for a keyword
def create_prompt(keyword):
    return f"Write an abstract about {keyword}."

# Preparing the final examples
text_gen_data = []

for item in keyword_generation_dataset:
    prompt = create_prompt(item["keyword"])
    text_gen_data.append({
        "pmid": item["pmid"],
        "keyword": item["keyword"],
        "input": prompt,
        "target": item["abstract"]
    })

print(f"Prepared {len(text_gen_data)} prompt-based examples.")

# Create directory if does not exist
output_dir = "../../../../data/training/text_gen"
os.makedirs(output_dir, exist_ok=True)

# Output path
output_path = os.path.join(output_dir, "keyword_to_abstract.jsonl")

# Saving
with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(text_gen_data)

print(f"Saved {len(text_gen_data)} examples to:")
print(output_path)
