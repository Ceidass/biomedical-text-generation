# Basic imports
import os # Manipulating directories and files of our Operating System
import json # Manipulating json files
from tqdm import tqdm # For progression bars show
import re # Use of regular expressions

# Directory with raw data
raw_data_dir = "../../data/raw"
all_data = []

# Loading json files
for filename in os.listdir(raw_data_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(raw_data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                entries = json.load(f)
                all_data.extend(entries)
            except Exception as e:
                print(f"❌ Error in file {filename}: {e}")

print(f"Number of abstracts loaded: {len(all_data)}")

# Cleaning data and filtering abstracts
cleaned_data = []

for entry in tqdm(all_data):
    title = entry.get("title", "").strip()
    abstract = entry.get("abstract", "").strip()

    # Skip if one of two is missing
    if not title or not abstract:
        continue

    # Cleaning of strange characters
    abstract = re.sub(r'\s+', ' ', abstract)
    abstract = re.sub(r'[^a-zA-Z0-9Α-Ωα-ω.,;:()\[\]\'"\s]', '', abstract)

    cleaned_data.append({
        "pmid": entry.get("pmid", ""),
        "title": title,
        "abstract": abstract
    })

print(f"Number of clear entries: {len(cleaned_data)}")

# Saving in new JSON files

# Create cleaned directory if does not exist
os.makedirs("../../data/cleaned", exist_ok=True)

# Saving
output_file = "../../data/cleaned/all_abstracts_cleaned.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(cleaned_data)} entries in {output_file}")
