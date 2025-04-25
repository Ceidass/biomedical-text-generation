import os
import json
import jsonlines
from tqdm import tqdm

# Load enriched abstracts (with entities)
enriched_path = "/content/drive/MyDrive/biomedical_text_generation/data/enriched/abstracts_with_entities.json"

with open(enriched_path, "r", encoding="utf-8") as f:
    enriched_data = json.load(f)

# Index enriched abstracts by PMID for fast lookup
pmid_to_entry = {entry["pmid"]: entry for entry in enriched_data}
print(f"Loaded {len(pmid_to_entry)} enriched abstracts.")

# Directory where the raw keyword-based files are stored
raw_dir = "/content/drive/MyDrive/biomedical_text_generation/data/raw"

# List all JSON files (each corresponding to a search keyword)
keyword_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]

print(f"Found {len(keyword_files)} keyword files.")

combined_data = []

for file_name in tqdm(keyword_files):
    keyword = file_name.replace(".json", "")
    file_path = os.path.join(raw_dir, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        pmid = article.get("pmid")
        abstract = article.get("abstract")

        # Skip if abstract or PMID is missing
        if not pmid or not abstract:
            continue

        # Get entities from the enriched data
        enriched_entry = pmid_to_entry.get(pmid)
        if not enriched_entry:
            continue

        entities = enriched_entry.get("entities", [])
        if not entities:
            continue

        # Compose input prompt: keyword + entities
        all_terms = [keyword] + entities
        input_text = ", ".join(all_terms)

        combined_data.append({
            "pmid": pmid,
            "input": input_text,
            "target": abstract
        })

import os

# Define output path
output_dir = "/content/drive/MyDrive/biomedical_text_generation/data/training/text_gen"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "keywords_entities_to_text.jsonl")

# Write to jsonlines format
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(combined_data)

print(f"Saved {len(combined_data)} samples to:")
print(output_path)
