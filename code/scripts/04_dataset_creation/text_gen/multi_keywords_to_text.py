import os
import json
from tqdm import tqdm

# Path to raw folder
raw_path = "/content/drive/MyDrive/biomedical_text_generation/data/raw/"

# List all json files inside raw/
raw_files = [f for f in os.listdir(raw_path) if f.endswith(".json")]

print(f"Found {len(raw_files)} raw files.")

from collections import defaultdict

# Map every pmid to keywords that contains it
pmid_to_keywords = defaultdict(set)

for file in tqdm(raw_files):
    keyword = file.replace(".json", "")  # get the keyword from the filename
    with open(os.path.join(raw_path, file), "r", encoding="utf-8") as f:
        entries = json.load(f)
        for entry in entries:
            pmid = entry.get("pmid")
            if pmid:
                pmid_to_keywords[pmid].add(keyword)

print(f"Collected keywords for {len(pmid_to_keywords)} abstracts.")

with open("/content/drive/MyDrive/biomedical_text_generation/data/enriched/abstracts_with_entities.json", "r", encoding="utf-8") as f:
    all_abstracts = json.load(f)

print(f"Loaded {len(all_abstracts)} enriched abstracts.")

multi_keyword_data = []

for entry in tqdm(all_abstracts):
    pmid = entry.get("pmid")
    abstract = entry.get("abstract")

    if not pmid or not abstract:
        continue

    keywords = list(pmid_to_keywords.get(pmid, []))
    
    if len(keywords) < 2:
        continue  # Keep only those linked to 2+ keywords

    prompt = " & ".join(keywords)

    multi_keyword_data.append({
        "pmid": pmid,
        "keywords": keywords,
        "input": prompt,
        "target": abstract,
        "abstract": abstract
    })

print(f"Prepared {len(multi_keyword_data)} examples with multiple keywords.")

import jsonlines
import os

output_dir = "/content/drive/MyDrive/biomedical_text_generation/data/training/text_gen"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "multi_keyword_to_text.jsonl")

with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(multi_keyword_data)

print(f"Saved multi-keyword dataset with {len(multi_keyword_data)} entries to:")
print(output_path)
