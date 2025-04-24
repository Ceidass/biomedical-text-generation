import json
from tqdm import tqdm
import os
import jsonlines

# Load the enriched abstracts with biomedical entities
input_path = "../../../../data/enriched/abstracts_with_entities.json"

with open(input_path, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Loaded {len(abstracts)} abstracts.")

# Prepare the dataset
entity_to_text = []

for entry in tqdm(abstracts):
    pmid = entry.get("pmid")
    abstract = entry["abstract"]
    entities = entry.get("entities", [])

    for entity in entities:
        if len(entity.split()) < 2:
            continue  # Skip overly generic terms

        input_text = f"Write a biomedical paragraph about {entity}."

        entity_to_text.append({
            "pmid": pmid,
            "entity": entity,
            "abstract": abstract,
            "input": input_text,
            "target": abstract
        })

# Output directory
output_dir = "../../../../data/training/text_gen"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "entity_to_text.jsonl")

with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(entity_to_text)

print(f"Saved {len(entity_to_text)} entries to {output_path}")
