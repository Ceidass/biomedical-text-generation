import json
from tqdm import tqdm
import os
import jsonlines

input_path = "../../../../data/enriched/abstracts_with_entities.json"

with open(input_path, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Loaded {len(abstracts)} abstracts.")

multi_entity_to_text = []

for entry in tqdm(abstracts):
    pmid = entry.get("pmid")
    abstract = entry["abstract"]
    entities = entry.get("entities", [])

    # Filter out overly short ones and deduplicate
    filtered_entities = list(set([ent for ent in entities if len(ent.split()) >= 2]))

    if len(filtered_entities) < 2:
        continue  # We want multi-entity prompts

    input_text = f"Write a biomedical paragraph using the terms: {', '.join(filtered_entities)}."

    multi_entity_to_text.append({
        "pmid": pmid,
        "entities": filtered_entities,
        "abstract": abstract,
        "input": input_text,
        "target": abstract
    })


output_dir = "../../../../data/training/text_gen"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "multi_entity_to_text.jsonl")

with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(multi_entity_to_text)

print(f"Saved {len(multi_entity_to_text)} multi-entity entries to {output_path}")
