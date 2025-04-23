import json

with open("/content/drive/MyDrive/biomedical_text_generation/data/enriched/abstracts_with_entities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total abstracts: {len(data)}")

multi_entity_pairs = []

for entry in data:
    abstract = entry["abstract"]
    pmid = entry.get("pmid")
    entities = entry.get("entities", [])
    
    # Filter entities with at least 2 words to keep meaningful ones
    filtered = [e for e in entities if len(e.split()) >= 2]
    
    # Only keep if there are at least 2-3 meaningful entities
    if len(filtered) >= 2:
        prompt = "Summarize findings involving: " + ", ".join(filtered[:5])  # Optional: limit to top 5 entities
        multi_entity_pairs.append({
            "input": prompt,
            "output": abstract,
            "pmid": pmid,
            "entities_used": filtered[:5]
        })

print(f"Generated {len(multi_entity_pairs)} training pairs.")

import os
import jsonlines

output_path = "/content/drive/MyDrive/biomedical_text_generation/data/training/summarization/"
os.makedirs(output_path, exist_ok=True)

with jsonlines.open(os.path.join(output_path, "multi_entity_to_abstract.jsonl"), mode="w") as writer:
    writer.write_all(multi_entity_pairs)

print("Saved multi-entity dataset.")
