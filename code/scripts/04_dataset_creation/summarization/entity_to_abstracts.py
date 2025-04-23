import json
import pandas as pd

# Load the enriched abstracts that include entities
with open("../../../../data/enriched/abstracts_with_entities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total abstracts loaded: {len(data)}")

df = pd.DataFrame(data)

# Confirm structure
df[["pmid", "title", "entities", "abstract"]].head(3)

# Store generated training examples here
training_pairs = []

# Loop through all abstracts
for entry in data:
    abstract = entry["abstract"]
    pmid = entry.get("pmid", None)
    for entity in entry["entities"]:
        # Create a templated prompt
        prompt = f"Summarize findings about {entity}."
        training_pairs.append({
            "input": prompt,
            "output": abstract,
            "pmid": pmid,
            "entity": entity  # optional: to trace which entity was used
        })

print(f"Generated {len(training_pairs)} training pairs.")

import os
import jsonlines

# Create output folder
output_path = "../../../../data/training/"
os.makedirs(output_path, exist_ok=True)

# Save as JSON Lines (one JSON object per line)
with jsonlines.open(os.path.join(output_path, "entity_to_abstract.jsonl"), mode="w") as writer:
    writer.write_all(training_pairs)

print("Saved dataset to entity_to_abstract.jsonl")
