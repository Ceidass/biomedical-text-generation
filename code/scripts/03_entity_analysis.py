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
