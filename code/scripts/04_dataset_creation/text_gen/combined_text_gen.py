import os
import jsonlines
from tqdm import tqdm

# Define paths to individual datasets
base_path = "../../../../data/training/text_gen"

input_files = [
    "entity_to_text.jsonl",
    "multi_entity_to_text.jsonl",
    "keywords_to_text.jsonl",
    "multi_keywords_to_text.jsonl",
    "keywords_entities_to_text.jsonl"
]

input_paths = [os.path.join(base_path, fname) for fname in input_files]

# Load all entries from the jsonl files
all_entries = []

for path in input_paths:
    with jsonlines.open(path) as reader:
        for obj in reader:
            all_entries.append(obj)

print(f"Total entries before deduplication: {len(all_entries)}")

# Remove exact duplicates based on (input, output)
unique = {}
for entry in all_entries:
    key = (entry["input"], entry["target"])
    unique[key] = entry  # overwrites duplicates

deduplicated = list(unique.values())
print(f"Entries after deduplication: {len(deduplicated)}")

# Output path
output_path = os.path.join(base_path, "combined_text_gen.jsonl")

# Save to jsonl
with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(deduplicated)

print(f" Combined dataset saved to:\n{output_path}")
