import jsonlines

# Paths to the two datasets
entity_file = "../../../../data/training/summarization/entity_to_abstracts.jsonl"
multi_entity_file = "../../../../data/training/summarization/multi_entity_to_abstracts.jsonl"
# Load both datasets into a combined list
all_samples = []

def load_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        return list(reader)

all_samples += load_jsonl(entity_file)
all_samples += load_jsonl(multi_entity_file)

print(f"Total loaded samples (before deduplication): {len(all_samples)}")

# Remove exact duplicates using a set of (input, output) tuples
seen = set()
unique_samples = []

for sample in all_samples:
    pair = (sample["input"], sample["output"])
    if pair not in seen:
        seen.add(pair)
        unique_samples.append(sample)

print(f"Samples after deduplication: {len(unique_samples)}")

# Save combined dataset
output_path = "../../../../data/training/summarization/combined_summarization_dataset.jsonl"

with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(unique_samples)

print("Combined dataset saved:", output_path)

# Some statistics
from collections import Counter

# How many pmids do we have?
pmid_counts = Counter([s["pmid"] for s in unique_samples])
print("Unique abstracts (pmids):", len(pmid_counts))
print("Average prompts per abstract:", round(len(unique_samples) / len(pmid_counts), 2))
