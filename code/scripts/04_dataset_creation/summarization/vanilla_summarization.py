import json
import jsonlines
from tqdm import tqdm

# Path to the enriched abstracts
path = "/content/drive/MyDrive/biomedical_text_generation/data/enriched/abstracts_with_entities.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total abstracts loaded: {len(data)}")

# Create list of dicts in the format {"input": abstract, "output": title}
vanilla_dataset = []

for entry in tqdm(data):
    abstract = entry.get("abstract", "").strip()
    title = entry.get("title", "").strip()

    # Skip empty or corrupted entries
    if abstract and title:
        vanilla_dataset.append({
            "input": abstract,
            "output": title
        })

print(f"Total samples in summarization dataset: {len(vanilla_dataset)}")

# Define output path
output_path = "/content/drive/MyDrive/biomedical_text_generation/data/training/summarization/vanilla_summarization.jsonl"

# Save the dataset
with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(vanilla_dataset)

print(f"Saved summarization dataset to: {output_path}")
