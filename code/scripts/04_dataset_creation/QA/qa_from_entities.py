# Imports
import json
from tqdm import tqdm
import random
import jsonlines

# Load enriched abstracts with biomedical entities
input_path = "../../../../data/enriched/abstracts_with_entities.json"

with open(input_path, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Loaded {len(abstracts)} abstracts.")

# Question templates for entity-based QA
QUESTION_TEMPLATES = [
    "What is the role of {}?",
    "How does {} affect cancer?",
    "What do we know about {}?",
    "What is {}?",
    "How is {} used in treatment?"
]

# Generate QA-style entries using biomedical entities
qa_data = []

for entry in tqdm(abstracts):
    pmid = entry.get("pmid")
    abstract = entry["abstract"]
    entities = entry.get("entities", [])

    for entity in entities:
        if len(entity.split()) < 2:
            continue  # Skip overly generic entities

        question = random.choice(QUESTION_TEMPLATES).format(entity)

        qa_data.append({
            "pmid": pmid,
            "context": abstract,
            "question": question,
            "answer": abstract  # Weak supervision: using full abstract as answer
        })

import os

# Ensure the target directory exists
output_dir = "../../../../data/training/QA"
os.makedirs(output_dir, exist_ok=True)

# Save dataset to JSONL file
output_path = os.path.join(output_dir, "qa_dataset.jsonl")

with jsonlines.open(output_path, mode="w") as writer:
    writer.write_all(qa_data)

print(f"Saved {len(qa_data)} QA pairs to:")
print(output_path)
