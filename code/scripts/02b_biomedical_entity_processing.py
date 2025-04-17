import spacy
import en_core_sci_lg
from tqdm import tqdm
import json
import os

# Load the pretrained SciSpaCy biomedical model
nlp = en_core_sci_lg.load()

# Load cleaned abstracts (assumed already preprocessed and cleaned)
with open("/content/drive/MyDrive/biomedical_text_generation/data/cleaned/all_abstracts_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Output number of abstracts loaded
print(f"Total abstracts loaded: {len(data)}")

# Initialize list to store processed abstracts with entities
enriched_data = []

# Process a sample (first 200 entries) for faster iteration
for entry in tqdm(data):
    abstract = entry["abstract"]

    # Apply the biomedical NLP pipeline on the abstract
    doc = nlp(abstract)

    # Extract unique named entities with length > 2 (to skip generic short tokens)
    entities = list(set(ent.text for ent in doc.ents if len(ent.text) > 2))

    # Append the extracted entities to the original entry
    entry["entities"] = entities

    # Add to final enriched dataset
    enriched_data.append(entry)

# Create the output folder if it doesn't exist
os.makedirs("/content/drive/MyDrive/biomedical_text_generation/data/enriched", exist_ok=True)

# Save the enriched data (with extracted biomedical entities) to a JSON file
with open("/content/drive/MyDrive/biomedical_text_generation/data/enriched/abstracts_with_entities.json", "w", encoding="utf-8") as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=2)

# Print confirmation message
print("Enriched dataset saved with biomedical entities.")

import pandas as pd

# Create a DataFrame to preview titles and extracted entities
df = pd.DataFrame(enriched_data)
df[["title", "entities"]].head(10)
