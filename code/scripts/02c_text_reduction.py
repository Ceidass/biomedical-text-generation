# Update with your actual file path
input_path = '../../data/enriched/abstracts_with_entities.json'

#
output_path = '../../data/enriched/abstracts_to_text.json'

import yake
import json
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
# Download the 'punkt_tab' resource
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize YAKE extractor
yake_kw_extractor = yake.KeywordExtractor(lan="en", n=10, top=20)

with open(input_path, "r") as f:
    data = json.load(f)

# Optional. For testing purposes
#data = data[:10]

from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def phrase_contains_any(word_set, phrase):
    return any(word in phrase.lower().split() for word in word_set)

def deduplicate_phrases(phrases):
    phrases_sorted = sorted(phrases, key=lambda x: -len(x))
    result = []
    seen = set()
    for phrase in phrases_sorted:
        if not any(phrase.lower() in p.lower() and phrase.lower() != p.lower() for p in result):
            result.append(phrase)
            seen.add(phrase.lower())
    return result

for entry in tqdm(data):
    abstract = entry.get("abstract", "")
    entities = entry.get("entities", [])

    # 1. Extract YAKE keywords
    yake_keywords = [kw for kw, score in yake_kw_extractor.extract_keywords(abstract)]

    # 2. Save all_entities: union of raw YAKE and entities, deduplicated
    all_entities = list(set(map(str.lower, entities + yake_keywords)))
    entry["all_entities"] = all_entities

    # 3. Filter YAKE keywords that overlap with any entity
    entity_words = set(e.lower() for e in entities)
    filtered_yake_keywords = [kw for kw in yake_keywords if phrase_contains_any(entity_words, kw)]

    # 4. Deduplicate overlapping phrases
    deduped_keywords = deduplicate_phrases(filtered_yake_keywords)
    entry["combined_keywords"] = deduped_keywords

    # 5. Match sentences from abstract
    sentences = sent_tokenize(abstract)
    matched_sentences = [
        sent.strip() for sent in sentences
        if any(kw.lower() in sent.lower() for kw in deduped_keywords)
    ]

    # 6. Deduplicate matched sentences
    seen = set()
    unique_matched_sentences = []
    for sent in matched_sentences:
        if sent not in seen:
            unique_matched_sentences.append(sent)
            seen.add(sent)

    # 7. Compose matched text
    entry["matched_text"] = " ".join(
        sent if sent.endswith(".") else sent + "." for sent in unique_matched_sentences
    )

    # 8. Optional cleanup: remove original entities field
    entry.pop("entities", None)

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f" Output saved to: {output_path}")
