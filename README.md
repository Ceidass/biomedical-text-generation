# Utilizing State-of-the-Art Methods for Biomedical Text Generation

This repository contains the code and resources for a final-year master thesis project at the University of Patras, 
Department of Computer Engineering and Informatics.
The aim is to build a biomedical text generation system using modern Natural Language Processing (NLP) techniques such as BERT, T5, and BioGPT.

## Project Objectives

- Collect biomedical data from PubMed and related sources.
- Preprocess and structure the data for different tasks (summarization, QA, generation).
- Fine-tune pretrained transformer models on domain-specific biomedical tasks.
- Evaluate output quality both quantitatively and manually.
- Deploy the model or evaluate in a testbed setup.

## Repository Structure

```
biomedical-text-generation/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original data from PubMed etc.
│   │   └── keywords.json          # Json files of every keyword
│   │
│   ├── cleaned/                   # Cleaned and normalized data
│   │   └── all_abstracts_cleaned.json
│   │
│   ├── processed/                 # Tokenized or ready-to-train datasets
│   │   ├── top_multiword_entities.json
│   │   └── abstracts_with_tokens.json
│   │
│   ├── training/
│   │   ├── summarization/
│   │   │   ├── combined_summarization.jsonl
│   │   │   ├── entity_to_abstracts.jsonl
│   │   │   ├── vanilla_summarization.jsonl
│   │   │   └── multi_entity_to_abstracts.jsonl
│   │   │
│   │   ├── text_gen/
│   │   │   ├── combined_text_gen.jsonl
│   │   │   ├── keywords_entities_to_text.jsonl
│   │   │   ├── entity_to_text.jsonl
│   │   │   ├── multi_entity_to_text.jsonl
│   │   │   ├── multi_keywords_to_text.jsonl
│   │   │   └── keywords_to_text.jsonl
│   │   │
│   │   └── QA/
│   │       └── qa_dataset.jsonl
│   │
│   └── enriched/                  # Abstracts with biomedical entities
│       └── abstracts_with_entities.json
│
├── notebooks/                     # Jupyter / Colab notebooks for experimentation
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 02b_biomedical_entity_processing.ipynb   # Entity extraction using SciSpaCy
│   ├── 02c_text_reduction.ipynb    # Abstract reduction using YAKE
│   ├── 03_analysis.ipynb
│   ├── 04_dataset_creation/
│   │   ├── text_gen/
│   │   │   ├── combined_text_gen.ipynb
│   │   │   ├── key_ent_to_text.ipynb
│   │   │   ├── entity_to_text.ipynb
│   │   │   ├── multi_entity_to_text.ipynb
│   │   │   ├── multi_keywords_to_text.ipynb
│   │   │   └── keywords_to_text.ipynb
│   │   │
│   │   ├── summarization/
│   │   │   ├── combine_summarization_tasks.ipynb
│   │   │   ├── entity_to_abstracts.ipynb
│   │   │   ├── multi_entity_to_abstracts.ipynb
│   │   │   └── vanilla_summarization.ipynb
│   │   │
│   │   └── QA/
│   │       └── qa_from_entities.ipynb
│   │
│   ├── 05_finetuning/
│   │   ├── text_gen/
│   │   │
│   │   │
│   │   ├── summarization/
│   │   │   ├── bart_fine.ipynb
│   │   │   └── t5_fine.ipynb
│   │   │
│   │   └── QA/
│   │
│   │
│   └── 06_evaluation.ipynb
│
├── scripts/                       # Modular Python scripts
│   ├── 01_data_collection.py
│   ├── 02_preprocessing.py
│   ├── 02b_biomedical_entity_processing.py   # Entity extraction using SciSpaCy
│   ├── 02c_text_reduction.py           # Abstract reduction using YAKE
│   ├── 03_analysis.py
│   ├── 04_dataset_creation/
│   │   ├── text_gen/
│   │   │   ├── combined_text_gen.py
│   │   │   ├── key_ent_to_text.py
│   │   │   ├── entity_to_text.py
│   │   │   ├── multi_entity_to_text.py
│   │   │   ├── multi_keywords_to_text.py
│   │   │   └── keywords_to_text.py
│   │   │
│   │   ├── summarization/
│   │   │   ├── combine_summarization_tasks.py
│   │   │   ├── entity_to_abstracts.py
│   │   │   ├── multi_entity_to_abstracts.py
│   │   │   └── vanilla_summarization.py
│   │   │
│   │   └── QA/
│   │       └── qa_from_entities.py
│   │
│   ├── 05_finetuning/
│   │   ├── text_gen/
│   │   │
│   │   │
│   │   ├── summarization/
│   │   │   ├── bart_fine.py
│   │   │   └── t5_fine.py
│   │   │
│   │   └── QA/
│   │
│   │
│   └── 06_evaluation.py
│
├── models/                        # Fine-tuned models and config files
│   └── t5_summary_model/
│
├── configs/                       # Training configuration files
│   └── t5_config.yaml
│
├── outputs/                       # Generated text, logs, and visualizations
│   ├── 01_data_collection.txt
│   ├── 02_preprocessing.txt
│   ├── 02b_biomedical_entity_processing.txt
│   ├── 02c_text_reduction.txt
│   ├── 03_analysis.txt
│   ├── 04_dataset_creation/
│   │   ├── text_gen/
│   │   │   ├── combined_text_gen.txt
│   │   │   ├── key_ent_to_text.txt
│   │   │   ├── entity_to_text.txt
│   │   │   ├── multi_entity_to_text.txt
│   │   │   ├── multi_keywords_to_text.txt
│   │   │   └── keywords_to_text.txt
│   │   │
│   │   ├── summarization/
│   │   │   ├── combine_summarization_tasks.txt
│   │   │   ├── entity_to_abstracts.txt
│   │   │   ├── multi_entity_to_abstracts.txt
│   │   │   └── vanilla_summarization.txt
│   │   │
│   │   │
│   │   └── QA/
│   │       └── qa_from_entities.txt
│   │
│   ├── 05_finetuning/
│   │   ├── text_gen/
│   │   │
│   │   │
│   │   ├── summarization/
│   │   │   ├── bart_fine.txt
│   │   │   └── t5_fine.txt
│   │   │
│   │   └── QA/
│   │
│   │
│   └── 06_evaluation.txt
│
├── docs/                          # Diagrams, documentation, report sections
│   └── architecture_diagram.png
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files and folders to exclude from Git
└── LICENSE                        # Project license


```


## Technologies Used

- Python 3.10+
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- ScispaCy / spaCy
- PubMed Entrez API
- PyTorch
- Google Colab

##  Tasks Implemented

- Biomedical Summarization (T5 / BART)
- Question Answering (BioGPT / Med-PaLM style)
- Free-form Text Generation (GPT-style)
- NER + entity linking with SciSpacy


##  Setup Instructions

To set up and run this project locally or in Google Colab:

1. Clone the repository:
```bash
git clone https://github.com/Ceidass/biomedical-text-generation.git
cd biomedical-text-generation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Alternatively, open the notebooks in Google Colab for experimentation using free GPU resources.
