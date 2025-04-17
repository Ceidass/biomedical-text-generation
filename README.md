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
│   │   └── abstracts_with_tokens.json
│   │
│   └── enriched/                  # Abstracts with biomedical entities
│       └── abstracts_with_entities.json
│
├── notebooks/                     # Jupyter / Colab notebooks for experimentation
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 02b_biomedical_entity_processing.ipynb   # NEW: Entity extraction using SciSpaCy
│   ├── 03_token_stats_and_analysis.ipynb
│   ├── 04_finetuning.ipynb
│   └── 05_evaluation.ipynb
│
├── scripts/                       # Modular Python scripts
│   ├── 01_data_collection.py
│   ├── 02_preprocessing.py
│   ├── 02b_biomedical_entity_processing.py
│   ├── 03_token_stats_and_analysis.py
│   ├── train_model.py
│   └── evaluate_model.py
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
│   ├── 03_token_stats_and_analysis.txt
│   └── evaluation_results/
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
