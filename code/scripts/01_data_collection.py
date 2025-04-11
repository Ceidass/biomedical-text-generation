# Import libraries
from Bio import Entrez # API for PubMed
import json # Manipulating json files
import os   # Manipulating directories and files of our Operating System
import time # For time measures etc

# Directory that you want to store the data.
output_path = "../../data/raw"

# For creating the directory if it does not exist.
os.makedirs(output_path, exist_ok=True)

# Enter your email for Entrez API use
Entrez.email = "yourEmail@here.you"

# Keywords list for returning the desirable papers. We created them manually by
# using a well known LLM.
keywords = [
    "cancer", "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer",
    "pancreatic cancer", "ovarian cancer", "leukemia", "melanoma", "lymphoma",
    "immunotherapy", "radiotherapy", "chemotherapy", "metastasis",
    "tumor microenvironment", "oncogenes", "tumor suppressor genes",
    "cancer biomarkers", "precision oncology", "targeted therapy"
]

# Max returns per keyword. Change this according to your needs
retmax = 1000 # NCBI recommends not asking for more results without special permission

# Looping through keywords to search for publications
for keyword in keywords:
    print(f"Searching for: {keyword}")
    try:
        # Articles searching
        search_handle = Entrez.esearch(db="pubmed", term=keyword, retmax=retmax)
        search_results = Entrez.read(search_handle)
        id_list = search_results["IdList"]
        search_handle.close()

        if not id_list:
            print(f"No results found for: {keyword}")
            continue

        # Abstracts retreival
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        abstracts = [] # List for storing articles
        for article in records['PubmedArticle']:
            try:
                pmid = article['MedlineCitation']['PMID']
                title = article['MedlineCitation']['Article']['ArticleTitle']
                abstract_text = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                abstract_str = ' '.join(abstract_text)
                abstracts.append({
                    "pmid": str(pmid),
                    "title": title,
                    "abstract": abstract_str
                })
            except Exception as e:
                # Skip articles without abstract
                continue

        # Save in Google Drive per keyword
        filename = os.path.join(output_path, f"{keyword.replace(' ', '_')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(abstracts, f, ensure_ascii=False, indent=2)

        print(f"A number of {len(abstracts)} abstracts have been saved for: {keyword}")
        time.sleep(1)  # for not overloading the API

    except Exception as e:
        print(f"Error with keyword '{keyword}': {e}")

