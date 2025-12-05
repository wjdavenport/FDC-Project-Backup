# FA25 UIUC MCS Online Foundations of Data Curation Project

This repository contains all public-shareable artifacts for a semester-long data curation project exploring the scientific literature on consciousness. The project follows the USGS Data Lifecycle Model (Plan → Acquire → Process → Analyze) and demonstrates:

Ethical and policy-compliant acquisition of PubMed metadata, reproducible programmatic workflows in R and Python, manual and machine-assisted relevance labeling, development and evaluation of an interpretable classifier, provenance capture for all workflow decisions, and metadata documentation following DataCite 4.4 elements.

The curated dataset derives from ~65,000 PubMed citations (1843–2025), of which a 500-record human-verified subset is used to train and evaluate a relevance classifier.

## Set-up Instructions for Cloning: ##

One-time setup on a new machine (MacOS with Brew):

brew install git-lfs
git lfs install

Clone with submodules (data repo included):

git clone --recurse-submodules https://github.com/wjdavenport/FDC-Project-Backup.git
cd FDC-Project-Backup

Ensure LFS pulls the big files inside the submodule:

cd data && git lfs pull && cd ..



## Step-by-Step Instructions (General Workflow) ##
The workflow below allows any user to either fully reproduce the dataset acquisition or begin directly from stored data artifacts. All steps assume use of either RStudio (for acquisition) or Python 3.12 (for processing and analysis).

1. (Optional) Reproduce the PubMed Metadata Download
To regenerate the raw MEDLINE file for a date range of your choice:
1. Open RStudio and run the acquisition script:
[acquire/r-project/scripts/01_download_pubmed_consciousness.R](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/acquire/r-project/scripts/01_download_pubmed_consciousness.R)
This script retrieves PubMed records using NCBI E-utilities and writes a MEDLINE file such as:
pubmed_consciousness_1843-2025.medline.
Alternatively, to skip acquisition and use the project's archived version, download the existing MEDLINE file at:
[data/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness_1843-2025.medline](https://github.com/wjdavenport/FDC-Project-Data/blob/main/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness_1843-2025.medline)
(from the FDC-Project-Data repository).  Please note that you will likely receive a big file notice.

2. Convert MEDLINE to CSV
Using Python, run the first processing script:
[consciousness-ezr/scripts/01_export_pubmed_csv.py](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/scripts/01_export_pubmed_csv.py)
This script parses the MEDLINE file and outputs a structured CSV (pubmed_consciousness.csv) containing PMIDs, titles, abstracts, MeSH terms, and derived metadata fields.
The processed CSV is also archived in the data repository for convenience ([https://github.com/wjdavenport/FDC-Project-Data/blob/main/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness.csv](https://github.com/wjdavenport/FDC-Project-Data/blob/main/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness.csv)).

3. Generate or Inspect Initial Labels
Run the script:
[consciousness-ezr/scripts/02_make_seed_labels.py](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/scripts/02_make_seed_labels.py)
This creates an initial labeling template for manual review.
A fully curated human-labeled set of 500 records - used for classifier training - is provided at:
[consciousness-ezr/data/labels_and_reviews/full_labeled_set.csv](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/data/labels_and_reviews/full_labeled_set.csv)
Each row contains a PMID and a binary label (1 = relevant, 0 = irrelevant), with labeling criteria documented in the Final Report.

4. Train the Relevance Classifier
Run:
[consciousness-ezr/scripts/03_train_baseline.py](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/scripts/03_train_baseline.py)
This script trains an interpretable logistic regression classifier (TF-IDF features) using the human-labeled dataset. It outputs:
- Top weighted positive and negative n-grams
- AUC and confusion matrix statistics
- Model parameters
- Logged provenance information
Console output and saved result files are written to
[consciousness-ezr/data/baseline_lr_metrics.csv/](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/data/baseline_lr_metrics.csv) each time the classifier is run.

5. Run the provided [jobAnalysis.ipynb](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/data/labels_and_reviews/jobAnalysis.ipynb) script by clicking 'Open in Colab' script and environment from withing Google Colab.  You will need to drag 2 files into the Colab file space (top directory level) to run the analysis.  Those 2 files are created on running the current classifier, or current copies at time of report can be found at [https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/models/lr_tfidf_meta.joblib](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/models/lr_tfidf_meta.joblib) and  
[https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/models/tfidf.joblib](https://github.com/wjdavenport/FDC-Project-Backup/blob/main/consciousness-ezr/models/tfidf.joblib)

## Notes ##

Data Structure and Reproducibility

Raw Data (Submodule: FDC-Project-Data)

Raw MEDLINE data is stored in a separate repository and included here as a git submodule under:

```data/```

The authoritative MEDLINE file used in this project is located at:

```data/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness_1843-2025.medline```

Derived CSV Output:
The script ```consciousness-ezr/scripts/01_export_pubmed_csv.py``` converts the MEDLINE file into a structured CSV suitable for downstream processing.  It reads from the raw MEDLINE file in the submodule and writes the derived CSV to: ```consciousness-ezr/data/exports/pubmed_consciousness.csv.```  This CSV (~111 MB) is not stored in version control.

Note on copyright:  data placed in this repository originates in part from the National Library of Medicine (NLM) (https://www.nlm.nih.gov/databases/download.html).  NLM has not endorsed this application; this application does not reflect the most current/accurate data from NLM.  The manager of this repository believes that the data herein are shared in the manner of fair use, where the use is for nonprofit educational purposes.

Version information:
R: Version 4.4.1, R Core Team (2024). _R: A Language and Environment for Statistical
  Computing_. R Foundation for Statistical Computing, Vienna, Austria. https://www.R-project.org
RStudio: Version 2024.09.1+394 (2024.09.1+394), Posit team (2024). RStudio: Integrated Development Environment for R. Posit Software, PBC, Boston, MA. URL http://www.posit.co/
Python: Version 3.12.2, Python Software Foundation. (2024). https://www.python.org
