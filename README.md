# FA25 UIUC MCS Online Foundations of Data Curation Project
The use case for this project is to curate a dataset of publications (periodical print, book, and electronic equivalents) related to the field of consciousness.  Such a database will be useful for several reasons, including allowing an analysis of current and past research efforts, organizing the field and tracking its progress, and serving as a common point of reference for future studies.

Instructions for cloning:

One-time setup on a new machine (MacOS with Brew):

brew install git-lfs
git lfs install

Clone with submodules (data repo included):

git clone --recurse-submodules https://github.com/wjdavenport/FDC-Project-Backup.git
cd FDC-Project-Backup

Ensure LFS pulls the big files inside the submodule:

cd data && git lfs pull && cd ..

Data Structure and Reproducibility

Raw Data (Submodule: FDC-Project-Data)

Raw MEDLINE data is stored in a separate repository and included here as a git submodule under:

```data/```

The authoritative MEDLINE file use in this project is located at:

```data/raw/2025-09-27_pubmed_consciousness/pubmed_consciousness_1843-2025.medline```

Derived CSV Output:
The script ```consciousness-ezr/scripts/01_export_pubmed_csv.py``` converts the MEDLINE file into a structured CSV suitable for downstream processing.  It reads from the raw MEDLINE file in the submodule and writes the derived CSV to: ```consciousness-ezr/data/exports/pubmed_consciousness.csv.```  This CSV (~111 MB) is not stored in version control.


Note on copyright:  data placed in this repository originates in part from the National Library of Medicine (NLM) (https://www.nlm.nih.gov/databases/download.html).  NLM has not endorsed this application; this application does not reflect the most current/accurate data from NLM.  The manager of this repository believes that the data herein are shared in the manner of fair use, where the use is for nonprofit educational purposes.

Version information:
R: Version 4.4.1, R Core Team (2024). _R: A Language and Environment for Statistical
  Computing_. R Foundation for Statistical Computing, Vienna, Austria. https://www.R-project.org
RStudio: Version 2024.09.1+394 (2024.09.1+394), Posit team (2024). RStudio: Integrated Development Environment for R. Posit Software, PBC, Boston, MA. URL http://www.posit.co/
Python: Version 3.12.2, Python Software Foundation. (2024). https://www.python.org
