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

Note on copyright:  data placed in this repository originates in part from the National Library of Medicine (NLM) (https://www.nlm.nih.gov/databases/download.html).  NLM has not endorsed this application; this application does not reflect the most current/accurate data from NLM.  The manager of this repository believes that the data herein are shared in the manner of fair use, where the use is for nonprofit educational purposes.
