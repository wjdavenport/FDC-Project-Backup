#!/usr/bin/env python
import csv, re, sys
from pathlib import Path
from Bio import Medline
from tqdm import tqdm

# Directory containing this script: .../FDC-Project-Backup/consciousness-ezr
SCRIPT_DIR = Path(__file__).resolve().parent
# Repository root: .../FDC-Project-Backup
REPO_ROOT = SCRIPT_DIR.parent

# RAW MEDLINE input lives in the data submodule:
#   FDC-Project-Backup/data/raw/2025-09-27_pubmed_consciousness/...
IN_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "2025-09-27_pubmed_consciousness"
    / "pubmed_consciousness_1843-2025.medline"
)

# Write the derived CSV into the analysis repo, not into the data submodule.
OUT_CSV = SCRIPT_DIR / "derived" / "pubmed_consciousness.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

mesh_keep = {"Consciousness", "Awareness", "Consciousness Disorders"}
mesh_exclude = {"Religion", "Spirituality", "Pastoral Care"}  # tune later

def norm(s): 
    return re.sub(r"\s+", " ", s.strip()) if s else ""

def has_mesh(terms, targets):
    return any((t.split("/")[0] in targets) for t in (terms or []))

def has_mesh_major(terms, targets):
    # MEDLINE marks major topics with asterisk, e.g. "Consciousness/physiology*"
    for mh in terms or []:
        topic = mh.split("/")[0]
        major = mh.endswith("*") or "*" in mh
        if major and topic in targets:
            return True
    return False

def main():
    with IN_PATH.open("r", encoding="utf-8", errors="ignore") as fh, \
         OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "pmid","year","journal",
            "title","abstract",
            "mesh_terms","pub_types",
            "has_mesh_consciousness","has_mesh_consciousness_major",
            "has_mesh_spirit_religion","n_mesh"
        ])
        writer.writeheader()

        records = Medline.parse(fh)
        for rec in tqdm(records, desc="Parsing MEDLINE"):
            pmid   = rec.get("PMID","")
            year   = (rec.get("DP","") or rec.get("EDAT",""))[:4]
            jrnl   = rec.get("JT","")
            title  = rec.get("TI","")
            ab     = rec.get("AB","")
            mesh   = rec.get("MH", [])  # list
            ptypes = rec.get("PT", [])  # list

            row = {
                "pmid": pmid,
                "year": year,
                "journal": norm(jrnl),
                "title": norm(title),
                "abstract": norm(ab),
                "mesh_terms": " | ".join(mesh),
                "pub_types":  " | ".join(ptypes),
                "has_mesh_consciousness": int(has_mesh(mesh, mesh_keep)),
                "has_mesh_consciousness_major": int(has_mesh_major(mesh, mesh_keep)),
                "has_mesh_spirit_religion": int(has_mesh(mesh, mesh_exclude)),
                "n_mesh": len(mesh or [])
            }
            writer.writerow(row)

if __name__ == "__main__":
    sys.exit(main())

