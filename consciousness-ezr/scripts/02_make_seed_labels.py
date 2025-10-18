#!/usr/bin/env python
import pandas as pd
from pathlib import Path

SRC = Path("../data/exports/pubmed_consciousness.csv")
OUT = Path("../data/exports/seed_label_batch.csv")

N = 300  # first batch; stratify on simple heuristics
df = pd.read_csv(SRC)
# simple stratified sample: prioritize no-abstract + abstract, and mesh flags
bucket_a = df[df["has_mesh_consciousness_major"]==1].sample(min(120, len(df)), random_state=1073)
bucket_b = df[(df["has_mesh_consciousness_major"]==0) & (df["has_mesh_consciousness"]==1)].sample(min(120, len(df)), random_state=1073)
bucket_c = df[(df["has_mesh_consciousness"]==0)].sample(min(60, len(df)), random_state=1073)

seed = pd.concat([bucket_a, bucket_b, bucket_c]).sample(frac=1, random_state=1073).head(N).copy()
seed["label"] = ""  # fill with 1=relevant, 0=irrelevant
keep_cols = ["pmid","year","journal","title","abstract","mesh_terms","pub_types",
             "has_mesh_consciousness_major","has_mesh_consciousness","has_mesh_spirit_religion","n_mesh","label"]
seed[keep_cols].to_csv(OUT, index=False)
print(f"Wrote {OUT}")

