# consciousness-ezr/scripts/04_merge_labels.py
#!/usr/bin/env python
import pandas as pd
from pathlib import Path

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
EZR_ROOT = SCRIPT_DIR.parent
DATA_DIR = EZR_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
LAR_DIR = DATA_DIR / "labels_and_reviews"

SEED_CSV = EXPORTS_DIR / "seed_label_batch_working_copy_07.csv"
REVIEWED_CSV = LAR_DIR / "unlabeled_model_sample_01_reviewed.csv"
OUT_CSV = LAR_DIR / "full_labeled_set.csv"

def normalize_label(col):
    """
    Normalize labels to 0/1.
    Accepts 0/1, '0'/'1', 'true'/'false', 'yes'/'no', etc.
    """
    ok_map = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0
    }
    s = col.astype(str).str.strip().str.lower()
    mask = s.isin(ok_map.keys())
    return s.where(mask).map(ok_map).astype("Int64")  # nullable int


def main():
    # 1) Seed batch: 300 articles
    seed = pd.read_csv(SEED_CSV)
    # We only really need pmid + label for training
    seed_labels = seed[["pmid", "label"]].copy()
    seed_labels["label"] = normalize_label(seed_labels["label"])

    # 2) Reviewed unlabeled sample: 200 articles
    rev = pd.read_csv(REVIEWED_CSV)
    # Column is named 'Human label'; ignore old model_prob/pred/abs_margin for training
    rev_labels = rev[["pmid", "Human label"]].copy()
    rev_labels = rev_labels.rename(columns={"Human label": "label"})
    rev_labels["label"] = normalize_label(rev_labels["label"])

    # 3) Concatenate and handle duplicates
    combined = pd.concat([seed_labels, rev_labels], axis=0, ignore_index=True)

    # Optional: inspect duplicates before resolving
    dup_count = combined.duplicated("pmid").sum()
    if dup_count:
        print(f"[merge] Found {dup_count} duplicate pmid(s); keeping last occurrence.")

    combined = combined.dropna(subset=["label"])
    combined = combined.drop_duplicates(subset=["pmid"], keep="last").reset_index(drop=True)

    # Sort by pmid for sanity
    combined = combined.sort_values("pmid").reset_index(drop=True)

    LAR_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"[merge] Wrote {len(combined)} labeled rows → {OUT_CSV}")

if __name__ == "__main__":
    main()

