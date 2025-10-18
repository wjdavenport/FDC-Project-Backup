#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack

DATA_ALL = Path(__file__).resolve().parents[1] / "data" / "exports" / "pubmed_consciousness.csv"
SEED_CSV = Path(__file__).resolve().parents[1] / "data" / "exports" / "seed_label_batch_working_copy_06.csv"

def build_text(df):
    return (df["title"].fillna("") + " \n " + df["abstract"].fillna("")).values

def build_meta(df):
    meta = df[["has_mesh_consciousness_major","has_mesh_consciousness","has_mesh_spirit_religion","n_mesh"]].fillna(0).astype(float)
    return csr_matrix(meta.values)

def main():
    df = pd.read_csv(DATA_ALL, low_memory=False)

    seed = pd.read_csv(SEED_CSV)
    seed = seed.dropna(subset=["label"])
    # normalize labels to 0/1
    ok_map = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0}
    seed = seed[
        seed["label"].astype(str).str.strip().str.lower().isin(ok_map.keys()) | seed["label"].isin([0,1])
    ]
    seed["label"] = seed["label"].astype(str).str.strip().str.lower().map(ok_map).astype(int)

    # Join to get text/meta columns
    seed = seed.merge(df, on=["pmid"], suffixes=("", "_all"), how="left")

    # Features
    X_text = build_text(seed)
    X_meta = build_meta(seed)
    y = seed["label"].values

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95)
    X_t = tfidf.fit_transform(X_text)
    X = hstack([X_t, X_meta])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1073, stratify=y)

    # ---- Baseline: Logistic Regression ----
    lr = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        n_jobs=1,
        random_state=1073
    )
    
    lr.fit(Xtr, ytr)

    # map vectorizer features to LR coefficients
    meta_cols = ["has_mesh_consciousness_major",
             "has_mesh_consciousness",
             "has_mesh_spirit_religion",
             "n_mesh"]

    tf_names   = list(tfidf.get_feature_names_out())
    meta_names = [f"__META_{c}" for c in meta_cols]

    feat_names = np.array(tf_names + meta_names)
    assert X.shape[1] == len(feat_names), \
        f"Mismatch: X has {X.shape[1]} cols but names list has {len(feat_names)}"

    coef = lr.coef_.ravel()

    # top n-grams pushing toward class 1 (relevant) vs 0 (irrelevant)
    topk = 20
    ix_pos = np.argsort(coef)[-topk:][::-1]
    ix_neg = np.argsort(coef)[:topk]

    print("\nTop positive n-grams (evidence for class 1):")
    for i in ix_pos:
        print(f"{feat_names[i]:40s}  {coef[i]: .4f}")

    print("\nTop negative n-grams (evidence for class 0):")
    for i in ix_neg:
        print(f"{feat_names[i]:40s}  {coef[i]: .4f}")
    
    proba = lr.predict_proba(Xte)[:, 1]         # probabilities for class 1
    pred  = (proba >= 0.5).astype(int)          # or lr.predict(Xte)

    print("\n=== Logistic Regression ===")
    print(classification_report(yte, pred, digits=3))

    if len(set(yte)) == 2:                      # both classes present
        roc = roc_auc_score(yte, proba)
        ap  = average_precision_score(yte, proba)
        print(f"ROC AUC: {roc:.4f}")
        print(f"PR-AUC (Average Precision): {ap:.4f}")
    else:
        print("AUCs skipped (test set has a single class).")

    # For plot later:
    prec, rec, thr = precision_recall_curve(yte, proba)

    # Tiny, interpretable text-feature tree
    def boolean_text_features(df):
        txt = (df["title"].fillna("") + "\n" + df["abstract"].fillna("")).str.lower()
        return pd.DataFrame({
            "mentions_consciousness": txt.str.contains(r"\bconscious(?:ness)?\b"),
            "mentions_awareness":     txt.str.contains(r"\bawareness\b"),
            "mentions_anesthesia":    txt.str.contains(r"\banesth(e|a)si"),
            "mentions_coma":          txt.str.contains(r"\bcoma\b"),
            "mentions_neural":        txt.str.contains(r"\bneural|\bneuron|\bconnectiv"),
            "mentions_philosophy":    txt.str.contains(r"\bphilosoph"),
            "mentions_religion":      txt.str.contains(r"\breligio|\bspiritu"),
            "len_abstract_over_250":  df["abstract"].fillna("").str.len() > 250,
        }).astype(int)

    # Build the compact text-feature frame on the *same* seed subset:
    text_feat = boolean_text_features(seed)
    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
        text_feat, seed["label"].astype(int),
        test_size=0.25, random_state=1073, stratify=seed["label"].astype(int)
    )

    from sklearn.tree import DecisionTreeClassifier, export_text
    tree2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=1073)
    tree2.fit(Xtr_t, ytr_t)
    pred_t = tree2.predict(Xte_t)

    print("\n=== Interpretable tree on simple text flags ===")
    print(classification_report(yte_t, pred_t, digits=3))
    print(export_text(tree2, feature_names=list(text_feat.columns)))
    

    # ---- ezr (interpretable rules on compact meta features) ----

    if os.getenv("RUN_EZR", "0") == "1":
        try:
            from ezr import EZR
            meta_cols = ["has_mesh_consciousness_major","has_mesh_consciousness","has_mesh_spirit_religion","n_mesh"]
            ezr_X = seed[meta_cols].fillna(0).astype(int)
            ezr_y = seed["label"].astype(int)

            Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(
                ezr_X, ezr_y, test_size=0.25, random_state=1073, stratify=ezr_y
            )

            model = EZR()
            model.fit(Xtr_e, ytr_e)

            # Try proba; fall back to hard labels if not available
            if hasattr(model, "predict_proba"):
                ezr_proba = model.predict_proba(Xte_e)
                if ezr_proba is not None and np.ndim(ezr_proba) == 1:
                    # some versions return 1D probs; others 2D [:,1]
                    ezr_p1 = ezr_proba if ezr_proba.ndim == 1 else ezr_proba[:, 1]
                    ezr_pred = (ezr_p1 >= 0.5).astype(int)
                    print("\n=== ezr (meta-features) ===")
                    print(classification_report(yte_e, ezr_pred, digits=3))
                    if len(set(yte_e)) == 2:
                        print(f"PR-AUC (Average Precision): {average_precision_score(yte_e, ezr_p1):.4f}")
                else:
                    # unknown shape -> just print rules + hard preds
                    ezr_pred = model.predict(Xte_e)
                    print("\n=== ezr (meta-features) ===")
                    print(classification_report(yte_e, ezr_pred, digits=3))
            else:
                ezr_pred = model.predict(Xte_e)
                print("\n=== ezr (meta-features) ===")
                print(classification_report(yte_e, ezr_pred, digits=3))

            print("Rules:\n", model)

        except Exception as e:
            print("\n[ezr not run] Install/usage issue:", repr(e))
            print("Tip: keep ezr features simple (booleans/ints), then print(model) to audit rules.")

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, "models/tfidf.joblib")
    joblib.dump(lr, "models/lr_tfidf_meta.joblib")

if __name__ == "__main__":
    main()
