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
        """
        Build a compact, interpretable set of boolean text features capturing:
        - Generic consciousness-related terms
        - Clinical / anesthetic / neural terms
        - Named theories/models of consciousness
        - Proper names of consciousness theorists
        - (Optional) Kuhn 'Landscape of Consciousness' taxonomy terms

        These are used only for a small depth-3 decision tree, to provide
        human-readable rules complementary to the full TF-IDF + LR model.
        """

        # Canonical lowercased text (title + abstract)
        txt = (df["title"].fillna("") + "\n" + df["abstract"].fillna("")).str.lower()

        # Base generic / clinical features
        base_patterns = {
            "mentions_consciousness":       r"\bconscious(?:ness)?\b",
            "mentions_awareness":           r"\bawareness\b",
            "mentions_anesthesia":          r"\banesth(?:e|a)si",
            "mentions_coma":                r"\bcoma\b",
            "mentions_vegetative":          r"\bvegetative\b",
            "mentions_minimally_conscious": r"\bminimally conscious|\bminimally-conscious",
            "mentions_locked_in":           r"\blocked-?in\b",
            "mentions_sedation":            r"\bsedat|\bpropofol|\bmidazolam",
            "mentions_arousal":             r"\barousal\b|\bwakeful",

            "mentions_neural":              r"\b(neural|neuron|neuronal|connectiv)",

            "mentions_philosophy":          r"\bphilosoph",
            "mentions_qualia":              r"\bqualia\b",
            "mentions_subjective":          r"\bsubjective\b|\bsubjectivity\b",
            "mentions_phenomenology":       r"\bphenomenolog",
            "mentions_report":              r"\breport\b|\breportability\b",

            "mentions_religion":            r"\b(religio|spiritu)",
        }

        # Named theories / models of consciousness
        theory_patterns = {
            "mentions_global_workspace":    r"\bglobal workspace\b",
            "mentions_iit":                 r"\bintegrated information theor(?:y|ies)\b|\bIIT\b",
            "mentions_mind_brain_identity": r"\bmind[- ]brain identity\b",
            "mentions_orch_or":             r"\borchestrated objective reduction|\borch[- ]or\b",
            "mentions_panpsychism":         r"\bpanpsychis",
            "mentions_dualism":             r"\bdualism\b|\bsubstance dualism\b|\bproperty dualism\b",
            "mentions_monism":              r"\bmonism\b|\bneutral monism\b",
            "mentions_electromagnetic":     r"\belectromagnetic (?:field|theor)",
            "mentions_neural_correlates":   r"\bneural correlates? of consciousness\b|\bNCC\b",
            "mentions_physicalism":         r"\bphysicalis",
        }

        # Proper names of theorists associated with major models
        author_patterns = {
            "mentions_edelman":    r"\bedelman\b",
            "mentions_crick":      r"\bcrick\b",
            "mentions_koch":       r"\bkoch\b",
            "mentions_dennett":    r"\bdennett\b",
            "mentions_baars":      r"\bbaars\b",
            "mentions_mcfadden":   r"\bmcfadden\b",
            "mentions_searle":     r"\bsearle\b",
            "mentions_chalmers":   r"\bchalmers\b",
            "mentions_dehaene":    r"\bdehaene\b",
            "mentions_tononi":     r"\btononi\b",
            "mentions_friston":    r"\bfriston\b",
        }

        # Kuhn 'Landscape of Consciousness' taxonomy terms (pending if needed)
        kuhn_landscape_patterns = {
            # e.g.,
            # "kuhn_physicalist_cluster": r"...",
            # "kuhn_dualist_cluster":     r"...",
        }

        # Build a single dictionary of all patterns
        all_patterns = {}
        all_patterns.update(base_patterns)
        all_patterns.update(theory_patterns)
        all_patterns.update(author_patterns)
        all_patterns.update(kuhn_landscape_patterns)

        # Evaluate contains() for each pattern
        feat_dict = {}
        for name, pattern in all_patterns.items():
            feat_dict[name] = txt.str.contains(pattern, regex=True, na=False)

        # Length-based feature
        feat_dict["len_abstract_over_250"] = df["abstract"].fillna("").str.len() > 250

        return pd.DataFrame(feat_dict).astype(int)


    # Build the compact text-feature frame on the same seed subset:
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

    # ---- Ranked audit view of ALL seed labels ----
    # Use LR probabilities for the *full* seed set (not just the test split)
    lr_proba_full = lr.predict_proba(X)[:, 1]

    export_seed_ranking_for_audit(
        seed_df=seed,
        text_feat=text_feat,
        proba=lr_proba_full,
        outpath="seed_model_audit.csv"
    )


    def export_seed_ranking_for_audit(seed_df, text_feat, proba, outpath="seed_model_audit.csv"):
        """
        Export all seed-labeled references, ranked for audit.

        Columns:
        - pmid, title, abstract
        - label (original human label 0/1)
        - model_prob (P(class 1) from LR)
        - pred (model hard prediction at threshold 0.5)
        - mismatch (True if pred != label)
        - abs_margin (distance from 0.5: model's confidence)

        Rows are sorted so that high-confidence mismatches appear first.
        """

        # Basic alignment checks
        assert len(seed_df) == len(text_feat) == len(proba), \
            "Lengths differ: seed_df, text_feat, and proba must align."

        # Base table
        df_out = pd.concat(
            [
                seed_df[["pmid", "title", "abstract", "label"]].reset_index(drop=True),
                pd.Series(proba, name="model_prob"),
                text_feat.reset_index(drop=True),
            ],
            axis=1
        )

        # Hard prediction and diagnostics
        df_out["pred"] = (df_out["model_prob"] >= 0.5).astype(int)
        df_out["mismatch"] = df_out["pred"] != df_out["label"]
        df_out["abs_margin"] = (df_out["model_prob"] - 0.5).abs()

        # Rank: mismatches first, then most confident
        df_out = df_out.sort_values(
            by=["mismatch", "abs_margin"],
            ascending=[False, False]
        )

        df_out.to_csv(outpath, index=False)
        print(f"[audit] Exported ranked seed set → {outpath}")
       

    
    # Export manual review set (full-seed predictions)
    # Probabilities for all seed rows, not only the test subset.
    lr_proba_full = lr.predict_proba(X)[:, 1]

    export_manual_review_sample(
        seed_df=seed,
        text_feat=text_feat,
        proba=lr_proba_full,
        n=200,
        outpath="manual_review_sample.csv"
    )



    # ---- ezr (interpretable rules on compact meta features) ---- not used yet

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
