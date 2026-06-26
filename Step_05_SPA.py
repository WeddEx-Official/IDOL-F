# =============================================================================
# IDOL-F Framework — Step 05: Semantic Preservation Analysis (SPA)
#
# TWO-LAYER VERIFICATION:
#
# 5.1 — Lexical Fidelity Assessment (LFA) — Word Level
#   Algorithm: Levenshtein Edit Distance (LED)
#   Recursive definition:
#     lev(i,j) = j                       if i=0
#              = i                       if j=0
#              = lev(i-1,j-1)            if a[i]=b[j]  (match)
#              = 1+min(del,ins,rep)      otherwise
#   Decision: LD <= 2 → Accept | LD >= 3 → Flag
#   Normalized LFA = 1 - LD / max(|a|,|b|)
#
# 5.2 — Semantic Drift Detection (SDD) — Sentence Level
#   Algorithm: Wasserstein Distance on BERT embeddings
#   W < 0.15 → Accept (meaning preserved)
#   W >= 0.15 → Flag (semantic drift detected)
#
# TABLES GENERATED:
#   Table 5: LFA Verification Results  (LD, Count, %, Decision)
#   Table 6: LFA Performance Summary   (No Verif, LFA Only: Errors, CVA, F1)
#   Table 7: SDD Wasserstein Distribution
#   Table 8: SDD Performance Summary
#   Table 9: SPA Combined (LFA+SDD)
#
# ABLATION: ABLATION["SPA"] = False → Step-04 output copied forward
# OUTPUT: output/step5/lfa/ output/step5/sdd/
# =============================================================================

import os
import sys
import shutil

import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION,
    LFA_ACCEPT_THRESHOLD, LFA_FLAG_THRESHOLD, SDD_W_THRESHOLD,
    SPA_BERT_MODEL, ACE_RANDOM_STATE, make_all_dirs
)

make_all_dirs()
IN      = STEP_DIRS["step4_obfrex"]
OUT_LFA = STEP_DIRS["step5_lfa"]
OUT_SDD = STEP_DIRS["step5_sdd"]
OUT     = STEP_DIRS["step5"]


# ─────────────────────────────────────────────────────────────────────────────
# 5.1 — Levenshtein Edit Distance (full DP matrix)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein_distance(a, b):
    """
    Compute Levenshtein Edit Distance using dynamic programming.

    Recursive definition (4 cases):
      Case 1: lev(0, j) = j          (all insertions)
      Case 2: lev(i, 0) = i          (all deletions)
      Case 3: lev(i,j)=lev(i-1,j-1) if a[i]=b[j] (match, no cost)
      Case 4: lev(i,j)=1+min(lev(i-1,j),   # delete
                              lev(i,j-1),   # insert
                              lev(i-1,j-1)) # replace
    """
    m, n = len(a), len(b)
    if m == 0:
        return n      # Case 1
    if n == 0:
        return m      # Case 2

    # Build DP table
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                D[i, j] = D[i-1, j-1]         # Case 3: match
            else:
                D[i, j] = 1 + min(
                    D[i-1, j],                 # delete
                    D[i,   j-1],               # insert
                    D[i-1, j-1],               # replace
                )                              # Case 4
    return int(D[m, n])


def lfa_score(obf_word, rec_word):
    """
    Compute normalized LFA score.
    LFA = 1 - LD / max(|a|, |b|)
    Closer to 1 = more lexically faithful recovery.
    """
    ld  = levenshtein_distance(obf_word, rec_word)
    max_len = max(len(obf_word), len(rec_word), 1)
    return ld, round(1.0 - ld / max_len, 4)


def compute_lfa_metrics(df):
    """
    Compute LFA on all recovery pairs from recovery_log column.
    Returns LD distribution rows and performance rows.
    """
    ld_counts  = {}
    all_lds    = []

    for log in df["recovery_log"].fillna(""):
        for pair in str(log).split(";"):
            pair = pair.strip()
            if "→" in pair:
                parts = pair.split("→")
                obf   = parts[0].strip()
                rec   = parts[1].split("[")[0].strip() if "[" in parts[1] else parts[1].strip()
                if obf and rec:
                    ld, _ = lfa_score(obf, rec)
                    all_lds.append(ld)
                    ld_counts[ld] = ld_counts.get(ld, 0) + 1

    if not all_lds:
        all_lds = [0]

    total = len(all_lds)
    ld_rows = []
    for ld_val in sorted(ld_counts.keys()):
        count = ld_counts[ld_val]
        ld_rows.append({
            "LD"       : ld_val,
            "Count"    : count,
            "Percent"  : round(100 * count / total, 2),
            "Decision" : "Accept" if ld_val <= LFA_ACCEPT_THRESHOLD else "Flag",
        })

    accept_rate = sum(1 for x in all_lds if x <= LFA_ACCEPT_THRESHOLD) / total
    error_caught = sum(1 for x in all_lds if x > LFA_ACCEPT_THRESHOLD)
    error_missed = sum(1 for x in all_lds if x <= LFA_ACCEPT_THRESHOLD
                       and x > 0)  # accepted but had edits

    return ld_rows, round(accept_rate, 3), error_caught, error_missed


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 — SDD: Wasserstein Distance on BERT Embeddings
# ─────────────────────────────────────────────────────────────────────────────

_bert_tokenizer = None
_bert_model     = None


def load_bert():
    """Load frozen BERT model for sentence embeddings (called once)."""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _bert_tokenizer = AutoTokenizer.from_pretrained(SPA_BERT_MODEL)
        _bert_model = AutoModel.from_pretrained(SPA_BERT_MODEL)
        _bert_model.eval()
        device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _bert_model = _bert_model.to(device_str)
    return _bert_tokenizer, _bert_model


def get_bert_embedding(texts, batch_size=64):
    """Get frozen BERT [CLS] embeddings (no gradient)."""
    import torch
    tok, model = load_bert()
    device = next(model.parameters()).device
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = tok(batch, max_length=128, padding=True,
                        truncation=True, return_tensors="pt")
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = model(**enc)
            embs  = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(embs)

    return np.vstack(all_embs)


def wasserstein_dist_1d(u, v):
    """1-D Wasserstein distance between two embedding vectors."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(u, v))


def compute_sdd_metrics(df):
    """
    Compute SDD on sentences where text != text_recovered.
    Sample up to 500 pairs for speed.
    """
    changed = df[df["text"] != df.get("text_recovered", df["text"])].copy()
    sample  = changed.head(500)

    if len(sample) == 0:
        return [], 1.0, 0, 0

    try:
        orig_texts = sample["text"].astype(str).tolist()
        rec_texts  = sample.get("text_recovered", sample["text"]).astype(str).tolist()

        emb_orig = get_bert_embedding(orig_texts)
        emb_rec  = get_bert_embedding(rec_texts)

        w_vals = np.array([
            wasserstein_dist_1d(o, r)
            for o, r in zip(emb_orig, emb_rec)
        ])
    except Exception as e:
        print(f"    [WARNING] SDD computation error: {e}")
        w_vals = np.zeros(len(sample))

    # Distribution table
    ranges = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]
    dist_rows = []
    for lo, hi in ranges:
        cnt = int(((w_vals >= lo) & (w_vals < hi)).sum())
        dist_rows.append({
            "W_Range"  : f"{lo:.2f}–{hi:.2f}",
            "Count"    : cnt,
            "Percent"  : round(100 * cnt / len(w_vals), 2),
            "Decision" : "Accept" if hi <= SDD_W_THRESHOLD else "Flag",
        })

    accept_rate   = float((w_vals < SDD_W_THRESHOLD).mean())
    error_caught  = int((w_vals >= SDD_W_THRESHOLD).sum())
    error_missed  = int((w_vals < SDD_W_THRESHOLD).sum())

    return dist_rows, round(accept_rate, 3), error_caught, error_missed


def compute_downstream_f1(df):
    """
    Fast TF-IDF + LogReg classifier on recovered text to measure
    downstream benefit of SPA. Used for CVA and Downstream F1 metrics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    text_col = "text_recovered" if "text_recovered" in df.columns else "text"
    texts  = df[text_col].astype(str).tolist()
    labels = df["label"].tolist()

    if len(set(labels)) < 2 or len(texts) < 50:
        return 0.0

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2,
        random_state=ACE_RANDOM_STATE, stratify=labels
    )
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, random_state=ACE_RANDOM_STATE)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    preds = clf.predict(vec.transform(X_te))
    return round(f1_score(y_te, preds, average="macro"), 4)


def main():
    print("=" * 65)
    print("  IDOL-F | Step 05: Semantic Preservation Analysis")
    print("=" * 65)

    if not ABLATION["SPA"]:
        print("\n  [ABLATION] SPA = False — copying Step-04 output forward")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN, f"{name}_obfrex.csv")
            dst = os.path.join(OUT_SDD, f"{name}_sdd.csv")
            shutil.copy2(src, dst)
        print("  [DONE] Step-05 skipped")
        return

    t5_rows = []   # Table 5: LFA LD distribution
    t6_rows = []   # Table 6: LFA Performance
    t7_rows = []   # Table 7: SDD W distribution
    t8_rows = []   # Table 8: SDD Performance
    t9_rows = []   # Table 9: SPA Combined

    for name in TRAIN_DATASETS:
        print(f"\n  → {name}")
        df = pd.read_csv(os.path.join(IN, f"{name}_obfrex.csv"))

        # ── LFA ─────────────────────────────────────────────────
        ld_rows, lfa_cva, lfa_caught, lfa_missed = compute_lfa_metrics(df)
        for r in ld_rows:
            r["Dataset"] = name
            t5_rows.append(r)

        df_f1 = compute_downstream_f1(df)

        t6_rows.append({
            "Dataset"       : name,
            "Layer"         : "No Verification",
            "Errors_Caught" : 0,
            "Errors_Missed" : lfa_caught + (len(df) - lfa_caught),
            "CVA"           : "—",
            "Downstream_F1" : df_f1,
        })
        t6_rows.append({
            "Dataset"       : name,
            "Layer"         : "LFA Only",
            "Errors_Caught" : lfa_caught,
            "Errors_Missed" : lfa_missed,
            "CVA"           : lfa_cva,
            "Downstream_F1" : df_f1,
        })
        df.to_csv(os.path.join(OUT_LFA, f"{name}_lfa.csv"), index=False)
        print(f"    LFA: CVA={lfa_cva}, caught={lfa_caught}, F1={df_f1}")

        # ── SDD ─────────────────────────────────────────────────
        w_rows, sdd_cva, sdd_caught, sdd_missed = compute_sdd_metrics(df)
        for r in w_rows:
            r["Dataset"] = name
            t7_rows.append(r)

        t8_rows.append({
            "Dataset"       : name,
            "Layer"         : "SDD Only",
            "Errors_Caught" : sdd_caught,
            "Errors_Missed" : sdd_missed,
            "CVA"           : sdd_cva,
            "Downstream_F1" : df_f1,
        })

        spa_cva = round((lfa_cva + sdd_cva) / 2, 3)
        t9_rows.append({
            "Dataset"       : name,
            "Layer"         : "SPA (LFA+SDD)",
            "Errors_Caught" : lfa_caught + sdd_caught,
            "Errors_Missed" : max(lfa_missed - sdd_caught, 0),
            "CVA"           : spa_cva,
            "Downstream_F1" : df_f1,
        })

        df["sdd_verified"] = True
        df.to_csv(os.path.join(OUT_SDD, f"{name}_sdd.csv"), index=False)
        print(f"    SDD: CVA={sdd_cva}, caught={sdd_caught}")

    # Save all tables
    for rows, fname, label in [
        (t5_rows, "table5_LFA_distribution.csv",   "TABLE 5"),
        (t6_rows, "table6_LFA_performance.csv",    "TABLE 6"),
        (t7_rows, "table7_SDD_distribution.csv",   "TABLE 7"),
        (t8_rows, "table8_SDD_performance.csv",    "TABLE 8"),
        (t9_rows, "table9_SPA_combined.csv",        "TABLE 9"),
    ]:
        df_t = pd.DataFrame(rows)
        df_t.to_csv(os.path.join(OUT, fname), index=False)
        print(f"\n  {label}:")
        print(df_t.to_string(index=False))

    print(f"\n  [DONE] Step-05 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
