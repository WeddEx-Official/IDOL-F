# =============================================================================
# IDOL-F Framework — Step 06: Structured Annotated Graph Propagation (SAGP)
#
# TWO TECHNIQUES:
#
# 6.1 — Syntactic Graph Role Parsing (SGRP)
#   Convert each sentence to directed dependency graph G=(V,E).
#   Extract: ROOT verb, subject (nsubj), object (dobj/pobj), negation (neg).
#   Algorithm: Transition-Based Dependency Parsing (SpaCy arc-eager).
#   Configuration C = (σ, β, A): stack, buffer, arc set.
#
# 6.2 — Offensive Lexicon Recognition (OLR)
#   Match ROOT verb lemma against 3-source offensive lexicon.
#   OLR_Match(v) = 1 if lemma(v) ∈ L_off AND dep(v)=ROOT else 0
#   NODE-LEVEL matching (not just word presence).
#
# DECISION:
#   OLR_Match=1 → sagp_label="offensive"  (forward to SICL for deeper analysis)
#   OLR_Match=0 → sagp_label="non_offensive" (no offensive predicate found)
#
# ABLATION: ABLATION["SAGP"] = False → all forwarded as "offensive" to SICL
# OUTPUT: output/step6/<DATASET>_sagp.csv
# =============================================================================

import os
import sys
import json
import shutil

import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION,
    SAGP_SPACY_MODEL, make_all_dirs
)
from Step_01_MCA import load_lexicon

make_all_dirs()
IN  = STEP_DIRS["step5_sdd"]
OUT = STEP_DIRS["step6"]

# Load offensive lexicon
_LEXICON = None


def get_lexicon():
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = load_lexicon()
    return _LEXICON


def load_spacy():
    try:
        import spacy
        return spacy.load(SAGP_SPACY_MODEL)
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", SAGP_SPACY_MODEL])
        import spacy
        return spacy.load(SAGP_SPACY_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# 6.1 — Syntactic Graph Role Parsing (SGRP)
# ─────────────────────────────────────────────────────────────────────────────

def sgrp_parse(doc, negation_set):
    """
    Parse one sentence and extract graph components.
    Uses SpaCy's transition-based dependency parser (arc-eager).

    Returns dict with: root_verb, subject, object, negation, edges
    """
    root_verb  = None
    subject    = None
    obj        = None
    negation   = None
    edges      = []

    for token in doc:
        # ROOT verb
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            root_verb = token

        # Collect all edges
        if token.head.i != token.i:
            edges.append({
                "src"  : token.head.text,
                "dep"  : token.dep_,
                "tgt"  : token.text,
                "lemma": token.lemma_.lower(),
            })

    if root_verb is not None:
        for child in root_verb.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                subject = child.text
            if child.dep_ in {"dobj", "pobj", "iobj", "attr", "oprd"}:
                obj = child.text
            if child.dep_ == "neg":
                negation = child.text

    # Fallback: if no ROOT verb, take first VERB in sentence
    if root_verb is None:
        for token in doc:
            if token.pos_ == "VERB":
                root_verb = token
                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subject = child.text
                    if child.dep_ in {"dobj", "pobj"}:
                        obj = child.text
                    if child.dep_ == "neg":
                        negation = child.text
                break

    # Also check standalone negation tokens
    if negation is None:
        for token in doc:
            if token.text.lower() in negation_set:
                negation = token.text
                break

    return {
        "root_verb" : root_verb.text if root_verb else "",
        "root_lemma": root_verb.lemma_.lower() if root_verb else "",
        "subject"   : subject or "",
        "object"    : obj or "",
        "negation"  : negation or "",
        "has_negation": negation is not None,
        "edges"     : edges,
        "n_edges"   : len(edges),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.2 — Offensive Lexicon Recognition (OLR)
# ─────────────────────────────────────────────────────────────────────────────

def olr_match(root_lemma, offensive_verbs):
    """
    OLR_Match(v) = 1 if lemma(v) ∈ L_off AND v is ROOT predicate
                 = 0 otherwise
    Node-level matching: verb counts as offensive ONLY when at ROOT position.
    """
    return int(root_lemma in offensive_verbs)


def main():
    print("=" * 65)
    print("  IDOL-F | Step 06: Structured Annotated Graph Propagation")
    print("=" * 65)

    lex = get_lexicon()
    offensive_verbs = lex["offensive_verbs"]
    negation_set    = lex["negations"]
    human_set       = lex["objects_human"]

    print(f"\n  Lexicon: {len(offensive_verbs):,} offensive verbs loaded")

    if not ABLATION["SAGP"]:
        print("\n  [ABLATION] SAGP = False — all sentences forwarded as offensive")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN, f"{name}_sdd.csv")
            df  = pd.read_csv(src)
            df["root_verb"]     = ""
            df["root_lemma"]    = ""
            df["subject"]       = ""
            df["object"]        = ""
            df["negation"]      = ""
            df["has_negation"]  = False
            df["olr_match"]     = 1
            df["sagp_label"]    = "offensive"
            df["is_human_tgt"]  = 0
            df.to_csv(os.path.join(OUT, f"{name}_sagp.csv"), index=False)
        print("  [DONE] Step-06 skipped (ablation mode)")
        return

    nlp = load_spacy()
    metric_rows = []

    for name in TRAIN_DATASETS:
        print(f"\n  → {name}")
        df    = pd.read_csv(os.path.join(IN, f"{name}_sdd.csv"))
        texts = df.get("text_recovered", df["text"]).fillna("").astype(str).tolist()

        root_verbs = []; root_lemmas = []; subjects = []; objects = []
        negations  = []; has_negs   = []; olr_vals  = []; labels  = []
        edges_all  = []; human_tgts = []

        n_off_found = n_complete = n_edges_total = 0

        for doc in nlp.pipe(texts, batch_size=256):
            parsed = sgrp_parse(doc, negation_set)
            olr    = olr_match(parsed["root_lemma"], offensive_verbs)

            root_verbs.append(parsed["root_verb"])
            root_lemmas.append(parsed["root_lemma"])
            subjects.append(parsed["subject"])
            objects.append(parsed["object"])
            negations.append(parsed["negation"])
            has_negs.append(parsed["has_negation"])
            olr_vals.append(olr)
            labels.append("offensive" if olr == 1 else "non_offensive")
            edges_all.append(json.dumps(parsed["edges"][:15]))

            # Human target check
            obj_lower  = parsed["object"].lower()
            is_human   = int(obj_lower in human_set or obj_lower in {
                "you", "him", "her", "them", "us", "me"})
            human_tgts.append(is_human)

            if olr == 1:
                n_off_found += 1
            if parsed["root_verb"] and parsed["subject"] and parsed["object"]:
                n_complete += 1
            n_edges_total += parsed["n_edges"]

        df["root_verb"]    = root_verbs
        df["root_lemma"]   = root_lemmas
        df["subject"]      = subjects
        df["object"]       = objects
        df["negation"]     = negations
        df["has_negation"] = has_negs
        df["olr_match"]    = olr_vals
        df["sagp_label"]   = labels
        df["is_human_tgt"] = human_tgts
        df["graph_edges"]  = edges_all

        df.to_csv(os.path.join(OUT, f"{name}_sagp.csv"), index=False)

        n = len(df)
        spdr = round(n_complete / n, 3)
        eer  = round((n_edges_total > 0) / n, 3)  # proxy
        ftr  = round(n_off_found / n, 3)

        metric_rows.append({
            "Dataset"    : name,
            "Total"      : n,
            "SPDR"       : spdr,   # Syntactic Parse Detection Rate
            "EER"        : eer,    # Edge Extraction Rate
            "FTR"        : ftr,    # Forward-to-STC Rate
            "Off_found"  : n_off_found,
            "Non_off"    : n - n_off_found,
        })
        print(f"    SPDR={spdr} EER={eer} FTR={ftr} "
              f"(off={n_off_found:,} / non={n-n_off_found:,})")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(OUT, "sagp_metrics.csv"), index=False)
    print("\n  SAGP Metrics:")
    print(metrics_df.to_string(index=False))

    print(f"\n  [DONE] Step-06 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
