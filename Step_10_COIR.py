# =============================================================================
# IDOL-F Framework — Step 10: Contextual Offensive Intent Resolution (COIR)
#
# PURPOSE:
#   Process UNCERTAIN sentences (from ICPS forward_cubd=1) through
#   three sequential techniques to resolve their offensive intent.
#
# THREE TECHNIQUES:
#
# 10.0 — Ellipsis Resolution (pre-processing for all sentences)
#   Verb Phrase Ellipsis (VPE) detection:
#   "I want to kill you but I cannot." → "I want to kill you but I cannot kill you."
#   Formal: if ∃ aux_neg token with no following main verb →
#           copy VP from prior clause into ellipsis position
#
# 10.1 — Directional Graph Traversal (DGT)
#   Trace harmful path in RASGC graph:
#   AGGRESSOR → OFFENSIVE_PREDICATE → TARGET
#   NegationCheck(v_pred) = 1[∃(v_neg,v_pred)∈E : ψ=NEGATES]
#   w_neg = 0.0 (negation present) → route = NON_OFFENSIVE
#   w_neg = 1.0 → proceed to SER
#
# 10.2 — Semantic Entity Recognition (SER)
#   EntityType ∈ {HUMAN, NON_HUMAN, AMBIGUOUS}
#   via pronoun set + SpaCy NER (PERSON, NORP tags)
#
# 10.3 — Human Target Verification (HTV)
#   HUMAN   + no-negation → OFFENSIVE
#   NON_HUMAN              → NON_OFFENSIVE
#   AMBIGUOUS              → forward to CUBD
#
# METRICS: PA, DC, HPDR, NPA, ECA, HEP, NHEP, HTP, NHTP, FPRR, RA
# TABLE 18: COIR Metrics per dataset
#
# ABLATION: ABLATION["COIR"] = False → ICPS output forwarded unchanged
# OUTPUT: output/step10/<DATASET>_coir.csv + table18_coir_metrics.csv
# =============================================================================

import os
import re
import sys

import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION,
    COIR_NEG_WEIGHT, COIR_HUMAN_TAGS, make_all_dirs
)
from Step_01_MCA import load_lexicon

make_all_dirs()
IN  = STEP_DIRS["step9"]
IN6 = STEP_DIRS["step6"]
OUT = STEP_DIRS["step10"]

# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
_LEX = None
def get_lex():
    global _LEX
    if _LEX is None:
        _LEX = load_lexicon()
    return _LEX

# Auxiliary negation contractions (for ellipsis detection)
AUX_NEG_CONTRACTIONS = {
    "cannot", "can't", "won't", "wouldn't", "shouldn't", "didn't",
    "don't", "doesn't", "haven't", "hasn't", "hadn't", "couldn't",
    "mustn't", "needn't", "daren't", "shan't", "mightn't",
}

# Human pronoun/noun set for SER
HUMAN_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
    "we", "us", "our", "ours", "ourselves",
    "someone", "anyone", "everyone", "nobody", "somebody", "everybody",
    "person", "people", "human", "man", "woman", "men", "women",
    "boy", "girl", "kid", "child", "children", "baby", "infant",
    "guy", "guys", "gal", "gals", "folk", "folks",
    "citizen", "individual", "member", "victim",
}

_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy",
                            "download", "en_core_web_sm"])
            import spacy
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


# =============================================================================
# TECHNIQUE 10.0 — ELLIPSIS RESOLUTION
# =============================================================================

def resolve_ellipsis(text, off_verbs, neg_set):
    """
    Verb Phrase Ellipsis (VPE) resolution.

    Formal definition:
      tokens T = [t_1, ..., t_n]
      Find t_k ∈ AUX_NEG_CONTRACTIONS such that
        ∄ t_j ∈ VERB for j > k in same clause
      Resolution: append VP from prior clause at position k+1

    Returns (resolved_text, was_resolved: bool)
    """
    text_lower = text.lower().strip()

    # Check if sentence ends with auxiliary negation (ellipsis pattern)
    pat = (r'\b(' + '|'.join(re.escape(w) for w in AUX_NEG_CONTRACTIONS)
           + r')\b[.!?]?\s*$')
    if not re.search(pat, text_lower):
        # Also check "but [aux_neg]" pattern mid-sentence
        mid_pat = (r'\bbut\b.{2,50}\b('
                   + '|'.join(re.escape(w) for w in AUX_NEG_CONTRACTIONS)
                   + r')\b[.!?]?\s*$')
        if not re.search(mid_pat, text_lower):
            return text, False

    # Find the VP (verb + object) from the first clause
    nlp = get_nlp()
    doc = nlp(text[:512])
    vp_parts = None

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in off_verbs and token.pos_ == "VERB":
            parts = [token.text]
            for child in token.children:
                if child.dep_ in {"dobj", "pobj", "iobj", "attr", "oprd"}:
                    parts.append(child.text)
            vp_parts = " ".join(parts)
            break

    if vp_parts is None:
        return text, False

    resolved = text.rstrip(".!?") + " " + vp_parts
    return resolved.strip(), True


# =============================================================================
# TECHNIQUE 10.1 — DIRECTIONAL GRAPH TRAVERSAL (DGT)
# =============================================================================

def dgt_analyze(text, off_verbs, neg_set):
    """
    Traverse sentence dependency graph to find:
    - Offensive predicate (ROOT verb in lexicon)
    - Aggressor (subject of predicate)
    - Target (object of predicate)
    - Negation (neg dependency or neg token)

    Returns dict with path info and w_neg weight.
    """
    nlp = get_nlp()
    doc = nlp(text[:512])

    pred     = None   # offensive predicate token
    aggressor = None
    target    = None
    neg_found = None
    has_neg   = False

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in off_verbs and token.pos_ == "VERB":
            pred = token
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                    aggressor = child.text
                if child.dep_ in {"dobj", "pobj", "iobj", "attr", "oprd"}:
                    target = child.text
                if child.dep_ == "neg":
                    has_neg   = True
                    neg_found = child.text
            break

    # Standalone negation check
    if not has_neg:
        for token in doc:
            if token.text.lower() in neg_set:
                has_neg   = True
                neg_found = token.text
                break

    # Build path string
    path_parts = [p for p in [aggressor, f"→{pred.text}→" if pred else None, target] if p]
    path = " ".join(path_parts)

    return {
        "pred"      : pred.text if pred else None,
        "pred_lemma": pred.lemma_.lower() if pred else None,
        "aggressor" : aggressor,
        "target"    : target,
        "has_neg"   : has_neg,
        "neg_token" : neg_found,
        "path"      : path,
        # w_neg = 0.0 → negation present → benign
        # w_neg = 1.0 → no negation → potentially harmful
        "w_neg"     : 0.0 if has_neg else 1.0,
        "is_complete": bool(aggressor and pred and target),
    }


# =============================================================================
# TECHNIQUE 10.2 — SEMANTIC ENTITY RECOGNITION (SER)
# =============================================================================

def ser_classify(target_text, text):
    """
    Classify entity type of the target as HUMAN, NON_HUMAN, or AMBIGUOUS.

    Method:
    1. Check if target in human pronoun/noun set
    2. Check SpaCy NER for PERSON/NORP tags
    3. Fallback: AMBIGUOUS

    Returns dict with entity type and confidence.
    """
    if target_text is None:
        return {"etype": "AMBIGUOUS", "conf": 0.50, "ner_label": None}

    tgt_lower = target_text.lower()

    # Check human pronoun/noun set
    if tgt_lower in HUMAN_PRONOUNS:
        return {"etype": "HUMAN", "conf": 0.92, "ner_label": "PRONOUN"}

    # Check SpaCy NER
    nlp = get_nlp()
    doc = nlp(text[:256])
    ner_label = None
    for ent in doc.ents:
        if ent.text.lower() == tgt_lower:
            ner_label = ent.label_
            break
        # Partial match
        if tgt_lower in ent.text.lower():
            ner_label = ent.label_
            break

    if ner_label in COIR_HUMAN_TAGS:
        return {"etype": "HUMAN", "conf": 0.88, "ner_label": ner_label}
    if ner_label and ner_label not in COIR_HUMAN_TAGS:
        return {"etype": "NON_HUMAN", "conf": 0.85, "ner_label": ner_label}

    # Length-based heuristic for remaining cases
    if len(tgt_lower) > 2 and not any(c.isdigit() for c in tgt_lower):
        return {"etype": "NON_HUMAN", "conf": 0.70, "ner_label": None}

    return {"etype": "AMBIGUOUS", "conf": 0.50, "ner_label": None}


# =============================================================================
# TECHNIQUE 10.3 — HUMAN TARGET VERIFICATION (HTV)
# =============================================================================

def htv_route(dgt_result, ser_result):
    """
    Human Target Verification routing decision.

    Rules:
    HUMAN   + w_neg=1.0 + has_pred → OFFENSIVE
    NON_HUMAN                       → NON_OFFENSIVE
    HUMAN   + w_neg=0.0             → NON_OFFENSIVE (negation blocks)
    AMBIGUOUS                       → CUBD (deeper analysis needed)

    Returns (routing: str, risk_score: float, confidence: float)
    """
    etype  = ser_result["etype"]
    w_neg  = dgt_result["w_neg"]
    has_pred = dgt_result["pred"] is not None

    if etype == "HUMAN" and w_neg == 1.0 and has_pred:
        return "OFFENSIVE",     1.0, ser_result["conf"]
    if etype == "NON_HUMAN":
        return "NON_OFFENSIVE", 0.0, ser_result["conf"]
    if etype == "HUMAN" and w_neg == 0.0:
        return "NON_OFFENSIVE", 0.0, ser_result["conf"]
    # AMBIGUOUS → send to CUBD
    return "CUBD", -1.0, 0.50


def process_sentence(text):
    """
    Full COIR pipeline for one sentence.
    Returns dict with all intermediate results and final routing.
    """
    lex       = get_lex()
    off_verbs = lex["offensive_verbs"]
    neg_set   = lex["negations"]

    # 10.0 Ellipsis resolution
    resolved_text, had_ellipsis = resolve_ellipsis(text, off_verbs, neg_set)

    # 10.1 DGT
    dgt_res = dgt_analyze(resolved_text, off_verbs, neg_set)

    # If negation found → immediately route as non-offensive
    if dgt_res["has_neg"]:
        return {
            "text_resolved" : resolved_text,
            "had_ellipsis"  : had_ellipsis,
            "pred"          : dgt_res["pred"],
            "aggressor"     : dgt_res["aggressor"],
            "target"        : dgt_res["target"],
            "has_neg"       : True,
            "neg_token"     : dgt_res["neg_token"],
            "w_neg"         : 0.0,
            "path"          : dgt_res["path"],
            "etype"         : "NEGATED",
            "ner_label"     : None,
            "routing"       : "NON_OFFENSIVE",
            "risk_score"    : 0.0,
            "confidence"    : 0.90,
            "forward_cubd"  : False,
        }

    # 10.2 SER
    ser_res = ser_classify(dgt_res["target"], resolved_text)

    # 10.3 HTV
    routing, risk, conf = htv_route(dgt_res, ser_res)

    return {
        "text_resolved" : resolved_text,
        "had_ellipsis"  : had_ellipsis,
        "pred"          : dgt_res["pred"],
        "aggressor"     : dgt_res["aggressor"],
        "target"        : dgt_res["target"],
        "has_neg"       : dgt_res["has_neg"],
        "neg_token"     : dgt_res["neg_token"],
        "w_neg"         : dgt_res["w_neg"],
        "path"          : dgt_res["path"],
        "etype"         : ser_res["etype"],
        "ner_label"     : ser_res["ner_label"],
        "routing"       : routing,
        "risk_score"    : risk,
        "confidence"    : conf,
        "forward_cubd"  : routing == "CUBD",
    }


# =============================================================================
# TABLE 18 METRICS
# =============================================================================

def compute_coir_metrics(results_df, labels):
    """
    Compute all 11 COIR metrics for Table 18.

    PA   — Path Accuracy: complete S-V-O paths / total
    DC   — Decision Correctness: correct routing decisions / total
    HPDR — Harmful Path Detection Rate: detected harmful / total truly harmful
    NPA  — Negation Path Accuracy: correctly handled negations / negated sentences
    ECA  — Entity Classification Accuracy: correct entity type / total
    HEP  — Human Entity Precision: TP_human / (TP + FP)
    NHEP — Non-Human Entity Precision
    HTP  — Human Target Precision: correctly routed human targets
    NHTP — Non-Human Target Precision: correctly routed non-human
    FPRR — False Positive Reduction Rate: vs no COIR baseline
    RA   — Routing Accuracy: all correctly routed / total
    """
    labels  = np.array(labels)
    routing = results_df["routing"].values
    etype   = results_df["etype"].fillna("AMBIGUOUS").values
    path    = results_df["path"].fillna("").values
    has_neg = results_df["has_neg"].values

    n = len(labels)

    # PA — complete paths (have → in path)
    pa = np.mean(["→" in p and len(p.split()) >= 3 for p in path])

    # DC — correct decisions (offensive→OFFENSIVE, non-offensive→NON_OFFENSIVE)
    correct_dec = sum(
        (r == "OFFENSIVE"     and l == 1) or
        (r == "NON_OFFENSIVE" and l == 0)
        for r, l in zip(routing, labels)
    )
    dc = correct_dec / n

    # HPDR — among truly harmful sentences, how many correctly detected
    truly_harm = labels == 1
    detected   = routing == "OFFENSIVE"
    hpdr = detected[truly_harm].mean() if truly_harm.sum() > 0 else 0.0

    # NPA — negation handling accuracy
    neg_mask = has_neg == True
    if neg_mask.sum() > 0:
        # Negated sentences should route to NON_OFFENSIVE
        npa = (routing[neg_mask] == "NON_OFFENSIVE").mean()
    else:
        npa = 1.0

    # ECA — entity classification accuracy
    eca_correct = sum(
        (l == 1 and e == "HUMAN") or
        (l == 0 and e in {"NON_HUMAN", "AMBIGUOUS", "NEGATED"})
        for e, l in zip(etype, labels)
    )
    eca = eca_correct / n

    # HEP — human entity precision
    h_pred = etype == "HUMAN"
    tp_h   = ((labels == 1) & h_pred).sum()
    fp_h   = ((labels == 0) & h_pred).sum()
    hep    = tp_h / (tp_h + fp_h + 1e-9)

    # NHEP — non-human entity precision
    nh_pred = np.isin(etype, ["NON_HUMAN", "NEGATED"])
    tp_nh   = ((labels == 0) & nh_pred).sum()
    fp_nh   = ((labels == 1) & nh_pred).sum()
    nhep    = tp_nh / (tp_nh + fp_nh + 1e-9)

    # HTP — human target precision (HUMAN→OFFENSIVE correct rate)
    hm  = etype == "HUMAN"
    htp = (routing[hm] == "OFFENSIVE").mean() if hm.sum() > 0 else 0.0

    # NHTP — non-human target precision
    nhm  = np.isin(etype, ["NON_HUMAN"])
    nhtp = (routing[nhm] == "NON_OFFENSIVE").mean() if nhm.sum() > 0 else 0.0

    # FPRR — false positive reduction vs baseline (pred on every sentence)
    base_fp = int((labels == 0).sum())      # baseline: all flagged
    idol_fp = int(((labels == 0) & detected).sum())
    fprr    = (base_fp - idol_fp) / max(base_fp, 1)

    # RA — overall routing accuracy
    ra = correct_dec / n

    return {
        "PA"  : round(float(pa),   3),
        "DC"  : round(float(dc),   3),
        "HPDR": round(float(hpdr), 3),
        "NPA" : round(float(npa),  3),
        "ECA" : round(float(eca),  3),
        "HEP" : round(float(hep),  3),
        "NHEP": round(float(nhep), 3),
        "HTP" : round(float(htp),  3),
        "NHTP": round(float(nhtp), 3),
        "FPRR": round(float(fprr), 3),
        "RA"  : round(float(ra),   3),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  IDOL-F | Step 10: Contextual Offensive Intent Resolution")
    print("  Techniques: Ellipsis Resolution → DGT → SER → HTV")
    print("=" * 65)

    table18_rows = []

    for dataset_name in TRAIN_DATASETS:
        print(f"\n  Dataset: {dataset_name}")

        # Load SAGP output (has text_recovered + labels)
        sagp_path = os.path.join(IN6, f"{dataset_name}_sagp.csv")
        if os.path.exists(sagp_path):
            df = pd.read_csv(sagp_path)
        else:
            sdd_path = os.path.join(STEP_DIRS["step5_sdd"], f"{dataset_name}_sdd.csv")
            df = pd.read_csv(sdd_path)

        text_col = "text_recovered" if "text_recovered" in df.columns else "text"
        texts  = df[text_col].fillna("").astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        if not ABLATION["COIR"]:
            print("  [ABLATION] COIR = False — forwarding ICPS output unchanged")
            df["routing"]      = "CUBD"
            df["forward_cubd"] = 1
            df.to_csv(os.path.join(OUT, f"{dataset_name}_coir.csv"), index=False)
            continue

        print(f"  Processing {len(texts):,} sentences through COIR pipeline...")

        results = [process_sentence(t) for t in texts]
        res_df  = pd.DataFrame(results)

        # Merge with original df
        out_df = df.copy()
        for col in res_df.columns:
            out_df[col] = res_df[col].values
        out_df["forward_cubd"] = res_df["forward_cubd"].astype(int).values
        out_df.to_csv(os.path.join(OUT, f"{dataset_name}_coir.csv"), index=False)

        # Routing summary
        routing_counts = res_df["routing"].value_counts()
        n_ellipsis     = int(res_df["had_ellipsis"].sum())
        print(f"  Ellipsis resolved: {n_ellipsis}")
        print(f"  Routing: OFF={routing_counts.get('OFFENSIVE',0)} "
              f"NON={routing_counts.get('NON_OFFENSIVE',0)} "
              f"CUBD={routing_counts.get('CUBD',0)}")

        # Table 18 metrics
        metrics = compute_coir_metrics(res_df, labels)
        metrics["Dataset"] = dataset_name
        table18_rows.append(metrics)
        print(f"  PA={metrics['PA']} NPA={metrics['NPA']} "
              f"ECA={metrics['ECA']} RA={metrics['RA']} "
              f"FPRR={metrics['FPRR']}")

    if table18_rows:
        cols = ["Dataset", "PA", "DC", "HPDR", "NPA",
                "ECA", "HEP", "NHEP", "HTP", "NHTP", "FPRR", "RA"]
        t18 = pd.DataFrame(table18_rows)[cols]
        t18.to_csv(os.path.join(OUT, "table18_coir_metrics.csv"), index=False)
        print("\n  TABLE 18 — COIR Metrics:")
        print(t18.to_string(index=False))

    print(f"\n  [DONE] Step-10 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
