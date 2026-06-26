# =============================================================================
# IDOL-F Framework — Step 01: Multi-Source Corpus Aggregation (MCA)
#                           + Lexicon Extraction
#
# PURPOSE (MCA):
#   1. Load 5 datasets (HASOC, HSLL, HTEval, HTXplain, OLID)
#   2. Standardise labels → binary (1=offensive, 0=non-offensive)
#   3. Compute Inter-Dataset Diversity Score (IDS) as datasets are added
#   4. Save per-dataset standardised CSVs
#
# PURPOSE (Lexicon Extraction — happens HERE as instructed):
#   Extract ALL verbs, subjects, objects, negations authentically from
#   the 5 datasets using SpaCy. These are saved to output/lexicon/ and
#   loaded by downstream steps (06, 07, 08, 10).
#
# IDS FORMULA:
#   IDS = 1 − CosSim(E_prior, E_new)
#   E(·) = mean TF-IDF embedding of a text corpus
#   Higher IDS → new dataset adds more diverse content
#
# LEXICON SOURCES (3-source fusion as per SAGP methodology):
#   Source 1 — Manual core seed list (curated offensive verbs)
#   Source 2 — WordNet synset expansion of core list
#   Source 3 — Statistical extraction from datasets
#              (ratio = P(verb|offensive) / P(verb|non-offensive) ≥ 2.0)
#
# TABLES GENERATED:
#   Table 1: Effect of Dataset Combination on Model Performance
#             Columns: Configuration, Datasets, Records, F1-Macro*,
#                      MCC*, AUC-ROC*, IDS
#             * filled later in Step-12 after model evaluation
#
# OUTPUT:
#   output/step1/<DATASET>.csv              standardised (text, label)
#   output/step1/table1_IDS.csv             Table 1 skeleton
#   output/step1/step1_summary.csv          dataset statistics
#   output/lexicon/offensive_verbs.txt
#   output/lexicon/subjects.txt
#   output/lexicon/objects_human.txt
#   output/lexicon/objects_nonhuman.txt
#   output/lexicon/negations.txt
#   output/lexicon/lexicon_stats.csv
# =============================================================================

import os
import sys
import ast
from collections import Counter

import numpy as np
import pandas as pd

# ── Path resolution (works in .py and Jupyter Notebook) ──────────────────────
_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    DATASETS_DIR, STEP_DIRS, TRAIN_DATASETS, DATASET_CONFIG,
    MCA_TFIDF_MAX_FEATURES, MCA_TFIDF_STOP_WORDS,
    SAGP_MIN_VERB_RATIO, SAGP_SPACY_MODEL,
    make_all_dirs, ABLATION,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

make_all_dirs()

OUT_STEP1   = STEP_DIRS["step1"]
OUT_LEXICON = STEP_DIRS["lexicon"]

# ─────────────────────────────────────────────────────────────────────────────
# PART A: DATASET LOADING & STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(label_entry):
    """
    HTXplain: labels column contains lists of annotator votes.
    Example: ['hatespeech', 'offensive', 'normal', 'normal']
    Rule: if offensive/hatespeech votes >= total/2 → label 1 else 0
    """
    try:
        votes = ast.literal_eval(str(label_entry))
        if not isinstance(votes, list) or len(votes) == 0:
            return None
        off_keywords = {"hatespeech", "offensive", "offensive_language"}
        off_count = sum(1 for v in votes if str(v).lower() in off_keywords)
        return 1 if off_count >= len(votes) / 2 else 0
    except Exception:
        return None


def load_and_standardise(dataset_name):
    """
    Load one dataset CSV, extract text+label, apply label mapping,
    drop invalid rows. Returns clean DataFrame (text, label).
    """
    cfg  = DATASET_CONFIG[dataset_name]
    path = os.path.join(DATASETS_DIR, cfg["file"])

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Please place {cfg['file']} in {DATASETS_DIR}")

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Strip leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    text_col  = cfg["text_col"]
    label_col = cfg["label_col"]

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).str.strip()

    # ── Apply label mapping ───────────────────────────────────────
    if cfg["type"] == "majority_vote":
        # HTXplain: annotator vote lists → majority vote
        out["label"] = df[label_col].apply(majority_vote)
    else:
        out["label"] = df[label_col].map(cfg["mapping"])

    # ── Clean: drop missing, empty text, unmapped labels ─────────
    before = len(out)
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].str.len() > 0]
    out = out[out["text"] != "nan"]
    out["label"] = out["label"].astype(int)
    dropped = before - len(out)

    return out.reset_index(drop=True), dropped


# ─────────────────────────────────────────────────────────────────────────────
# PART B: INTER-DATASET DIVERSITY SCORE (IDS)
#
# Formula: IDS = 1 − CosSim(E_prior, E_new)
# E(·) = mean TF-IDF embedding of the corpus
# Datasets added cumulatively: HASOC → +HSLL → +HTEval → +HTXplain → +OLID
# ─────────────────────────────────────────────────────────────────────────────

def compute_ids_cumulative(datasets_dict):
    """
    Compute IDS as each new dataset is added to the prior corpus.
    Returns list of dicts for Table 1.
    """
    # Fit single TF-IDF on all text so space is comparable
    all_text = []
    for name in TRAIN_DATASETS:
        all_text.extend(datasets_dict[name]["text"].tolist())

    vectorizer = TfidfVectorizer(
        max_features=MCA_TFIDF_MAX_FEATURES,
        stop_words=MCA_TFIDF_STOP_WORDS,
    )
    vectorizer.fit(all_text)

    prior_text   = []
    cumulative_n = 0
    table_rows   = []

    for i, name in enumerate(TRAIN_DATASETS):
        new_text  = datasets_dict[name]["text"].tolist()
        cumulative_n += len(new_text)

        if i == 0:
            # Baseline — no prior corpus to compare against
            ids_val = "—"
        else:
            # IDS = 1 − CosSim(mean_tfidf(prior), mean_tfidf(new))
            E_prior = np.asarray(
                vectorizer.transform(prior_text).mean(axis=0))
            E_new   = np.asarray(
                vectorizer.transform(new_text).mean(axis=0))
            cos_sim = float(cosine_similarity(E_prior, E_new)[0][0])
            ids_val = round(1.0 - cos_sim, 4)

        table_rows.append({
            "Configuration" : f"Config-{i + 1}",
            "Datasets"      : " + ".join(TRAIN_DATASETS[:i + 1]),
            "Records"       : cumulative_n,
            # F1-Macro, MCC, AUC-ROC filled in Step-12 after training
            "F1-Macro"      : "",
            "MCC"           : "",
            "AUC-ROC"       : "",
            "IDS"           : ids_val,
        })

        prior_text.extend(new_text)

    return table_rows


# ─────────────────────────────────────────────────────────────────────────────
# PART C: LEXICON EXTRACTION (runs here, used by Steps 06, 07, 08, 10)
#
# We parse ALL sentences from all 5 datasets with SpaCy and collect:
#   - Offensive verbs (3-source fusion)
#   - Subjects (nsubj / nsubjpass)
#   - Objects: human vs non-human (dobj / pobj / iobj)
#   - Negation markers (neg dependency + seed list)
# ─────────────────────────────────────────────────────────────────────────────

# Source 1 — Manually curated offensive verb seed list
CORE_OFFENSIVE_VERBS = {
    # Violence / Physical harm
    "kill", "murder", "slaughter", "assassinate", "stab", "strangle",
    "beat", "shoot", "bomb", "torture", "execute", "massacre", "butcher",
    # Destruction
    "destroy", "demolish", "obliterate", "eliminate", "annihilate",
    "exterminate", "wipe", "erase", "ruin", "wreck", "devastate",
    # Harm / Abuse
    "harm", "hurt", "injure", "wound", "damage", "abuse", "attack",
    "assault", "rape", "molest", "maim", "cripple",
    # Hatred / Degradation
    "hate", "despise", "loathe", "detest", "abhor", "discriminate",
    "humiliate", "degrade", "demean", "belittle", "mock", "ridicule",
    "insult", "slur", "defame", "denigrate",
    # Threat / Intimidation
    "threaten", "intimidate", "terrorize", "bully", "harass", "stalk",
    "coerce", "menace",
    # Other offensive actions
    "oppress", "exploit", "persecute",
}

# Human pronouns/nouns for object classification
HUMAN_IDENTIFIERS = {
    # Pronouns
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
    "we", "us", "our", "ours", "ourselves",
    # Human nouns
    "someone", "anyone", "everyone", "nobody", "somebody", "everybody",
    "person", "people", "human", "man", "woman", "men", "women",
    "boy", "girl", "kid", "child", "children", "baby",
    "guy", "guys", "gal", "gals", "folk", "folks",
    "citizen", "individual", "member",
}

# Negation seed words
NEGATION_SEEDS = {
    "not", "never", "no", "nor", "neither", "nobody", "nothing",
    "nowhere", "n't", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "didn't", "doesn't", "don't",
    "haven't", "hadn't", "hasn't", "barely", "hardly", "scarcely",
}


def extract_lexicon_from_datasets(datasets_dict):
    """
    Parse all 5 datasets with SpaCy.
    Collect verbs, subjects, objects, negations authentically.
    Returns dict of sets/lists.
    """
    try:
        import spacy
        nlp = spacy.load(SAGP_SPACY_MODEL, disable=["ner"])
    except OSError:
        print("  [INFO] Downloading SpaCy model...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", SAGP_SPACY_MODEL],
            check=True
        )
        import spacy
        nlp = spacy.load(SAGP_SPACY_MODEL, disable=["ner"])

    # Counters for statistical extraction
    verb_off_count = Counter()
    verb_non_count = Counter()
    subjects       = Counter()
    objects_all    = Counter()
    negations      = Counter(NEGATION_SEEDS)  # seed + dataset-extracted
    stats_rows     = []

    total_sentences = 0

    for name in TRAIN_DATASETS:
        df     = datasets_dict[name]
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].tolist()
        total_sentences += len(texts)

        print(f"    Parsing {name} ({len(texts):,} sentences)...")

        n_verbs = n_subj = n_obj = n_neg = 0

        # Batch parse with nlp.pipe for speed
        for doc, label in zip(nlp.pipe(texts, batch_size=512), labels):
            for token in doc:
                lemma = token.lemma_.lower()
                if not lemma.isalpha():
                    continue

                # ── Verbs — count per class ───────────────────────
                if token.pos_ == "VERB":
                    if label == 1:
                        verb_off_count[lemma] += 1
                    else:
                        verb_non_count[lemma] += 1
                    n_verbs += 1

                # ── Subjects ─────────────────────────────────────
                if token.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
                    subjects[lemma] += 1
                    n_subj += 1

                # ── Objects ──────────────────────────────────────
                if token.dep_ in {"dobj", "pobj", "iobj", "attr", "oprd"}:
                    objects_all[lemma] += 1
                    n_obj += 1

                # ── Negations ────────────────────────────────────
                if token.dep_ == "neg" or lemma in NEGATION_SEEDS:
                    negations[lemma] += 1
                    n_neg += 1

        stats_rows.append({
            "Dataset"      : name,
            "Sentences"    : len(texts),
            "Verb_tokens"  : n_verbs,
            "Subjects"     : n_subj,
            "Objects"      : n_obj,
            "Negations"    : n_neg,
        })

    # ── Source 2: WordNet synset expansion ────────────────────────
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("kill")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn

        wordnet_verbs = set(CORE_OFFENSIVE_VERBS)
        for core_verb in CORE_OFFENSIVE_VERBS:
            for syn in wn.synsets(core_verb, pos=wn.VERB):
                for lemma in syn.lemmas():
                    name_clean = lemma.name().lower().replace("_", " ")
                    if " " not in name_clean and name_clean.isalpha():
                        wordnet_verbs.add(name_clean)
        print(f"  WordNet expansion: {len(CORE_OFFENSIVE_VERBS)} core "
              f"→ {len(wordnet_verbs)} after expansion")
    except ImportError:
        print("  [WARNING] NLTK not available — skipping WordNet expansion")
        wordnet_verbs = set(CORE_OFFENSIVE_VERBS)

    # ── Source 3: Statistical extraction from datasets ─────────────
    total_off = sum(verb_off_count.values()) + 1
    total_non = sum(verb_non_count.values()) + 1
    eps = 1e-9

    dataset_verbs = set()
    MIN_COUNT = 5  # verb must appear at least 5 times total

    for verb in set(verb_off_count) | set(verb_non_count):
        c_off = verb_off_count.get(verb, 0)
        c_non = verb_non_count.get(verb, 0)
        if c_off + c_non < MIN_COUNT:
            continue
        ratio = (c_off / total_off + eps) / (c_non / total_non + eps)
        if ratio >= SAGP_MIN_VERB_RATIO and c_off >= MIN_COUNT:
            dataset_verbs.add(verb)

    print(f"  Statistical extraction: {len(dataset_verbs)} offensive verbs "
          f"(ratio ≥ {SAGP_MIN_VERB_RATIO})")

    # ── Merge all three sources ────────────────────────────────────
    all_offensive_verbs = sorted(
        CORE_OFFENSIVE_VERBS | wordnet_verbs | dataset_verbs
    )

    # ── Classify objects as human / non-human ─────────────────────
    objects_human    = []
    objects_nonhuman = []
    for obj, _ in objects_all.most_common():
        if obj in HUMAN_IDENTIFIERS:
            objects_human.append(obj)
        else:
            objects_nonhuman.append(obj)

    # Add dataset-extracted human objects not already in list
    for obj, _ in objects_all.most_common():
        if obj not in objects_human and obj not in objects_nonhuman:
            objects_nonhuman.append(obj)

    return {
        "offensive_verbs"  : all_offensive_verbs,
        "subjects"         : [w for w, _ in subjects.most_common()],
        "objects_human"    : objects_human,
        "objects_nonhuman" : objects_nonhuman,
        "negations"        : sorted(negations.keys()),
        "stats_rows"       : stats_rows,
        "n_core"           : len(CORE_OFFENSIVE_VERBS),
        "n_wordnet"        : len(wordnet_verbs),
        "n_dataset"        : len(dataset_verbs),
        "total_sentences"  : total_sentences,
    }


def save_lexicon(lexicon_dict):
    """Save lexicon resources to output/lexicon/ directory."""
    def write_list(fname, items):
        path = os.path.join(OUT_LEXICON, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in items))
        print(f"  Saved {fname:28s} ({len(items):,} entries)")

    write_list("offensive_verbs.txt",  lexicon_dict["offensive_verbs"])
    write_list("subjects.txt",         lexicon_dict["subjects"])
    write_list("objects_human.txt",    lexicon_dict["objects_human"])
    write_list("objects_nonhuman.txt", lexicon_dict["objects_nonhuman"])
    write_list("negations.txt",        lexicon_dict["negations"])

    pd.DataFrame(lexicon_dict["stats_rows"]).to_csv(
        os.path.join(OUT_LEXICON, "lexicon_stats.csv"), index=False
    )

    # Summary
    with open(os.path.join(OUT_LEXICON, "lexicon_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"IDOL-F Lexicon Summary\n")
        f.write(f"Total offensive verbs : {len(lexicon_dict['offensive_verbs'])}\n")
        f.write(f"  Source 1 (core)     : {lexicon_dict['n_core']}\n")
        f.write(f"  Source 2 (wordnet)  : {lexicon_dict['n_wordnet']}\n")
        f.write(f"  Source 3 (dataset)  : {lexicon_dict['n_dataset']}\n")
        f.write(f"Subjects extracted    : {len(lexicon_dict['subjects'])}\n")
        f.write(f"Human objects         : {len(lexicon_dict['objects_human'])}\n")
        f.write(f"Non-human objects     : {len(lexicon_dict['objects_nonhuman'])}\n")
        f.write(f"Negation markers      : {len(lexicon_dict['negations'])}\n")
        f.write(f"Total sentences parsed: {lexicon_dict['total_sentences']:,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD LEXICON HELPER — called by downstream steps
# ─────────────────────────────────────────────────────────────────────────────

def load_lexicon():
    """
    Load saved lexicon from output/lexicon/.
    Called by Steps 06, 07, 08, 10.
    Returns dict of sets.
    """
    def read_file(fname):
        path = os.path.join(OUT_LEXICON, fname)
        if not os.path.exists(path):
            print(f"  [WARNING] Lexicon file missing: {fname}")
            print(f"  Run Step-01 first to extract lexicon.")
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    return {
        "offensive_verbs"  : read_file("offensive_verbs.txt"),
        "subjects"         : read_file("subjects.txt"),
        "objects_human"    : read_file("objects_human.txt"),
        "objects_nonhuman" : read_file("objects_nonhuman.txt"),
        "negations"        : read_file("negations.txt"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  IDOL-F | Step 01: Multi-Source Corpus Aggregation + Lexicon")
    print("=" * 65)

    # ── A: Load and standardise all datasets ─────────────────────
    print("\n  [A] Loading and standardising datasets...")

    datasets_dict = {}
    summary_rows  = []

    for name in TRAIN_DATASETS:
        print(f"\n  → {name}")
        df, dropped = load_and_standardise(name)
        datasets_dict[name] = df

        # Save standardised dataset
        out_path = os.path.join(OUT_STEP1, f"{name}.csv")
        df.to_csv(out_path, index=False)

        n_off = int((df["label"] == 1).sum())
        n_non = int((df["label"] == 0).sum())

        summary_rows.append({
            "Dataset"      : name,
            "Total"        : len(df),
            "Offensive"    : n_off,
            "Non-offensive": n_non,
            "Dropped"      : dropped,
            "Imbalance_%"  : round(100 * n_off / len(df), 1),
        })

        print(f"     Total   : {len(df):,}")
        print(f"     Off(1)  : {n_off:,}")
        print(f"     Non(0)  : {n_non:,}")
        print(f"     Dropped : {dropped}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT_STEP1, "step1_summary.csv"), index=False
    )
    print("\n  Dataset Summary:")
    print(summary_df.to_string(index=False))

    # ── B: Compute IDS cumulative ─────────────────────────────────
    print("\n  [B] Computing Inter-Dataset Diversity Score (IDS)...")
    table1_rows = compute_ids_cumulative(datasets_dict)
    table1_df   = pd.DataFrame(table1_rows)
    table1_df.to_csv(
        os.path.join(OUT_STEP1, "table1_IDS.csv"), index=False
    )
    print("\n  TABLE 1: Effect of Dataset Combination on Model Performance")
    print(table1_df.to_string(index=False))

    # ── C: Lexicon extraction ─────────────────────────────────────
    print("\n  [C] Extracting lexicon from all 5 datasets...")
    print("  (This parses every sentence with SpaCy — may take a few minutes)")
    print()

    lexicon_dict = extract_lexicon_from_datasets(datasets_dict)
    save_lexicon(lexicon_dict)

    print(f"\n  Lexicon Summary:")
    print(f"    Offensive verbs : {len(lexicon_dict['offensive_verbs']):,}")
    print(f"      Core seed     : {lexicon_dict['n_core']}")
    print(f"      + WordNet     : {lexicon_dict['n_wordnet']}")
    print(f"      + Dataset     : {lexicon_dict['n_dataset']}")
    print(f"    Subjects        : {len(lexicon_dict['subjects']):,}")
    print(f"    Human objects   : {len(lexicon_dict['objects_human']):,}")
    print(f"    Non-human obj.  : {len(lexicon_dict['objects_nonhuman']):,}")
    print(f"    Negations       : {len(lexicon_dict['negations']):,}")
    print(f"    Total parsed    : {lexicon_dict['total_sentences']:,} sentences")

    print(f"\n  [DONE] Step-01 complete.")
    print(f"  Output: {OUT_STEP1}")
    print(f"  Lexicon: {OUT_LEXICON}")
    print("=" * 65)

    return datasets_dict


if __name__ == "__main__":
    main()
