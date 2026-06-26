# =============================================================================
# IDOL-F Framework — Step 00: Central Configuration
#
# PURPOSE:
#   Single source of truth for the entire IDOL-F pipeline.
#   Every other step imports from this file.
#
# WHAT THIS FILE CONTAINS:
#   1. Dataset paths (training + cross-transfer)
#   2. Output directory structure
#   3. ABLATION STUDY toggles (TRUE/FALSE per component)
#   4. Six language model configurations (HuggingFace names + hyperparameters)
#   5. Per-step algorithm constants
#   6. Hardware / CUDA settings
#   7. OpenAI API key (for Step-03 IPHNS)
#
# ABLATION STUDY USAGE:
#   Set any component to False to disable it. The pipeline will
#   automatically route around disabled components. All results
#   are collected in Step-12 for comparison.
#
#   Example: Set ABLATION["SAGP"] = False
#   → SAGP step is skipped, raw text forwarded to SICL
#   → Step-12 shows performance WITHOUT SAGP
# =============================================================================

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Base directory — change this to your local path
BASE_DIR = r"C:\Users\WorkStation\Videos\Paper-10"

# Training dataset directory
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")

# Cross-transfer evaluation datasets
CROSS_TRANSFER_DIR = os.path.join(BASE_DIR, "Datasets", "Cross_Transfer_Datasets")
CROSS_TRANSFER_DATASETS = {
    "AggPars": os.path.join(CROSS_TRANSFER_DIR, "AggPars.csv"),
    "TwtPars": os.path.join(CROSS_TRANSFER_DIR, "TwtPars.csv"),
}
# Cross-transfer dataset columns: "Text" (input), "Label" (output, binary 0/1)

# Output directory structure
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STEP_DIRS = {
    "step1"         : os.path.join(OUTPUT_DIR, "step1"),
    "lexicon"       : os.path.join(OUTPUT_DIR, "lexicon"),
    "step2"         : os.path.join(OUTPUT_DIR, "step2"),
    "step3"         : os.path.join(OUTPUT_DIR, "step3"),
    "step4"         : os.path.join(OUTPUT_DIR, "step4"),
    "step4_prenorm" : os.path.join(OUTPUT_DIR, "step4", "prenorm"),
    "step4_obfdetx" : os.path.join(OUTPUT_DIR, "step4", "obfdetx"),
    "step4_obfrex"  : os.path.join(OUTPUT_DIR, "step4", "obfrex"),
    "step5"         : os.path.join(OUTPUT_DIR, "step5"),
    "step5_lfa"     : os.path.join(OUTPUT_DIR, "step5", "lfa"),
    "step5_sdd"     : os.path.join(OUTPUT_DIR, "step5", "sdd"),
    "step6"         : os.path.join(OUTPUT_DIR, "step6"),
    "step7"         : os.path.join(OUTPUT_DIR, "step7"),
    "step8"         : os.path.join(OUTPUT_DIR, "step8"),
    "step9"         : os.path.join(OUTPUT_DIR, "step9"),
    "step10"        : os.path.join(OUTPUT_DIR, "step10"),
    "step11"        : os.path.join(OUTPUT_DIR, "step11"),
    "step12"        : os.path.join(OUTPUT_DIR, "step12"),
    "ablation"      : os.path.join(OUTPUT_DIR, "ablation"),
    "models"        : os.path.join(OUTPUT_DIR, "models"),
}

# Training dataset names (in order for MCA cumulative IDS computation)
TRAIN_DATASETS = ["HASOC", "HSLL", "HTEval", "HTXplain", "OLID"]

# Each dataset: file name, text column, label column, label mapping
DATASET_CONFIG = {
    "HASOC": {
        "file"      : "HASOC.csv",
        "text_col"  : "text",
        "label_col" : "task_1",
        # HOF = Hate/Offensive → 1, NOT = Not offensive → 0
        "mapping"   : {"HOF": 1, "NOT": 0},
        "type"      : "direct",
    },
    "HSLL": {
        "file"      : "HSLL.csv",
        "text_col"  : "tweet",
        "label_col" : "class",
        # 0 = hate speech → 1, 1 = offensive → 1, 2 = neither → 0
        "mapping"   : {0: 1, 1: 1, 2: 0},
        "type"      : "direct",
    },
    "HTEval": {
        "file"      : "HTEval.csv",
        "text_col"  : "text",
        "label_col" : "HS",
        "mapping"   : {1: 1, 0: 0},
        "type"      : "direct",
    },
    "HTXplain": {
        "file"      : "HTXplain.csv",
        "text_col"  : "text",
        "label_col" : "labels",
        "mapping"   : None,
        # Labels are lists of annotator votes → majority vote
        "type"      : "majority_vote",
    },
    "OLID": {
        "file"      : "OLID.csv",
        "text_col"  : "text",
        "label_col" : "label",
        "mapping"   : {1: 1, 0: 0},
        "type"      : "direct",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ABLATION STUDY — TRUE/FALSE TOGGLES
#
# Set any component to False to DISABLE it.
# The pipeline will skip that step and forward previous output.
# Results with each component disabled are compared in Step-12.
#
# To run ablation:
#   1. Set one component to False
#   2. Run full pipeline (Step-01 to Step-12)
#   3. Step-12 will log results under that configuration name
#   4. Repeat for each component
# ─────────────────────────────────────────────────────────────────────────────

ABLATION = {
    # Step-01: Multi-Source Corpus Aggregation
    "MCA"       : True,

    # Step-02: Adaptive Corpus Equilibrium (class balancing)
    "ACE"       : True,

    # Step-03: Intent-Preserve Hard Negative Synthesis
    "IPHNS"     : True,

    # Step-04: Obfuscation Detection and Mitigation System
    "ODMS"      : True,

    # Step-05: Semantic Preservation Analysis (LFA + SDD)
    "SPA"       : True,

    # Step-06: Structured Annotated Graph Propagation
    "SAGP"      : True,

    # Step-07: Semantic Intent Contrastive Learning
    "SICL"      : True,

    # Step-08: Role-Annotated Semantic Graph Construction + ARGP
    "RASGC_ARGP": True,

    # Step-09: Intent Cluster Polarity Separation
    "ICPS"      : True,

    # Step-10: Contextual Offensive Intent Resolution
    "COIR"      : True,

    # Step-11: Conformal Uncertainty Boundary Detection
    "CUBD"      : True,
}


def get_ablation_config_name():
    """Return a readable name for the current ablation configuration."""
    disabled = [k for k, v in ABLATION.items() if not v]
    if not disabled:
        return "Full_IDOL-F"
    return "w/o_" + "+".join(disabled)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SIX LANGUAGE MODELS — CONFIGURATIONS
#
# Each model is trained SEPARATELY on each of the 5 datasets.
# Hyperparameters are tuned for RTX 4070 Ti Super (16 GB VRAM).
#
# Dataset split: 70% train / 15% validation / 15% test
# Split is performed inside Step-07 (first model training step).
# ─────────────────────────────────────────────────────────────────────────────

# Data splitting ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# Six model configurations
MODEL_CONFIGS = {

    # Qwen2-0.5B — small decoder LLM, lower LR for stability
    "Qwen2": {
        "hf_name"      : "Qwen/Qwen2-0.5B",
        "learning_rate": 3e-5,
        "batch_size"   : 16,
        "weight_decay" : 0.01,
        "epochs"       : 12,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },

    # RoBERTa-base — strong encoder, standard LR
    "RoBERTa": {
        "hf_name"      : "roberta-base",
        "learning_rate": 2e-5,
        "batch_size"   : 32,
        "weight_decay" : 0.01,
        "epochs"       : 8,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },

    # DeBERTa-v3-base — best encoder, slower convergence
    "DeBERTa": {
        "hf_name"      : "microsoft/deberta-v3-base",
        "learning_rate": 2e-5,
        "batch_size"   : 24,
        "weight_decay" : 0.01,
        "epochs"       : 6,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },

    # HateBERT — domain-specific BERT pre-trained on hate speech
    "HateBERT": {
        "hf_name"      : "GroNLP/hateBERT",
        "learning_rate": 2e-5,
        "batch_size"   : 32,
        "weight_decay" : 0.01,
        "epochs"       : 7,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },

    # XLNet-base — autoregressive encoder, slightly lower LR
    "XLNet": {
        "hf_name"      : "xlnet-base-cased",
        "learning_rate": 1.5e-5,
        "batch_size"   : 24,
        "weight_decay" : 0.01,
        "epochs"       : 5,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },

    # SmolLM2-360M — small efficient LLM
    "SmolLM2": {
        "hf_name"      : "HuggingFaceTB/SmolLM2-360M",
        "learning_rate": 2e-5,
        "batch_size"   : 16,
        "weight_decay" : 0.01,
        "epochs"       : 6,
        "max_len"      : 256,
        "warmup_ratio" : 0.1,
        "grad_clip"    : 1.0,
        "scheduler"    : "linear_warmup",
        "dropout"      : 0.1,
        "label_smoothing": 0.05,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PER-STEP ALGORITHM CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Step-01: MCA ─────────────────────────────────────────────────────────────
# IDS = 1 - CosSim(E_prior, E_new)
MCA_TFIDF_MAX_FEATURES = 5000
MCA_TFIDF_STOP_WORDS   = "english"

# ── Step-02: ACE ─────────────────────────────────────────────────────────────
# N_min = min(N_off, N_non)
# Final size = 2 × N_min
ACE_RANDOM_STATE = 42

# ── Step-03: IPHNS ───────────────────────────────────────────────────────────
OPENAI_API_KEY        = "sk-proj-ABCDE"  # It is secret KEY.
IPHNS_MODEL           = "gpt-4o-mini"
IPHNS_PER_CATEGORY    = 200      # sentences per category (5 categories total)
IPHNS_BATCH_SIZE      = 50       # API batch size
IPHNS_MAX_RETRIES     = 3

# ── Step-04: ODMS ─────────────────────────────────────────────────────────────
# LexPreNorm: 13 sequential cleaning steps (no constants needed — rule-based)
# ObfDetX: 3 detection rules
LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "9": "g", "@": "a", "$": "s",
    "!": "i", "+": "t", "€": "e",
}
INWORD_SYMBOLS = set("@$!*#%+")

# ── Step-05: SPA ─────────────────────────────────────────────────────────────
# LFA — Levenshtein Edit Distance thresholds
LFA_ACCEPT_THRESHOLD  = 2        # LD <= 2 → Accept (faithful recovery)
LFA_FLAG_THRESHOLD    = 3        # LD >= 3 → Flag (suspicious)
# SDD — Wasserstein Distance threshold
SDD_W_THRESHOLD       = 0.15     # W < 0.15 → Accept (meaning preserved)
SPA_BERT_MODEL        = "bert-base-uncased"   # frozen BERT for embeddings

# ── Step-06: SAGP ────────────────────────────────────────────────────────────
# OLR: offensive lexicon matching at ROOT node level
SAGP_SPACY_MODEL      = "en_core_web_sm"
SAGP_WORDNET_DEPTH    = 2        # synset expansion depth
SAGP_MIN_VERB_RATIO   = 2.0      # P(verb|off)/P(verb|non) threshold

# ── Step-07: SICL ────────────────────────────────────────────────────────────
# STC — Semantic Triplet Construction
SICL_MARGIN           = 1.0      # Triplet margin m
# CICA — Cross-Intent Contrastive Alignment
SICL_TEMPERATURE      = 0.07     # InfoNCE temperature τ
# vMF Distribution
SICL_VMF_KAPPA        = 10.0     # concentration κ
SICL_PROJ_DIM         = 128      # projection head output dimension
# Combined SICL Loss weights
SICL_LAMBDA_CE        = 1.0      # CrossEntropy weight λ_CE
SICL_LAMBDA_TRIP      = 0.5      # Triplet loss weight α
SICL_LAMBDA_NCE       = 0.5      # InfoNCE weight β
SICL_LAMBDA_VMF       = 0.3      # vMF weight γ

# ── Step-08: RASGC + ARGP ────────────────────────────────────────────────────
# ARGP hyperparameters
ARGP_HEADS            = 4        # attention heads
ARGP_LAYERS           = 2        # GAT layers
ARGP_BETA_INIT        = 0.5      # negation attenuation init
ARGP_EPOCHS           = 37
ARGP_LR               = 2e-4
ARGP_HIDDEN_DIM       = 768      # node embedding dimension

# ── Step-09: ICPS ────────────────────────────────────────────────────────────
ICPS_N_CLUSTERS       = 3        # Offensive, Non-offensive, Uncertain
ICPS_EM_MAX_ITER      = 100
ICPS_N_INIT           = 10       # GMM restarts for stability
ICPS_UNCERTAIN_THRESH = 0.40     # max_probability < 0.40 → uncertain cluster
ICPS_COV_TYPE         = "full"   # GMM covariance type

# ── Step-10: COIR ────────────────────────────────────────────────────────────
# DGT — Directional Graph Traversal
COIR_NEG_WEIGHT       = 0.0      # w_neg when negation present → benign
# SER — Semantic Entity Recognition
COIR_HUMAN_TAGS       = {"PERSON", "NORP"}   # NER tags for human targets

# ── Step-11: CUBD ────────────────────────────────────────────────────────────
CUBD_ALPHA            = 0.10     # error level → 90% coverage guarantee
CUBD_ALPHA_LEVELS     = [0.05, 0.10, 0.15, 0.20]   # calibration table
CUBD_RENYI_ALPHA      = 2.0      # Rényi entropy order
# Confidence zones for routing
CUBD_CONFIDENT_HIGH   = 0.80     # > 0.80 → confident offensive
CUBD_CONFIDENT_LOW    = 0.40     # < 0.40 → confident non-offensive
# Between 0.40 and 0.80 → uncertain → CUBD analysis

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: HARDWARE / CUDA
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Return CUDA device if available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [GPU] {gpu_name} | VRAM: {vram_gb:.1f} GB")
        else:
            device = torch.device("cpu")
            print("  [CPU] CUDA not available — using CPU")
        return device
    except ImportError:
        print("  [WARNING] PyTorch not installed")
        return None

# Mixed precision training (FP16) for RTX 4070 Ti Super
USE_FP16 = True
# Number of DataLoader workers
NUM_WORKERS = 4
# Pin memory for faster GPU transfer
PIN_MEMORY = True

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_all_dirs():
    """Create all output directories if they don't exist."""
    for name, path in STEP_DIRS.items():
        os.makedirs(path, exist_ok=True)
    print(f"  [OK] All output directories created under: {OUTPUT_DIR}")


def get_notebook_dir():
    """
    Notebook-safe directory resolution.
    Works in both .py scripts and Jupyter Notebooks.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Running in Jupyter Notebook — use current working directory
        return os.path.abspath(".")


def add_code_dir_to_path():
    """Add the code directory to sys.path for imports."""
    code_dir = get_notebook_dir()
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run this file directly to verify setup
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  IDOL-F Framework — Configuration Verification")
    print("=" * 65)

    # Check directories
    print(f"\n  Base dir     : {BASE_DIR}")
    print(f"  Datasets dir : {DATASETS_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")

    # Create output directories
    print("\n  Creating output directories...")
    make_all_dirs()

    # Check datasets exist
    print("\n  Checking training datasets...")
    all_ok = True
    for name, cfg in DATASET_CONFIG.items():
        path = os.path.join(DATASETS_DIR, cfg["file"])
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"    {name:10s}: {status} → {cfg['file']}")
        if not exists:
            all_ok = False

    # Check cross-transfer datasets
    print("\n  Checking cross-transfer datasets...")
    for name, path in CROSS_TRANSFER_DATASETS.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"    {name:10s}: {status}")

    # Check GPU
    print("\n  Checking GPU...")
    device = get_device()

    # Show ablation status
    print("\n  Ablation study configuration:")
    print(f"    Config name : {get_ablation_config_name()}")
    for comp, enabled in ABLATION.items():
        status = "ON " if enabled else "OFF"
        print(f"    {comp:15s}: {status}")

    # Show models
    print(f"\n  Language models ({len(MODEL_CONFIGS)} total):")
    for name, cfg in MODEL_CONFIGS.items():
        print(f"    {name:10s}: {cfg['hf_name']}")
        print(f"              LR={cfg['learning_rate']} "
              f"BS={cfg['batch_size']} "
              f"Epochs={cfg['epochs']} "
              f"MaxLen={cfg['max_len']}")

    # OpenAI key check
    print(f"\n  OpenAI API key: "
          f"{'SET' if 'sk-' in OPENAI_API_KEY else 'NOT SET — update OPENAI_API_KEY'}")

    if all_ok:
        print("\n  [READY] Configuration looks good. Run Step-01 next.")
    else:
        print("\n  [WARNING] Some datasets missing. Check DATASETS_DIR path.")

    print("=" * 65)
