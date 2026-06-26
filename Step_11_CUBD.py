# =============================================================================
# IDOL-F Framework — Step 11: Conformal Uncertainty Boundary Detection (CUBD)
#
# COVERAGE GUARANTEE: P(y_true ∈ C(x)) ≥ 1 − α
#
# ALGORITHM (6 steps):
# 1. Train classifier → get P(y=1|x), P(y=0|x) for each sentence
# 2. Calibration nonconformity: α_i = 1 − p̂(y_i|x_i)
# 3. Threshold: τ̂ = Quantile_{1-α}({α_i}_{i=1}^{m})
# 4. P-value: p(y) = (|{i : α_i ≥ α_new(y)}| + 1) / (m + 1)
# 5. Rényi entropy: H_α = 1/(1-α)·log Σ_k p_k^α  (α=2, collision entropy)
# 6. Prediction set: C(x) = {y : p(y) > α}
#    Routing:
#      |C(x)| = 1 → confident classification
#      |C(x)| = 2 → uncertain → flag for review
#      |C(x)| = 0 → reject (very high uncertainty)
#
# TABLES:
#   Table 19: CUBD Calibration Verification (α=0.05,0.10,0.15,0.20)
#   Table 20: CUBD Before vs After (F1, ECE, Coverage, Fast-path%)
#
# METRICS: CGR, PSE, UCS, FPE, ECE, F1-Macro
#
# ABLATION: ABLATION["CUBD"] = False → direct argmax, no conformal guarantee
# OUTPUT: output/step11/table19_calibration.csv + table20_before_after.csv
# =============================================================================

import os
import sys
import math

import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, MODEL_CONFIGS,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO,
    CUBD_ALPHA, CUBD_ALPHA_LEVELS, CUBD_RENYI_ALPHA, make_all_dirs
)

make_all_dirs()
IN  = STEP_DIRS["step10"]
IN6 = STEP_DIRS["step6"]
OUT = STEP_DIRS["step11"]

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Classifier model (lightweight for CUBD calibration)
# ─────────────────────────────────────────────────────────────────────────────

class CUBDClassifier(nn.Module):
    """
    Backbone + 2-class head.
    Outputs softmax probabilities for conformal prediction.
    """
    def __init__(self, hf_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs     = out.last_hidden_state
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        logits = self.head(pooled)
        return torch.softmax(logits, dim=-1)   # (B, 2) probabilities


class SimpleDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tok, self.ml = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.ml,
                       padding="max_length", truncation=True,
                       return_tensors="pt")
        return (enc["input_ids"].squeeze(0),
                enc["attention_mask"].squeeze(0),
                torch.tensor(self.labels[idx], dtype=torch.long))


def train_and_get_probs(texts, labels, cfg):
    """
    Train classifier and return calibration+test probabilities.
    Split: 70% train / 15% calibration / 15% test
    """
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n   = len(texts)
    ntr = int(TRAIN_RATIO * n)
    ncal= int(VAL_RATIO   * n)

    tr_txt, tr_lbl   = texts[:ntr],          labels[:ntr]
    cal_txt, cal_lbl = texts[ntr:ntr+ncal],  labels[ntr:ntr+ncal]
    te_txt, te_lbl   = texts[ntr+ncal:],     labels[ntr+ncal:]

    model = CUBDClassifier(cfg["hf_name"], dropout=cfg["dropout"]).to(DEVICE)

    train_dl = DataLoader(SimpleDS(tr_txt, tr_lbl, tok, cfg["max_len"]),
                          batch_size=cfg["batch_size"], shuffle=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                              weight_decay=cfg["weight_decay"])
    sch  = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(cfg["warmup_ratio"] * len(train_dl) * cfg["epochs"]),
        num_training_steps=len(train_dl) * cfg["epochs"],
    )
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    model.train()
    for ep in range(cfg["epochs"]):
        for ids, mask, lbl in train_dl:
            opt.zero_grad()
            probs = model(ids.to(DEVICE), mask.to(DEVICE))
            loss  = crit(torch.log(probs + 1e-9), lbl.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sch.step()
        print(f"    ep{ep+1}/{cfg['epochs']}: loss={loss.item():.4f}")

    def extract_probs(t_list, l_list):
        ds = SimpleDS(t_list, l_list, tok, cfg["max_len"])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
        P_list, L_list = [], []
        model.eval()
        with torch.no_grad():
            for ids, mask, lbl in dl:
                p = model(ids.to(DEVICE), mask.to(DEVICE))
                P_list.append(p.cpu().numpy())
                L_list.extend(lbl.numpy())
        return np.vstack(P_list), np.array(L_list)

    cal_probs, cal_labels = extract_probs(cal_txt, cal_lbl)
    te_probs,  te_labels  = extract_probs(te_txt,  te_lbl)

    del model
    torch.cuda.empty_cache()
    return cal_probs, cal_labels, te_probs, te_labels


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION CORE
# ─────────────────────────────────────────────────────────────────────────────

def nonconformity_score(probs, labels):
    """
    α_i = 1 − p̂(y_i|x_i)
    Model's "surprise" at the true label.
    Higher α → model less confident → less conforming.
    """
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_threshold(cal_scores, alpha):
    """
    τ̂ = Quantile_{1-α}({α_i}_{i=1}^m)
    The (1-α) quantile of calibration nonconformity scores.
    """
    return float(np.quantile(cal_scores, 1.0 - alpha))


def p_value(cal_scores, new_score):
    """
    p(y) = (|{i : α_i ≥ α_new}| + 1) / (m + 1)
    Proportion of calibration scores at least as extreme as the new one.
    +1 in numerator and denominator: Laplace smoothing for validity.
    """
    return (int((cal_scores >= new_score).sum()) + 1) / (len(cal_scores) + 1)


def renyi_entropy(prob_vector, alpha=CUBD_RENYI_ALPHA):
    """
    Rényi Entropy of order α:
    H_α = 1/(1-α) · log(Σ_k p_k^α)

    Special case α=1: Shannon entropy H = -Σ p log p
    α=2: Collision entropy (most used for uncertainty)

    Higher H_α → more uncertain prediction.
    """
    p = np.clip(prob_vector, 1e-9, 1.0)
    if abs(alpha - 1.0) < 1e-6:
        return float(-np.sum(p * np.log(p)))
    return float(1.0 / (1.0 - alpha) * np.log(np.sum(p ** alpha)))


def get_prediction_set(cal_scores, test_probs_i, alpha):
    """
    C(x) = {y : p(y) > α}
    Compute p-value for each label and include those above alpha.
    Returns (prediction_set: list, p_values: dict)
    """
    pred_set = []
    p_vals   = {}
    for y in [0, 1]:
        nc_score = 1.0 - test_probs_i[y]
        pv       = p_value(cal_scores, nc_score)
        p_vals[y] = pv
        if pv > alpha:
            pred_set.append(y)
    return pred_set, p_vals


def route_prediction(pred_set):
    """
    Routing based on prediction set size:
    |C|=1 → confident → classify
    |C|=2 → uncertain → flag
    |C|=0 → reject (very uncertain) → human review
    """
    if len(pred_set) == 1:
        label   = pred_set[0]
        routing = "OFFENSIVE" if label == 1 else "NON_OFFENSIVE"
        return routing, "CONFIDENT"
    if len(pred_set) == 2:
        return "UNCERTAIN", "FLAG"
    return "REJECT", "HUMAN_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_cubd_metrics(te_labels, pred_sets, te_probs, alpha):
    """
    CGR — Coverage Guarantee Rate:  P(y ∈ C(x)) (should be ≥ 1-α)
    PSE — Prediction Set Size:      mean(|C(x)|)
    UCS — Uncertainty Control Score: 1 - |CGR - (1-α)|
    FPE — Fast-Path Efficiency:     fraction with |C|=1 (confident)
    ECE — Expected Calibration Error
    F1  — Macro F1 on confident predictions
    """
    n   = len(te_labels)
    cgr = sum(l in s for l, s in zip(te_labels, pred_sets)) / n
    pse = np.mean([len(s) for s in pred_sets])
    ucs = 1.0 - abs(cgr - (1.0 - alpha))
    fpe = sum(len(s) == 1 for s in pred_sets) / n

    # ECE
    conf  = te_probs[:, 1]
    preds = (conf > 0.5).astype(int)
    edges = np.linspace(0, 1, 11)
    ece   = 0.0
    for i in range(10):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum():
            ece += (m.sum() / n) * abs((preds[m] == te_labels[m]).mean() - conf[m].mean())

    # F1 on confident predictions only
    conf_preds = [s[0] for s in pred_sets if len(s) == 1]
    conf_true  = [l for l, s in zip(te_labels, pred_sets) if len(s) == 1]
    f1 = f1_score(conf_true, conf_preds, average="macro", zero_division=0) \
         if conf_true else 0.0

    return {
        "CGR": round(cgr, 4),
        "PSE": round(pse, 4),
        "UCS": round(ucs, 4),
        "FPE": round(fpe, 4),
        "ECE": round(float(ece), 4),
        "F1" : round(f1, 4),
    }


def calibration_table(cal_scores, cal_labels):
    """
    Table 19: Calibration verification at multiple α levels.
    Shows expected vs empirical coverage.
    """
    rows = []
    for alpha in CUBD_ALPHA_LEVELS:
        sets = []
        for probs_i, lbl in zip(
            # Reconstruct approximate probs from scores
            np.column_stack([cal_scores, 1.0 - cal_scores]),
            cal_labels
        ):
            ps, _ = get_prediction_set(cal_scores, probs_i, alpha)
            sets.append(ps)
        coverage = sum(l in s for l, s in zip(cal_labels, sets)) / len(cal_labels)
        gap      = abs(coverage - (1.0 - alpha))
        rows.append({
            "Alpha"            : alpha,
            "Expected_Coverage": round((1.0 - alpha) * 100, 1),
            "Empirical_Coverage": round(coverage * 100, 1),
            "Gap"              : round(gap, 4),
            "UCS"              : round(1.0 - gap, 4),
        })
    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  IDOL-F | Step 11: Conformal Uncertainty Boundary Detection")
    print(f"  Coverage target: {(1-CUBD_ALPHA)*100:.0f}% (α={CUBD_ALPHA})")
    print("=" * 65)

    all_metric_rows  = []
    cal_table_rows   = []
    before_after_rows = []

    for dataset_name in TRAIN_DATASETS:
        print(f"\n  Dataset: {dataset_name}")

        # Load SAGP/COIR output
        coir_path = os.path.join(IN, f"{dataset_name}_coir.csv")
        sagp_path = os.path.join(IN6, f"{dataset_name}_sagp.csv")
        if os.path.exists(coir_path):
            df = pd.read_csv(coir_path)
        elif os.path.exists(sagp_path):
            df = pd.read_csv(sagp_path)
        else:
            print(f"  [WARNING] No input file for {dataset_name}, skipping")
            continue

        text_col = "text_recovered" if "text_recovered" in df.columns else "text"
        texts  = df[text_col].fillna("").astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        for model_name, cfg in MODEL_CONFIGS.items():
            print(f"\n  >> {model_name}")

            try:
                cal_probs, cal_labels, te_probs, te_labels = \
                    train_and_get_probs(texts, labels, cfg)

                if not ABLATION["CUBD"]:
                    # Direct argmax (no conformal wrapper)
                    preds  = (te_probs[:, 1] > 0.5).astype(int)
                    f1_dir = round(f1_score(te_labels, preds,
                                            average="macro", zero_division=0), 4)
                    all_metric_rows.append({
                        "Model"  : f"IDOL-F+{model_name}",
                        "Dataset": dataset_name,
                        "CGR": "—", "PSE": "—", "UCS": "—",
                        "FPE": "—", "ECE": "—", "F1": f1_dir,
                    })
                    print(f"  [ABLATION] CUBD=False → direct F1={f1_dir}")
                    continue

                # ── CONFORMAL PREDICTION ──────────────────────────────
                cal_scores = nonconformity_score(cal_probs, cal_labels)

                # Build calibration table (once per dataset, first model)
                if not cal_table_rows:
                    cal_table_rows.extend(calibration_table(cal_scores, cal_labels))

                # Prediction sets for test
                pred_sets    = []
                routings     = []
                renyi_scores = []

                for i in range(len(te_probs)):
                    ps, _ = get_prediction_set(cal_scores, te_probs[i], CUBD_ALPHA)
                    pred_sets.append(ps)
                    routing, _ = route_prediction(ps)
                    routings.append(routing)
                    renyi_scores.append(renyi_entropy(te_probs[i]))

                metrics = compute_cubd_metrics(te_labels, pred_sets, te_probs, CUBD_ALPHA)

                # Before CUBD — direct argmax F1
                preds_dir = (te_probs[:, 1] > 0.5).astype(int)
                f1_before = round(f1_score(te_labels, preds_dir,
                                           average="macro", zero_division=0), 4)

                all_metric_rows.append({
                    "Model"  : f"IDOL-F+{model_name}",
                    "Dataset": dataset_name,
                    **metrics,
                })
                before_after_rows.append({
                    "Model"     : f"IDOL-F+{model_name}",
                    "Dataset"   : dataset_name,
                    "F1_Before" : f1_before,
                    "F1_After"  : metrics["F1"],
                    "ECE_After" : metrics["ECE"],
                    "Coverage"  : f"{metrics['CGR']*100:.1f}%",
                    "Fast_path%": f"{metrics['FPE']*100:.1f}%",
                })

                print(f"  CGR={metrics['CGR']} PSE={metrics['PSE']} "
                      f"UCS={metrics['UCS']} FPE={metrics['FPE']} "
                      f"F1_before={f1_before} F1_after={metrics['F1']}")

            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
                import traceback; traceback.print_exc()

    # ── Save Table 19: Calibration Verification ───────────────────
    if cal_table_rows:
        t19 = pd.DataFrame(cal_table_rows)
        t19.to_csv(os.path.join(OUT, "table19_calibration.csv"), index=False)
        print("\n  TABLE 19 — CUBD Calibration Verification:")
        print(t19.to_string(index=False))

    # ── Save Table 20: Before vs After ────────────────────────────
    if before_after_rows:
        t20 = pd.DataFrame(before_after_rows)
        t20.to_csv(os.path.join(OUT, "table20_before_after.csv"), index=False)
        print("\n  TABLE 20 — Before vs After CUBD:")
        print(t20.to_string(index=False))

    # ── Save CUBD metrics summary ─────────────────────────────────
    if all_metric_rows:
        tm = pd.DataFrame(all_metric_rows)
        tm.to_csv(os.path.join(OUT, "cubd_metrics.csv"), index=False)

    print(f"\n  [DONE] Step-11 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
