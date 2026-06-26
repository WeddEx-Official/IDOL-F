# =============================================================================
# IDOL-F Framework — Step 07: Semantic Intent Contrastive Learning (SICL)
#
# DATASET SPLIT HAPPENS HERE (first model training step):
#   70% train / 15% validation / 15% test  (random_seed=42)
#
# THREE TRAINING STAGES (cumulative):
#   Stage 1 — Baseline : CrossEntropy loss only
#   Stage 2 — +STC     : + Triplet Margin Loss
#                         L_trip = max(0, ‖za-zp‖ - ‖za-zn‖ + m)
#   Stage 3 — +CICA    : + InfoNCE + vMF
#                         L_nce = -E[log exp(za·zp/τ) / Σ exp(za·zj/τ)]
#                         L_vmf = -1/N Σ log exp(κ μ_y·z) / Σ exp(κ μ_c·z)
#   SICL Full           : L = λCE·LCE + α·Ltrip + β·Lnce + γ·Lvmf
#
# SIX MODELS trained separately, each with own hyperparameters from config.
#
# TABLES GENERATED:
#   Table 10: Baseline + STC Embedding Quality (MIE, ESU, EAS, ICA)
#   Table 11: CICA Embedding Quality (MIE, ESU, EAS, ICA)
#   Table 12: CICA Training Convergence (InfoNCE, ISS, MIB, LEA per epoch)
#   Table 13: VMF Distribution (κ, Log-L, NLL, KS, Intra-cos, Inter-cos, Sep, Ang-Margin)
#   Table 14: SICL Full Embedding Quality
#
# ABLATION: ABLATION["SICL"] = False → no contrastive training, just CE baseline
# OUTPUT: output/step7/<MODEL>_<DATASET>_sicl.pth + embedding CSVs + tables
# =============================================================================

import os
import sys
import math
import shutil

import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, MODEL_CONFIGS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    SICL_MARGIN, SICL_TEMPERATURE, SICL_VMF_KAPPA, SICL_PROJ_DIM,
    SICL_LAMBDA_CE, SICL_LAMBDA_TRIP, SICL_LAMBDA_NCE, SICL_LAMBDA_VMF,
    USE_FP16, make_all_dirs
)

make_all_dirs()
IN  = STEP_DIRS["step6"]
OUT = STEP_DIRS["step7"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize as sk_normalize
from scipy.stats import ks_2samp

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class SICLModel(nn.Module):
    """
    Backbone LM + Classifier head + Projection head.
    Projection head maps to unit sphere for contrastive learning.
    Works for both encoder (BERT-style) and decoder (GPT-style) models.
    """

    def __init__(self, hf_name, proj_dim=SICL_PROJ_DIM, dropout=0.1):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(hf_name)
        hidden          = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
        )

    def mean_pool(self, last_hidden, attention_mask):
        """Mean pool over non-padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed  = (last_hidden * mask_expanded).sum(dim=1)
        counts  = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.mean_pool(out.last_hidden_state, attention_mask)
        logits  = self.classifier(pooled)
        # L2 normalize projection for unit sphere (for vMF and InfoNCE)
        z       = F.normalize(self.projector(pooled), p=2, dim=-1)
        return logits, z


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def triplet_margin_loss(z_a, z_p, z_n, margin=SICL_MARGIN):
    """
    Semantic Triplet Construction (STC) loss.
    L_trip = max(0, ‖za-zp‖₂ - ‖za-zn‖₂ + m)
    Pushes anchor closer to positive and farther from hard negative.
    """
    d_pos = F.pairwise_distance(z_a, z_p, p=2)
    d_neg = F.pairwise_distance(z_a, z_n, p=2)
    loss  = F.relu(d_pos - d_neg + margin)
    return loss.mean()


def infonce_loss(z_a, z_p, temperature=SICL_TEMPERATURE):
    """
    Cross-Intent Contrastive Alignment (CICA) — InfoNCE loss.
    L_nce = -E[log exp(za·zp/τ) / Σ_j exp(za·zj/τ)]

    All other batch samples are treated as negatives.
    Mutual information lower bound: I(Za;Zp) >= log(N) - L_nce
    """
    N   = z_a.shape[0]
    sim = torch.matmul(z_a, z_p.T) / temperature   # (N, N)
    labels = torch.arange(N, device=z_a.device)
    return F.cross_entropy(sim, labels)


def vmf_loss(z, labels, kappa=SICL_VMF_KAPPA):
    """
    vMF (von Mises-Fisher) cluster loss.
    L_vmf = -1/N Σ log exp(κ μ_y·z) / Σ_c exp(κ μ_c·z)

    Pulls embeddings toward their class cluster center on the unit sphere.
    μ_c = normalize(mean(z_i : y_i = c))
    """
    # Compute cluster means per class
    unique_labels = torch.unique(labels)
    mu_list = []
    label_to_idx = {}

    for i, lbl in enumerate(unique_labels):
        mask    = (labels == lbl)
        mu      = F.normalize(z[mask].mean(dim=0), p=2, dim=0)
        mu_list.append(mu)
        label_to_idx[lbl.item()] = i

    mu = torch.stack(mu_list)   # (C, proj_dim)

    # Scores: κ × μ_c^T z_i  for each sample and each class
    scores  = kappa * torch.matmul(z, mu.T)   # (N, C)
    # Map original labels to cluster indices
    mapped  = torch.tensor(
        [label_to_idx[l.item()] for l in labels],
        device=labels.device
    )
    return F.cross_entropy(scores, mapped)


def label_smoothing_ce(logits, labels, smoothing=0.05):
    """
    CrossEntropy with label smoothing for better generalization.
    Reduces overconfidence and improves calibration.
    """
    n_classes = logits.shape[1]
    with torch.no_grad():
        smooth_labels = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    log_prob = F.log_softmax(logits, dim=-1)
    return -(smooth_labels * log_prob).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Standard text classification dataset."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer          = tokenizer
        self.max_len            = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class TripletDataset(Dataset):
    """
    Returns (anchor, positive, hard-negative) triplets.
    Positive: same label as anchor, similar SAGP reason.
    Hard Negative: different label but surface-level aggressive text.
    """

    def __init__(self, df, tokenizer, max_len):
        self.tok     = tokenizer
        self.max_len = max_len
        self.off_texts  = df[df["label"] == 1].get(
            "text_recovered", df[df["label"] == 1]["text"]
        ).astype(str).tolist()
        self.non_texts  = df[df["label"] == 0].get(
            "text_recovered", df[df["label"] == 0]["text"]
        ).astype(str).tolist()

        if not self.off_texts:
            self.off_texts = ["offensive text placeholder"]
        if not self.non_texts:
            self.non_texts = ["non-offensive text placeholder"]

    def _encode(self, text):
        enc = self.tok(
            text, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.off_texts)

    def __getitem__(self, idx):
        anchor   = self.off_texts[idx]
        positive = self.off_texts[(idx + 1) % len(self.off_texts)]
        negative = self.non_texts[idx % len(self.non_texts)]

        ai, am = self._encode(anchor)
        pi, pm = self._encode(positive)
        ni, nm = self._encode(negative)

        return ai, am, pi, pm, ni, nm


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mie(nce_loss_val, batch_size):
    """
    Mutual Information Estimate (MIE).
    MIE = log(N) - L_nce
    Higher MIE = embeddings encode more intent information.
    """
    return round(math.log(max(batch_size, 1)) - nce_loss_val, 4)


def compute_esu(Z):
    """
    Embedding Space Uniformity (ESU).
    U(Z) = log(1/N² ΣΣ exp(-2‖zi-zj‖²))
    Measures how uniformly embeddings fill the hypersphere.
    Lower (less negative) = more uniform.
    """
    Z_t  = torch.tensor(Z, dtype=torch.float32)
    sim  = torch.matmul(Z_t, Z_t.T)
    sq   = 2.0 - 2.0 * sim.clamp(-1, 1)
    avg  = torch.exp(-2.0 * sq).mean()
    return round(math.log(avg.item() + 1e-9), 4)


def compute_eas(Z, y):
    """
    Embedding Alignment Score (EAS).
    A(Z) = -1/|T| Σ ‖za-zp‖² over same-label pairs.
    Higher (less negative) = tighter same-intent clusters.
    """
    Z_t   = torch.tensor(Z, dtype=torch.float32)
    y_arr = np.array(y)
    total = count = 0.0

    for cls in np.unique(y_arr):
        idxs = np.where(y_arr == cls)[0]
        # Limit to first 50 per class for speed
        idxs = idxs[:50]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                total += (Z_t[idxs[a]] - Z_t[idxs[b]]).pow(2).sum().item()
                count += 1

    return round(-total / count, 4) if count > 0 else 0.0


def compute_ica(Z, y):
    """
    Intent Cluster Accuracy (ICA) — linear probe.
    Train logistic regression on embeddings, evaluate test accuracy.
    Higher = embeddings are more linearly separable by intent.
    """
    if len(np.unique(y)) < 2 or len(Z) < 10:
        return 0.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        Z, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    clf  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr, y_tr)
    return round(accuracy_score(y_te, clf.predict(X_te)), 4)


def compute_iss(Z, y):
    """
    Intent Separation Score (ISS).
    ISS = d_inter / d_intra
    Higher = clusters more separated.
    """
    Z_t   = torch.tensor(Z, dtype=torch.float32)
    y_arr = np.array(y)
    intra, inter = [], []

    for a in range(len(Z_t)):
        for b in range(a + 1, min(a + 20, len(Z_t))):
            d = (Z_t[a] - Z_t[b]).pow(2).sum().sqrt().item()
            if y_arr[a] == y_arr[b]:
                intra.append(d)
            else:
                inter.append(d)

    d_intra = np.mean(intra) if intra else 1e-9
    d_inter = np.mean(inter) if inter else 0.0
    return round(d_inter / (d_intra + 1e-9), 4)


def compute_vmf_metrics(Z, y):
    """
    Table 13: VMF Distribution Metrics.
    κ (concentration), Log-L, NLL, KS, Intra-cos, Inter-cos, Sep, Angular-Margin.
    """
    Z_norm = sk_normalize(Z, norm="l2")
    y_arr  = np.array(y)
    Z0     = Z_norm[y_arr == 0]
    Z1     = Z_norm[y_arr == 1]

    def est_kappa(Zc):
        if len(Zc) == 0:
            return 0.0
        r_bar = np.linalg.norm(Zc.mean(axis=0))
        d     = Zc.shape[1]
        k     = r_bar * (d - r_bar**2) / (1.0 - r_bar**2 + 1e-9)
        return round(float(max(k, 0.0)), 3)

    kappa_off = est_kappa(Z1)
    kappa_non = est_kappa(Z0)

    def log_likelihood(Zc, kappa):
        if len(Zc) == 0 or kappa == 0:
            return 0.0
        mu  = sk_normalize(Zc.mean(axis=0).reshape(1, -1))[0]
        return float((kappa * Zc.dot(mu)).sum())

    ll  = round(log_likelihood(Z1, kappa_off) + log_likelihood(Z0, kappa_non), 2)
    nll = round(-ll / max(len(Z), 1), 4)

    # KS test between distributions
    if len(Z0) > 10 and len(Z1) > 10:
        mu1 = sk_normalize(Z1.mean(axis=0).reshape(1, -1))[0]
        mu0 = sk_normalize(Z0.mean(axis=0).reshape(1, -1))[0]
        scores1 = Z_norm.dot(mu1)
        scores0 = Z_norm.dot(mu0)
        ks_stat = round(float(ks_2samp(scores1, scores0)[0]), 4)
    else:
        ks_stat = 0.0

    # Cosine similarity metrics
    def mean_cosine(A, B=None):
        if B is None:
            B = A
        if len(A) == 0 or len(B) == 0:
            return 0.0
        n_sample = min(200, len(A), len(B))
        ia = np.random.choice(len(A), n_sample, replace=False)
        ib = np.random.choice(len(B), n_sample, replace=False)
        return round(float(A[ia].dot(B[ib].T).mean()), 4)

    intra = round((mean_cosine(Z1) + mean_cosine(Z0)) / 2, 4)
    inter = mean_cosine(Z0, Z1)
    sep   = round(intra - inter, 4)

    # Angular margin between cluster centers
    if len(Z0) > 0 and len(Z1) > 0:
        mu0 = sk_normalize(Z0.mean(axis=0).reshape(1, -1))[0]
        mu1 = sk_normalize(Z1.mean(axis=0).reshape(1, -1))[0]
        cos_val   = np.clip(mu0.dot(mu1), -1.0, 1.0)
        ang_margin = round(math.degrees(math.acos(cos_val)), 3)
    else:
        ang_margin = 0.0

    return {
        "kappa_off"  : kappa_off,
        "kappa_non"  : kappa_non,
        "Log-L"      : ll,
        "NLL"        : nll,
        "KS"         : ks_stat,
        "Intra-cos"  : intra,
        "Inter-cos"  : inter,
        "Sep"        : sep,
        "Ang-Margin" : ang_margin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader):
    """Extract projection embeddings from trained model."""
    model.eval()
    Z_list, Y_list = [], []
    for ids, mask, lbls in loader:
        _, z = model(ids.to(DEVICE), mask.to(DEVICE))
        Z_list.append(z.cpu().numpy())
        Y_list.append(lbls.numpy())
    return np.vstack(Z_list), np.concatenate(Y_list)


def train_one_stage(model, train_dl, triplet_dl, cfg, stage,
                    n_epochs=None, label_smooth=0.05):
    """
    Train model for one SICL stage.
    stage: "baseline", "stc", "cica", "full"
    """
    epochs = n_epochs or cfg["epochs"]
    opt    = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_dl) * epochs
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if USE_FP16 and DEVICE.type == "cuda" else None

    epoch_logs = []
    last_nce   = 2.8

    for ep in range(epochs):
        model.train()
        total_loss  = 0.0
        nce_vals    = []
        trip_iter   = iter(triplet_dl)

        for ids, mask, lbls in train_dl:
            ids, mask, lbls = ids.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()

            def forward_pass():
                logits, z = model(ids, mask)
                # CrossEntropy (always)
                loss = SICL_LAMBDA_CE * label_smoothing_ce(logits, lbls, label_smooth)

                # STC: Triplet loss
                if stage in ("stc", "cica", "full"):
                    try:
                        ai, am, pi, pm, ni, nm = next(trip_iter)
                    except StopIteration:
                        ai, am, pi, pm, ni, nm = next(iter(triplet_dl))
                    _, za = model(ai.to(DEVICE), am.to(DEVICE))
                    _, zp = model(pi.to(DEVICE), pm.to(DEVICE))
                    _, zn = model(ni.to(DEVICE), nm.to(DEVICE))
                    loss = loss + SICL_LAMBDA_TRIP * triplet_margin_loss(za, zp, zn)

                # CICA: InfoNCE + vMF
                if stage in ("cica", "full"):
                    half = z.shape[0] // 2
                    if half > 1:
                        nce = infonce_loss(z[:half], z[half:2*half])
                        nce_vals.append(nce.item())
                        loss = loss + SICL_LAMBDA_NCE * nce
                    if len(torch.unique(lbls)) > 1:
                        loss = loss + SICL_LAMBDA_VMF * vmf_loss(z, lbls)

                return loss

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = forward_pass()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss = forward_pass()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_dl), 1)
        if nce_vals:
            last_nce = float(np.mean(nce_vals))

        epoch_logs.append({
            "epoch"   : ep + 1,
            "avg_loss": round(avg_loss, 4),
            "InfoNCE" : round(last_nce, 4),
        })
        print(f"      ep{ep+1}/{epochs}: loss={avg_loss:.4f} nce={last_nce:.4f}")

    return last_nce, epoch_logs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  IDOL-F | Step 07: Semantic Intent Contrastive Learning")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # Accumulators for all tables
    t10_rows  = []   # Baseline + STC
    t11_rows  = []   # CICA
    t12_rows  = []   # CICA convergence
    t13_rows  = []   # VMF distribution
    t14_rows  = []   # SICL Full

    for dataset_name in TRAIN_DATASETS:
        print(f"\n  {'='*50}")
        print(f"  Dataset: {dataset_name}")
        print(f"  {'='*50}")

        df = pd.read_csv(os.path.join(IN, f"{dataset_name}_sagp.csv"))
        text_col = "text_recovered" if "text_recovered" in df.columns else "text"
        texts  = df[text_col].fillna("").astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        # ── DATASET SPLIT (70/15/15) — happens here first time ───
        X_tv, X_test, y_tv, y_test = train_test_split(
            texts, labels, test_size=TEST_RATIO,
            random_state=RANDOM_SEED, stratify=labels
        )
        val_ratio_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_ratio_adj,
            random_state=RANDOM_SEED, stratify=y_tv
        )
        print(f"  Split: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

        # Save split indices for downstream steps
        split_df = df.copy()
        split_indices = {
            "train": list(range(len(X_train))),
            "val"  : list(range(len(X_train), len(X_train)+len(X_val))),
            "test" : list(range(len(X_train)+len(X_val), len(texts))),
        }
        import json
        with open(os.path.join(OUT, f"{dataset_name}_split_indices.json"), "w") as f:
            json.dump(split_indices, f)

        for model_name, cfg in MODEL_CONFIGS.items():
            print(f"\n  >> Model: {model_name} ({cfg['hf_name']})")

            try:
                tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                train_df = df.iloc[:len(X_train)]
                ml = cfg["max_len"]

                train_ds  = TextDataset(X_train, y_train, tokenizer, ml)
                test_ds   = TextDataset(X_test,  y_test,  tokenizer, ml)
                triplet_ds = TripletDS(train_df, tokenizer, ml)

                train_dl  = DataLoader(train_ds,   batch_size=cfg["batch_size"],
                                       shuffle=True,  drop_last=True,  num_workers=0)
                test_dl   = DataLoader(test_ds,    batch_size=cfg["batch_size"],
                                       shuffle=False, num_workers=0)
                triplet_dl = DataLoader(triplet_ds, batch_size=cfg["batch_size"],
                                        shuffle=True,  drop_last=True,  num_workers=0)

                bs = cfg["batch_size"]

                if not ABLATION["SICL"]:
                    # Baseline only (CE loss, no contrastive)
                    print(f"  [ABLATION] SICL=False — training baseline only")
                    model = SICLModel(cfg["hf_name"], proj_dim=SICL_PROJ_DIM,
                                      dropout=cfg["dropout"]).to(DEVICE)
                    nce, _ = train_one_stage(model, train_dl, triplet_dl, cfg, "baseline")
                    Z, Y   = extract_embeddings(model, test_dl)
                    row = {
                        "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                        "Stage": "Baseline",
                        "MIE": compute_mie(nce, bs), "ESU": compute_esu(Z),
                        "EAS": compute_eas(Z, Y), "ICA": compute_ica(Z, Y),
                    }
                    t10_rows.append(row)
                    torch.save(model.state_dict(),
                        os.path.join(OUT, f"{model_name}_{dataset_name}_sicl.pth"))
                    del model
                    torch.cuda.empty_cache()
                    continue

                # Stage 1 — Baseline
                print("    Stage 1: Baseline (CE only)")
                model = SICLModel(cfg["hf_name"], proj_dim=SICL_PROJ_DIM,
                                  dropout=cfg["dropout"]).to(DEVICE)
                nce_b, _ = train_one_stage(model, train_dl, triplet_dl, cfg, "baseline")
                Z_b, Y_b = extract_embeddings(model, test_dl)
                t10_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "Stage": "Baseline",
                    "MIE": compute_mie(nce_b, bs), "ESU": compute_esu(Z_b),
                    "EAS": compute_eas(Z_b, Y_b), "ICA": compute_ica(Z_b, Y_b),
                })
                del model; torch.cuda.empty_cache()

                # Stage 2 — +STC
                print("    Stage 2: +STC (Triplet loss)")
                model = SICLModel(cfg["hf_name"], proj_dim=SICL_PROJ_DIM,
                                  dropout=cfg["dropout"]).to(DEVICE)
                nce_stc, _ = train_one_stage(model, train_dl, triplet_dl, cfg, "stc")
                Z_s, Y_s   = extract_embeddings(model, test_dl)
                t10_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "Stage": "+STC",
                    "MIE": compute_mie(nce_stc, bs), "ESU": compute_esu(Z_s),
                    "EAS": compute_eas(Z_s, Y_s), "ICA": compute_ica(Z_s, Y_s),
                })
                del model; torch.cuda.empty_cache()

                # Stage 3 — +CICA
                print("    Stage 3: +CICA (InfoNCE + vMF)")
                model = SICLModel(cfg["hf_name"], proj_dim=SICL_PROJ_DIM,
                                  dropout=cfg["dropout"]).to(DEVICE)
                nce_c, epoch_logs = train_one_stage(model, train_dl, triplet_dl,
                                                    cfg, "cica")
                Z_c, Y_c = extract_embeddings(model, test_dl)
                t11_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "Stage": "+CICA",
                    "MIE": compute_mie(nce_c, bs), "ESU": compute_esu(Z_c),
                    "EAS": compute_eas(Z_c, Y_c), "ICA": compute_ica(Z_c, Y_c),
                })
                # Table 12: CICA convergence per epoch
                for ep_log in epoch_logs:
                    Z_ep, Y_ep = Z_c, Y_c
                    t12_rows.append({
                        "Model"    : f"IDOL-F+{model_name}",
                        "Dataset"  : dataset_name,
                        "Epoch"    : ep_log["epoch"],
                        "InfoNCE"  : ep_log["InfoNCE"],
                        "ISS"      : compute_iss(Z_ep, Y_ep),
                        "MIB"      : compute_mie(ep_log["InfoNCE"], bs),
                        "LEA"      : compute_ica(Z_ep, Y_ep),
                    })
                del model; torch.cuda.empty_cache()

                # Stage 4 — SICL Full
                print("    Stage 4: SICL Full (CE + Triplet + InfoNCE + vMF)")
                model = SICLModel(cfg["hf_name"], proj_dim=SICL_PROJ_DIM,
                                  dropout=cfg["dropout"]).to(DEVICE)
                nce_f, _ = train_one_stage(model, train_dl, triplet_dl, cfg, "full")
                Z_f, Y_f  = extract_embeddings(model, test_dl)
                t14_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "Stage": "SICL Full",
                    "MIE": compute_mie(nce_f, bs), "ESU": compute_esu(Z_f),
                    "EAS": compute_eas(Z_f, Y_f), "ICA": compute_ica(Z_f, Y_f),
                })
                # Table 13: VMF metrics
                vmf = compute_vmf_metrics(Z_f, Y_f)
                t13_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    **vmf,
                })
                # Save best model
                torch.save(model.state_dict(),
                    os.path.join(OUT, f"{model_name}_{dataset_name}_sicl.pth"))
                # Save embeddings for downstream
                np.save(os.path.join(OUT, f"{model_name}_{dataset_name}_Z.npy"), Z_f)
                np.save(os.path.join(OUT, f"{model_name}_{dataset_name}_Y.npy"), Y_f)

                print(f"    SICL Full — MIE={compute_mie(nce_f,bs)} "
                      f"ICA={compute_ica(Z_f,Y_f)} VMF_sep={vmf['Sep']}")
                del model; torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [ERROR] {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── Save all tables ────────────────────────────────────────────
    for rows, fname, label in [
        (t10_rows, "table10_baseline_stc.csv",     "TABLE 10 — Baseline + STC"),
        (t11_rows, "table11_cica.csv",             "TABLE 11 — CICA"),
        (t12_rows, "table12_cica_convergence.csv", "TABLE 12 — CICA Convergence"),
        (t13_rows, "table13_vmf.csv",              "TABLE 13 — VMF Distribution"),
        (t14_rows, "table14_sicl_full.csv",        "TABLE 14 — SICL Full"),
    ]:
        if rows:
            df_t = pd.DataFrame(rows)
            df_t.to_csv(os.path.join(OUT, fname), index=False)
            print(f"\n  {label}:")
            print(df_t.head(12).to_string(index=False))

    print(f"\n  [DONE] Step-07 complete. Output: {OUT}")
    print("=" * 65)


# Fix missing import in TripletDataset
class TripletDS(TripletDataset):
    pass


if __name__ == "__main__":
    main()
