# =============================================================================
# IDOL-F Framework — Step 09: Intent Cluster Polarity Separation (ICPS)
#
# Algorithm: GMM (K=3) + EM + Platt Scaling
# Clusters: Offensive (0), Non-Offensive (1), Uncertain (2)
#
# GMM: p(z) = Σ_k π_k N(z|μ_k, Σ_k)
# E-step: γ_ik = π_k N(z_i|μ_k,Σ_k) / Σ_j π_j N(z_i|μ_j,Σ_j)
# M-step: μ_k = Σ γ_ik z_i / Σ γ_ik
# Platt:  P_cal = 1 / (1 + exp(-(A·s + B)))
#
# Uncertainty: max_k P_cal(k|z) < ICPS_UNCERTAIN_THRESH → uncertain cluster
#
# METRICS: CSS, DBI, Silhouette, ECE, URR, F1-Macro
# TABLE: step9_ICPS_metrics.csv
# ABLATION: ABLATION["ICPS"] = False → all forwarded as uncertain
# =============================================================================

import os, sys
import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, MODEL_CONFIGS,
    RANDOM_SEED, ICPS_N_CLUSTERS, ICPS_EM_MAX_ITER,
    ICPS_N_INIT, ICPS_UNCERTAIN_THRESH, ICPS_COV_TYPE, make_all_dirs
)

make_all_dirs()
IN  = STEP_DIRS["step8"]
IN7 = STEP_DIRS["step7"]
OUT = STEP_DIRS["step9"]

import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (f1_score, silhouette_score,
                              davies_bouldin_score)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize as sk_normalize

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(model_name, dataset_name):
    """Load SICL projection embeddings saved by Step-07."""
    Z_path = os.path.join(IN7, f"{model_name}_{dataset_name}_Z.npy")
    Y_path = os.path.join(IN7, f"{model_name}_{dataset_name}_Y.npy")
    if os.path.exists(Z_path) and os.path.exists(Y_path):
        return np.load(Z_path), np.load(Y_path)
    return None, None


def compute_css(gmm):
    """
    Cluster Separation Score (CSS).
    CSS = ‖μ_off - μ_non‖ / (σ_off + σ_non)
    Higher = clusters more separated.
    """
    means = gmm.means_
    covs  = gmm.covariances_
    if covs.ndim == 3:
        sds = [np.sqrt(np.trace(covs[k])) for k in range(len(means))]
    else:
        sds = [np.sqrt(np.trace(covs))] * len(means)
    return round(float(np.linalg.norm(means[0] - means[1]) / (sds[0] + sds[1] + 1e-9)), 3)


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE).
    ECE = Σ_bins (|B_m|/N) |acc(B_m) - conf(B_m)|
    Lower = better calibrated.
    """
    conf   = probs[:, 1]
    preds  = (conf > 0.5).astype(int)
    labels = np.array(labels)
    edges  = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    N      = len(labels)
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i+1])
        if mask.sum() == 0:
            continue
        acc  = (preds[mask] == labels[mask]).mean()
        conf_mean = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf_mean)
    return round(float(ece), 4)


def run_icps_for_model(model_name, dataset_name, Z, Y):
    """Run GMM + EM + Platt for one model-dataset pair."""
    Z_norm = sk_normalize(Z, norm="l2")

    # GMM with warm start from label hints
    gmm = GaussianMixture(
        n_components=ICPS_N_CLUSTERS,
        covariance_type=ICPS_COV_TYPE,
        max_iter=ICPS_EM_MAX_ITER,
        n_init=ICPS_N_INIT,
        random_state=RANDOM_SEED,
    )
    gmm.fit(Z_norm)
    hard    = gmm.predict(Z_norm)
    probs_g = gmm.predict_proba(Z_norm)

    # Platt scaling on confident clusters (not uncertain)
    mask = hard != 2
    if mask.sum() > 30 and len(np.unique(hard[mask])) > 1:
        y_platt = (hard[mask] == 0).astype(int)
        try:
            clf = CalibratedClassifierCV(
                LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
                method="sigmoid", cv=3
            )
            clf.fit(Z_norm[mask], y_platt)
            probs_cal = clf.predict_proba(Z_norm)
        except Exception:
            probs_cal = np.column_stack([probs_g[:, 1], probs_g[:, 0]])
    else:
        probs_cal = np.column_stack([probs_g[:, 0], probs_g[:, 1]])

    # Uncertain routing
    max_prob  = probs_cal.max(axis=1)
    uncertain = max_prob < ICPS_UNCERTAIN_THRESH
    cluster   = np.where(uncertain, "uncertain",
                np.where(probs_cal[:, 1] >= 0.5, "offensive", "non_offensive"))

    # Metrics
    idx_sample = np.random.choice(len(Z_norm), min(2000, len(Z_norm)), replace=False)
    try:
        ss = round(float(silhouette_score(Z_norm[idx_sample], hard[idx_sample],
                                          metric="cosine")), 3)
    except Exception:
        ss = 0.0
    try:
        dbi = round(float(davies_bouldin_score(Z_norm, hard)), 3)
    except Exception:
        dbi = 0.0

    # Map cluster to binary prediction for F1
    bin_pred = np.where(hard == 0, 1, np.where(hard == 1, 0, -1))
    mask2    = bin_pred != -1
    f1 = round(f1_score(Y[mask2], bin_pred[mask2],
                        average="macro", zero_division=0), 3) if mask2.sum() else 0.0

    ece = compute_ece(probs_cal, Y)
    css = compute_css(gmm)
    urr = round(float(uncertain.mean()), 3)

    return {
        "CSS"         : css,
        "DBI"         : dbi,
        "Silhouette"  : ss,
        "ECE"         : ece,
        "URR"         : urr,
        "F1"          : f1,
    }, cluster, uncertain, probs_cal


def main():
    print("=" * 65)
    print("  IDOL-F | Step 09: Intent Cluster Polarity Separation")
    print("=" * 65)

    metric_rows = []

    for dataset_name in TRAIN_DATASETS:
        print(f"\n  Dataset: {dataset_name}")
        df = pd.read_csv(os.path.join(IN, f"rasgc_{dataset_name}.csv")
                         if os.path.exists(os.path.join(IN, f"rasgc_{dataset_name}.csv"))
                         else os.path.join(IN7, f"{list(MODEL_CONFIGS.keys())[0]}_{dataset_name}_sicl.pth").replace(".pth", ""))
        # Load the base dataset
        sagp_path = os.path.join(STEP_DIRS["step6"], f"{dataset_name}_sagp.csv")
        if os.path.exists(sagp_path):
            df = pd.read_csv(sagp_path)
        else:
            df = pd.read_csv(os.path.join(STEP_DIRS["step5_sdd"], f"{dataset_name}_sdd.csv"))

        labels = df["label"].astype(int).tolist()

        for model_name in MODEL_CONFIGS:
            print(f"  >> {model_name}")

            if not ABLATION["ICPS"]:
                print("  [ABLATION] ICPS = False — all forwarded as uncertain")
                metric_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "CSS":0, "DBI":0, "Silhouette":0, "ECE":0, "URR":1.0, "F1":0
                })
                df["icps_cluster"]  = "uncertain"
                df["forward_cubd"]  = 1
                continue

            Z, Y = load_embeddings(model_name, dataset_name)
            if Z is None:
                print(f"    [WARNING] No embeddings for {model_name}+{dataset_name}")
                metric_rows.append({
                    "Model": f"IDOL-F+{model_name}", "Dataset": dataset_name,
                    "CSS":0, "DBI":0, "Silhouette":0, "ECE":0, "URR":0, "F1":0
                })
                continue

            try:
                metrics, cluster, uncertain, probs_cal = run_icps_for_model(
                    model_name, dataset_name, Z, Y
                )
                # Save clustering output
                n_test = len(cluster)
                out_df = df.tail(n_test).copy()
                out_df["icps_cluster"]   = cluster
                out_df["forward_cubd"]   = uncertain.astype(int)
                out_df["p_offensive"]    = probs_cal[:, 1]
                out_df["p_nonoffensive"] = probs_cal[:, 0]
                out_df.to_csv(os.path.join(OUT, f"{model_name}_{dataset_name}_icps.csv"),
                              index=False)

                metric_rows.append({
                    "Model"       : f"IDOL-F+{model_name}",
                    "Dataset"     : dataset_name,
                    **metrics,
                })
                print(f"    CSS={metrics['CSS']} DBI={metrics['DBI']} "
                      f"SS={metrics['Silhouette']} ECE={metrics['ECE']} "
                      f"URR={metrics['URR']} F1={metrics['F1']}")

            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback; traceback.print_exc()

    if metric_rows:
        df_m = pd.DataFrame(metric_rows)
        df_m.to_csv(os.path.join(OUT, "step9_ICPS_metrics.csv"), index=False)
        print("\n  ICPS Metrics:")
        print(df_m.to_string(index=False))

    print(f"\n  [DONE] Step-09 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
