# =============================================================================
# IDOL-F Framework — Step 02: Adaptive Corpus Equilibrium (ACE)
#
# PURPOSE:
#   Balance each dataset so offensive and non-offensive classes have
#   equal representation. This prevents majority-class bias during training.
#
# PROBLEM:
#   After Step-01, datasets have class imbalance:
#   - HASOC  : 2,261 off vs 3,591 non (39% offensive)
#   - HSLL   : 20,620 off vs 4,163 non (83% offensive)
#   - HTEval : 5,640 off vs 7,860 non (42% offensive)
#   If trained on imbalanced data, model predicts majority class always.
#
# ALGORITHM:
#   N_min = min(N_off, N_non)
#   Majority class is randomly undersampled to N_min.
#   Final balanced size = 2 × N_min
#
# FORMULA:
#   N_min = min(N_offensive, N_non-offensive)
#   balanced_dataset = sample(majority_class, N_min) + minority_class
#   |balanced_dataset| = 2 × N_min
#
# ABLATION:
#   ABLATION["ACE"] = False → Step-01 output copied forward unchanged
#   → Model trained on imbalanced data (shows ACE's contribution)
#
# TABLE GENERATED:
#   step2_ACE_summary.csv — per dataset before/after statistics
#
# OUTPUT:
#   output/step2/<DATASET>_balanced.csv
# =============================================================================

import os
import sys
import shutil

import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, ACE_RANDOM_STATE, make_all_dirs
)

make_all_dirs()
IN  = STEP_DIRS["step1"]
OUT = STEP_DIRS["step2"]


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM: Adaptive Corpus Equilibrium
#
# N_min = min(N_off, N_non)
# Step 1: Identify majority and minority class
# Step 2: Randomly undersample majority class to N_min
# Step 3: Combine both classes and shuffle
# Final: |balanced| = 2 × N_min
# ─────────────────────────────────────────────────────────────────────────────

def balance_dataset(df, dataset_name):
    """
    Apply ACE to one dataset.

    Formula:
        N_min = min(N_off, N_non)
        Final size = 2 × N_min

    Uses random_state=42 for reproducibility.
    """
    offensive     = df[df["label"] == 1].copy()
    non_offensive = df[df["label"] == 0].copy()

    n_off = len(offensive)
    n_non = len(non_offensive)

    # N_min = min(N_off, N_non)
    n_min = min(n_off, n_non)

    # Undersample majority class
    if n_off > n_min:
        offensive = offensive.sample(
            n=n_min, random_state=ACE_RANDOM_STATE)
    if n_non > n_min:
        non_offensive = non_offensive.sample(
            n=n_min, random_state=ACE_RANDOM_STATE)

    # Combine and shuffle
    balanced = pd.concat([offensive, non_offensive], ignore_index=True)
    balanced = balanced.sample(
        frac=1.0, random_state=ACE_RANDOM_STATE
    ).reset_index(drop=True)

    return balanced, n_min


def main():
    print("=" * 65)
    print("  IDOL-F | Step 02: Adaptive Corpus Equilibrium (ACE)")
    print("=" * 65)

    # ── ABLATION: if ACE disabled, copy Step-01 output forward ────
    if not ABLATION["ACE"]:
        print("\n  [ABLATION] ACE = False")
        print("  Copying Step-01 output forward without balancing...")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN,  f"{name}.csv")
            dst = os.path.join(OUT, f"{name}_balanced.csv")
            shutil.copy2(src, dst)
            df = pd.read_csv(dst)
            print(f"  {name}: {len(df):,} records (unbalanced)")
        print("\n  [DONE] Step-02 skipped (ablation mode)")
        return

    # ── ACE ENABLED ───────────────────────────────────────────────
    print("\n  Balancing datasets...")
    summary_rows = []

    for name in TRAIN_DATASETS:
        # Load Step-01 output
        df = pd.read_csv(os.path.join(IN, f"{name}.csv"))

        before_off = int((df["label"] == 1).sum())
        before_non = int((df["label"] == 0).sum())
        before_total = len(df)

        # Apply ACE
        balanced, n_min = balance_dataset(df, name)

        # Save balanced dataset (keep only text and label)
        balanced[["text", "label"]].to_csv(
            os.path.join(OUT, f"{name}_balanced.csv"), index=False
        )

        after_off  = int((balanced["label"] == 1).sum())
        after_non  = int((balanced["label"] == 0).sum())
        reduction  = round(100 * (1 - len(balanced) / before_total), 1)

        summary_rows.append({
            "Dataset"       : name,
            "Before_Total"  : before_total,
            "Before_Off"    : before_off,
            "Before_Non"    : before_non,
            "N_min"         : n_min,
            "After_Total"   : len(balanced),
            "After_Off"     : after_off,
            "After_Non"     : after_non,
            "Reduction_%"   : reduction,
        })

        print(f"\n  {name}:")
        print(f"    Before: {before_total:,} "
              f"(off={before_off:,}, non={before_non:,})")
        print(f"    N_min = min({before_off:,}, {before_non:,}) = {n_min:,}")
        print(f"    After : {len(balanced):,} "
              f"(off={after_off:,}, non={after_non:,}) "
              f"[{reduction}% reduction]")

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT, "step2_ACE_summary.csv"), index=False
    )

    print("\n  ACE Summary:")
    print(summary_df.to_string(index=False))

    print(f"\n  [DONE] Step-02 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
