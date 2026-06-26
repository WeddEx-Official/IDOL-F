# =============================================================================
# IDOL-F Framework — run_all.py
# Full pipeline runner: Step 00 → Step 12
#
# Usage:
#   python run_all.py                   # full pipeline
#   python run_all.py --from 7          # start from step 7
#   python run_all.py --only 1          # run only step 1
#   python run_all.py --ablation SICL   # disable SICL, run all, compare
#
# HOW ABLATION WORKS:
#   --ablation COMPONENT sets that component to False before running.
#   Step-12 saves results under "w/o_COMPONENT" config name.
# =============================================================================

import sys
import os
import argparse
import importlib
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import Step_00_Config as cfg

STEPS = [
    (1,  "Step_01_MCA"),
    (2,  "Step_02_ACE"),
    (3,  "Step_03_IPHNS"),
    (4,  "Step_04_ODMS"),
    (5,  "Step_05_SPA"),
    (6,  "Step_06_SAGP"),
    (7,  "Step_07_SICL"),
    (8,  "Step_08_RASGC_ARGP"),
    (9,  "Step_09_ICPS"),
    (10, "Step_10_COIR"),
    (11, "Step_11_CUBD"),
    (12, "Step_12_Final_Classification"),
]


def run_step(module_name, step_num):
    """Import and run one step's main() function."""
    print(f"\n{'='*65}")
    print(f"  RUNNING STEP {step_num:02d}: {module_name}")
    print(f"{'='*65}")
    t0  = time.time()
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  Step {step_num:02d} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="IDOL-F Pipeline Runner")
    parser.add_argument("--from",   type=int, default=1,
                        dest="from_step", help="Start from step N")
    parser.add_argument("--to",     type=int, default=12,
                        dest="to_step",   help="Run up to step N")
    parser.add_argument("--only",   type=int, default=None,
                        help="Run only step N")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Disable component (e.g. SICL, ACE, CUBD...)")
    args = parser.parse_args()

    # ── Handle ablation ────────────────────────────────────────────
    if args.ablation:
        comp = args.ablation.upper()
        if comp in cfg.ABLATION:
            cfg.ABLATION[comp] = False
            print(f"\n  [ABLATION MODE] {comp} = False")
        else:
            valid = list(cfg.ABLATION.keys())
            print(f"  [ERROR] Unknown component '{comp}'. Valid: {valid}")
            sys.exit(1)

    config_name = cfg.get_ablation_config_name()
    print(f"\n{'='*65}")
    print(f"  IDOL-F FRAMEWORK — FULL PIPELINE")
    print(f"  Configuration : {config_name}")
    print(f"  Ablation state: {cfg.ABLATION}")
    print(f"{'='*65}")

    # ── Create output dirs ─────────────────────────────────────────
    cfg.make_all_dirs()

    # ── Select steps to run ────────────────────────────────────────
    if args.only is not None:
        selected = [(n, m) for n, m in STEPS if n == args.only]
    else:
        selected = [(n, m) for n, m in STEPS
                    if args.from_step <= n <= args.to_step]

    if not selected:
        print("  [ERROR] No steps selected.")
        sys.exit(1)

    print(f"\n  Steps to run: {[n for n,_ in selected]}\n")

    # ── Run selected steps ─────────────────────────────────────────
    t_total = time.time()
    errors  = []

    for step_num, module_name in selected:
        try:
            run_step(module_name, step_num)
        except Exception as e:
            print(f"\n  [ERROR] Step {step_num} failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append((step_num, str(e)))
            answer = input(f"\n  Continue despite error in step {step_num}? [y/N]: ")
            if answer.lower() != "y":
                break

    total_time = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Config     : {config_name}")
    print(f"  Total time : {total_time/60:.1f} minutes")
    if errors:
        print(f"  Errors     : {len(errors)} step(s) failed")
        for sn, msg in errors:
            print(f"    Step {sn}: {msg}")
    else:
        print(f"  Status     : All steps completed successfully")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
