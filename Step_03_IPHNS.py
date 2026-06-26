# =============================================================================
# IDOL-F Framework — Step 03: Intent-Preserve Hard Negative Synthesis (IPHNS)
#
# PURPOSE:
#   Generate synthetic hard negative sentences using GPT-4o Mini API.
#   Hard negatives LOOK offensive (contain aggressive vocabulary) but are
#   actually NON-OFFENSIVE because the target is a non-human entity.
#
# WHY NEEDED:
#   After ACE, datasets contain clear offensive and non-offensive sentences.
#   Missing: boundary cases where aggressive words are used harmlessly.
#   Example: "I will kill this exam" — "kill" present but target=exam, not human.
#   Without hard negatives, model classifies ANY sentence with "kill" as offensive.
#
# 5 CATEGORIES → 5 DATASETS:
#   Category 1 — Academic Frustration  → HASOC
#   Category 2 — Gaming and Sports     → HSLL
#   Category 3 — Professional/Workplace → HTEval
#   Category 4 — Daily Life Frustration → HTXplain
#   Category 5 — Negated Offensive Intent → OLID
#
# Each category generates IPHNS_PER_CATEGORY sentences (default=200).
# All generated sentences get label=0 (NON-OFFENSIVE).
#
# ABLATION:
#   ABLATION["IPHNS"] = False → Step-02 output copied forward unchanged
#
# OUTPUT:
#   output/step3/<DATASET>_balanced_IPHNS.csv
#   output/step3/step3_IPHNS_summary.csv
# =============================================================================

import os
import sys
import json
import time
import shutil

import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION,
    OPENAI_API_KEY, IPHNS_MODEL, IPHNS_PER_CATEGORY,
    IPHNS_BATCH_SIZE, IPHNS_MAX_RETRIES, ACE_RANDOM_STATE, make_all_dirs
)

make_all_dirs()
IN  = STEP_DIRS["step2"]
OUT = STEP_DIRS["step3"]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (shared across all 5 categories)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert linguistic annotator specializing in offensive language detection research. Your task is to generate adversarial hard negative sentences for training a robust offensive language detection framework.

A hard negative sentence is defined as:
- A sentence that APPEARS offensive on the surface due to the presence of aggressive or violent vocabulary
- But is actually NON-OFFENSIVE in meaning and intent
- Because the target of the aggressive word is a non-human entity

STRICT RULES:
1. Every sentence MUST contain at least one word from this list: kill, destroy, murder, hate, crush, attack, smash, eliminate, wipe out, annihilate, burn, ruin, obliterate, devastate, demolish, wreck, harm, hurt, damage, wound, butcher, slaughter
2. The target MUST be non-human only
3. Sentence must sound natural — like a real person frustrated or joking
4. Do NOT target any person, group, or living being
5. Every sentence must be unique in structure and vocabulary
6. Return ONLY a valid JSON array of strings. No markdown, no explanations.

Example output format:
["I want to destroy this spreadsheet.", "I could murder this assignment right now.", "I hate this traffic so much."]"""

# ─────────────────────────────────────────────────────────────────────────────
# 5 USER PROMPTS — one per category/dataset
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_CONFIGS = {

    # Category 1 → HASOC
    "HASOC": {
        "category"   : "Academic Frustration",
        "user_prompt": """Generate {n} hard negative sentences for offensive language detection research.

CONTEXT: Academic frustration ONLY.
All sentences must be about student frustration toward ACADEMIC TASKS, not people.
Targets must be: exam, test, assignment, deadline, thesis, dissertation, project, presentation, subject, chapter, course, grade, paper, quiz, homework, module, semester, syllabus, report, textbook, lecture, professor's assignment (not the professor themselves)

RULES:
- Every sentence MUST contain aggressive vocabulary from the approved list
- Target must always be an academic object or concept, NEVER a person
- No two sentences should have the same structure
- Vary length: some short (5-8 words), some long (12-20 words)
- Use different aggressive words — do not repeat the same word more than 4 times
- Sound like a real frustrated student's social media post

Return ONLY a JSON array of {n} strings. Nothing else.""",
    },

    # Category 2 → HSLL
    "HSLL": {
        "category"   : "Gaming and Sports",
        "user_prompt": """Generate {n} hard negative sentences for offensive language detection research.

CONTEXT: Gaming and Sports frustration ONLY.
Targets must be: video game, match, level, score, boss, opponent (in-game), tournament, record, character, mission, team (not individual players), game, round, season, leaderboard, challenge, achievement, quest, map, strategy

RULES:
- Every sentence MUST contain aggressive vocabulary from the approved list
- Target must always be a gaming/sports object or concept, NEVER a real person
- Sentences must sound like frustrated gamer or sports fan social media posts
- Vary length and structure
- Use different aggressive words

Return ONLY a JSON array of {n} strings. Nothing else.""",
    },

    # Category 3 → HTEval
    "HTEval": {
        "category"   : "Professional Workplace",
        "user_prompt": """Generate {n} hard negative sentences for offensive language detection research.

CONTEXT: Professional/Workplace frustration ONLY.
Targets must be: meeting, deadline, code, bug, presentation, client project, task, target, report, spreadsheet, database, system, server, process, workflow, ticket, software, application, document, email chain

RULES:
- Every sentence MUST contain aggressive vocabulary from the approved list
- Target must always be a work object or concept, NEVER a person or colleague
- Sentences must sound like frustrated professional's social media posts
- Vary length and sentence patterns
- Avoid repeating same aggressive word more than 3 times

Return ONLY a JSON array of {n} strings. Nothing else.""",
    },

    # Category 4 → HTXplain
    "HTXplain": {
        "category"   : "Daily Life Frustration",
        "user_prompt": """Generate {n} hard negative sentences for offensive language detection research.

CONTEXT: Daily Life frustration ONLY.
Targets must be: traffic, weather, technology, appliance, queue, price, internet connection, food, commute, parking, bill, phone, laptop, TV, electricity, Wi-Fi, weather forecast, flight, train, taxi, delivery

RULES:
- Every sentence MUST contain aggressive vocabulary from the approved list
- Target must always be a daily life object or situation, NEVER a person
- Sentences must sound like ordinary person's frustrated social media posts
- Mix short and long sentences
- Avoid repeating aggressive words

Return ONLY a JSON array of {n} strings. Nothing else.""",
    },

    # Category 5 → OLID
    "OLID": {
        "category"   : "Negated Offensive Intent",
        "user_prompt": """Generate {n} hard negative sentences for offensive language detection research.

CONTEXT: Negated offensive intent ONLY.
These sentences contain aggressive words but the intent is clearly negated or hypothetical.
Patterns to use:
- "I would never..." + aggressive word
- "I cannot bring myself to..." + aggressive word
- "Even though I feel like... I would not"
- "I would love to... but I won't"
- "I thought about... but decided against it"
- "I wish I could... but I can't"

RULES:
- Every sentence MUST contain aggressive vocabulary from the approved list
- The negative/contrary intent must be crystal clear
- No ambiguity — reader must immediately understand the intent is NOT harmful
- Vary the negation patterns — do not repeat same pattern more than 3 times
- Sentences should sound natural

Return ONLY a JSON array of {n} strings. Nothing else.""",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GPT API CALL WITH RETRY
# ─────────────────────────────────────────────────────────────────────────────

def call_gpt_api(client, dataset_name, n_sentences):
    """
    Call GPT-4o Mini to generate hard negative sentences for one category.
    Sends in batches of IPHNS_BATCH_SIZE with IPHNS_MAX_RETRIES retries.
    """
    cfg         = CATEGORY_CONFIGS[dataset_name]
    all_sents   = []
    batch_size  = min(IPHNS_BATCH_SIZE, n_sentences)

    while len(all_sents) < n_sentences:
        need  = min(batch_size, n_sentences - len(all_sents))
        prompt = cfg["user_prompt"].format(n=need)

        for attempt in range(IPHNS_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=IPHNS_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=1.0,
                    max_tokens=4096,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences if present
                raw = raw.replace("```json", "").replace("```", "").strip()

                # Parse JSON
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    valid = [s for s in parsed if isinstance(s, str) and len(s) > 5]
                    all_sents.extend(valid)
                    print(f"    Batch OK: +{len(valid)} sentences "
                          f"(total {len(all_sents)}/{n_sentences})")
                    break
                else:
                    raise ValueError("Response is not a JSON list")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"    Attempt {attempt+1}/{IPHNS_MAX_RETRIES} failed: {e}")
                if attempt < IPHNS_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
            except Exception as e:
                print(f"    API error: {e}")
                if attempt < IPHNS_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        else:
            print(f"    [WARNING] Batch failed after {IPHNS_MAX_RETRIES} attempts")
            break

    return all_sents[:n_sentences]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  IDOL-F | Step 03: Intent-Preserve Hard Negative Synthesis")
    print("=" * 65)

    # ── ABLATION: if IPHNS disabled, copy Step-02 output forward ──
    if not ABLATION["IPHNS"]:
        print("\n  [ABLATION] IPHNS = False")
        print("  Copying Step-02 output forward without hard negatives...")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN,  f"{name}_balanced.csv")
            dst = os.path.join(OUT, f"{name}_balanced_IPHNS.csv")
            shutil.copy2(src, dst)
            df = pd.read_csv(dst)
            print(f"  {name}: {len(df):,} records (no hard negatives added)")
        print("\n  [DONE] Step-03 skipped (ablation mode)")
        return

    # ── Check API key ─────────────────────────────────────────────
    if "PASTE" in OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
        print("\n  [ERROR] OpenAI API key not set!")
        print("  Update OPENAI_API_KEY in Step_00_Config.py")
        print("  Falling back to pass-through mode...")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN,  f"{name}_balanced.csv")
            dst = os.path.join(OUT, f"{name}_balanced_IPHNS.csv")
            shutil.copy2(src, dst)
        return

    # ── Initialize OpenAI client ──────────────────────────────────
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("  [ERROR] openai package not installed. Run: pip install openai")
        return

    # ── Generate hard negatives per dataset ───────────────────────
    print(f"\n  Generating {IPHNS_PER_CATEGORY} hard negatives per dataset...")
    print(f"  Model: {IPHNS_MODEL}\n")

    summary_rows = []

    for name in TRAIN_DATASETS:
        cfg = CATEGORY_CONFIGS[name]
        print(f"\n  → {name} | Category: {cfg['category']}")

        # Load Step-02 balanced dataset
        df = pd.read_csv(os.path.join(IN, f"{name}_balanced.csv"))
        before = len(df)

        # Generate hard negatives
        hard_negs = call_gpt_api(client, name, IPHNS_PER_CATEGORY)

        if hard_negs:
            # All hard negatives are NON-OFFENSIVE → label = 0
            hn_df = pd.DataFrame({
                "text" : hard_negs,
                "label": [0] * len(hard_negs),
            })

            # Concatenate with balanced dataset and shuffle
            combined = pd.concat(
                [df[["text", "label"]], hn_df],
                ignore_index=True
            )
            combined = combined.sample(
                frac=1.0, random_state=ACE_RANDOM_STATE
            ).reset_index(drop=True)
        else:
            print(f"    [WARNING] No hard negatives generated — using original")
            combined = df[["text", "label"]].copy()

        # Save combined dataset
        combined.to_csv(
            os.path.join(OUT, f"{name}_balanced_IPHNS.csv"), index=False
        )

        added = len(combined) - before
        summary_rows.append({
            "Dataset"         : name,
            "Category"        : cfg["category"],
            "Original"        : before,
            "HardNeg_Added"   : added,
            "Final_Total"     : len(combined),
            "Label_0_pct"     : round(100 * (combined["label"]==0).mean(), 1),
            "Label_1_pct"     : round(100 * (combined["label"]==1).mean(), 1),
        })
        print(f"    Original: {before:,} | +Hard negs: {added} | "
              f"Final: {len(combined):,}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT, "step3_IPHNS_summary.csv"), index=False
    )

    print("\n  IPHNS Summary:")
    print(summary_df.to_string(index=False))

    print(f"\n  [DONE] Step-03 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
