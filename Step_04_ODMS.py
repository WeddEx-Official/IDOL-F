# =============================================================================
# IDOL-F Framework — Step 04: Obfuscation Detection and Mitigation System
#
# THREE SUB-MODULES (sequential):
#
# 4.1 — Lexical PreNorm Surface
#   13 sequential cleaning steps that normalize raw social media text
#   WITHOUT recovering obfuscations (that is ObfReX's job).
#
# 4.2 — ObfDetX (Obfuscation Detection)
#   Three detection rules:
#   Rule 1 — Special character substitution inside words (h@te, sh!t, $hit)
#   Rule 2 — Number-as-letter leet-speak (k1ll, d3stroy, h4te)
#   Rule 3 — Masking/elongation (f*ck, f***, fuuuck, killl)
#
# 4.3 — ObfReX (Obfuscation Recovery)
#   Recovers detected obfuscated tokens to their original readable form
#   using leet map + closest-match against the offensive lexicon.
#   Verifies that recovery preserves word-level meaning.
#
# METRICS (computed on audit sample with synthetic injection):
#   ODP — Obfuscation Detection Precision  = TP / (TP + FP)
#   ODR — Obfuscation Detection Recall     = TP / (TP + FN)
#   PCR — Pattern Coverage Rate            = pattern_types_detected / 3
#   RCR — Recovery Completion Rate         = tokens_recovered / tokens_detected
#   TRA — Token Recovery Accuracy          = correct_recoveries / recovered
#   SSR — Semantic Shift Rate              = shifted / recovered_sentences
#
# TABLES GENERATED:
#   Table 2: Lexical PreNorm (Records, Changes, Change %, Dropped)
#   Table 3: ObfDetX — ODP, ODR, PCR
#   Table 4: ObfReX  — RCR, TRA, SSR
#
# ABLATION:
#   ABLATION["ODMS"] = False → input forwarded with text_recovered = text
#
# OUTPUT:
#   output/step4/prenorm/<DATASET>_prenorm.csv
#   output/step4/obfdetx/<DATASET>_obfdetx.csv
#   output/step4/obfrex/<DATASET>_obfrex.csv   (has text_recovered column)
#   output/step4/table2_prenorm.csv
#   output/step4/table3_obfdetx.csv
#   output/step4/table4_obfrex.csv
# =============================================================================

import os
import re
import sys
import html
import shutil
import random
from collections import Counter

import pandas as pd
import numpy as np

# ── Path resolution ───────────────────────────────────────────────────────────
_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, LEET_MAP, INWORD_SYMBOLS,
    ACE_RANDOM_STATE, make_all_dirs
)
from Step_01_MCA import load_lexicon

make_all_dirs()
random.seed(ACE_RANDOM_STATE)

IN       = STEP_DIRS["step3"]
OUT_MAIN = STEP_DIRS["step4"]
OUT_PN   = STEP_DIRS["step4_prenorm"]
OUT_DET  = STEP_DIRS["step4_obfdetx"]
OUT_REC  = STEP_DIRS["step4_obfrex"]


# =============================================================================
# 4.1 — LEXICAL PRENORM SURFACE (13 sequential steps)
# =============================================================================

def step1_lowercase(t):
    """Step 1: Convert all text to lowercase."""
    return t.lower()

def step2_remove_urls(t):
    """Step 2: Remove URLs (https/http, www, bare link/url/pic tokens)."""
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"www\.\S+", " ", t)
    t = re.sub(r"\b(?:url|link|pic\.twitter\S*)\b", " ", t)
    return t

def step3_remove_mentions(t):
    """Step 3: Remove @mentions (leading @, preserves in-word @ like h@te)."""
    return re.sub(r"(?<!\w)@\w+", " ", t)

def step4_remove_hashtags(t):
    """Step 4: Remove #hashtags."""
    return re.sub(r"(?<!\w)#\w+", " ", t)

def step5_decode_html_entities(t):
    """Step 5: Decode HTML entities (&amp; → and, others removed)."""
    t = t.replace("&amp;", " and ")
    t = html.unescape(t)
    t = re.sub(r"&\w+;", " ", t)
    return t

def step6_remove_rt_token(t):
    """Step 6: Remove RT (retweet markers)."""
    return re.sub(r"\brt\b", " ", t)

def step7_remove_non_ascii(t):
    """Step 7: Remove emojis and non-ASCII characters."""
    return t.encode("ascii", errors="ignore").decode()

def step8_remove_action_emotes(t):
    """
    Step 8: Remove *action* emotes at token level.
    ONLY removes tokens that are purely alphabetic inside asterisks
    (e.g. *sighs*, *laughs*).
    Preserves f***ing, n***er, f*ck (genuine obfuscation).
    """
    tokens = t.split()
    filtered = [
        tok for tok in tokens
        if not re.fullmatch(r"\*[a-zA-Z]{2,20}\*", tok)
    ]
    return " ".join(filtered)

def step9_remove_standalone_numbers(t):
    """
    Step 9: Remove pure numeric tokens.
    PRESERVES leet tokens like k1ll, d3stroy, h4te (have letters too).
    """
    tokens = t.split()
    filtered = [tok for tok in tokens if not tok.isdigit()]
    return " ".join(filtered)

def step10_remove_punctuation_only_tokens(t):
    """Step 10: Remove tokens with no alphanumeric characters."""
    tokens = t.split()
    filtered = [tok for tok in tokens if re.search(r"[a-zA-Z0-9]", tok)]
    return " ".join(filtered)

def step11_strip_trailing_punctuation(t):
    """
    Step 11: Strip trailing punctuation from word ends.
    PRESERVES internal special chars (sh!t, h@te, f*ck).
    """
    def strip_tok(tok):
        last = len(tok)
        while last > 0 and tok[last-1] in ".,;:?!\"'`)(":
            # Check if this char is internal (not last occurrence)
            # Only strip truly trailing chars
            if any(c.isalnum() or c in "@$!*#%+" for c in tok[:last-1]):
                last -= 1
            else:
                break
        return tok[:last] if last > 0 else tok

    return " ".join(strip_tok(tok) for tok in t.split())

def step12_collapse_repeated_punctuation(t):
    """Step 12: Collapse repeated punctuation (!!!!! → !)."""
    return re.sub(r"([!?.,])\1{1,}", r"\1", t)

def step13_cleanup_whitespace(t):
    """Step 13: Remove extra whitespace."""
    return re.sub(r"\s+", " ", t).strip()


def lexical_prenorm(text):
    """
    Apply all 13 Lexical PreNorm Surface steps in sequence.
    Returns cleaned text (obfuscated words preserved for ObfDetX).
    """
    t = str(text)
    t = step1_lowercase(t)
    t = step2_remove_urls(t)
    t = step3_remove_mentions(t)
    t = step4_remove_hashtags(t)
    t = step5_decode_html_entities(t)
    t = step6_remove_rt_token(t)
    t = step7_remove_non_ascii(t)
    t = step8_remove_action_emotes(t)
    t = step9_remove_standalone_numbers(t)
    t = step10_remove_punctuation_only_tokens(t)
    t = step11_strip_trailing_punctuation(t)
    t = step12_collapse_repeated_punctuation(t)
    t = step13_cleanup_whitespace(t)
    return t


# =============================================================================
# 4.2 — ObfDetX: THREE DETECTION RULES
# =============================================================================

def decode_leet(token):
    """Map leet characters to letters using LEET_MAP from config."""
    return "".join(LEET_MAP.get(c, c) for c in token.lower())


def detect_obfuscation(token, lexicon):
    """
    Apply three ObfDetX detection rules to a single token.

    Rule 1 — Special character substitution inside word:
        h@te, sh!t, $hit, f*ck — special char used as letter replacement
    Rule 2 — Number-as-letter leet substitution:
        k1ll, d3stroy, h4te — digits substituted for similar-looking letters
    Rule 3 — Asterisk masking or character elongation:
        f***, n***er — internal asterisk masking
        fuuuck, killl — character repeated 3+ times

    Returns: (is_obfuscated: bool, rule: str, decoded_guess: str)
    """
    tok = token.lower()
    if len(tok) < 2:
        return False, None, None

    # ── Rule 3a — Asterisk masking: f*ck, f***, n***er ──────────
    if "*" in tok:
        n_letters = sum(c.isalpha() for c in tok)
        if n_letters >= 1:
            decoded = decode_leet(tok)
            return True, "masking", decoded

    # ── Rule 1 — Special character substitution inside word ──────
    # Check for in-word special characters (not at word start as punctuation)
    inner = tok[1:-1] if len(tok) > 2 else ""
    has_inner_symbol = any(c in INWORD_SYMBOLS for c in inner)
    starts_with_dollar_at = tok and tok[0] in "@$" and sum(c.isalpha() for c in tok) >= 2

    if has_inner_symbol or starts_with_dollar_at:
        decoded = decode_leet(tok)
        if decoded != tok:
            return True, "special_char", decoded

    # ── Rule 2 — Leet digit substitution ─────────────────────────
    # Pattern: letters + digits + letters (sandwiched)
    if re.search(r"[a-z][013457890][a-z]", tok):
        decoded = decode_leet(tok)
        if decoded.isalpha() and decoded != tok:
            return True, "leet", decoded
    # Also catch: starts with letters, has digit, ends with letters
    if re.match(r"^[a-z]+[013457890][a-z]*$", tok) and not tok.isalpha():
        decoded = decode_leet(tok)
        if decoded.isalpha() and decoded != tok:
            return True, "leet", decoded

    # ── Rule 3b — Character elongation: fuuuck, killl ────────────
    if tok.isalpha() and re.search(r"(.)\1{2,}", tok):
        # Try collapsing to 1 or 2 repetitions
        collapsed1 = re.sub(r"(.)\1{2,}", r"\1", tok)
        collapsed2 = re.sub(r"(.)\1{1,}", r"\1", tok)
        for candidate in [collapsed1, collapsed2]:
            if candidate in lexicon:
                return True, "elongation", candidate
        # Return collapsed version even if not in lexicon
        if len(collapsed1) > 1:
            return True, "elongation", collapsed1

    return False, None, None


# =============================================================================
# 4.3 — ObfReX: RECOVERY
# =============================================================================

def closest_lexicon_match(decoded_token, lexicon):
    """
    For masked tokens like 'f*ck' (decoded as 'f*ck' with unresolved *):
    Find the lexicon word matching visible letters + same length.
    """
    if not decoded_token:
        return None

    # Build regex: replace * with [a-z]
    pattern = "^" + "".join(
        re.escape(c) if c != "*" else "[a-z]" for c in decoded_token
    ) + "$"
    rex = re.compile(pattern)

    for word in lexicon:
        if len(word) == len(decoded_token) and rex.match(word):
            return word

    # Fallback: same first letter, similar length (±1)
    if decoded_token:
        first = decoded_token.replace("*", "")[0] if decoded_token.replace("*", "") else ""
        for word in lexicon:
            if (first and word.startswith(first)
                    and abs(len(word) - len(decoded_token)) <= 1):
                return word
    return None


def recover_token(token, lexicon):
    """
    Recover one obfuscated token to its original form.
    Returns (recovered_token, was_recovered, rule_applied).
    """
    is_obf, rule, decoded = detect_obfuscation(token, lexicon)

    if not is_obf:
        return token, False, None

    if decoded is None:
        return token, False, None

    # Handle remaining asterisks (masking case)
    if "*" in decoded:
        match = closest_lexicon_match(decoded, lexicon)
        if match:
            return match, True, rule
        # Strip asterisks as last resort
        stripped = decoded.replace("*", "")
        if stripped.isalpha() and len(stripped) >= 2:
            return stripped, True, rule
        return token, False, None

    # For clean decoded string
    if decoded.isalpha() and len(decoded) >= 2:
        return decoded, True, rule

    return token, False, None


def recover_sentence(text, lexicon):
    """
    Recover ALL obfuscated tokens in a sentence.
    Returns (recovered_text, n_detected, n_recovered, recovery_log).
    """
    tokens         = str(text).split()
    recovered_toks = []
    n_detected     = 0
    n_recovered    = 0
    log_parts      = []

    for tok in tokens:
        is_obf, rule, _ = detect_obfuscation(tok, lexicon)
        if is_obf:
            n_detected += 1
            rec, ok, rule_used = recover_token(tok, lexicon)
            recovered_toks.append(rec)
            if ok and rec != tok:
                n_recovered += 1
                log_parts.append(f"{tok}→{rec}[{rule_used}]")
            else:
                log_parts.append(tok)
        else:
            recovered_toks.append(tok)

    return (
        " ".join(recovered_toks),
        n_detected,
        n_recovered,
        "; ".join(log_parts),
    )


# =============================================================================
# AUDIT METRICS — Synthetic injection for authentic ODP/ODR/TRA
# =============================================================================

def inject_obfuscation(word):
    """Create a known obfuscated variant of a clean word for audit."""
    strategies = ["leet", "symbol", "mask", "elongate"]
    strategy   = random.choice(strategies)

    if strategy == "leet":
        for char, leet in [("i", "1"), ("e", "3"), ("a", "4"),
                            ("o", "0"), ("s", "5")]:
            if char in word:
                return word.replace(char, leet, 1)

    if strategy == "symbol":
        for char, sym in [("a", "@"), ("s", "$"), ("i", "!")]:
            if char in word:
                return word.replace(char, sym, 1)

    if strategy == "mask" and len(word) > 3:
        return word[0] + "*" * (len(word) - 2) + word[-1]

    if strategy == "elongate":
        mid = len(word) // 2
        return word[:mid] + word[mid] * 3 + word[mid+1:]

    return word  # no modification possible


def compute_audit_metrics(df, lexicon, n_audit=500):
    """
    Compute ODP, ODR, PCR, RCR, TRA, SSR using synthetic injection.

    For each sampled sentence containing a lexicon word:
    1. Inject known obfuscation
    2. Run ObfDetX detection
    3. Run ObfReX recovery
    4. Compare with known original
    """
    tp_det = fp_det = fn_det = 0
    tp_rec = total_rec = 0
    patterns_found = set()
    shifted_sentences = 0
    recovered_sentences = 0

    # Collect candidate sentences (those with lexicon words)
    candidates = []
    for _, row in df.iterrows():
        words = str(row.get("text", "")).split()
        lexicon_words = [w.lower() for w in words if w.lower() in lexicon]
        if lexicon_words:
            candidates.append((str(row.get("text", "")), lexicon_words[0]))
        if len(candidates) >= n_audit:
            break

    for orig_text, clean_word in candidates:
        obf_word = inject_obfuscation(clean_word)
        if obf_word == clean_word:
            continue

        # Test detection
        is_obf, rule, _ = detect_obfuscation(obf_word, lexicon)
        if is_obf:
            tp_det += 1
            patterns_found.add(rule or "unknown")
            # Test recovery
            recovered, ok, _ = recover_token(obf_word, lexicon)
            total_rec += 1
            if ok and recovered == clean_word:
                tp_rec += 1
            # SSR: check if sentence meaning shifts
            if ok and recovered != clean_word:
                shifted_sentences += 1
            recovered_sentences += 1
        else:
            fn_det += 1

    # False positives: clean tokens incorrectly flagged
    clean_tokens = []
    for text in df["text"].head(200):
        clean_tokens.extend(
            w for w in str(text).split() if w.isalpha() and w not in lexicon
        )
    for tok in clean_tokens[:500]:
        is_obf, _, _ = detect_obfuscation(tok, lexicon)
        if is_obf:
            fp_det += 1

    # Compute metrics
    odp = tp_det / (tp_det + fp_det + 1e-9)
    odr = tp_det / (tp_det + fn_det + 1e-9)
    pcr = len(patterns_found) / 3.0        # 3 detection rule types
    rcr = total_rec / (tp_det + 1e-9)
    tra = tp_rec / (total_rec + 1e-9)
    ssr = shifted_sentences / (recovered_sentences + 1e-9)

    return {
        "ODP"             : round(odp, 3),
        "ODR"             : round(odr, 3),
        "PCR"             : round(pcr, 3),
        "RCR"             : round(rcr, 3),
        "TRA"             : round(tra, 3),
        "SSR"             : round(ssr, 3),
        "TP_detected"     : tp_det,
        "FP_detected"     : fp_det,
        "FN_detected"     : fn_det,
        "Patterns_found"  : sorted(patterns_found),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  IDOL-F | Step 04: Obfuscation Detection & Mitigation System")
    print("=" * 65)

    # Load offensive lexicon (extracted in Step-01)
    lexicon_dict = load_lexicon()
    lexicon      = lexicon_dict["offensive_verbs"]

    if not lexicon:
        print("  [ERROR] Offensive lexicon is empty!")
        print("  Please run Step-01 first to extract the lexicon.")
        return

    print(f"\n  Loaded offensive lexicon: {len(lexicon):,} verbs")

    # ── ABLATION: if ODMS disabled ────────────────────────────────
    if not ABLATION["ODMS"]:
        print("\n  [ABLATION] ODMS = False")
        print("  Copying Step-03 output forward (text_recovered = text)...")
        for name in TRAIN_DATASETS:
            src = os.path.join(IN, f"{name}_balanced_IPHNS.csv")
            df  = pd.read_csv(src)
            df["text_original"] = df["text"]
            df["text_recovered"] = df["text"]
            df["obf_detected"] = 0
            df["recovery_log"] = ""
            dst = os.path.join(OUT_REC, f"{name}_obfrex.csv")
            df.to_csv(dst, index=False)
        print("\n  [DONE] Step-04 skipped (ablation mode)")
        return

    # ── RUN ALL THREE SUB-MODULES ─────────────────────────────────
    t2_rows  = []   # Table 2: PreNorm
    t3_rows  = []   # Table 3: ObfDetX
    t4_rows  = []   # Table 4: ObfReX

    for name in TRAIN_DATASETS:
        print(f"\n  → {name}")
        df = pd.read_csv(os.path.join(IN, f"{name}_balanced_IPHNS.csv"))

        # ── 4.1 LexPreNorm ───────────────────────────────────────
        df["text_original"] = df["text"].copy()
        df["text"]          = df["text"].apply(lexical_prenorm)

        changed = int((df["text"] != df["text_original"].str.lower()).sum())
        before  = len(df)
        df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
        dropped = before - len(df)
        change_pct = round(100 * changed / max(before, 1), 2)

        df.to_csv(os.path.join(OUT_PN, f"{name}_prenorm.csv"), index=False)
        t2_rows.append({
            "Dataset"  : name,
            "Records"  : len(df),
            "Changes"  : changed,
            "Change_%" : change_pct,
            "Dropped"  : dropped,
        })
        print(f"    4.1 PreNorm  : {changed:,} changed ({change_pct}%), "
              f"{dropped} dropped")

        # ── 4.2 + 4.3 ObfDetX + ObfReX on every row ─────────────
        all_recovered   = []
        all_detected    = []
        all_rec_logs    = []
        all_has_obf     = []
        total_detected  = 0
        total_recovered = 0

        for text in df["text"]:
            rec_text, n_det, n_rec, log = recover_sentence(text, lexicon)
            all_recovered.append(rec_text)
            all_detected.append(n_det)
            all_rec_logs.append(log)
            all_has_obf.append(int(n_det > 0))
            total_detected  += n_det
            total_recovered += n_rec

        df["text_recovered"]    = all_recovered
        df["n_obf_detected"]    = all_detected
        df["recovery_log"]      = all_rec_logs
        df["has_obfuscation"]   = all_has_obf

        df.to_csv(os.path.join(OUT_DET, f"{name}_obfdetx.csv"), index=False)
        df.to_csv(os.path.join(OUT_REC, f"{name}_obfrex.csv"),  index=False)

        print(f"    4.2 ObfDetX  : {total_detected:,} tokens detected")
        print(f"    4.3 ObfReX   : {total_recovered:,} tokens recovered")

        # ── Audit metrics ────────────────────────────────────────
        metrics = compute_audit_metrics(df, lexicon)

        t3_rows.append({
            "Dataset" : name,
            "ODP"     : metrics["ODP"],
            "ODR"     : metrics["ODR"],
            "PCR"     : metrics["PCR"],
        })
        t4_rows.append({
            "Dataset" : name,
            "RCR"     : metrics["RCR"],
            "TRA"     : metrics["TRA"],
            "SSR"     : metrics["SSR"],
        })
        print(f"    Metrics      : ODP={metrics['ODP']} ODR={metrics['ODR']} "
              f"PCR={metrics['PCR']} RCR={metrics['RCR']} "
              f"TRA={metrics['TRA']} SSR={metrics['SSR']}")

    # ── Save tables ───────────────────────────────────────────────
    t2_df = pd.DataFrame(t2_rows)
    t3_df = pd.DataFrame(t3_rows)
    t4_df = pd.DataFrame(t4_rows)

    t2_df.to_csv(os.path.join(OUT_MAIN, "table2_prenorm.csv"),  index=False)
    t3_df.to_csv(os.path.join(OUT_MAIN, "table3_obfdetx.csv"),  index=False)
    t4_df.to_csv(os.path.join(OUT_MAIN, "table4_obfrex.csv"),   index=False)

    print("\n\n  TABLE 2 — Lexical PreNorm:")
    print(t2_df.to_string(index=False))
    print("\n  TABLE 3 — ObfDetX Metrics (ODP, ODR, PCR):")
    print(t3_df.to_string(index=False))
    print("\n  TABLE 4 — ObfReX Metrics (RCR, TRA, SSR):")
    print(t4_df.to_string(index=False))

    print(f"\n  [DONE] Step-04 complete. Output: {OUT_MAIN}")
    print("=" * 65)


if __name__ == "__main__":
    main()
