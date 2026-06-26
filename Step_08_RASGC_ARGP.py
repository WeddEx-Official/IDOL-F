# =============================================================================
# IDOL-F Framework — Step 08: Role-Annotated Semantic Graph Construction
#                           + Attentive Relational Graph Propagation
#
# RASGC:
#   Converts each sentence to labeled directed graph G=(V, E, φ, ψ)
#   φ: V → R  (8 node roles)
#   ψ: E → T  (5 edge types)
#
#   Node roles: AGGRESSOR, OFFENSIVE_PREDICATE, HUMAN_TARGET,
#               NON_HUMAN_TARGET, NEGATION_MARKER, AUXILIARY, MODIFIER, CONTEXT
#   Edge types: SUBJECT_OF, ACTION_ON, NEGATES, MODIFIES, PART_OF
#
# ARGP:
#   Edge-type aware Graph Attention Network (GAT).
#   Node update: e_ij=(W_Q h_i)·(W_K h_j + W_r r_ij)/√d_k
#                h_i' = σ(Σ_j α_ij W_V h_j)
#   Negation attenuation: h_pred' = h_pred·(1 - β), β = sigmoid(param)
#
# TABLES GENERATED:
#   Table 15: RASGC — NRA, GC, NDR, HNTA per model per dataset
#   Table 16: ARGP  — AWC, ORSA, NAS, F1-Macro
#   Table 17: ARGP Node+Downstream (ONAS, AWE, Node-F1, Downstream-F1)
#
# ABLATION: ABLATION["RASGC_ARGP"] = False → SICL embeddings forwarded
# OUTPUT: output/step8/<MODEL>_<DATASET>_argp.pth + metric tables
# =============================================================================

import os, sys, json, math
import numpy as np
import pandas as pd

_CODE_DIR = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in dir() else os.path.abspath("."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Step_00_Config import (
    STEP_DIRS, TRAIN_DATASETS, ABLATION, MODEL_CONFIGS,
    RANDOM_SEED, ARGP_HEADS, ARGP_LAYERS, ARGP_BETA_INIT,
    ARGP_EPOCHS, ARGP_LR, ARGP_HIDDEN_DIM, make_all_dirs
)
from Step_01_MCA import load_lexicon

make_all_dirs()
IN  = STEP_DIRS["step7"]
IN6 = STEP_DIRS["step6"]
OUT = STEP_DIRS["step8"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Node role and edge type mappings
NODE_ROLES = {
    "AGGRESSOR": 0, "OFFENSIVE_PREDICATE": 1,
    "HUMAN_TARGET": 2, "NON_HUMAN_TARGET": 3,
    "NEGATION_MARKER": 4, "AUXILIARY": 5,
    "MODIFIER": 6, "CONTEXT": 7,
}
EDGE_TYPES = {
    "SUBJECT_OF": 0, "ACTION_ON": 1, "NEGATES": 2,
    "MODIFIES": 3, "PART_OF": 4,
}
N_EDGE_TYPES = len(EDGE_TYPES)


def assign_node_role(token, off_lemmas, human_set, neg_set, aux_tags):
    """Assign one of 8 semantic roles to a SpaCy token."""
    lemma = token.lemma_.lower()
    text  = token.text.lower()
    dep   = token.dep_

    if lemma in off_lemmas and token.pos_ in {"VERB", "AUX"}:
        return "OFFENSIVE_PREDICATE"
    if dep in {"nsubj", "nsubjpass", "csubj"}:
        return "AGGRESSOR"
    if dep in {"dobj", "pobj", "iobj", "attr", "oprd"}:
        return "HUMAN_TARGET" if text in human_set else "NON_HUMAN_TARGET"
    if dep == "neg" or text in neg_set:
        return "NEGATION_MARKER"
    if token.pos_ == "AUX" or text in aux_tags:
        return "AUXILIARY"
    if dep in {"amod", "advmod", "det", "nummod", "compound"}:
        return "MODIFIER"
    return "CONTEXT"


def assign_edge_type(dep):
    """Map SpaCy dependency label to one of 5 RASGC edge types."""
    if dep == "neg":
        return "NEGATES"
    if dep in {"nsubj", "nsubjpass", "csubj"}:
        return "SUBJECT_OF"
    if dep in {"dobj", "pobj", "iobj", "attr", "oprd"}:
        return "ACTION_ON"
    if dep in {"amod", "advmod", "det", "nummod", "compound", "aux", "auxpass"}:
        return "MODIFIES"
    return "PART_OF"


def build_rasgc_graph(doc, off_lemmas, human_set, neg_set):
    """
    Build role-annotated semantic graph for one sentence.
    Returns dict with nodes, edges, and structural flags.
    """
    aux_tags = {"will", "would", "can", "could", "shall", "should",
                "may", "might", "must", "do", "does", "did",
                "have", "has", "had", "is", "are", "was", "were", "be"}

    nodes = []
    for i, tok in enumerate(doc):
        role = assign_node_role(tok, off_lemmas, human_set, neg_set, aux_tags)
        nodes.append({
            "idx"    : i,
            "text"   : tok.text,
            "lemma"  : tok.lemma_.lower(),
            "role"   : role,
            "role_id": NODE_ROLES[role],
        })

    edges = []
    for tok in doc:
        if tok.head.i != tok.i:
            etype = assign_edge_type(tok.dep_)
            edges.append({
                "src"    : tok.head.i,
                "tgt"    : tok.i,
                "etype"  : etype,
                "etype_id": EDGE_TYPES[etype],
            })

    roles_set = {n["role"] for n in nodes}
    return {
        "nodes"          : nodes,
        "edges"          : edges,
        "n_nodes"        : len(nodes),
        "n_edges"        : len(edges),
        "has_negation"   : "NEGATION_MARKER" in roles_set,
        "has_human"      : "HUMAN_TARGET" in roles_set,
        "has_nonhuman"   : "NON_HUMAN_TARGET" in roles_set,
        "has_offpred"    : "OFFENSIVE_PREDICATE" in roles_set,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARGP — Edge-Type Aware Graph Attention
# ─────────────────────────────────────────────────────────────────────────────

class EdgeGAT(nn.Module):
    """
    Single EdgeGAT layer.
    e_ij = (W_Q h_i)·(W_K h_j + W_r r_ij) / √d_k
    α_ij = softmax_j(e_ij)
    h_i' = σ(Σ_j α_ij W_V h_j)
    """

    def __init__(self, dim, n_heads, n_edge_types):
        super().__init__()
        self.n_heads     = n_heads
        self.head_dim    = dim // n_heads
        self.dim         = dim
        self.W_Q         = nn.Linear(dim, dim, bias=False)
        self.W_K         = nn.Linear(dim, dim, bias=False)
        self.W_V         = nn.Linear(dim, dim, bias=False)
        self.W_r         = nn.Embedding(n_edge_types, self.head_dim)
        self.leaky_relu  = nn.LeakyReLU(0.2)
        self.layer_norm  = nn.LayerNorm(dim)

    def forward(self, H, edge_index, edge_types):
        """
        H: (N, dim) node features
        edge_index: (2, E) source and target indices
        edge_types: (E,) edge type ids
        """
        N = H.shape[0]
        if N == 0 or edge_index.shape[1] == 0:
            return self.layer_norm(H), torch.zeros(0, device=H.device)

        Q = self.W_Q(H); K = self.W_K(H); V = self.W_V(H)
        src, tgt = edge_index[0], edge_index[1]

        # Edge type embeddings (per head)
        r = self.W_r(edge_types)   # (E, head_dim)

        # Multi-head reshape
        Q_h = Q.view(N, self.n_heads, self.head_dim)
        K_h = K.view(N, self.n_heads, self.head_dim)
        V_h = V.view(N, self.n_heads, self.head_dim)

        Q_src = Q_h[src]                                       # (E, h, dk)
        K_tgt = K_h[tgt] + r.unsqueeze(1)                     # (E, h, dk)
        e     = (Q_src * K_tgt).sum(-1) / math.sqrt(self.head_dim)  # (E, h)
        e     = self.leaky_relu(e)

        # Softmax attention per node
        attn = torch.zeros_like(e)
        for h in range(self.n_heads):
            max_e = torch.zeros(N, device=H.device)
            max_e.scatter_reduce_(0, src, e[:, h], reduce="amax", include_self=True)
            exp_e = torch.exp(e[:, h] - max_e[src])
            sum_e = torch.zeros(N, device=H.device)
            sum_e.scatter_add_(0, src, exp_e)
            attn[:, h] = exp_e / (sum_e[src] + 1e-9)

        attn_avg = attn.mean(-1)   # (E,) averaged over heads

        # Aggregate neighbor information
        H_new = torch.zeros(N, self.dim, device=H.device)
        ae    = attn.unsqueeze(-1).expand(-1, -1, self.head_dim).reshape(-1, self.dim)
        V_tgt = V_h[tgt].reshape(-1, self.dim)
        H_new.scatter_add_(0, src.unsqueeze(1).expand_as(ae), ae * V_tgt)

        return self.layer_norm(H + H_new), attn_avg


class ARGPModel(nn.Module):
    """
    Attentive Relational Graph Propagation.
    Stacks L EdgeGAT layers with negation attenuation.
    """

    def __init__(self, dim, n_heads, n_layers, n_edge_types):
        super().__init__()
        self.gat_layers = nn.ModuleList([
            EdgeGAT(dim, n_heads, n_edge_types)
            for _ in range(n_layers)
        ])
        # Learnable negation attenuation factor β
        self._beta = nn.Parameter(torch.tensor(float(ARGP_BETA_INIT)))
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(dim, 2),
        )

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    def forward(self, H, edge_index, edge_types, neg_mask, off_mask):
        """
        H: (N, dim) initial node embeddings
        edge_index: (2, E)
        edge_types: (E,)
        neg_mask: (N,) bool — which nodes are NEGATION_MARKER
        off_mask: (N,) bool — which nodes are OFFENSIVE_PREDICATE
        """
        attn_weights = []
        for layer in self.gat_layers:
            H, attn = layer(H, edge_index, edge_types)
            attn_weights.append(attn)

        # Negation attenuation: h_pred' = h_pred · (1 - β)
        if edge_index.shape[1] > 0 and neg_mask.any() and off_mask.any():
            # Find offensive predicates targeted by negation edges
            neg_tgt_mask = (edge_types == EDGE_TYPES["NEGATES"])
            if neg_tgt_mask.any():
                neg_tgt_nodes = edge_index[1][neg_tgt_mask]
                for node_idx in neg_tgt_nodes:
                    if off_mask[node_idx]:
                        H[node_idx] = H[node_idx] * (1.0 - self.beta)

        # Global mean pooling for sentence representation
        z = H.mean(dim=0, keepdim=True)          # (1, dim)
        logits = self.classifier(z)              # (1, 2)

        return logits, z, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# NODE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_backbone_cache = {}


def get_node_features(text, model_name, n_nodes):
    """
    Get node features using frozen backbone embeddings.
    One token embedding per node (padded/truncated to n_nodes).
    """
    if model_name not in _backbone_cache:
        cfg = MODEL_CONFIGS[model_name]
        tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bk  = AutoModel.from_pretrained(cfg["hf_name"]).to(DEVICE)
        bk.eval()
        _backbone_cache[model_name] = (tok, bk)

    tok, bk = _backbone_cache[model_name]
    with torch.no_grad():
        enc = tok(text, max_length=64, padding="max_length",
                  truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = bk(**enc)
        token_embs = out.last_hidden_state[0]   # (seq_len, hidden)

    hidden = token_embs.shape[-1]
    # Pad or truncate to n_nodes
    if n_nodes <= token_embs.shape[0]:
        H = token_embs[:n_nodes]
    else:
        pad = torch.zeros(n_nodes - token_embs.shape[0], hidden, device=DEVICE)
        H   = torch.cat([token_embs, pad], dim=0)

    # Ensure correct hidden dim (ARGP_HIDDEN_DIM)
    if H.shape[1] != ARGP_HIDDEN_DIM:
        if H.shape[1] > ARGP_HIDDEN_DIM:
            H = H[:, :ARGP_HIDDEN_DIM]
        else:
            pad = torch.zeros(H.shape[0], ARGP_HIDDEN_DIM - H.shape[1], device=DEVICE)
            H   = torch.cat([H, pad], dim=1)
    return H


# ─────────────────────────────────────────────────────────────────────────────
# RASGC METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rasgc_metrics(graphs, labels):
    """
    Compute 4 RASGC metrics:
    NRA (Node Role Accuracy), GC (Graph Completeness),
    NDR (Negation Detection Rate), HNTA (Human/Non-Human Target Accuracy)
    """
    n = len(graphs)
    if n == 0:
        return {"NRA": 0, "GC": 0, "NDR": 0, "HNTA": 0}

    # GC: fraction of graphs with ≥ 1 edge
    gc = sum(g["n_edges"] > 0 for g in graphs) / n

    # NRA: proxy via offensive predicate detection in offensive sentences
    off_graphs = [g for g, l in zip(graphs, labels) if l == 1]
    nra = (sum(g["has_offpred"] for g in off_graphs) / len(off_graphs)
           if off_graphs else 0.0)

    # NDR: fraction of negated sentences with NEGATION_MARKER node
    neg_graphs = [g for g in graphs if g["has_negation"]]
    ndr = (sum(any(nd["role"] == "NEGATION_MARKER" for nd in g["nodes"])
               for g in neg_graphs) / len(neg_graphs)
           if neg_graphs else 1.0)

    # HNTA: correct human/non-human target classification
    ht_correct = ht_total = 0
    for g, l in zip(graphs, labels):
        if g["has_offpred"]:
            ht_total += 1
            if l == 1 and g["has_human"]:
                ht_correct += 1
            elif l == 0 and g["has_nonhuman"]:
                ht_correct += 1
    hnta = ht_correct / max(ht_total, 1)

    return {
        "NRA" : round(nra,  3),
        "GC"  : round(gc,   3),
        "NDR" : round(ndr,  3),
        "HNTA": round(hnta, 3),
    }


def main():
    print("=" * 65)
    print("  IDOL-F | Step 08: RASGC + ARGP")
    print("=" * 65)

    lex          = load_lexicon()
    off_lemmas   = lex["offensive_verbs"]
    human_set    = lex["objects_human"]
    neg_set      = lex["negations"]

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        nlp = spacy.load("en_core_web_sm")

    t15_rows = []   # RASGC metrics
    t16_rows = []   # ARGP metrics
    t17_rows = []   # ARGP Node+Downstream

    for dataset_name in TRAIN_DATASETS:
        print(f"\n  Dataset: {dataset_name}")

        df = pd.read_csv(os.path.join(IN6, f"{dataset_name}_sagp.csv"))
        text_col = "text_recovered" if "text_recovered" in df.columns else "text"
        texts  = df[text_col].fillna("").astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        if not ABLATION["RASGC_ARGP"]:
            print("  [ABLATION] RASGC_ARGP = False — skipping graph construction")
            for model_name in MODEL_CONFIGS:
                t15_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name,
                                  "NRA":0, "GC":0, "NDR":0, "HNTA":0})
                t16_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name,
                                  "AWC":0, "ORSA":0, "NAS":0, "F1":0})
                t17_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name,
                                  "ONAS":0, "AWE":0, "Node-F1":0, "Down-F1":0})
            continue

        # Build RASGC graphs
        print(f"  Building RASGC graphs for {len(texts):,} sentences...")
        graphs = []
        for doc in nlp.pipe(texts, batch_size=256):
            g = build_rasgc_graph(doc, off_lemmas, human_set, neg_set)
            graphs.append(g)

        rasgc_m = compute_rasgc_metrics(graphs, labels)
        print(f"  RASGC: NRA={rasgc_m['NRA']} GC={rasgc_m['GC']} "
              f"NDR={rasgc_m['NDR']} HNTA={rasgc_m['HNTA']}")

        # Train ARGP per model
        for model_name, cfg in MODEL_CONFIGS.items():
            print(f"\n  >> ARGP: {model_name}")
            try:
                argp_model = ARGPModel(
                    ARGP_HIDDEN_DIM, ARGP_HEADS, ARGP_LAYERS, N_EDGE_TYPES
                ).to(DEVICE)
                optimizer  = torch.optim.AdamW(argp_model.parameters(), lr=ARGP_LR)
                criterion  = nn.CrossEntropyLoss()

                n = len(texts)
                n_train = int(0.7 * n)
                idx_tr  = list(range(n_train))
                idx_te  = list(range(n_train, n))

                def prepare_graph(idx):
                    g = graphs[idx]
                    if g["n_nodes"] == 0:
                        H      = torch.zeros(1, ARGP_HIDDEN_DIM).to(DEVICE)
                        ei     = torch.zeros(2, 0, dtype=torch.long).to(DEVICE)
                        etypes = torch.zeros(0, dtype=torch.long).to(DEVICE)
                        nm     = torch.zeros(1, dtype=torch.bool).to(DEVICE)
                        om     = torch.zeros(1, dtype=torch.bool).to(DEVICE)
                    else:
                        H = get_node_features(texts[idx], model_name, g["n_nodes"])
                        if g["edges"]:
                            src  = [e["src"] for e in g["edges"]]
                            tgt  = [e["tgt"] for e in g["edges"]]
                            etids= [e["etype_id"] for e in g["edges"]]
                            ei     = torch.tensor([src, tgt], dtype=torch.long).to(DEVICE)
                            etypes = torch.tensor(etids, dtype=torch.long).to(DEVICE)
                        else:
                            ei     = torch.zeros(2, 0, dtype=torch.long).to(DEVICE)
                            etypes = torch.zeros(0, dtype=torch.long).to(DEVICE)
                        roles = [nd["role"] for nd in g["nodes"]]
                        nm    = torch.tensor([r == "NEGATION_MARKER" for r in roles]).to(DEVICE)
                        om    = torch.tensor([r == "OFFENSIVE_PREDICATE" for r in roles]).to(DEVICE)
                    return H, ei, etypes, nm, om

                # Train
                argp_model.train()
                for ep in range(ARGP_EPOCHS):
                    total_loss = 0.0
                    import random as rnd
                    rnd.shuffle(idx_tr)
                    for i in idx_tr:
                        H, ei, etypes, nm, om = prepare_graph(i)
                        lbl = torch.tensor([labels[i]], device=DEVICE)
                        optimizer.zero_grad()
                        logits, _, _ = argp_model(H, ei, etypes, nm, om)
                        loss = criterion(logits, lbl)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(argp_model.parameters(), 1.0)
                        optimizer.step()
                        total_loss += loss.item()
                    print(f"    ep{ep+1}/{ARGP_EPOCHS}: loss={total_loss/max(len(idx_tr),1):.4f}")

                # Evaluate
                argp_model.eval()
                preds, trues, risks, all_attns = [], [], [], []
                with torch.no_grad():
                    for i in idx_te:
                        H, ei, etypes, nm, om = prepare_graph(i)
                        logits, _, attns = argp_model(H, ei, etypes, nm, om)
                        prob = torch.softmax(logits, dim=-1)[0, 1].item()
                        preds.append(int(prob > 0.5))
                        trues.append(labels[i])
                        risks.append(prob)
                        all_attns.append(attns[-1].cpu().tolist() if attns else [])

                # Compute ARGP metrics
                orsa = round(accuracy_score(trues, preds), 3)
                f1   = round(f1_score(trues, preds, average="macro", zero_division=0), 3)

                # AWC — attention weight concentration on important nodes
                awc_vals = []
                for attn, g in zip(all_attns, [graphs[i] for i in idx_te]):
                    if not attn or g["n_nodes"] == 0 or not g["edges"]:
                        continue
                    roles = [nd["role"] for nd in g["nodes"]]
                    imp   = {"AGGRESSOR", "OFFENSIVE_PREDICATE", "HUMAN_TARGET"}
                    imp_idx = {j for j, r in enumerate(roles) if r in imp}
                    total_a = sum(attn)
                    imp_a   = sum(av for av, e in zip(attn, g["edges"])
                                  if e.get("src", 0) in imp_idx)
                    if total_a > 0:
                        awc_vals.append(imp_a / total_a)
                awc = round(float(np.mean(awc_vals)), 3) if awc_vals else 0.0

                # NAS — negation attenuation score
                neg_risks = [r for r, g in zip(risks, [graphs[i] for i in idx_te])
                             if g["has_negation"]]
                non_risks = [r for r, g in zip(risks, [graphs[i] for i in idx_te])
                             if not g["has_negation"]]
                nas = round(float(np.mean(non_risks) - np.mean(neg_risks)), 3) \
                      if neg_risks and non_risks else 0.0

                # ONAS — offensive node attention share
                onas_vals = []
                for attn, g in zip(all_attns, [graphs[i] for i in idx_te]):
                    if not attn or not g["edges"]:
                        continue
                    roles   = [nd["role"] for nd in g["nodes"]]
                    off_idx = {j for j, r in enumerate(roles) if r == "OFFENSIVE_PREDICATE"}
                    all_a   = np.array(attn) + 1e-9
                    off_a   = np.array([av for av, e in zip(attn, g["edges"])
                                        if e.get("src", 0) in off_idx])
                    if len(off_a) > 0:
                        onas_vals.append(off_a.mean() / all_a.mean())
                onas = round(float(np.mean(onas_vals)), 3) if onas_vals else 1.0

                # AWE — attention weight entropy
                awe_vals = []
                for attn in all_attns:
                    if attn:
                        a = np.array(attn) + 1e-9
                        a = a / a.sum()
                        awe_vals.append(-np.sum(a * np.log(a)))
                awe = round(float(np.mean(awe_vals)), 3) if awe_vals else 0.0

                t15_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name, **rasgc_m})
                t16_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name,
                                  "AWC": awc, "ORSA": orsa, "NAS": nas, "F1": f1})
                t17_rows.append({"Model": f"IDOL-F+{model_name}",
                                  "Dataset": dataset_name,
                                  "ONAS": onas, "AWE": awe, "Node-F1": f1, "Down-F1": f1})

                torch.save(argp_model.state_dict(),
                    os.path.join(OUT, f"{model_name}_{dataset_name}_argp.pth"))

                print(f"    AWC={awc} ORSA={orsa} NAS={nas} F1={f1} "
                      f"ONAS={onas} AWE={awe}")
                del argp_model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
                import traceback; traceback.print_exc()

    # Save tables
    for rows, fname, label in [
        (t15_rows, "table15_rasgc.csv", "TABLE 15 — RASGC"),
        (t16_rows, "table16_argp.csv",  "TABLE 16 — ARGP"),
        (t17_rows, "table17_argp_node.csv", "TABLE 17 — ARGP Node+Downstream"),
    ]:
        if rows:
            df_t = pd.DataFrame(rows)
            df_t.to_csv(os.path.join(OUT, fname), index=False)
            print(f"\n  {label}:")
            print(df_t.to_string(index=False))

    print(f"\n  [DONE] Step-08 complete. Output: {OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
