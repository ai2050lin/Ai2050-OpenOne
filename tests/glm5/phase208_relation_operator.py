"""
Phase 208: Relation Operator Recovery & Grammatical Covariance
==============================================================

THEORETICAL BREAKTHROUGH (from user analysis):
  Phase 207 proved semantic relations exist beyond token collapse.
  BUT: cosine similarity is the WRONG tool for measuring relations.

  KEY INSIGHT: RELATION ≠ SIMILARITY
  - Subject-verb agreement is a STRUCTURAL DEPENDENCY, not vector proximity
  - The correct framework: R(i,j) = h_i^T A h_j (bilinear relation operator)
  - A is ASYMMETRIC, task-dependent, context-dependent
  - cos(h_i, h_j) = h_i^T I h_j (identity = no relational structure)

  This is the most important conceptual shift in the entire project:
  From "how similar are tokens" → "what relational operator connects tokens"

CORE EXPERIMENTS:
  Exp1: Bilinear Relation Operator Recovery (THE CORE)
    - Learn A_sv, A_vo such that h_i^T A h_j separates true vs random pairs
    - Compare AUC: operator vs cosine — does A capture BEYOND similarity?
    - If operator AUC >> cosine AUC → structural relations exist
    - If operator AUC ≈ cosine AUC → relations are just similarity

  Exp2: Role Swap Invariance (COUNTERFACTUAL TEST)
    - "The cat chases the dog" vs "The dog chases the cat"
    - Train A on ACTIVE sentences, test on SWAPPED sentences
    - If A captures STRUCTURAL ROLES → should generalize despite word swap
    - If A captures WORD PATTERNS → should fail on swapped

  Exp3: Number Agreement (GRAMMATICAL COVARIANCE)
    - "The cat chases" vs "The cats chase"
    - Train A on singular, test on plural
    - If A captures AGREEMENT → should recognize both as valid S-V
    - This tests whether the operator is "surface-form independent"

  Exp4: Asymmetry Test (DIRECTIONAL DEPENDENCY)
    - Compare: h_s^T A h_v (subject→verb) vs h_v^T A h_s (verb→subject)
    - Cosine is always symmetric → cannot capture this
    - If A is asymmetric → it captures DIRECTIONAL dependency

  Exp5: Attention→Relation Mapping
    - Does attention weight α_ij predict h_i^T A h_j?
    - If yes → attention implements relation operators
    - If no → relations are computed elsewhere (e.g., in residual stream)

MODELS: Qwen3, GLM4, DS7B (bf16 + device_map="auto", NO 8-bit)
DATA: 60 base sentences × 3 variants (active/swapped/plural) = 180 sentences
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score

from model_utils import (get_model_info, release_model, get_layers, get_W_U,
                          MODEL_CONFIGS)

warnings.filterwarnings('ignore')

LITE = os.environ.get('LITE', '0') == '1'

# ========================================================================
# Sentence Data — 60 base triplets with 3 variants each
# Pattern: "The {subject} {verb_3sg} the {object}"
# ========================================================================
# Each triple: (subject, verb_3sg, verb_base, object)
# Selected so both subject-object and object-subject orderings are grammatically valid
SENTENCE_TRIPLES = [
    # Animals (30 pairs)
    ("cat", "chases", "chase", "dog"),
    ("dog", "follows", "follow", "cat"),
    ("bear", "watches", "watch", "wolf"),
    ("wolf", "avoids", "avoid", "bear"),
    ("lion", "finds", "find", "tiger"),
    ("tiger", "fears", "fear", "lion"),
    ("horse", "meets", "meet", "deer"),
    ("deer", "sees", "see", "horse"),
    ("fox", "tracks", "track", "rabbit"),
    ("rabbit", "hears", "hear", "fox"),
    ("whale", "joins", "join", "shark"),
    ("shark", "follows", "follow", "whale"),
    ("monkey", "greets", "greet", "elephant"),
    ("elephant", "trusts", "trust", "monkey"),
    ("bird", "watches", "watch", "eagle"),
    ("eagle", "finds", "find", "bird"),
    ("snake", "avoids", "avoid", "frog"),
    ("frog", "hears", "hear", "snake"),
    ("fish", "follows", "follow", "seal"),
    ("seal", "catches", "catch", "fish"),
    ("cow", "meets", "meet", "horse"),
    ("sheep", "follows", "follow", "goat"),
    ("duck", "watches", "watch", "goose"),
    ("mouse", "fears", "fear", "rat"),
    ("rat", "chases", "chase", "mouse"),
    ("hawk", "finds", "find", "owl"),
    ("owl", "watches", "watch", "hawk"),
    ("ant", "follows", "follow", "bee"),
    ("bee", "joins", "join", "ant"),
    ("crab", "meets", "meet", "fish"),
    # Humans (30 pairs)
    ("teacher", "helps", "help", "student"),
    ("student", "thanks", "thank", "teacher"),
    ("doctor", "treats", "treat", "patient"),
    ("patient", "trusts", "trust", "doctor"),
    ("king", "protects", "protect", "queen"),
    ("queen", "guides", "guide", "king"),
    ("captain", "leads", "lead", "soldier"),
    ("soldier", "follows", "follow", "captain"),
    ("chef", "helps", "help", "baker"),
    ("baker", "thanks", "thank", "chef"),
    ("writer", "meets", "meet", "reader"),
    ("reader", "thanks", "thank", "writer"),
    ("singer", "joins", "join", "dancer"),
    ("dancer", "follows", "follow", "singer"),
    ("judge", "watches", "watch", "lawyer"),
    ("lawyer", "finds", "find", "judge"),
    ("driver", "meets", "meet", "rider"),
    ("rider", "thanks", "thank", "driver"),
    ("builder", "helps", "help", "painter"),
    ("painter", "guides", "guide", "builder"),
    ("farmer", "feeds", "feed", "worker"),
    ("worker", "helps", "help", "farmer"),
    ("priest", "guides", "guide", "monk"),
    ("monk", "follows", "follow", "priest"),
    ("nurse", "helps", "help", "clerk"),
    ("clerk", "thanks", "thank", "nurse"),
    ("master", "teaches", "teach", "pupil"),
    ("pupil", "thanks", "thank", "master"),
    ("hero", "protects", "protect", "child"),
    ("child", "trusts", "trust", "hero"),
]


def generate_sentences():
    """Generate 3 variants of each base sentence."""
    active = []
    swapped = []
    plural = []

    for subj, v3sg, vbase, obj in SENTENCE_TRIPLES:
        # Active: "The X verbs the Y"
        active.append(f"The {subj} {v3sg} the {obj}")

        # Swapped: "The Y verbs the X" (role reversal)
        swapped.append(f"The {obj} {v3sg} the {subj}")

        # Plural: "The Xs verb the Y" (number agreement)
        # Simple: add 's' to subject (works for most English nouns)
        subj_plural = subj + "s" if not subj.endswith("s") else subj + "es"
        # Special cases
        irregular_plurals = {
            "child": "children", "mouse": "mice", "goose": "geese",
            "fish": "fish", "deer": "deer", "sheep": "sheep",
        }
        if subj in irregular_plurals:
            subj_plural = irregular_plurals[subj]

        plural.append(f"The {subj_plural} {vbase} the {obj}")

    return active, swapped, plural


# ========================================================================
# Model Loading (bf16 + device_map="auto", with SDPA for flash attention)
# ========================================================================
def load_model_bf16(model_name: str):
    """BF16加载模型 — bfloat16 + device_map="auto" + flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try SDPA (flash-like) first, then eager as fallback
    model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            print(f"[load] Using attn_implementation={attn_impl}")
            break
        except Exception as e:
            print(f"[load] attn_implementation={attn_impl} failed: {e}")
            model = None
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[load] {model_name}: GPU={gpu_count} components, CPU={cpu_count}, "
              f"GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


# ========================================================================
# SVO Annotation — Find subject-verb-object positions
# ========================================================================
def find_svo_positions(token_ids, tokenizer):
    """Find SVO pattern "The X V the Y" in tokenized sentence."""
    tokens = []
    for tid in token_ids:
        decoded = tokenizer.decode([tid]).strip().lower()
        tokens.append(decoded)

    determiners = {"the", "a", "an"}

    for i in range(len(tokens) - 4):
        if tokens[i] in determiners:
            if tokens[i+3] in determiners:
                return {
                    'det1': i,
                    'subject': i + 1,
                    'verb': i + 2,
                    'det2': i + 3,
                    'object': i + 4,
                }
    return None


# ========================================================================
# Hidden State Collection
# ========================================================================
def collect_hidden_states(model, tokenizer, device, sentences, n_layers,
                          collect_attention=False, max_len=64):
    """
    Collect hidden states (and optionally attention weights) for all sentences.

    Returns:
        all_hidden: {layer_idx: [seq_len, d_model] per sentence}
        all_annotations: [dict or None] per sentence
        all_attentions: {layer_idx: [n_heads, seq_len, seq_len] per sentence} or None
    """
    all_hidden = {l: [] for l in range(n_layers + 1)}
    all_annotations = []
    all_attentions = {} if collect_attention else None

    if collect_attention:
        for l in range(n_layers):
            all_attentions[l] = []

    n_sents = len(sentences)
    t0 = time.time()

    for si, sent in enumerate(sentences):
        if LITE and si >= 20:
            break

        # Periodic logging
        if si % 10 == 0:
            elapsed = time.time() - t0
            rate = (si + 1) / max(elapsed, 0.1)
            eta = (n_sents - si - 1) / max(rate, 0.1)
            print(f"    [{si+1}/{n_sents}] '{sent[:50]}...' "
                  f"({rate:.1f} sent/s, ETA {eta:.0f}s)")

        toks = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)

        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=collect_attention,
                )
                for li in range(n_layers + 1):
                    hs = out.hidden_states[li].float().cpu().numpy()
                    all_hidden[li].append(hs[0])  # [seq_len, d_model]

                if collect_attention and out.attentions is not None:
                    for li in range(n_layers):
                        attn = out.attentions[li].float().cpu().numpy()  # [1, n_heads, seq, seq]
                        all_attentions[li].append(attn[0])  # [n_heads, seq, seq]

            except Exception as e:
                print(f"    [WARN] Forward failed for sentence {si}: {e}")
                for li in range(n_layers + 1):
                    all_hidden[li].append(None)
                if collect_attention:
                    for li in range(n_layers):
                        all_attentions[li].append(None)

        # Annotate
        input_ids_np = input_ids[0].cpu().numpy()
        svo = find_svo_positions(input_ids_np, tokenizer)
        all_annotations.append(svo)

        # Periodic memory cleanup
        if si % 20 == 0:
            torch.cuda.empty_cache()

    n_annotated = sum(1 for a in all_annotations if a is not None)
    print(f"  Collected {len(all_annotations)} sentences, {n_annotated} with SVO annotation")

    return all_hidden, all_annotations, all_attentions


# ========================================================================
# Bilinear Relation Operator
# ========================================================================
class BilinearRelationOperator(nn.Module):
    """
    Learnable bilinear relation operator A = U V^T

    Score(h_i, h_j) = h_i^T A h_j = (U^T h_i) · (V^T h_j)

    This captures ASYMMETRIC relations (U ≠ V).
    Compare with cosine: cos(h_i, h_j) = h_i^T I h_j (symmetric, no structure)
    """
    def __init__(self, d_model, rank=32):
        super().__init__()
        self.rank = rank
        # U: projects "left" element of pair (e.g., subject)
        # V: projects "right" element of pair (e.g., verb)
        self.U = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def score_pair(self, h_i, h_j):
        """Score for a single pair: h_i^T A h_j = (U^T h_i) · (V^T h_j)"""
        p_i = self.U @ h_i  # [rank]
        p_j = self.V @ h_j  # [rank]
        return (p_i * p_j).sum()

    def score_batch(self, h_i_batch, h_j_batch):
        """Score for batch of pairs: [batch_size]"""
        p_i = h_i_batch @ self.U.T  # [batch, rank]
        p_j = h_j_batch @ self.V.T  # [batch, rank]
        return (p_i * p_j).sum(dim=1)  # [batch]

    def reverse_score_batch(self, h_i_batch, h_j_batch):
        """Score for reversed pairs: h_j^T A h_i = (U^T h_j) · (V^T h_i)"""
        p_j = h_j_batch @ self.U.T  # [batch, rank]
        p_i = h_i_batch @ self.V.T  # [batch, rank]
        return (p_j * p_i).sum(dim=1)  # [batch]


def train_relation_operator(pos_pairs, neg_pairs, d_model, rank=32,
                             epochs=80, lr=0.001, verbose=False):
    """
    Train bilinear relation operator to separate positive from negative pairs.

    Args:
        pos_pairs: list of (h_i, h_j) numpy arrays for TRUE relation pairs
        neg_pairs: list of (h_i, h_j) numpy arrays for RANDOM pairs
        d_model: hidden dimension
        rank: operator rank
        epochs: training epochs
        lr: learning rate

    Returns:
        trained model, training loss history
    """
    if len(pos_pairs) < 5 or len(neg_pairs) < 5:
        return None, []

    # Convert to tensors
    pos_h_i = torch.tensor(np.array([p[0] for p in pos_pairs]), dtype=torch.float32)
    pos_h_j = torch.tensor(np.array([p[1] for p in pos_pairs]), dtype=torch.float32)
    neg_h_i = torch.tensor(np.array([p[0] for p in neg_pairs]), dtype=torch.float32)
    neg_h_j = torch.tensor(np.array([p[1] for p in neg_pairs]), dtype=torch.float32)

    model = BilinearRelationOperator(d_model, rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Positive scores
        pos_scores = model.score_batch(pos_h_i, pos_h_j)
        # Negative scores
        neg_scores = model.score_batch(neg_h_i, neg_h_j)

        # Logistic loss: maximize pos_scores, minimize neg_scores
        # L = -log(σ(pos - neg)) for each pair
        # Use mean over all positive-negative combinations
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [n_pos, n_neg]
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and epoch % 20 == 0:
            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()
            print(f"    Epoch {epoch}: loss={loss_val:.4f}, "
                  f"pos_score={pos_mean:.4f}, neg_score={neg_mean:.4f}")

    return model, loss_history


def compute_auc(model, pos_pairs, neg_pairs, d_model):
    """Compute AUC for the relation operator on given pairs."""
    if model is None or len(pos_pairs) < 3 or len(neg_pairs) < 3:
        return 0.5

    model.eval()
    with torch.no_grad():
        pos_h_i = torch.tensor(np.array([p[0] for p in pos_pairs]), dtype=torch.float32)
        pos_h_j = torch.tensor(np.array([p[1] for p in pos_pairs]), dtype=torch.float32)
        neg_h_i = torch.tensor(np.array([p[0] for p in neg_pairs]), dtype=torch.float32)
        neg_h_j = torch.tensor(np.array([p[1] for p in neg_pairs]), dtype=torch.float32)

        pos_scores = model.score_batch(pos_h_i, pos_h_j).numpy()
        neg_scores = model.score_batch(neg_h_i, neg_h_j).numpy()

    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.5

    return float(auc)


def compute_cosine_auc(pos_pairs, neg_pairs):
    """Compute AUC using cosine similarity as score."""
    if len(pos_pairs) < 3 or len(neg_pairs) < 3:
        return 0.5

    pos_scores = []
    for h_i, h_j in pos_pairs:
        ni = np.linalg.norm(h_i)
        nj = np.linalg.norm(h_j)
        if ni > 1e-10 and nj > 1e-10:
            pos_scores.append(float(np.dot(h_i, h_j) / (ni * nj)))
        else:
            pos_scores.append(0.0)

    neg_scores = []
    for h_i, h_j in neg_pairs:
        ni = np.linalg.norm(h_i)
        nj = np.linalg.norm(h_j)
        if ni > 1e-10 and nj > 1e-10:
            neg_scores.append(float(np.dot(h_i, h_j) / (ni * nj)))
        else:
            neg_scores.append(0.0)

    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.5

    return float(auc)


# ========================================================================
# Pair Extraction Utilities
# ========================================================================
def extract_pairs(all_hidden, annotations, relation_type='subject_verb',
                   normalize=True, max_pairs_per_sent=5):
    """
    Extract positive and negative pairs from hidden states.

    Args:
        all_hidden: {layer: [seq_len, d_model]} per sentence
        annotations: [dict or None] per sentence
        relation_type: 'subject_verb', 'verb_object', or 'det_noun'
        normalize: L2-normalize hidden states before extracting

    Returns:
        pos_pairs: list of (h_i, h_j) for TRUE relation pairs
        neg_pairs: list of (h_i, h_j) for RANDOM pairs
    """
    pos_pairs = []
    neg_pairs = []

    for si, hs in enumerate(all_hidden):
        if hs is None:
            continue
        svo = annotations[si] if si < len(annotations) else None
        if svo is None:
            continue

        n_tok = hs.shape[0]
        s, v, o = svo['subject'], svo['verb'], svo['object']
        d1, d2 = svo['det1'], svo['det2']

        # L2 normalize
        if normalize:
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            hs_norm = hs / norms
        else:
            hs_norm = hs

        # Define positive pair positions
        if relation_type == 'subject_verb':
            pos_i, pos_j = s, v
        elif relation_type == 'verb_object':
            pos_i, pos_j = v, o
        elif relation_type == 'det_noun':
            # Use first det-noun pair
            pos_i, pos_j = d1, s
        else:
            continue

        if pos_i >= n_tok or pos_j >= n_tok:
            continue

        # Positive pair
        pos_pairs.append((hs_norm[pos_i].copy(), hs_norm[pos_j].copy()))

        # Negative pairs: other token pairs from same sentence
        neg_count = 0
        for i in range(n_tok):
            for j in range(i+1, n_tok):
                # Skip the positive pair
                if i == pos_i and j == pos_j:
                    continue
                # Skip other specific syntactic pairs (to make it harder)
                if relation_type == 'subject_verb':
                    if (i == v and j == o):  # skip V-O
                        continue
                neg_pairs.append((hs_norm[i].copy(), hs_norm[j].copy()))
                neg_count += 1
                if neg_count >= max_pairs_per_sent:
                    break
            if neg_count >= max_pairs_per_sent:
                break

    return pos_pairs, neg_pairs


# ========================================================================
# Exp1: Bilinear Relation Operator Recovery (THE CORE)
# ========================================================================
def run_exp1_operator_recovery(all_hidden_active, annotations_active,
                                 n_layers, sample_layers, d_model):
    """
    Exp1: Learn bilinear relation operators for S-V and V-O relations.
    Compare with cosine baseline.
    
    KEY METRIC: operator_AUC - cosine_AUC
    If Δ > 0 → operator captures structural info BEYOND similarity
    If Δ ≈ 0 → relation is just similarity
    """
    print(f"\n{'='*70}")
    print("Exp1: Bilinear Relation Operator Recovery (THE CORE)")
    print(f"{'='*70}")

    results = {}
    relation_types = ['subject_verb', 'verb_object']
    operator_rank = 32

    for li in sample_layers:
        hs_list = all_hidden_active.get(li, [])
        if not hs_list:
            continue

        for rel_type in relation_types:
            # Extract ALL pairs first
            all_pos, all_neg = extract_pairs(hs_list, annotations_active, rel_type)

            if len(all_pos) < 8:
                continue

            # Split pairs 70/30 into train/test
            n_pos = len(all_pos)
            n_train_pos = max(5, int(0.7 * n_pos))
            train_pos = all_pos[:n_train_pos]
            test_pos = all_pos[n_train_pos:]

            # For negatives: sample proportionally
            n_neg = len(all_neg)
            n_train_neg = min(int(0.7 * n_neg), n_train_pos * 5)
            train_neg = all_neg[:n_train_neg]
            test_neg = all_neg[n_train_neg:n_train_neg + len(test_pos) * 5]

            if len(test_pos) < 3 or len(test_neg) < 3:
                continue

            # Train operator
            model_op, loss_hist = train_relation_operator(
                train_pos, train_neg, d_model, rank=operator_rank,
                epochs=80, lr=0.001, verbose=(li == sample_layers[len(sample_layers)//2])
            )

            # Evaluate
            if model_op is not None:
                op_auc = compute_auc(model_op, test_pos, test_neg, d_model)
                cos_auc = compute_cosine_auc(test_pos, test_neg)
                op_auc_train = compute_auc(model_op, train_pos, train_neg, d_model)
                cos_auc_train = compute_cosine_auc(train_pos, train_neg)

                improvement = op_auc - cos_auc

                key = f"L{li}_{rel_type}"
                results[key] = {
                    'operator_auc_test': op_auc,
                    'cosine_auc_test': cos_auc,
                    'improvement': improvement,
                    'operator_auc_train': op_auc_train,
                    'cosine_auc_train': cos_auc_train,
                    'overfitting': op_auc_train - op_auc,
                    'n_train_pos': len(train_pos),
                    'n_train_neg': len(train_neg),
                    'n_test_pos': len(test_pos),
                    'n_test_neg': len(test_neg),
                }

                if li in [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]:
                    marker = "★★★" if improvement > 0.05 else ("★★" if improvement > 0.02 else ("★" if improvement > 0 else "✗"))
                    print(f"  L{li} {rel_type}: op_AUC={op_auc:.4f}, "
                          f"cos_AUC={cos_auc:.4f}, Δ={improvement:+.4f} {marker}")

    return results


# ========================================================================
# Exp2: Role Swap Invariance (COUNTERFACTUAL TEST)
# ========================================================================
def run_exp2_role_swap(all_hidden_active, annotations_active,
                        all_hidden_swapped, annotations_swapped,
                        n_layers, sample_layers, d_model):
    """
    Exp2: Train operator on ACTIVE, test on SWAPPED sentences.

    KEY QUESTION: Does A capture STRUCTURAL ROLES or WORD PATTERNS?

    "The cat chases the dog" → "The dog chases the cat"
    In ACTIVE: cat=subject, dog=object
    In SWAPPED: dog=subject, cat=object

    If A captures structural roles → should work on swapped (AUC similar)
    If A captures word patterns → should fail on swapped (AUC drops)
    """
    print(f"\n{'='*70}")
    print("Exp2: Role Swap Invariance (COUNTERFACTICAL TEST)")
    print(f"{'='*70}")

    results = {}
    relation_types = ['subject_verb', 'verb_object']

    for li in sample_layers:
        hs_active = all_hidden_active.get(li, [])
        hs_swapped = all_hidden_swapped.get(li, [])
        if not hs_active or not hs_swapped:
            continue

        for rel_type in relation_types:
            # Extract pairs from both distributions
            active_pos, active_neg = extract_pairs(hs_active, annotations_active, rel_type)
            swapped_pos, swapped_neg = extract_pairs(hs_swapped, annotations_swapped, rel_type)

            if len(active_pos) < 8 or len(swapped_pos) < 3:
                continue

            # Split ACTIVE into 70/30 train/test
            n_train = max(5, int(0.7 * len(active_pos)))
            train_pos = active_pos[:n_train]
            active_test_pos = active_pos[n_train:]
            train_neg = active_neg[:n_train * 5]
            active_test_neg = active_neg[n_train * 5:n_train * 5 + len(active_test_pos) * 5]

            if len(active_test_pos) < 3:
                active_test_pos = active_pos  # fallback: use all
                active_test_neg = active_neg[:len(active_pos) * 5]

            # Train on ACTIVE train split
            model_op, _ = train_relation_operator(
                train_pos, train_neg, d_model, rank=32,
                epochs=80, lr=0.001
            )
            if model_op is None:
                continue

            # Test on ACTIVE test split (same-domain baseline)
            auc_active = compute_auc(model_op, active_test_pos, active_test_neg, d_model)

            # Test on SWAPPED (cross-domain — THE KEY TEST)
            auc_swapped = compute_auc(model_op, swapped_pos, swapped_neg, d_model)

            # Cosine baselines
            cos_auc_active = compute_cosine_auc(active_test_pos, active_test_neg)
            cos_auc_swapped = compute_cosine_auc(swapped_pos, swapped_neg)

            generalization_drop = auc_active - auc_swapped

            key = f"L{li}_{rel_type}"
            results[key] = {
                'operator_auc_active': auc_active,
                'operator_auc_swapped': auc_swapped,
                'cosine_auc_active': cos_auc_active,
                'cosine_auc_swapped': cos_auc_swapped,
                'generalization_drop': generalization_drop,
                # Does operator generalize BETTER than cosine?
                'operator_swap_vs_cosine_swap': auc_swapped - cos_auc_swapped,
            }

            if li in [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]:
                verdict = "STRUCTURAL ROLE" if generalization_drop < 0.1 else \
                          ("PARTIAL" if generalization_drop < 0.2 else "WORD PATTERN")
                print(f"  L{li} {rel_type}: active={auc_active:.4f} → swapped={auc_swapped:.4f} "
                      f"(drop={generalization_drop:+.4f}) [{verdict}]")
                print(f"    cosine: active={cos_auc_active:.4f} → swapped={cos_auc_swapped:.4f}")

    return results


# ========================================================================
# Exp3: Number Agreement (GRAMMATICAL COVARIANCE)
# ========================================================================
def run_exp3_number_agreement(all_hidden_active, annotations_active,
                                all_hidden_plural, annotations_plural,
                                n_layers, sample_layers, d_model):
    """
    Exp3: Train operator on singular (ACTIVE), test on plural (PLURAL).

    KEY QUESTION: Does A capture AGREEMENT (a structural dependency)?
    "The cat chases" vs "The cats chase"
    If A captures agreement → should recognize both as valid S-V
    """
    print(f"\n{'='*70}")
    print("Exp3: Number Agreement (GRAMMATICAL COVARIANCE)")
    print(f"{'='*70}")

    results = {}

    for li in sample_layers:
        hs_active = all_hidden_active.get(li, [])
        hs_plural = all_hidden_plural.get(li, [])
        if not hs_active or not hs_plural:
            continue

        # Train on ACTIVE
        train_pos, train_neg = extract_pairs(hs_active, annotations_active, 'subject_verb')
        if len(train_pos) < 5:
            continue

        model_op, _ = train_relation_operator(
            train_pos, train_neg, d_model, rank=32,
            epochs=80, lr=0.001
        )
        if model_op is None:
            continue

        # Test on PLURAL
        plural_pos, plural_neg = extract_pairs(hs_plural, annotations_plural, 'subject_verb')
        if len(plural_pos) < 3:
            continue

        auc_active = compute_auc(model_op, train_pos[:10], train_neg[:10], d_model)
        auc_plural = compute_auc(model_op, plural_pos, plural_neg, d_model)
        cos_auc_plural = compute_cosine_auc(plural_pos, plural_neg)

        key = f"L{li}_sv"
        results[key] = {
            'operator_auc_active': auc_active,
            'operator_auc_plural': auc_plural,
            'cosine_auc_plural': cos_auc_plural,
            'agreement_drop': auc_active - auc_plural,
        }

        if li in [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]:
            drop = auc_active - auc_plural
            verdict = "AGREEMENT CAPTURED" if drop < 0.1 else ("PARTIAL" if drop < 0.2 else "SURFACE FORM")
            print(f"  L{li}: active_AUC={auc_active:.4f}, plural_AUC={auc_plural:.4f}, "
                  f"drop={drop:+.4f} [{verdict}]")

    return results


# ========================================================================
# Exp4: Asymmetry Test (DIRECTIONAL DEPENDENCY)
# ========================================================================
def run_exp4_asymmetry(all_hidden_active, annotations_active,
                         n_layers, sample_layers, d_model):
    """
    Exp4: Test whether the relation operator is ASYMMETRIC.

    Compare: h_s^T A h_v (subject→verb) vs h_v^T A h_s (verb→subject)
    If A captures directional dependency → these should be different
    Cosine is always symmetric → cannot capture this
    """
    print(f"\n{'='*70}")
    print("Exp4: Asymmetry Test (DIRECTIONAL DEPENDENCY)")
    print(f"{'='*70}")

    results = {}

    for li in sample_layers:
        hs_list = all_hidden_active.get(li, [])
        if not hs_list:
            continue

        # Train S-V operator
        train_pos, train_neg = extract_pairs(hs_list, annotations_active, 'subject_verb')
        if len(train_pos) < 5:
            continue

        model_op, _ = train_relation_operator(
            train_pos, train_neg, d_model, rank=32,
            epochs=80, lr=0.001
        )
        if model_op is None:
            continue

        # Compute forward and reverse scores for test pairs
        n_test = min(30, len(train_pos))
        test_pos = train_pos[:n_test]

        model_op.eval()
        with torch.no_grad():
            h_i = torch.tensor(np.array([p[0] for p in test_pos]), dtype=torch.float32)
            h_j = torch.tensor(np.array([p[1] for p in test_pos]), dtype=torch.float32)

            forward_scores = model_op.score_batch(h_i, h_j).numpy()
            reverse_scores = model_op.reverse_score_batch(h_i, h_j).numpy()

        # Asymmetry metrics
        diff = forward_scores - reverse_scores
        abs_diff = np.abs(diff)
        mean_forward = float(np.mean(forward_scores))
        mean_reverse = float(np.mean(reverse_scores))
        mean_abs_diff = float(np.mean(abs_diff))

        # Relative asymmetry: |forward - reverse| / (|forward| + |reverse|)
        denom = np.abs(forward_scores) + np.abs(reverse_scores) + 1e-10
        rel_asym = float(np.mean(abs_diff / denom))

        # Compare with cosine (always symmetric)
        cosine_vals = []
        for h_i_np, h_j_np in test_pos:
            ni = np.linalg.norm(h_i_np)
            nj = np.linalg.norm(h_j_np)
            if ni > 1e-10 and nj > 1e-10:
                cosine_vals.append(float(np.dot(h_i_np, h_j_np) / (ni * nj)))
        mean_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0

        key = f"L{li}"
        results[key] = {
            'forward_score': mean_forward,
            'reverse_score': mean_reverse,
            'mean_abs_diff': mean_abs_diff,
            'relative_asymmetry': rel_asym,
            'mean_cosine': mean_cosine,
            'asymmetry_vs_cosine': rel_asym,  # Cosine is always 0 (symmetric)
        }

        if li in [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]:
            verdict = "STRONGLY ASYMMETRIC" if rel_asym > 0.3 else \
                      ("MODERATELY ASYMMETRIC" if rel_asym > 0.1 else "WEAKLY ASYMMETRIC")
            print(f"  L{li}: forward={mean_forward:.4f}, reverse={mean_reverse:.4f}, "
                  f"rel_asym={rel_asym:.4f}, cosine={mean_cosine:.4f} [{verdict}]")

    return results


# ========================================================================
# Exp5: Attention→Relation Mapping
# ========================================================================
def run_exp5_attention_relation(all_hidden_active, annotations_active,
                                 all_attentions, n_layers, sample_layers, d_model):
    """
    Exp5: Does attention weight α_ij predict the relation operator score?

    If yes → attention IMPLEMENTS relation operators
    If no → relations are computed elsewhere
    """
    print(f"\n{'='*70}")
    print("Exp5: Attention→Relation Mapping")
    print(f"{'='*70}")

    if all_attentions is None:
        print("  [SKIP] No attention data collected")
        return {}

    results = {}

    for li in sample_layers:
        hs_list = all_hidden_active.get(li, [])
        attn_list = all_attentions.get(li, [])
        if not hs_list or not attn_list:
            continue

        # Train S-V operator first
        train_pos, train_neg = extract_pairs(hs_list, annotations_active, 'subject_verb')
        if len(train_pos) < 5:
            continue

        model_op, _ = train_relation_operator(
            train_pos, train_neg, d_model, rank=32,
            epochs=80, lr=0.001
        )
        if model_op is None:
            continue

        # Collect attention weights and operator scores for S-V pairs
        attn_sv_scores = []  # attention weights from verb→subject (averaged over heads)
        op_sv_scores = []    # operator scores for S-V pairs

        for si, hs in enumerate(hs_list):
            if hs is None or si >= len(attn_list) or attn_list[si] is None:
                continue

            svo = annotations_active[si] if si < len(annotations_active) else None
            if svo is None:
                continue

            n_tok = hs.shape[0]
            s, v = svo['subject'], svo['verb']

            if s >= n_tok or v >= n_tok:
                continue

            # Get attention: [n_heads, seq, seq]
            attn = attn_list[si]
            if attn.shape[0] == 0 or attn.shape[1] <= v or attn.shape[2] <= s:
                continue

            # Attention from verb→subject: α_{v,s} = how much verb attends to subject
            # Average over heads
            attn_vs = float(attn[:, v, s].mean())

            # Operator score
            hs_norm = hs / np.maximum(np.linalg.norm(hs, axis=1, keepdims=True), 1e-10)
            h_s = torch.tensor(hs_norm[s], dtype=torch.float32)
            h_v = torch.tensor(hs_norm[v], dtype=torch.float32)

            with torch.no_grad():
                op_score = model_op.score_pair(h_s, h_v).item()

            attn_sv_scores.append(attn_vs)
            op_sv_scores.append(op_score)

        if len(attn_sv_scores) < 5:
            continue

        # Correlation
        attn_arr = np.array(attn_sv_scores)
        op_arr = np.array(op_sv_scores)

        # Pearson correlation
        if np.std(attn_arr) > 1e-10 and np.std(op_arr) > 1e-10:
            corr = float(np.corrcoef(attn_arr, op_arr)[0, 1])
        else:
            corr = 0.0

        # Spearman correlation
        from scipy.stats import spearmanr
        try:
            spearman_corr, p_val = spearmanr(attn_arr, op_arr)
        except:
            spearman_corr, p_val = 0.0, 1.0

        key = f"L{li}"
        results[key] = {
            'pearson_corr': corr,
            'spearman_corr': float(spearman_corr),
            'spearman_p': float(p_val),
            'n_pairs': len(attn_sv_scores),
            'mean_attn': float(attn_arr.mean()),
            'mean_op_score': float(op_arr.mean()),
        }

        if li in [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]:
            verdict = "ATTENTION=OPERATOR" if abs(corr) > 0.5 else \
                      ("PARTIAL" if abs(corr) > 0.3 else "INDEPENDENT")
            print(f"  L{li}: pearson={corr:.4f}, spearman={spearman_corr:.4f} "
                  f"(n={len(attn_sv_scores)}) [{verdict}]")

    return results


# ========================================================================
# Main Function
# ========================================================================
def run_phase208(model_name: str):
    print(f"\n{'='*70}")
    print(f"Phase 208: Relation Operator Recovery & Grammatical Covariance — {model_name}")
    print(f"{'='*70}")
    t_start = time.time()

    # ---- Load model ----
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}, class={info.model_class}")

    # ---- Sample layers ----
    if n_layers <= 12:
        sample_layers = list(range(n_layers + 1))
    else:
        step = max(1, n_layers // 10)
        sample_layers = sorted(set(list(range(0, n_layers + 1, step)) + [n_layers - 1]))
    print(f"  Sample layers: {sample_layers}")

    # ---- Generate sentences ----
    active_sents, swapped_sents, plural_sents = generate_sentences()
    n_sents = len(active_sents)
    print(f"  Sentences: {n_sents} × 3 variants = {n_sents * 3} total")

    # ---- Collect hidden states ----
    print(f"\n--- Collecting ACTIVE hidden states ---")
    hidden_active, annot_active, _ = collect_hidden_states(
        model, tokenizer, device, active_sents, n_layers, collect_attention=False
    )

    print(f"\n--- Collecting SWAPPED hidden states ---")
    hidden_swapped, annot_swapped, _ = collect_hidden_states(
        model, tokenizer, device, swapped_sents, n_layers, collect_attention=False
    )

    print(f"\n--- Collecting PLURAL hidden states ---")
    hidden_plural, annot_plural, _ = collect_hidden_states(
        model, tokenizer, device, plural_sents, n_layers, collect_attention=False
    )

    # ---- Collect attention for a subset (Exp5) ----
    print(f"\n--- Collecting ATTENTION data (subset of 20 sentences) ---")
    attn_sents = active_sents[:20]
    _, _, attentions = collect_hidden_states(
        model, tokenizer, device, attn_sents, n_layers, collect_attention=True
    )

    # ---- Release model ----
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nModel released.")

    # ====================================================================
    # Run all experiments
    # ====================================================================
    all_results = {}

    # Exp1: Bilinear Relation Operator Recovery
    exp1 = run_exp1_operator_recovery(
        hidden_active, annot_active, n_layers, sample_layers, d_model
    )
    all_results['exp1_operator_recovery'] = exp1

    # Exp2: Role Swap Invariance
    exp2 = run_exp2_role_swap(
        hidden_active, annot_active,
        hidden_swapped, annot_swapped,
        n_layers, sample_layers, d_model
    )
    all_results['exp2_role_swap'] = exp2

    # Exp3: Number Agreement
    exp3 = run_exp3_number_agreement(
        hidden_active, annot_active,
        hidden_plural, annot_plural,
        n_layers, sample_layers, d_model
    )
    all_results['exp3_number_agreement'] = exp3

    # Exp4: Asymmetry Test
    exp4 = run_exp4_asymmetry(
        hidden_active, annot_active,
        n_layers, sample_layers, d_model
    )
    all_results['exp4_asymmetry'] = exp4

    # Exp5: Attention→Relation Mapping
    # Need to re-collect hidden states with attention for the subset
    # Use the already collected attention data + hidden states from subset
    hidden_attn_subset = {}
    annot_attn_subset = []
    for li in sample_layers:
        hidden_attn_subset[li] = hidden_active[li][:20]
    annot_attn_subset = annot_active[:20]

    exp5 = run_exp5_attention_relation(
        hidden_attn_subset, annot_attn_subset,
        attentions, n_layers, sample_layers, d_model
    )
    all_results['exp5_attention_relation'] = exp5

    # ====================================================================
    # Cross-experiment summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("PHASE 208 SUMMARY")
    print(f"{'='*70}")

    # Exp1 summary: operator vs cosine
    print("\n--- Exp1: Operator vs Cosine (S-V relation) ---")
    for key, val in sorted(all_results['exp1_operator_recovery'].items()):
        if 'subject_verb' in key:
            imp = val.get('improvement', 0)
            marker = "★★★" if imp > 0.05 else ("★★" if imp > 0.02 else ("★" if imp > 0 else "✗"))
            print(f"  {key}: op_AUC={val['operator_auc_test']:.4f}, "
                  f"cos_AUC={val['cosine_auc_test']:.4f}, Δ={imp:+.4f} {marker}")

    # Exp2 summary: role swap
    print("\n--- Exp2: Role Swap Invariance ---")
    for key, val in sorted(all_results['exp2_role_swap'].items()):
        drop = val.get('generalization_drop', val.get('swap_drop', 0))
        verdict = "STRUCTURAL" if drop < 0.1 else ("PARTIAL" if drop < 0.2 else "WORD-PATTERN")
        print(f"  {key}: drop={drop:+.4f} [{verdict}]")

    # Exp3 summary: number agreement
    print("\n--- Exp3: Number Agreement ---")
    for key, val in sorted(all_results['exp3_number_agreement'].items()):
        drop = val.get('agreement_drop', 0)
        verdict = "AGREEMENT" if drop < 0.1 else ("PARTIAL" if drop < 0.2 else "SURFACE")
        print(f"  {key}: drop={drop:+.4f} [{verdict}]")

    # Exp4 summary: asymmetry
    print("\n--- Exp4: Asymmetry ---")
    for key, val in sorted(all_results['exp4_asymmetry'].items()):
        asym = val.get('relative_asymmetry', 0)
        verdict = "STRONG" if asym > 0.3 else ("MODERATE" if asym > 0.1 else "WEAK")
        print(f"  {key}: rel_asym={asym:.4f} [{verdict}]")

    # Exp5 summary: attention→relation
    print("\n--- Exp5: Attention→Relation ---")
    for key, val in sorted(all_results['exp5_attention_relation'].items()):
        corr = val.get('pearson_corr', 0)
        verdict = "STRONG" if abs(corr) > 0.5 else ("MODERATE" if abs(corr) > 0.3 else "WEAK")
        print(f"  {key}: pearson={corr:.4f} [{verdict}]")

    # ====================================================================
    # Save results
    # ====================================================================
    save_dir = Path(__file__).parent.parent / "glm5_temp"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"phase208_{model_name}_results.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            return str(obj)

    json_results = make_serializable({
        "experiments": all_results,
        "metadata": {
            "model": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "n_sentences": n_sents,
            "sentence_variants": ["active", "swapped", "plural"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f}min)")
    return json_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase208(model_name)
