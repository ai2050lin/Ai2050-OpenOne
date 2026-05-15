"""
Phase 173: Constraint Propagation Tensor (约束传播张量)
=========================================================

★★★ PARADIGM SHIFT: From "hidden-state dynamics" → "constraint dynamics"! ★★★

User's core insight (ALL CORRECT):

1. Hidden state ≠ semantic state. It's a "constraint cache" for next-token prediction.
   h_l is not "meaning" but "which constraints are satisfied/pending".

2. Attention α_ij is just "who can influence whom" — dynamic adjacency matrix.
   The real information is in V_j, not in α_ij.

3. Late-layer "curvature" = probability projection curvature, not semantic curvature.
   The model must compress h → logits in the final layers.

4. ★★★ The correct object is ∂C_{l+1}/∂C_l, not ∂h_{l+1}/∂h_l ★★★
   Where C(h) is the constraint signal extracted from minimal pairs.

5. Language = "constraint propagation dynamical system":
   - Syntax = integrable propagation constraints
   - Reasoning = constraint closure
   - Translation = constraint isomorphism between coordinate systems
   - Knowledge = propagable topology structure

★★★ FIVE KEY EXPERIMENTS ★★★

Exp 1: ★★★ Number Constraint Propagation (30 pairs)
  - Minimal pairs: "The cat sleeps" vs "The cats sleep"
  - Measure constraint signal C_l(p) = h_l^+(p) - h_l^-(p) at each layer/position
  - KEY: How does singular/plural signal propagate from subject → verb?

Exp 2: ★★★ Gender Constraint Propagation (20 pairs)
  - Minimal pairs: "The woman said she" vs "The man said he"
  - Same measurements

Exp 3: ★★★ Constraint Closure Layer
  - At which layer does the model have enough information to correctly
    predict the constrained token (e.g., "sleeps" vs "sleep")?
  - Project h_l(verb_pos) through W_U at each layer → check logit ranking

Exp 4: ★★★ Cross-Position Constraint Transfer
  - How does constraint signal move from subject → intermediate → verb?
  - For long-distance pairs: which intermediate positions carry the constraint?

Exp 5: ★★ Constraint Signal in W_U Space
  - How much of C_l(p) is in the output projection space?
  - Early layers: constraint may be latent (outside W_U space)
  - Late layers: constraint must be in W_U space for correct prediction

Usage: python tests/glm5/phase173_constraint_propagation.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto" — same as model_demo_bf16.py)
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


# =====================================================================
# SENTENCE PAIR DEFINITIONS
# =====================================================================

# Number constraint: simple subject-verb agreement (30 pairs)
NUMBER_SIMPLE = [
    # (singular_sentence, plural_sentence)
    ("The cat sleeps", "The cats sleep"),
    ("The dog runs", "The dogs run"),
    ("The bird flies", "The birds fly"),
    ("The tree grows", "The trees grow"),
    ("The river flows", "The rivers flow"),
    ("The house stands", "The houses stand"),
    ("The car moves", "The cars move"),
    ("The star shines", "The stars shine"),
    ("The bell rings", "The bells ring"),
    ("The lamp glows", "The lamps glow"),
    ("The door opens", "The doors open"),
    ("The book closes", "The books close"),
    ("The light flickers", "The lights flicker"),
    ("The flower blooms", "The flowers bloom"),
    ("The wind blows", "The winds blow"),
    ("The rain falls", "The rains fall"),
    ("The snow melts", "The snows melt"),
    ("The fire burns", "The fires burn"),
    ("The stone rolls", "The stones roll"),
    ("The leaf drops", "The leaves drop"),
    ("The cloud drifts", "The clouds drift"),
    ("The wave crashes", "The waves crash"),
    ("The fish swims", "The fish swim"),
    ("The horse gallops", "The horses gallop"),
    ("The child plays", "The children play"),
    ("The man walks", "The men walk"),
    ("The woman reads", "The women read"),
    ("The student studies", "The students study"),
    ("The teacher writes", "The teachers write"),
    ("The doctor works", "The doctors work"),
]

# Number constraint: long-distance subject-verb agreement (15 pairs)
# Subject ... intervening noun ... verb
NUMBER_LONGDIST = [
    ("The cat that the dogs chased sleeps", "The cats that the dogs chased sleep"),
    ("The dog that the cats saw runs", "The dogs that the cats saw run"),
    ("The bird that the cats caught flies", "The birds that the cats caught fly"),
    ("The man who the women saw walks", "The men who the women saw walk"),
    ("The child that the dogs scared plays", "The children that the dogs scared play"),
    ("The tree that the workers cut grows", "The trees that the workers cut grow"),
    ("The star that the clouds hid shines", "The stars that the clouds hid shine"),
    ("The car that the drivers bought moves", "The cars that the drivers bought move"),
    ("The lamp that the students used glows", "The lamps that the students used glow"),
    ("The door that the workers fixed opens", "The doors that the workers fixed open"),
    ("The book that the students read closes", "The books that the students read close"),
    ("The cat that the dog chased runs", "The cats that the dog chased run"),
    ("The dog that the man saw walks", "The dogs that the man saw walk"),
    ("The bird that the cat caught flies", "The birds that the cat caught fly"),
    ("The fish that the bear caught swims", "The fish that the bear caught swim"),
]

# Gender constraint: noun → pronoun (20 pairs)
GENDER_PAIRS = [
    ("The woman said she", "The man said he"),
    ("The girl said she", "The boy said he"),
    ("The mother told her", "The father told his"),
    ("The sister called her", "The brother called his"),
    ("The queen ruled her", "The king ruled his"),
    ("The aunt visited her", "The uncle visited his"),
    ("The lady helped her", "The lord helped his"),
    ("The wife called her", "The husband called his"),
    ("The woman walked and she", "The man walked and he"),
    ("The girl studied and she", "The boy studied and he"),
    ("The mother smiled and she", "The father smiled and he"),
    ("The sister laughed and she", "The brother laughed and he"),
    ("The queen spoke and she", "The king spoke and he"),
    ("The aunt arrived and she", "The uncle arrived and he"),
    ("The lady left and she", "The lord left and he"),
    ("The woman opened the door and she", "The man opened the door and he"),
    ("The girl read the book and she", "The boy read the book and he"),
    ("The mother cooked the meal and she", "The father cooked the meal and he"),
    ("The sister wrote the letter and she", "The brother wrote the letter and he"),
    ("The queen visited the city and she", "The king visited the city and he"),
]


# =====================================================================
# CORE FUNCTIONS
# =====================================================================

def extract_hidden_states(model, tokenizer, device, sentence, n_layers):
    """Extract hidden states at all layers for all token positions."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    # hidden_states: tuple of (1, seq_len, d_model), length = n_layers + 1
    hs = out.hidden_states
    tokens = [tokenizer.decode([t]) for t in input_ids[0].cpu().numpy()]

    # Convert to numpy: {layer_idx: ndarray[seq_len, d_model]}
    hidden_np = {}
    for li in range(len(hs)):
        hidden_np[li] = hs[li][0].float().cpu().numpy()  # [seq_len, d_model]

    return hidden_np, tokens, input_ids[0].cpu().numpy()


def find_aligned_positions(tokens_a, tokens_b):
    """
    Find aligned positions between two token lists.
    Returns: dict with 'shared' positions (same token), 'diff' positions (different token)
    """
    n = min(len(tokens_a), len(tokens_b))
    shared = []
    diff = []
    for i in range(n):
        if tokens_a[i].strip() == tokens_b[i].strip():
            shared.append(i)
        else:
            diff.append(i)
    return {"shared": shared, "diff": diff, "n_aligned": n}


def compute_constraint_signal(hs_a, hs_b, n_layers_plus1):
    """
    Compute constraint signal C_l(p) = h_l^+(p) - h_l^-(p) at each layer and position.

    Returns:
        css: {layer: {pos: float}} — Constraint Signal Strength ||C_l(p)||
        signals: {layer: ndarray[seq_len, d_model]} — raw constraint signals
    """
    css = {}
    signals = {}
    for li in range(n_layers_plus1):
        h_a = hs_a[li]  # [seq_len_a, d_model]
        h_b = hs_b[li]  # [seq_len_b, d_model]
        n = min(h_a.shape[0], h_b.shape[0])
        delta = h_a[:n] - h_b[:n]  # [n, d_model]
        signals[li] = delta
        css[li] = {}
        for p in range(n):
            css[li][p] = float(np.linalg.norm(delta[p]))
    return css, signals


def compute_constraint_propagation_ratio(css, n_layers_plus1, positions=None):
    """
    Compute Constraint Propagation Ratio: CSS(l+1)/CSS(l)
    CPR > 1: constraint amplified
    CPR < 1: constraint decayed
    CPR ≈ 1: constraint preserved
    """
    if positions is None:
        # Use all positions that exist in all layers
        positions = list(css[0].keys())

    cpr = {}
    for li in range(n_layers_plus1 - 1):
        cpr[li] = {}
        for p in positions:
            css_l = css[li].get(p, 0)
            css_lp1 = css[li + 1].get(p, 0)
            if css_l > 1e-10:
                cpr[li][p] = css_lp1 / css_l
            else:
                cpr[li][p] = 0.0
    return cpr


def compute_constraint_direction_stability(signals, n_layers_plus1, positions=None):
    """
    Compute CDS: cos(C_{l+1}(p), C_l(p)) — constraint direction stability.
    CDS ≈ 1: direction preserved
    CDS ≈ 0: direction rotated 90°
    CDS < 0: direction reversed
    """
    if positions is None:
        n = signals[0].shape[0]
        positions = list(range(n))

    cds = {}
    for li in range(n_layers_plus1 - 1):
        cds[li] = {}
        for p in positions:
            c_l = signals[li][p]
            c_lp1 = signals[li + 1][p]
            n_l = np.linalg.norm(c_l)
            n_lp1 = np.linalg.norm(c_lp1)
            if n_l > 1e-10 and n_lp1 > 1e-10:
                cds[li][p] = float(np.dot(c_l, c_lp1) / (n_l * n_lp1))
            else:
                cds[li][p] = 0.0
    return cds


def compute_cross_position_transfer(signals, n_layers_plus1, source_pos, target_pos):
    """
    Compute cross-position constraint transfer:
    cos(C_l(source_pos), C_{l+1}(target_pos))

    This measures how the constraint signal at source_pos at layer l
    relates to the constraint signal at target_pos at layer l+1.
    """
    transfer = {}
    for li in range(n_layers_plus1 - 1):
        c_src = signals[li][source_pos]
        c_tgt = signals[li + 1][target_pos]
        n_src = np.linalg.norm(c_src)
        n_tgt = np.linalg.norm(c_tgt)
        if n_src > 1e-10 and n_tgt > 1e-10:
            transfer[li] = float(np.dot(c_src, c_tgt) / (n_src * n_tgt))
        else:
            transfer[li] = 0.0
    return transfer


def compute_constraint_closure(hs, tokenizer, W_U, verb_tokens, n_layers_plus1, verb_pos):
    """
    Compute constraint closure: at which layer does the model have enough
    information to correctly predict the constrained token?

    For each layer, project h_l(verb_pos) through W_U and check if
    the correct verb form is ranked higher than the incorrect one.

    Returns:
        closure_layer: first layer where correct > incorrect
        logit_diff_by_layer: {layer: logit_correct - logit_incorrect}
    """
    # Get token IDs for verb forms
    correct_id = verb_tokens["correct_id"]
    incorrect_id = verb_tokens["incorrect_id"]

    logit_diff = {}
    for li in range(n_layers_plus1):
        h = hs[li][verb_pos]  # [d_model]
        logits = W_U @ h  # [vocab_size]
        diff = float(logits[correct_id] - logits[incorrect_id])
        logit_diff[li] = diff

    # Find closure layer
    closure_layer = None
    for li in range(n_layers_plus1):
        if logit_diff[li] > 0:
            closure_layer = li
            break

    return closure_layer, logit_diff


def compute_constraint_in_wu_space(signal, U_wut):
    """
    Compute how much of the constraint signal C_l(p) is in the W_U row space.
    Uses precomputed SVD result U_wut.
    """
    norm = np.linalg.norm(signal)
    if norm < 1e-10:
        return 0.0

    proj_coeffs = U_wut.T @ signal  # [k]
    proj_energy = np.sum(proj_coeffs ** 2)
    ratio = min(proj_energy / max(norm ** 2, 1e-20), 1.0)

    return ratio


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def run_experiment(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 173: Constraint Propagation Tensor — {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    print(f"  W_U: shape={W_U.shape}", flush=True)

    # Precompute W_U SVD for W_U space analysis (avoid repeated SVD)
    print("  Precomputing W_U SVD...", flush=True)
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)
    k_svd = min(100, min(W_U_T.shape) - 2)
    U_wut, _, _ = svds(W_U_T, k=k_svd)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k]
    print(f"  W_U SVD done: U_wut shape={U_wut.shape}", flush=True)

    n_layers_plus1 = info.n_layers + 1  # +1 for embedding layer

    # Results storage
    results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "number_simple": {},
        "number_longdist": {},
        "gender": {},
        "constraint_closure": {},
        "summary": {},
    }

    # =====================================================================
    # Exp 1: Number Constraint (Simple)
    # =====================================================================
    print(f"\n--- Exp 1: Number Constraint (Simple) — {len(NUMBER_SIMPLE)} pairs ---", flush=True)

    number_simple_agg = defaultdict(list)
    number_simple_detail = []

    for idx, (sing, plur) in enumerate(NUMBER_SIMPLE):
        if idx % 10 == 0:
            print(f"  Processing pair {idx+1}/{len(NUMBER_SIMPLE)}...", flush=True)

        # Extract hidden states
        hs_sing, toks_sing, ids_sing = extract_hidden_states(
            model, tokenizer, device, sing, info.n_layers)
        hs_plur, toks_plur, ids_plur = extract_hidden_states(
            model, tokenizer, device, plur, info.n_layers)

        # Find aligned positions
        alignment = find_aligned_positions(toks_sing, toks_plur)
        n_aligned = alignment["n_aligned"]

        # Compute constraint signal
        css, signals = compute_constraint_signal(hs_sing, hs_plur, n_layers_plus1)

        # Get verb token IDs for closure measurement
        # The verb is typically at the last or second-to-last position
        verb_sing = sing.split()[-1]  # e.g., "sleeps"
        verb_plur = plur.split()[-1]  # e.g., "sleep"
        verb_sing_ids = tokenizer.encode(verb_sing, add_special_tokens=False)
        verb_plur_ids = tokenizer.encode(verb_plur, add_special_tokens=False)

        # For constraint closure: we look at the LAST token position
        # (which should be the verb or near it)
        last_pos = n_aligned - 1
        second_last = n_aligned - 2

        # Try to identify which position is the verb
        verb_pos = last_pos  # default
        for p in alignment["diff"]:
            # Different positions are likely subject and verb
            if p > 0:  # skip BOS
                verb_pos_candidate = p

        # Constraint closure
        closure_data = {}
        if verb_sing_ids and verb_plur_ids:
            # Use singular sentence as reference (correct verb = singular)
            verb_tokens = {
                "correct_id": verb_sing_ids[0],
                "incorrect_id": verb_plur_ids[0],
            }
            closure_layer, logit_diff = compute_constraint_closure(
                hs_sing, tokenizer, W_U, verb_tokens, n_layers_plus1, verb_pos)
            closure_data = {
                "closure_layer": closure_layer,
                "verb_pos": verb_pos,
                "logit_diff_early": logit_diff.get(1, 0),
                "logit_diff_mid": logit_diff.get(info.n_layers // 2, 0),
                "logit_diff_late": logit_diff.get(info.n_layers - 1, 0),
            }

        # Aggregate CSS at subject and verb positions
        # Subject is typically the first "diff" position, verb is the second
        subj_pos = None
        verb_p = None
        diff_positions = sorted(alignment["diff"])
        if len(diff_positions) >= 2:
            subj_pos = diff_positions[0]
            verb_p = diff_positions[-1]
        elif len(diff_positions) == 1:
            verb_p = diff_positions[0]

        # CSS evolution at key positions
        subj_css_evolution = []
        verb_css_evolution = []
        shared_css_evolution = []

        for li in range(n_layers_plus1):
            if subj_pos is not None and subj_pos in css[li]:
                subj_css_evolution.append(css[li][subj_pos])
            if verb_p is not None and verb_p in css[li]:
                verb_css_evolution.append(css[li][verb_p])
            # Average CSS at shared positions
            shared_css = [css[li][p] for p in alignment["shared"] if p in css[li]]
            shared_css_evolution.append(np.mean(shared_css) if shared_css else 0)

        # CPR and CDS at verb position
        cpr_verb = []
        cds_verb = []
        for li in range(n_layers_plus1 - 1):
            if verb_p is not None and verb_p in css[li] and verb_p in css[li + 1]:
                if css[li][verb_p] > 1e-10:
                    cpr_verb.append(css[li + 1][verb_p] / css[li][verb_p])
                else:
                    cpr_verb.append(0)

        # Constraint signal in W_U space (at verb position, selected layers)
        wu_ratio = {}
        for li in [0, info.n_layers // 4, info.n_layers // 2,
                   3 * info.n_layers // 4, info.n_layers - 1]:
            if li < n_layers_plus1 and verb_p is not None:
                sig = signals[li][verb_p] if verb_p < signals[li].shape[0] else None
                if sig is not None:
                    wu_ratio[li] = round(compute_constraint_in_wu_space(sig, U_wut), 4)

        # Store per-pair detail
        pair_result = {
            "singular": sing, "plural": plur,
            "n_aligned": n_aligned,
            "subj_pos": subj_pos, "verb_pos": verb_p,
            "shared_positions": len(alignment["shared"]),
            "diff_positions": len(alignment["diff"]),
            "subj_css_evolution": [round(x, 4) for x in subj_css_evolution],
            "verb_css_evolution": [round(x, 4) for x in verb_css_evolution],
            "shared_css_mean_evolution": [round(x, 4) for x in shared_css_evolution],
            "closure": closure_data,
            "wu_ratio_verb": wu_ratio,
        }
        number_simple_detail.append(pair_result)

        # Aggregate
        number_simple_agg["subj_css_early"].append(
            subj_css_evolution[1] if len(subj_css_evolution) > 1 else 0)
        number_simple_agg["subj_css_late"].append(
            subj_css_evolution[-1] if len(subj_css_evolution) > 0 else 0)
        number_simple_agg["verb_css_early"].append(
            verb_css_evolution[1] if len(verb_css_evolution) > 1 else 0)
        number_simple_agg["verb_css_late"].append(
            verb_css_evolution[-1] if len(verb_css_evolution) > 0 else 0)
        number_simple_agg["shared_css_early"].append(
            shared_css_evolution[1] if len(shared_css_evolution) > 1 else 0)
        number_simple_agg["shared_css_late"].append(
            shared_css_evolution[-1] if len(shared_css_evolution) > 0 else 0)
        if closure_data.get("closure_layer") is not None:
            number_simple_agg["closure_layers"].append(closure_data["closure_layer"])

    # Summary for number simple
    results["number_simple"] = {
        "n_pairs": len(NUMBER_SIMPLE),
        "avg_subj_css_early": round(np.mean(number_simple_agg["subj_css_early"]), 4),
        "avg_subj_css_late": round(np.mean(number_simple_agg["subj_css_late"]), 4),
        "avg_verb_css_early": round(np.mean(number_simple_agg["verb_css_early"]), 4),
        "avg_verb_css_late": round(np.mean(number_simple_agg["verb_css_late"]), 4),
        "avg_shared_css_early": round(np.mean(number_simple_agg["shared_css_early"]), 4),
        "avg_shared_css_late": round(np.mean(number_simple_agg["shared_css_late"]), 4),
        "avg_closure_layer": round(np.mean(number_simple_agg["closure_layers"]), 2)
            if number_simple_agg["closure_layers"] else None,
        "closure_rate": round(len(number_simple_agg["closure_layers"]) / max(len(NUMBER_SIMPLE), 1), 4),
        "detail": number_simple_detail[:5],  # Save first 5 for reference
    }

    # Free memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # Exp 2: Number Constraint (Long-Distance)
    # =====================================================================
    print(f"\n--- Exp 2: Number Constraint (Long-Distance) — {len(NUMBER_LONGDIST)} pairs ---", flush=True)

    longdist_agg = defaultdict(list)
    longdist_detail = []

    for idx, (sing, plur) in enumerate(NUMBER_LONGDIST):
        if idx % 5 == 0:
            print(f"  Processing pair {idx+1}/{len(NUMBER_LONGDIST)}...", flush=True)

        hs_sing, toks_sing, ids_sing = extract_hidden_states(
            model, tokenizer, device, sing, info.n_layers)
        hs_plur, toks_plur, ids_plur = extract_hidden_states(
            model, tokenizer, device, plur, info.n_layers)

        alignment = find_aligned_positions(toks_sing, toks_plur)
        n_aligned = alignment["n_aligned"]
        css, signals = compute_constraint_signal(hs_sing, hs_plur, n_layers_plus1)

        # Identify positions: subject, verb, intervening nouns
        # For "The cat that the dogs chased sleeps":
        #   pos 0: The, pos 1: cat (SUBJECT), pos 2: that, pos 3: the,
        #   pos 4: dogs (INTERVENING), pos 5: chased, pos 6: sleeps (VERB)
        diff_positions = sorted(alignment["diff"])
        subj_pos = diff_positions[0] if diff_positions else None
        verb_pos = diff_positions[-1] if len(diff_positions) > 1 else None

        # Constraint CSS evolution at each position
        pos_css_by_layer = defaultdict(list)  # {pos: [css_l0, css_l1, ...]}
        for li in range(n_layers_plus1):
            for p in range(n_aligned):
                if p in css[li]:
                    pos_css_by_layer[p].append(css[li][p])

        # Cross-position transfer: subj → verb
        subj_verb_transfer = {}
        if subj_pos is not None and verb_pos is not None:
            subj_verb_transfer = compute_cross_position_transfer(
                signals, n_layers_plus1, subj_pos, verb_pos)

        # Also track subj → intermediate positions
        intermediate_transfers = {}
        if subj_pos is not None:
            for p in alignment["shared"]:
                if p != subj_pos and p != verb_pos:
                    t = compute_cross_position_transfer(
                        signals, n_layers_plus1, subj_pos, p)
                    # Take average across layers
                    if t:
                        avg_t = np.mean(list(t.values()))
                        if abs(avg_t) > 0.01:  # Only store notable transfers
                            intermediate_transfers[p] = round(avg_t, 4)

        # Constraint closure
        verb_sing = sing.split()[-1]
        verb_plur = plur.split()[-1]
        verb_sing_ids = tokenizer.encode(verb_sing, add_special_tokens=False)
        verb_plur_ids = tokenizer.encode(verb_plur, add_special_tokens=False)

        closure_data = {}
        if verb_sing_ids and verb_plur_ids and verb_pos is not None:
            verb_tokens = {"correct_id": verb_sing_ids[0], "incorrect_id": verb_plur_ids[0]}
            closure_layer, logit_diff = compute_constraint_closure(
                hs_sing, tokenizer, W_U, verb_tokens, n_layers_plus1, verb_pos)
            closure_data = {
                "closure_layer": closure_layer,
                "logit_diff_early": logit_diff.get(1, 0),
                "logit_diff_mid": logit_diff.get(info.n_layers // 2, 0),
                "logit_diff_late": logit_diff.get(info.n_layers - 1, 0),
            }

        # W_U space ratio at verb
        wu_ratio_verb = {}
        for li in [0, info.n_layers // 2, info.n_layers - 1]:
            if li < n_layers_plus1 and verb_pos is not None and verb_pos < signals[li].shape[0]:
                sig = signals[li][verb_pos]
                wu_ratio_verb[li] = round(compute_constraint_in_wu_space(sig, U_wut), 4)

        # CSS at each position for key layers (averaged)
        pos_css_summary = {}
        for li_name, li in [("early", 1), ("mid", info.n_layers // 2), ("late", info.n_layers - 1)]:
            if li < n_layers_plus1:
                pos_css_summary[li_name] = {
                    str(p): round(css[li].get(p, 0), 4) for p in range(min(n_aligned, 8))
                }

        pair_result = {
            "singular": sing, "plural": plur,
            "n_aligned": n_aligned,
            "subj_pos": subj_pos, "verb_pos": verb_pos,
            "diff_positions": diff_positions,
            "pos_css_summary": pos_css_summary,
            "subj_verb_transfer_avg": round(np.mean(list(subj_verb_transfer.values())), 4)
                if subj_verb_transfer else 0,
            "intermediate_transfers": intermediate_transfers,
            "closure": closure_data,
            "wu_ratio_verb": wu_ratio_verb,
        }
        longdist_detail.append(pair_result)

        # Aggregate
        if subj_pos is not None:
            longdist_agg["subj_css_early"].append(css.get(1, {}).get(subj_pos, 0))
            longdist_agg["subj_css_late"].append(css.get(info.n_layers - 1, {}).get(subj_pos, 0))
        if verb_pos is not None:
            longdist_agg["verb_css_early"].append(css.get(1, {}).get(verb_pos, 0))
            longdist_agg["verb_css_late"].append(css.get(info.n_layers - 1, {}).get(verb_pos, 0))
        if subj_verb_transfer:
            longdist_agg["subj_verb_transfer"].append(
                np.mean(list(subj_verb_transfer.values())))
        if closure_data.get("closure_layer") is not None:
            longdist_agg["closure_layers"].append(closure_data["closure_layer"])

    results["number_longdist"] = {
        "n_pairs": len(NUMBER_LONGDIST),
        "avg_subj_css_early": round(np.mean(longdist_agg["subj_css_early"]), 4)
            if longdist_agg["subj_css_early"] else 0,
        "avg_subj_css_late": round(np.mean(longdist_agg["subj_css_late"]), 4)
            if longdist_agg["subj_css_late"] else 0,
        "avg_verb_css_early": round(np.mean(longdist_agg["verb_css_early"]), 4)
            if longdist_agg["verb_css_early"] else 0,
        "avg_verb_css_late": round(np.mean(longdist_agg["verb_css_late"]), 4)
            if longdist_agg["verb_css_late"] else 0,
        "avg_subj_verb_transfer": round(np.mean(longdist_agg["subj_verb_transfer"]), 4)
            if longdist_agg["subj_verb_transfer"] else 0,
        "avg_closure_layer": round(np.mean(longdist_agg["closure_layers"]), 2)
            if longdist_agg["closure_layers"] else None,
        "closure_rate": round(len(longdist_agg["closure_layers"]) / max(len(NUMBER_LONGDIST), 1), 4),
        "detail": longdist_detail[:5],
    }

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # Exp 3: Gender Constraint
    # =====================================================================
    print(f"\n--- Exp 3: Gender Constraint — {len(GENDER_PAIRS)} pairs ---", flush=True)

    gender_agg = defaultdict(list)
    gender_detail = []

    for idx, (fem, masc) in enumerate(GENDER_PAIRS):
        if idx % 5 == 0:
            print(f"  Processing pair {idx+1}/{len(GENDER_PAIRS)}...", flush=True)

        hs_fem, toks_fem, ids_fem = extract_hidden_states(
            model, tokenizer, device, fem, info.n_layers)
        hs_masc, toks_masc, ids_masc = extract_hidden_states(
            model, tokenizer, device, masc, info.n_layers)

        alignment = find_aligned_positions(toks_fem, toks_masc)
        n_aligned = alignment["n_aligned"]
        css, signals = compute_constraint_signal(hs_fem, hs_masc, n_layers_plus1)

        diff_positions = sorted(alignment["diff"])
        subj_pos = diff_positions[0] if diff_positions else None
        pron_pos = diff_positions[-1] if len(diff_positions) > 1 else None

        # CSS evolution
        subj_css = []
        pron_css = []
        for li in range(n_layers_plus1):
            if subj_pos is not None:
                subj_css.append(css[li].get(subj_pos, 0))
            if pron_pos is not None:
                pron_css.append(css[li].get(pron_pos, 0))

        # Cross-position: subj → pron
        subj_pron_transfer = {}
        if subj_pos is not None and pron_pos is not None:
            subj_pron_transfer = compute_cross_position_transfer(
                signals, n_layers_plus1, subj_pos, pron_pos)

        # W_U space ratio at pronoun
        wu_ratio_pron = {}
        for li in [0, info.n_layers // 2, info.n_layers - 1]:
            if li < n_layers_plus1 and pron_pos is not None and pron_pos < signals[li].shape[0]:
                sig = signals[li][pron_pos]
                wu_ratio_pron[li] = round(compute_constraint_in_wu_space(sig, U_wut), 4)

        pair_result = {
            "feminine": fem, "masculine": masc,
            "n_aligned": n_aligned,
            "subj_pos": subj_pos, "pron_pos": pron_pos,
            "subj_css_early": round(subj_css[1], 4) if len(subj_css) > 1 else 0,
            "subj_css_late": round(subj_css[-1], 4) if subj_css else 0,
            "pron_css_early": round(pron_css[1], 4) if len(pron_css) > 1 else 0,
            "pron_css_late": round(pron_css[-1], 4) if pron_css else 0,
            "subj_pron_transfer_avg": round(np.mean(list(subj_pron_transfer.values())), 4)
                if subj_pron_transfer else 0,
            "wu_ratio_pron": wu_ratio_pron,
        }
        gender_detail.append(pair_result)

        gender_agg["subj_css_early"].append(subj_css[1] if len(subj_css) > 1 else 0)
        gender_agg["subj_css_late"].append(subj_css[-1] if subj_css else 0)
        gender_agg["pron_css_early"].append(pron_css[1] if len(pron_css) > 1 else 0)
        gender_agg["pron_css_late"].append(pron_css[-1] if pron_css else 0)
        if subj_pron_transfer:
            gender_agg["subj_pron_transfer"].append(np.mean(list(subj_pron_transfer.values())))

    results["gender"] = {
        "n_pairs": len(GENDER_PAIRS),
        "avg_subj_css_early": round(np.mean(gender_agg["subj_css_early"]), 4),
        "avg_subj_css_late": round(np.mean(gender_agg["subj_css_late"]), 4),
        "avg_pron_css_early": round(np.mean(gender_agg["pron_css_early"]), 4),
        "avg_pron_css_late": round(np.mean(gender_agg["pron_css_late"]), 4),
        "avg_subj_pron_transfer": round(np.mean(gender_agg["subj_pron_transfer"]), 4)
            if gender_agg["subj_pron_transfer"] else 0,
        "detail": gender_detail[:5],
    }

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # Exp 4: Detailed Constraint Propagation Analysis (Cross-Layer CSS)
    # =====================================================================
    print(f"\n--- Exp 4: Detailed CSS Evolution (10 representative pairs) ---", flush=True)

    # Use first 5 simple + 5 long-distance pairs for detailed analysis
    test_pairs = [("simple", s, p) for s, p in NUMBER_SIMPLE[:5]] + \
                 [("longdist", s, p) for s, p in NUMBER_LONGDIST[:5]]

    css_evolution_detail = []

    for pair_type, sing, plur in test_pairs:
        hs_sing, toks_sing, _ = extract_hidden_states(
            model, tokenizer, device, sing, info.n_layers)
        hs_plur, toks_plur, _ = extract_hidden_states(
            model, tokenizer, device, plur, info.n_layers)

        alignment = find_aligned_positions(toks_sing, toks_plur)
        n_aligned = alignment["n_aligned"]
        css, signals = compute_constraint_signal(hs_sing, hs_plur, n_layers_plus1)

        # Sample layers for compact output
        sample_layers = list(range(0, n_layers_plus1, max(1, info.n_layers // 8)))
        if info.n_layers - 1 not in sample_layers:
            sample_layers.append(info.n_layers - 1)

        # CSS at each position for sample layers
        pos_css = {}
        for li in sample_layers:
            pos_css[f"L{li}"] = {str(p): round(css[li].get(p, 0), 4)
                                  for p in range(min(n_aligned, 10))}

        # CPR at verb position for sample layers
        diff_positions = sorted(alignment["diff"])
        verb_pos = diff_positions[-1] if diff_positions else None
        cpr_verb = {}
        for li in sample_layers:
            if li + 1 >= n_layers_plus1:
                continue
            if verb_pos and verb_pos in css[li] and verb_pos in css[li + 1]:
                if css[li][verb_pos] > 1e-10:
                    cpr_verb[f"L{li}_L{li+1}"] = round(
                        css[li + 1][verb_pos] / css[li][verb_pos], 4)

        # CDS at verb position
        cds_verb = {}
        for li in sample_layers:
            if li + 1 >= n_layers_plus1:
                continue
            if verb_pos and verb_pos < signals[li].shape[0] and verb_pos < signals[li + 1].shape[0]:
                c_l = signals[li][verb_pos]
                c_lp1 = signals[li + 1][verb_pos]
                n_l = np.linalg.norm(c_l)
                n_lp1 = np.linalg.norm(c_lp1)
                if n_l > 1e-10 and n_lp1 > 1e-10:
                    cds_verb[f"L{li}_L{li+1}"] = round(
                        float(np.dot(c_l, c_lp1) / (n_l * n_lp1)), 4)

        css_evolution_detail.append({
            "type": pair_type,
            "sentence": sing,
            "n_aligned": n_aligned,
            "verb_pos": verb_pos,
            "pos_css": pos_css,
            "cpr_verb": cpr_verb,
            "cds_verb": cds_verb,
        })

    results["css_evolution_detail"] = css_evolution_detail

    # =====================================================================
    # Exp 5: Constraint Signal in W_U Space (Aggregated)
    # =====================================================================
    print(f"\n--- Exp 5: Constraint Signal in W_U Space ---", flush=True)

    # Use 10 pairs for W_U space analysis
    wu_analysis = {"number_simple": {}, "number_longdist": {}, "gender": {}}

    for pair_type, pairs, label in [
        ("number_simple", NUMBER_SIMPLE[:10], "Number Simple"),
        ("number_longdist", NUMBER_LONGDIST[:10], "Number Long-Dist"),
        ("gender", [(f, m) for f, m in GENDER_PAIRS[:10]], "Gender"),
    ]:
        wu_ratios_by_layer = defaultdict(list)

        for a, b in pairs:
            hs_a, toks_a, _ = extract_hidden_states(model, tokenizer, device, a, info.n_layers)
            hs_b, toks_b, _ = extract_hidden_states(model, tokenizer, device, b, info.n_layers)

            _, signals = compute_constraint_signal(hs_a, hs_b, n_layers_plus1)

            alignment = find_aligned_positions(toks_a, toks_b)
            diff_positions = sorted(alignment["diff"])
            target_pos = diff_positions[-1] if diff_positions else 0

            for li in range(n_layers_plus1):
                if target_pos < signals[li].shape[0]:
                    sig = signals[li][target_pos]
                    ratio = compute_constraint_in_wu_space(sig, U_wut)
                    wu_ratios_by_layer[li].append(ratio)

        # Average across pairs
        avg_wu = {}
        sample_layers = list(range(0, n_layers_plus1, max(1, info.n_layers // 6)))
        if info.n_layers - 1 not in sample_layers:
            sample_layers.append(info.n_layers - 1)
        for li in sample_layers:
            if li in wu_ratios_by_layer:
                avg_wu[f"L{li}"] = round(np.mean(wu_ratios_by_layer[li]), 4)

        wu_analysis[pair_type] = avg_wu
        print(f"  {label}: {avg_wu}")

    results["wu_space_analysis"] = wu_analysis

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — {model_name}")
    print(f"{'='*70}")

    # Number Simple
    ns = results["number_simple"]
    print(f"\n[Number Simple]")
    print(f"  Subj CSS: early={ns['avg_subj_css_early']:.4f}, late={ns['avg_subj_css_late']:.4f}, "
          f"ratio={ns['avg_subj_css_late']/max(ns['avg_subj_css_early'],1e-10):.2f}")
    print(f"  Verb CSS: early={ns['avg_verb_css_early']:.4f}, late={ns['avg_verb_css_late']:.4f}, "
          f"ratio={ns['avg_verb_css_late']/max(ns['avg_verb_css_early'],1e-10):.2f}")
    print(f"  Shared CSS: early={ns['avg_shared_css_early']:.4f}, late={ns['avg_shared_css_late']:.4f}")
    print(f"  Closure layer: {ns['avg_closure_layer']}, rate={ns['closure_rate']:.2f}")

    # Number Long-Distance
    nl = results["number_longdist"]
    print(f"\n[Number Long-Distance]")
    print(f"  Subj CSS: early={nl['avg_subj_css_early']:.4f}, late={nl['avg_subj_css_late']:.4f}")
    print(f"  Verb CSS: early={nl['avg_verb_css_early']:.4f}, late={nl['avg_verb_css_late']:.4f}")
    print(f"  Subj→Verb transfer: {nl['avg_subj_verb_transfer']:.4f}")
    print(f"  Closure layer: {nl['avg_closure_layer']}, rate={nl['closure_rate']:.2f}")

    # Gender
    g = results["gender"]
    print(f"\n[Gender]")
    print(f"  Subj CSS: early={g['avg_subj_css_early']:.4f}, late={g['avg_subj_css_late']:.4f}")
    print(f"  Pron CSS: early={g['avg_pron_css_early']:.4f}, late={g['avg_pron_css_late']:.4f}")
    print(f"  Subj→Pron transfer: {g['avg_subj_pron_transfer']:.4f}")

    # W_U Space
    print(f"\n[W_U Space Ratios]")
    for pair_type, ratios in wu_analysis.items():
        print(f"  {pair_type}: {ratios}")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase173_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    valid = ["qwen3", "glm4", "deepseek7b"]
    if model_name not in valid:
        print(f"Invalid model: {model_name}. Valid: {valid}")
        sys.exit(1)

    run_experiment(model_name)
    print(f"\nPhase 173 complete for {model_name}!")
