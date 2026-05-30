"""
Phase 298: Role Subspace Extraction & Causal Direction Test
============================================================
Goal: Extract role/frame subspaces via PCA, test causal effectiveness
of role directions via activation patching.

Key Questions:
1. What is the effective dimensionality of role increments?
2. What is the effective dimensionality of frame increments?
3. Are role and frame subspaces overlapping or orthogonal?
4. Does the extracted role direction have causal effect on model output?
5. How does this differ across Qwen3/GLM4/DS7B?

Measurements:
1. Role Increment PCA per layer → dimensionality, explained variance
2. Frame Increment PCA per layer → dimensionality, explained variance
3. Role vs Frame subspace overlap (principal angles)
4. Cross-token role direction consistency (leave-one-out)
5. Causal direction test (add role direction vs random control)

Usage:
  python tests/glm5/phase298_role_subspace_causal.py qwen3
  python tests/glm5/phase298_role_subspace_causal.py glm4
  python tests/glm5/phase298_role_subspace_causal.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase298_role_subspace")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# =====================================================================
# STIMULUS DESIGN — Reuse Phase 297 + new test sentences
# =====================================================================
def build_phase297_stimuli():
    """Same stimulus set as Phase 297 for subspace extraction."""
    stimuli = []
    adj_verb_tokens = {
        "open": {
            "adj": {
                "P1": ["the door is open", "the gate is open"],
                "P2": ["the door remains open", "the gate remains open"],
                "P3": ["the open door", "the open gate"],
                "P4": ["the shop seemed open", "the road seemed open"],
            },
            "verb": {
                "P1": ["they open the door", "they open the gate"],
                "P2": ["we open the door", "we open the gate"],
                "P3": ["the door will open", "the gate will open"],
                "P4": ["they began to open the shop", "they began to open the road"],
            },
        },
        "clear": {
            "adj": {
                "P1": ["the path is clear", "the road is clear"],
                "P2": ["the path remains clear", "the road remains clear"],
                "P3": ["the clear path", "the clear road"],
                "P4": ["the desk seemed clear", "the table seemed clear"],
            },
            "verb": {
                "P1": ["they clear the path", "they clear the road"],
                "P2": ["we clear the path", "we clear the road"],
                "P3": ["the path will clear", "the road will clear"],
                "P4": ["they began to clear the desk", "they began to clear the table"],
            },
        },
        "warm": {
            "adj": {
                "P1": ["the room is warm", "the house is warm"],
                "P2": ["the room remains warm", "the house remains warm"],
                "P3": ["the warm room", "the warm house"],
                "P4": ["the water seemed warm", "the food seemed warm"],
            },
            "verb": {
                "P1": ["they warm the room", "they warm the house"],
                "P2": ["we warm the room", "we warm the house"],
                "P3": ["the room will warm", "the house will warm"],
                "P4": ["they began to warm the water", "they began to warm the food"],
            },
        },
        "clean": {
            "adj": {
                "P1": ["the floor is clean", "the table is clean"],
                "P2": ["the floor remains clean", "the table remains clean"],
                "P3": ["the clean floor", "the clean table"],
                "P4": ["the room seemed clean", "the house seemed clean"],
            },
            "verb": {
                "P1": ["they clean the floor", "they clean the table"],
                "P2": ["we clean the floor", "we clean the table"],
                "P3": ["the floor will clean", "the table will clean"],
                "P4": ["they began to clean the room", "they began to clean the house"],
            },
        },
    }
    adj_noun_tokens = {
        "light": {
            "adj": {
                "P1": ["the bag is light", "the box is light"],
                "P2": ["the bag remains light", "the box remains light"],
                "P3": ["the light bag", "the light box"],
                "P4": ["the load seemed light", "the dress seemed light"],
            },
            "noun": {
                "P1": ["the light is bright", "the light is warm"],
                "P2": ["that light is bright", "that light is warm"],
                "P3": ["near the light", "by the light"],
                "P4": ["they saw the light", "they found the light"],
            },
        },
        "cold": {
            "adj": {
                "P1": ["the water is cold", "the wind is cold"],
                "P2": ["the water remains cold", "the wind remains cold"],
                "P3": ["the cold water", "the cold wind"],
                "P4": ["the room seemed cold", "the air seemed cold"],
            },
            "noun": {
                "P1": ["the cold is severe", "the cold is bitter"],
                "P2": ["that cold is severe", "that cold is bitter"],
                "P3": ["in the cold", "despite the cold"],
                "P4": ["they felt the cold", "they noticed the cold"],
            },
        },
    }
    noun_verb_tokens = {
        "fire": {
            "noun": {
                "P1": ["the fire is hot", "the fire is big"],
                "P2": ["that fire is hot", "that fire is big"],
                "P3": ["near the fire", "by the fire"],
                "P4": ["they saw the fire", "they started the fire"],
            },
            "verb": {
                "P1": ["they fire the gun", "they fire the worker"],
                "P2": ["they will fire the gun", "they will fire the worker"],
                "P3": ["the gun will fire", "the engine will fire"],
                "P4": ["they began to fire the gun", "they began to fire the worker"],
            },
        },
        "record": {
            "noun": {
                "P1": ["the record is old", "the record is broken"],
                "P2": ["that record is old", "that record is broken"],
                "P3": ["on the record", "for the record"],
                "P4": ["they broke the record", "they set the record"],
            },
            "verb": {
                "P1": ["they record music", "they record data"],
                "P2": ["they will record music", "they will record data"],
                "P3": ["the device will record", "the system will record"],
                "P4": ["they began to record music", "they began to record data"],
            },
        },
    }
    all_tokens = {}
    all_tokens.update(adj_verb_tokens)
    all_tokens.update(adj_noun_tokens)
    all_tokens.update(noun_verb_tokens)

    for token, roles in all_tokens.items():
        role_pair = "adj_verb" if token in adj_verb_tokens else (
            "adj_noun" if token in adj_noun_tokens else "noun_verb"
        )
        for role, pairs in roles.items():
            for pair_label, sentences in pairs.items():
                for sent in sentences:
                    stimuli.append({
                        "sentence": sent,
                        "target_word": token,
                        "token_label": token,
                        "role_label": role,
                        "pair_label": pair_label,
                        "role_pair": role_pair,
                        "group": "subspace",
                    })
    return stimuli


def build_causal_test_stimuli():
    """New sentences (not in Phase 297) for causal direction test."""
    stimuli = []
    # Same tokens as Phase 297, but with different content words
    # Each entry: (sentence, target_word, role, role_pair)
    test_pairs = [
        # adj-verb: open
        ("the window is open", "open", "adj", "adj_verb"),
        ("they open the window", "open", "verb", "adj_verb"),
        ("the market is open", "open", "adj", "adj_verb"),
        ("they open the market", "open", "verb", "adj_verb"),
        # adj-verb: clear
        ("the field is clear", "clear", "adj", "adj_verb"),
        ("they clear the field", "clear", "verb", "adj_verb"),
        # adj-verb: warm
        ("the meal is warm", "warm", "adj", "adj_verb"),
        ("they warm the meal", "warm", "verb", "adj_verb"),
        # adj-verb: clean
        ("the shirt is clean", "clean", "adj", "adj_verb"),
        ("they clean the shirt", "clean", "verb", "adj_verb"),
        # adj-noun: light
        ("the feather is light", "light", "adj", "adj_noun"),
        ("the light is on", "light", "noun", "adj_noun"),
        # adj-noun: cold
        ("the drink is cold", "cold", "adj", "adj_noun"),
        ("the cold is harsh", "cold", "noun", "adj_noun"),
        # noun-verb: fire
        ("the fire is bright", "fire", "noun", "noun_verb"),
        ("they fire the employee", "fire", "verb", "noun_verb"),
        # noun-verb: record
        ("the record is famous", "record", "noun", "noun_verb"),
        ("they record the song", "record", "verb", "noun_verb"),
    ]
    for sent, target, role, rp in test_pairs:
        stimuli.append({
            "sentence": sent,
            "target_word": target,
            "token_label": target,
            "role_label": role,
            "pair_label": "test",
            "role_pair": rp,
            "group": "causal_test",
        })
    return stimuli


# =====================================================================
# MODEL LOADING (bf16 + device_map="auto", flash attention)
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto, flash)...")

    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = None
    used_attn = "eager"
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl,
            )
            used_attn = attn_impl
            break
        except Exception as e:
            log(f"  attn={attn_impl} failed: {str(e)[:80]}")

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    gpu = torch.cuda.memory_allocated() / 1e9
    log(f"  Loaded with attn={used_attn}, GPU={gpu:.1f}GB")

    layers = get_layers(model)
    nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU layers: {len(gpu_l)}{' (' + str(gpu_l[0]) + '-' + str(gpu_l[-1]) + ')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' (' + str(cpu_l[0]) + '-' + str(cpu_l[-1]) + ')' if cpu_l else ''}")
    return model, tok


def _capture_single(model, tokenizer, sent, n_layers, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {}
    for li, h in enumerate(out.hidden_states):
        hs[li] = h.detach().cpu().float()
    return {
        "hidden": hs,
        "input_ids": inputs["input_ids"].cpu(),
        "tokens": [tokenizer.decode([t]).strip() for t in inputs["input_ids"][0].tolist()],
    }


def _find_token_pos(decoded_tokens, target):
    target_lower = target.lower()
    for i, t in enumerate(decoded_tokens):
        if t == target_lower:
            return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower:
            return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower:
                return i
    return None


def resolve_positions(stimuli, tokenizer):
    resolved = []
    skipped = 0
    for stim in stimuli:
        sent = stim["sentence"]
        target = stim["target_word"]
        toks = tokenizer.encode(sent, add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, target)
        if pos is not None:
            new_stim = dict(stim)
            new_stim["target_pos"] = pos
            new_stim["seq_len"] = len(toks)
            resolved.append(new_stim)
        else:
            skipped += 1
    if skipped > 0:
        log(f"  Skipped {skipped} stimuli with unresolved positions")
    return resolved


# =====================================================================
# MEASUREMENT 1: Role Increment PCA
# =====================================================================
def role_increment_pca(captures, stimuli, n_layers):
    """
    For each dual-role token, compute role increment (role2 - role1) for each pair.
    Stack all increments into a matrix and do PCA.
    Report: explained variance per PC, cumulative, effective dimensionality.
    """
    obs = defaultdict(list)
    for stim in stimuli:
        if stim.get("group") != "subspace":
            continue
        token = stim["token_label"]
        role = stim["role_label"]
        pair = stim["pair_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        # Compute cell means
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Compute role increments for each token, each pair
        increments = []
        inc_labels = []  # (token, pair, role_pair_type)
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            pairs_for_token = sorted(set(p for (t, r, p) in cell_means if t == token))
            for p in pairs_for_token:
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    delta = m2 - m1  # r2 - r1 (e.g., verb - adj)
                    increments.append(delta)
                    # Determine role pair type
                    rp = f"{r1}-{r2}"
                    inc_labels.append((token, p, rp))

        if len(increments) < 3:
            continue

        inc_matrix = np.array(increments)  # [n_increments, d_model]
        n_inc, d = inc_matrix.shape

        # PCA via SVD (center the data)
        inc_centered = inc_matrix - np.mean(inc_matrix, axis=0, keepdims=True)

        # Use SVD for numerical stability
        # For n < d, use economy SVD
        if n_inc < d:
            U, S, Vt = np.linalg.svd(inc_centered, full_matrices=False)
        else:
            U, S, Vt = np.linalg.svd(inc_centered, full_matrices=False)

        total_var = np.sum(S ** 2)
        explained_var = (S ** 2) / max(total_var, 1e-10)
        cumulative_var = np.cumsum(explained_var)

        # Effective dimensionality
        dim_50 = int(np.searchsorted(cumulative_var, 0.50)) + 1
        dim_80 = int(np.searchsorted(cumulative_var, 0.80)) + 1
        dim_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1

        # Top-1 direction
        top1_dir = Vt[0] if len(Vt) > 0 else np.zeros(d)
        top1_var = float(explained_var[0]) if len(explained_var) > 0 else 0.0

        results[li] = {
            "n_increments": n_inc,
            "total_variance": float(total_var),
            "explained_var_top10": [float(v) for v in explained_var[:10]],
            "cumulative_var_top10": [float(v) for v in cumulative_var[:10]],
            "dim_50": dim_50,
            "dim_80": dim_80,
            "dim_95": dim_95,
            "top1_var": top1_var,
            "top3_var": float(cumulative_var[2]) if len(cumulative_var) > 2 else 1.0,
            "top5_var": float(cumulative_var[4]) if len(cumulative_var) > 4 else 1.0,
            "top1_direction_norm": float(np.linalg.norm(top1_dir)),
            "mean_increment_norm": float(np.mean([np.linalg.norm(d) for d in increments])),
        }

    return results, inc_labels


# =====================================================================
# MEASUREMENT 2: Frame Increment PCA (within role)
# =====================================================================
def frame_increment_pca(captures, stimuli, n_layers):
    """
    Within each role, compute pair differences (P_j - P_mean) for each token.
    Stack all frame increments and do PCA.
    """
    obs = defaultdict(list)
    for stim in stimuli:
        if stim.get("group") != "subspace":
            continue
        token = stim["token_label"]
        role = stim["role_label"]
        pair = stim["pair_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    results = {}
    for li in range(n_layers + 1):
        # Compute cell means
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Compute frame increments within each role
        # For each (token, role), compute deviations from pair-mean
        frame_increments = []
        for (token, role, pair), mean_v in cell_means.items():
            # Compute mean across pairs for this (token, role)
            pair_means = []
            for p in ["P1", "P2", "P3", "P4"]:
                m = cell_means.get((token, role, p))
                if m is not None:
                    pair_means.append(m)
            if len(pair_means) < 2:
                continue
            pair_avg = np.mean(pair_means, axis=0)
            # Frame increment = deviation from pair average
            delta = mean_v - pair_avg
            frame_increments.append(delta)

        if len(frame_increments) < 3:
            continue

        inc_matrix = np.array(frame_increments)
        n_inc, d = inc_matrix.shape
        inc_centered = inc_matrix - np.mean(inc_matrix, axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(inc_centered, full_matrices=False)
        total_var = np.sum(S ** 2)
        explained_var = (S ** 2) / max(total_var, 1e-10)
        cumulative_var = np.cumsum(explained_var)

        dim_50 = int(np.searchsorted(cumulative_var, 0.50)) + 1
        dim_80 = int(np.searchsorted(cumulative_var, 0.80)) + 1
        dim_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1

        results[li] = {
            "n_increments": n_inc,
            "total_variance": float(total_var),
            "explained_var_top10": [float(v) for v in explained_var[:10]],
            "cumulative_var_top10": [float(v) for v in cumulative_var[:10]],
            "dim_50": dim_50,
            "dim_80": dim_80,
            "dim_95": dim_95,
            "top1_var": float(explained_var[0]) if len(explained_var) > 0 else 0.0,
            "top3_var": float(cumulative_var[2]) if len(cumulative_var) > 2 else 1.0,
            "top5_var": float(cumulative_var[4]) if len(cumulative_var) > 4 else 1.0,
        }

    return results


# =====================================================================
# MEASUREMENT 3: Role-Frame Subspace Overlap
# =====================================================================
def subspace_overlap(role_pca, frame_pca, captures, stimuli, n_layers):
    """
    Compute overlap between role and frame subspaces using principal angles.
    For each layer, extract top-k PCs from role and frame increments,
    then compute principal angles between the subspaces.
    """
    obs = defaultdict(list)
    for stim in stimuli:
        if stim.get("group") != "subspace":
            continue
        token = stim["token_label"]
        role = stim["role_label"]
        pair = stim["pair_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        if li not in role_pca or li not in frame_pca:
            continue

        # Recompute SVD for role increments to get Vt
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Role increments
        role_incs = []
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            for p in ["P1", "P2", "P3", "P4"]:
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    role_incs.append(m2 - m1)

        # Frame increments
        frame_incs = []
        for (token, role, pair), mean_v in cell_means.items():
            pair_means = [cell_means.get((token, role, p)) for p in ["P1", "P2", "P3", "P4"]]
            pair_means = [m for m in pair_means if m is not None]
            if len(pair_means) < 2:
                continue
            frame_incs.append(mean_v - np.mean(pair_means, axis=0))

        if len(role_incs) < 3 or len(frame_incs) < 3:
            continue

        # PCA for both
        role_mat = np.array(role_incs)
        frame_mat = np.array(frame_incs)
        role_centered = role_mat - np.mean(role_mat, axis=0, keepdims=True)
        frame_centered = frame_mat - np.mean(frame_mat, axis=0, keepdims=True)

        _, _, role_Vt = np.linalg.svd(role_centered, full_matrices=False)
        _, _, frame_Vt = np.linalg.svd(frame_centered, full_matrices=False)

        # Top-k subspaces
        k = min(5, role_Vt.shape[0], frame_Vt.shape[0])
        role_sub = role_Vt[:k]  # [k, d]
        frame_sub = frame_Vt[:k]  # [k, d]

        # Principal angles via SVD of role_sub @ frame_sub^T
        M = role_sub @ frame_sub.T  # [k, k]
        _, svals, _ = np.linalg.svd(M)
        # Principal angles = arccos(singular values / k approximation)
        # svals are cosines of principal angles (if subspaces are orthonormal)
        cos_angles = np.clip(svals, -1, 1)
        principal_angles = np.arccos(cos_angles)  # in radians

        avg_angle = float(np.mean(principal_angles))
        min_angle = float(np.min(principal_angles))

        results[li] = {
            "avg_principal_angle_rad": avg_angle,
            "min_principal_angle_rad": min_angle,
            "avg_principal_angle_deg": float(np.degrees(avg_angle)),
            "min_principal_angle_deg": float(np.degrees(min_angle)),
            "top1_cos_overlap": float(cos_angles[0]) if len(cos_angles) > 0 else 0.0,
            "top3_avg_cos_overlap": float(np.mean(cos_angles[:3])) if len(cos_angles) >= 3 else float(np.mean(cos_angles)),
            "k": k,
        }

    return results


# =====================================================================
# MEASUREMENT 4: Cross-Token Role Direction Generalization
# =====================================================================
def cross_token_generalization(captures, stimuli, n_layers):
    """
    Leave-one-token-out: extract role direction from N-1 tokens,
    test on held-out token.
    Report: cosine between predicted and actual role gap for held-out token.
    """
    obs = defaultdict(list)
    for stim in stimuli:
        if stim.get("group") != "subspace":
            continue
        token = stim["token_label"]
        role = stim["role_label"]
        pair = stim["pair_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set)
    token_pairs = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
        token_pairs[token].add(pair)
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])

    results = {}
    for li in range(n_layers + 1):
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # For each token, compute pair-averaged role delta
        token_deltas = {}
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            r1_means = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
            r2_means = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
            r1_means = [m for m in r1_means if m is not None]
            r2_means = [m for m in r2_means if m is not None]
            if r1_means and r2_means:
                token_deltas[token] = np.mean(r2_means, axis=0) - np.mean(r1_means, axis=0)

        if len(token_deltas) < 3:
            continue

        # Leave-one-out
        loo_cosines = []
        loo_norm_ratios = []
        all_tokens_list = sorted(token_deltas.keys())
        for held_out in all_tokens_list:
            # Direction from remaining tokens
            remaining = [t for t in all_tokens_list if t != held_out]
            avg_dir = np.mean([token_deltas[t] for t in remaining], axis=0)
            # Actual delta for held-out
            actual = token_deltas[held_out]
            # Cosine
            n1 = np.linalg.norm(avg_dir)
            n2 = np.linalg.norm(actual)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_val = float(np.dot(avg_dir, actual) / (n1 * n2))
                loo_cosines.append(cos_val)
                loo_norm_ratios.append(n1 / n2)

        results[li] = {
            "avg_loo_cosine": float(np.mean(loo_cosines)) if loo_cosines else 0.0,
            "std_loo_cosine": float(np.std(loo_cosines)) if loo_cosines else 0.0,
            "avg_norm_ratio": float(np.mean(loo_norm_ratios)) if loo_norm_ratios else 0.0,
            "n_tokens": len(loo_cosines),
        }

    return results


# =====================================================================
# MEASUREMENT 5: Causal Direction Test
# =====================================================================
def causal_direction_test(model, tokenizer, captures, subspace_stimuli,
                          causal_stimuli, n_layers, role_directions):
    """
    For each test sentence, add the role direction at key layers and
    measure logit shift.

    role_directions: dict of {layer_idx: {"mean_delta": np.array, "top1_pc": np.array}}

    Steps:
    1. Run baseline forward pass on test sentence → baseline_logits
    2. Run forward pass on paired sentence (other role) → target_logits
    3. Run forward pass with role direction added at layer l → patched_logits
    4. Measure: shift toward target role
    """
    # Organize causal test stimuli into pairs
    test_pairs = defaultdict(dict)  # token -> {role: stim}
    for stim in causal_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim

    # Identify dual-role test tokens
    dual_test_tokens = []
    for token, roles_dict in test_pairs.items():
        if len(roles_dict) >= 2:
            roles_list = sorted(roles_dict.keys())
            dual_test_tokens.append((token, roles_list))

    layers_obj = get_layers(model)
    input_device = next(model.parameters()).device

    # Select layers to test
    sample_layers = sorted(set(
        [0, 1] +
        list(range(0, n_layers, max(1, n_layers // 8))) +
        [n_layers - 2, n_layers - 1]
    ) & set(range(n_layers)))

    results = {}
    for token, roles_list in dual_test_tokens:
        if len(roles_list) != 2:
            continue
        r1, r2 = roles_list  # r1 < r2 (sorted)

        stim_r1 = test_pairs[token].get(r1)
        stim_r2 = test_pairs[token].get(r2)
        if stim_r1 is None or stim_r2 is None:
            continue
        if stim_r1.get("target_pos") is None or stim_r2.get("target_pos") is None:
            continue

        pos_r1 = stim_r1["target_pos"]
        pos_r2 = stim_r2["target_pos"]
        sent_r1 = stim_r1["sentence"]
        sent_r2 = stim_r2["sentence"]

        # Baseline forward passes
        inputs_r1 = tokenizer(sent_r1, return_tensors="pt", truncation=True, max_length=64)
        inputs_r1 = {k: v.to(input_device) for k, v in inputs_r1.items()}

        inputs_r2 = tokenizer(sent_r2, return_tensors="pt", truncation=True, max_length=64)
        inputs_r2 = {k: v.to(input_device) for k, v in inputs_r2.items()}

        with torch.no_grad():
            out_r1 = model(**inputs_r1, output_hidden_states=True)
            out_r2 = model(**inputs_r2, output_hidden_states=True)

        # Logits at target position
        logits_r1 = out_r1.logits[0, pos_r1].float().cpu().numpy()
        logits_r2 = out_r2.logits[0, pos_r2].float().cpu().numpy()

        # For each test layer, add role direction and measure effect
        for li in sample_layers:
            if li not in role_directions:
                continue

            direction_mean = role_directions[li].get("mean_delta")
            direction_top1 = role_directions[li].get("top1_pc")
            if direction_mean is None:
                continue

            for dir_name, direction in [("mean_delta", direction_mean), ("top1_pc", direction_top1)]:
                if direction is None:
                    continue

                dir_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=input_device)

                # --- Test 1: Add r2-r1 direction to r1 sentence → should shift toward r2 ---
                captured_patch = {}
                intervened = [False]

                def make_hook(target_pos, dir_t, alpha, flag):
                    def hook_fn(module, input, output):
                        if not flag[0] and isinstance(output, tuple):
                            h = output[0].clone()
                            device = h.device
                            dtype = h.dtype
                            h[0, target_pos, :] += (alpha * dir_t).to(dtype=dtype, device=device)
                            flag[0] = True
                            return (h,) + output[1:]
                        return output
                    return hook_fn

                hook = layers_obj[li].register_forward_hook(
                    make_hook(pos_r1, dir_tensor, 1.0, intervened)
                )
                with torch.no_grad():
                    out_patched = model(**inputs_r1)
                hook.remove()

                logits_patched = out_patched.logits[0, pos_r1].float().cpu().numpy()

                # Measure shift toward r2
                # 1. Cosine similarity with target logits
                cos_base_r2 = float(np.dot(logits_r1, logits_r2) /
                                    (np.linalg.norm(logits_r1) * np.linalg.norm(logits_r2) + 1e-10))
                cos_patch_r2 = float(np.dot(logits_patched, logits_r2) /
                                     (np.linalg.norm(logits_patched) * np.linalg.norm(logits_r2) + 1e-10))
                shift_toward_r2 = cos_patch_r2 - cos_base_r2

                # 2. KL divergence shift (using softmax)
                def softmax(x):
                    e = np.exp(x - np.max(x))
                    return e / (e.sum() + 1e-10)

                p_base = softmax(logits_r1)
                p_target = softmax(logits_r2)
                p_patched = softmax(logits_patched)

                kl_base_target = float(np.sum(p_base * np.log((p_base + 1e-10) / (p_target + 1e-10))))
                kl_patch_target = float(np.sum(p_patched * np.log((p_patched + 1e-10) / (p_target + 1e-10))))
                kl_shift = kl_base_target - kl_patch_target  # positive = moved toward target

                # --- Test 2: Random direction control ---
                random_dir = np.random.randn(len(direction))
                random_dir = random_dir / np.linalg.norm(random_dir) * np.linalg.norm(direction)
                random_tensor = torch.tensor(random_dir, dtype=torch.bfloat16, device=input_device)

                intervened2 = [False]
                hook2 = layers_obj[li].register_forward_hook(
                    make_hook(pos_r1, random_tensor, 1.0, intervened2)
                )
                with torch.no_grad():
                    out_random = model(**inputs_r1)
                hook2.remove()

                logits_random = out_random.logits[0, pos_r1].float().cpu().numpy()
                cos_random_r2 = float(np.dot(logits_random, logits_r2) /
                                      (np.linalg.norm(logits_random) * np.linalg.norm(logits_r2) + 1e-10))
                random_shift = cos_random_r2 - cos_base_r2

                key = f"{token}_{r1}→{r2}"
                if key not in results:
                    results[key] = {}
                results[key][f"L{li}_{dir_name}"] = {
                    "cos_shift_toward_target": round(shift_toward_r2, 6),
                    "kl_shift_toward_target": round(kl_shift, 6),
                    "random_cos_shift": round(random_shift, 6),
                    "specificity_ratio": round(shift_toward_r2 / max(abs(random_shift), 1e-10), 2),
                }

        # Log progress
        log(f"  Causal test: {token} ({r1}→{r2}) done for {len(sample_layers)} layers")

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase298_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 298: Role Subspace & Causal Direction Test — {model_name}")
    log(f"=" * 60)

    # ---- 1. Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")

    # ---- 2. Build stimuli ----
    sub_stimuli = build_phase297_stimuli()
    causal_stimuli = build_causal_test_stimuli()

    sub_stimuli = resolve_positions(sub_stimuli, tok)
    causal_stimuli = resolve_positions(causal_stimuli, tok)

    log(f"  Subspace stimuli: {len(sub_stimuli)}")
    log(f"  Causal test stimuli: {len(causal_stimuli)}")

    # Deduplicate sentences
    all_sentences = []
    sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]

    log(f"  Unique sentences: {len(all_sentences)}")

    # ---- 3. Capture hidden states ----
    log(f"\n--- Capturing hidden states for {len(all_sentences)} sentences ---")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent, nl)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  Captured {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"  Captured all in {time.time()-t0:.0f}s")

    # ---- 4. Measurement 1: Role Increment PCA ----
    log(f"\n--- Measurement 1: Role Increment PCA ---")
    role_pca, role_inc_labels = role_increment_pca(captures, sub_stimuli, nl)

    sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1]) & set(role_pca.keys()))
    log("  Layer | n_inc | top1% | top3% | top5% | dim50 | dim80 | dim95 | mean_norm")
    for li in sorted(role_pca.keys()):
        if li in sample_layers or li >= nl - 2:
            r = role_pca[li]
            log(f"  L{li:3d} | {r['n_increments']:5d} | {r['top1_var']*100:5.1f} | "
                f"{r['top3_var']*100:5.1f} | {r['top5_var']*100:5.1f} | "
                f"{r['dim_50']:5d} | {r['dim_80']:5d} | {r['dim_95']:5d} | "
                f"{r['mean_increment_norm']:.2f}")

    # ---- 5. Measurement 2: Frame Increment PCA ----
    log(f"\n--- Measurement 2: Frame Increment PCA ---")
    frame_pca = frame_increment_pca(captures, sub_stimuli, nl)

    log("  Layer | n_inc | top1% | top3% | top5% | dim50 | dim80 | dim95")
    for li in sorted(frame_pca.keys()):
        if li in sample_layers or li >= nl - 2:
            r = frame_pca[li]
            log(f"  L{li:3d} | {r['n_increments']:5d} | {r['top1_var']*100:5.1f} | "
                f"{r['top3_var']*100:5.1f} | {r['top5_var']*100:5.1f} | "
                f"{r['dim_50']:5d} | {r['dim_80']:5d} | {r['dim_95']:5d}")

    # ---- 6. Measurement 3: Role-Frame Subspace Overlap ----
    log(f"\n--- Measurement 3: Role-Frame Subspace Overlap ---")
    overlap = subspace_overlap(role_pca, frame_pca, captures, sub_stimuli, nl)

    log("  Layer | avg_angle° | min_angle° | top1_cos | top3_avg_cos")
    for li in sorted(overlap.keys()):
        if li in sample_layers or li >= nl - 2:
            r = overlap[li]
            log(f"  L{li:3d} | {r['avg_principal_angle_deg']:10.1f} | "
                f"{r['min_principal_angle_deg']:10.1f} | "
                f"{r['top1_cos_overlap']:+.4f} | {r['top3_avg_cos_overlap']:+.4f}")

    # ---- 7. Measurement 4: Cross-Token Generalization ----
    log(f"\n--- Measurement 4: Cross-Token Generalization (LOO) ---")
    loo = cross_token_generalization(captures, sub_stimuli, nl)

    log("  Layer | avg_LOO_cos | std_LOO_cos | norm_ratio | n_tokens")
    for li in sorted(loo.keys()):
        if li in sample_layers or li >= nl - 2:
            r = loo[li]
            log(f"  L{li:3d} | {r['avg_loo_cosine']:+.4f} | {r['std_loo_cosine']:.4f} | "
                f"{r['avg_norm_ratio']:.4f} | {r['n_tokens']}")

    # ---- 8. Extract role directions for causal test ----
    log(f"\n--- Extracting role directions for causal test ---")
    role_directions = {}
    for li in role_pca:
        # Mean delta direction (pair-averaged role increment)
        cell_means = {}
        obs = defaultdict(list)
        for stim in sub_stimuli:
            token = stim["token_label"]
            role = stim["role_label"]
            pair = stim["pair_label"]
            idx = stim.get("_idx")
            pos = stim.get("target_pos")
            if idx is not None and pos is not None:
                obs[(token, role, pair)].append((idx, pos))

        token_roles = defaultdict(set)
        for (token, role, pair) in obs:
            token_roles[token].add(role)
        dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Compute mean role delta across all tokens and pairs
        all_deltas = []
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            for p in ["P1", "P2", "P3", "P4"]:
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    all_deltas.append(m2 - m1)

        if all_deltas:
            mean_delta = np.mean(all_deltas, axis=0)

            # Top-1 PC direction (already have from role_pca)
            # Need to recompute Vt for this layer
            inc_matrix = np.array(all_deltas)
            inc_centered = inc_matrix - np.mean(inc_matrix, axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(inc_centered, full_matrices=False)
            top1_pc = Vt[0] if len(Vt) > 0 else np.zeros(d_model)
            # Scale top1_pc to same norm as mean_delta
            top1_pc_scaled = top1_pc * np.linalg.norm(mean_delta)

            role_directions[li] = {
                "mean_delta": mean_delta,
                "top1_pc": top1_pc_scaled,
            }

    log(f"  Extracted directions for {len(role_directions)} layers")

    # ---- 9. Measurement 5: Causal Direction Test ----
    log(f"\n--- Measurement 5: Causal Direction Test ---")
    causal_results = causal_direction_test(
        model, tok, captures, sub_stimuli, causal_stimuli, nl, role_directions
    )

    # Summarize causal results
    log("\n  Causal Test Summary (mean_delta direction):")
    log("  Token_pair | Layer | cos_shift | KL_shift | random_shift | specificity")
    for key in sorted(causal_results.keys()):
        for layer_key in sorted(causal_results[key].keys()):
            if "mean_delta" in layer_key:
                r = causal_results[key][layer_key]
                log(f"  {key:15s} | {layer_key:12s} | {r['cos_shift_toward_target']:+.6f} | "
                    f"{r['kl_shift_toward_target']:+.6f} | {r['random_cos_shift']:+.6f} | "
                    f"{r['specificity_ratio']:.1f}x")

    # ---- 10. Save results ----
    log(f"\n--- Saving results ---")

    def convert_keys(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        if isinstance(d, dict):
            return {str(k): convert_keys(v) for k, v in d.items()}
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d

    # Don't save raw direction arrays (too large), save summaries
    role_dirs_summary = {}
    for li, dirs in role_directions.items():
        role_dirs_summary[li] = {
            "mean_delta_norm": float(np.linalg.norm(dirs["mean_delta"])),
            "top1_pc_norm": float(np.linalg.norm(dirs["top1_pc"])),
            "cos_mean_top1": float(np.dot(dirs["mean_delta"], dirs["top1_pc"]) /
                                    (np.linalg.norm(dirs["mean_delta"]) * np.linalg.norm(dirs["top1_pc"]) + 1e-10)),
        }

    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "role_increment_pca": convert_keys(role_pca),
        "frame_increment_pca": convert_keys(frame_pca),
        "subspace_overlap": convert_keys(overlap),
        "cross_token_generalization": convert_keys(loo),
        "causal_direction_test": convert_keys(causal_results),
        "role_directions_summary": convert_keys(role_dirs_summary),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULT_DIR / f"{model_name}_role_subspace.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ---- 11. Summary ----
    log(f"\n{'='*60}")
    log(f"PHASE 298 SUMMARY — {model_name}")
    log(f"{'='*60}")

    log("\n  [A] Role Subspace Dimensionality:")
    for li in sorted(role_pca.keys()):
        if li in sample_layers or li >= nl - 2:
            r = role_pca[li]
            log(f"    L{li:3d}: dim50={r['dim_50']} dim80={r['dim_80']} dim95={r['dim_95']} "
                f"top1={r['top1_var']*100:.1f}% top3={r['top3_var']*100:.1f}% top5={r['top5_var']*100:.1f}%")

    log("\n  [B] Frame Subspace Dimensionality:")
    for li in sorted(frame_pca.keys()):
        if li in sample_layers or li >= nl - 2:
            r = frame_pca[li]
            log(f"    L{li:3d}: dim50={r['dim_50']} dim80={r['dim_80']} dim95={r['dim_95']} "
                f"top1={r['top1_var']*100:.1f}% top3={r['top3_var']*100:.1f}%")

    log("\n  [C] Role-Frame Subspace Overlap:")
    for li in sorted(overlap.keys()):
        if li in sample_layers or li >= nl - 2:
            r = overlap[li]
            log(f"    L{li:3d}: avg_angle={r['avg_principal_angle_deg']:.1f}° "
                f"top1_cos={r['top1_cos_overlap']:+.4f}")

    log("\n  [D] Cross-Token Generalization (LOO):")
    for li in sorted(loo.keys()):
        if li in sample_layers or li >= nl - 2:
            r = loo[li]
            log(f"    L{li:3d}: avg_LOO_cos={r['avg_loo_cosine']:+.4f} ± {r['std_loo_cosine']:.4f}")

    log("\n  [E] Causal Direction Test (mean_delta, mid-layer):")
    mid = nl // 2
    for key in sorted(causal_results.keys()):
        for layer_key in sorted(causal_results[key].keys()):
            li_str = layer_key.split("_")[0][1:]
            try:
                li_val = int(li_str)
            except:
                continue
            if "mean_delta" in layer_key and abs(li_val - mid) <= 2:
                r = causal_results[key][layer_key]
                log(f"    {key} L{li_val}: cos_shift={r['cos_shift_toward_target']:+.6f} "
                    f"specificity={r['specificity_ratio']:.1f}x")

    # Release
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    log(f"Phase 298 complete for {model_name}!")


if __name__ == "__main__":
    main()
