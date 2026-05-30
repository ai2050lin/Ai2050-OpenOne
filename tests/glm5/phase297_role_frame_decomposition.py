"""
Phase 297: Orthogonal Identity-Role-Frame Decomposition (v2)
=============================================================
Goal: Separate Role from syntactic frame by proper nested design.

Key Design Insight:
  - Frames are role-specific (adj frames ≠ verb frames)
  - So frame is NESTED within role, not crossed
  - We use a nested ANOVA: h = μ + I(token) + R(role) + F(role:frame) + IR + IF|role + ε

  Plus a critical within-frame-pair analysis:
  - Match frame indices across roles (F1_adj ↔ F1_verb, F2_adj ↔ F2_verb, etc.)
  - Each frame pair shares the same content words (same noun/object)
  - This gives us "controlled role gap" at each frame pair

Stimulus Design:
  8 dual-role tokens, 4 frame pairs per token
  Each frame pair: same content words, different syntax for each role
  2 variants per cell (different noun/object)
  Total: 8 × 2 × 4 × 2 = 128 sentences

Usage:
  python tests/glm5/phase297_role_frame_decomposition.py qwen3
  python tests/glm5/phase297_role_frame_decomposition.py glm4
  python tests/glm5/phase297_role_frame_decomposition.py deepseek7b
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

RESULT_DIR = Path("results/phase297_role_frame")
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
# STIMULUS DESIGN — Matched frame pairs
# =====================================================================
def build_stimulus_set():
    """
    Build stimuli with matched frame pairs across roles.

    Key: each "frame pair" (P1-P4) provides a content-controlled comparison.
    Same noun/object appears in both roles, only syntax differs.

    Example for "open":
      P1: "the door is open" (adj, copula)     ↔ "they open the door" (verb, transitive)
      P2: "the gate remains open" (adj, copula+) ↔ "we open the gate" (verb, transitive)
      P3: "the open door" (adj, prenominal)     ↔ "the door will open" (verb, intransitive)
      P4: "the open gate" (adj, prenominal)     ↔ "the gate will open" (verb, intransitive)

    Returns list of dicts with token_label, role_label, pair_label, sentence, target_word
    """
    stimuli = []

    # ============================================================
    # Adjective <-> Verb tokens (4)
    # Each pair uses matched content words
    # ============================================================
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

    # ============================================================
    # Adjective <-> Noun tokens (2)
    # ============================================================
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

    # ============================================================
    # Noun <-> Verb tokens (2)
    # ============================================================
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
                        "group": "three_factor",
                    })

    return stimuli


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto)...")

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
# MEASUREMENT 1: Two-way ANOVA per role (token × pair)
# =====================================================================
def anova_per_role(captures, stimuli, n_layers, normalize=False):
    """
    Within each role, do two-way ANOVA: h = μ + I(token) + P(pair) + I×P + ε
    This gives us:
      - Identity variance within role
      - Frame-pair variance within role
      - How much frame varies independent of token

    Then compare across roles to get:
      - Cross-role frame variance
    """
    obs = defaultdict(list)  # (token, role, pair) -> list of (idx, pos)
    for stim in stimuli:
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
        # Collect vectors per role
        role_data = defaultdict(list)  # role -> [(token, pair, vec)]
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                v = h[0, pos, :].numpy().copy()
                if normalize:
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-8:
                        continue
                    v = v / nrm
                role_data[role].append((token, pair, v))

        layer_result = {"roles": {}}

        for role, entries in role_data.items():
            if len(entries) < 10:
                continue

            all_v = np.array([v for _, _, v in entries])
            mu = np.mean(all_v, axis=0)
            N = len(entries)

            # Token means
            token_vecs = defaultdict(list)
            for token, pair, v in entries:
                token_vecs[token].append(v)
            token_means = {t: np.mean(vs, axis=0) for t, vs in token_vecs.items()}

            # Pair means
            pair_vecs = defaultdict(list)
            for token, pair, v in entries:
                pair_vecs[pair].append(v)
            pair_means = {p: np.mean(vs, axis=0) for p, vs in pair_vecs.items()}

            # Cell means
            cell_vecs = defaultdict(list)
            for token, pair, v in entries:
                cell_vecs[(token, pair)].append(v)
            cell_means = {k: np.mean(vs, axis=0) for k, vs in cell_vecs.items()}

            # SS
            total_ss = sum(np.sum((v - mu) ** 2) for _, _, v in entries)
            identity_ss = sum(len(token_vecs[t]) * np.sum((token_means[t] - mu) ** 2) for t in token_means)
            pair_ss = sum(len(pair_vecs[p]) * np.sum((pair_means[p] - mu) ** 2) for p in pair_means)
            interaction_ss = 0
            for (t, p), mean_v in cell_means.items():
                effect = mean_v - token_means[t] - pair_means[p] + mu
                interaction_ss += len(cell_vecs[(t, p)]) * np.sum(effect ** 2)
            residual_ss = max(total_ss - identity_ss - pair_ss - interaction_ss, 0)

            layer_result["roles"][role] = {
                "total_ss": float(total_ss),
                "identity_ratio": float(identity_ss / max(total_ss, 1e-10)),
                "pair_ratio": float(pair_ss / max(total_ss, 1e-10)),
                "interaction_ratio": float(interaction_ss / max(total_ss, 1e-10)),
                "residual_ratio": float(residual_ss / max(total_ss, 1e-10)),
                "r_squared": float((identity_ss + pair_ss) / max(total_ss, 1e-10)),
                "n_observations": N,
                "n_tokens": len(token_means),
                "n_pairs": len(pair_means),
            }

        results[li] = layer_result

    return results


# =====================================================================
# MEASUREMENT 2: Matched-pair role gap
# =====================================================================
def matched_pair_role_gap(captures, stimuli, n_layers, normalize=False):
    """
    For each dual-role token and each pair:
      - Compute role gap within that pair
      - This controls for content words (same noun/object in both roles)

    Compare:
      - Raw role gap (avg across all sentences)
      - Matched-pair role gap (avg across pairs)
      - Cross-pair consistency (cosine of role gap direction across pairs)
    """
    obs = defaultdict(list)
    for stim in stimuli:
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
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {
        "raw_role_gap": defaultdict(dict),
        "matched_pair_role_gap": defaultdict(dict),
        "cross_pair_consistency": defaultdict(dict),  # cos of role delta across pairs
        "pair_role_gap_per_pair": defaultdict(dict),  # layer -> {pair: avg_gap}
    }

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
                v = h[0, pos, :].numpy().copy()
                if normalize:
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-8:
                        continue
                    v = v / nrm
                vecs.append(v)
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list

            # 1. Raw role gap (average across all pairs)
            r1_all = [cell_means[(token, r1, p)] for p in token_pairs[token]
                      if (token, r1, p) in cell_means]
            r2_all = [cell_means[(token, r2, p)] for p in token_pairs[token]
                      if (token, r2, p) in cell_means]
            if not r1_all or not r2_all:
                continue
            r1_mean = np.mean(r1_all, axis=0)
            r2_mean = np.mean(r2_all, axis=0)
            results["raw_role_gap"][li][token] = round(float(np.linalg.norm(r1_mean - r2_mean)), 4)

            # 2. Matched-pair role gap (per pair, then average)
            per_pair_gaps = []
            per_pair_deltas = []
            for p in sorted(token_pairs[token]):
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    delta = m1 - m2
                    per_pair_gaps.append(float(np.linalg.norm(delta)))
                    per_pair_deltas.append(delta)

            if per_pair_gaps:
                results["matched_pair_role_gap"][li][token] = round(float(np.mean(per_pair_gaps)), 4)

                # Per-pair gap
                for pi, p in enumerate(sorted(token_pairs[token])):
                    if pi < len(per_pair_gaps):
                        if p not in results["pair_role_gap_per_pair"][li]:
                            results["pair_role_gap_per_pair"][li][p] = []
                        results["pair_role_gap_per_pair"][li][p].append(per_pair_gaps[pi])

                # 3. Cross-pair consistency
                if len(per_pair_deltas) >= 2:
                    cos_vals = []
                    for i in range(len(per_pair_deltas)):
                        for j in range(i + 1, len(per_pair_deltas)):
                            d1, d2 = per_pair_deltas[i], per_pair_deltas[j]
                            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                            if n1 > 1e-8 and n2 > 1e-8:
                                cos_vals.append(float(np.dot(d1, d2) / (n1 * n2)))
                    if cos_vals:
                        results["cross_pair_consistency"][li][token] = round(float(np.mean(cos_vals)), 4)

    return results


# =====================================================================
# MEASUREMENT 3: Cross-role ANOVA (token × role, averaging over pairs)
# =====================================================================
def cross_role_anova(captures, stimuli, n_layers, normalize=False):
    """
    After averaging over pairs, do token × role ANOVA.
    This gives us the "pair-averaged" identity and role variance,
    which should be a cleaner estimate of pure role effect.

    Also: compute pair-adjusted role effect using residuals.
    """
    obs = defaultdict(list)
    for stim in stimuli:
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
        # Compute cell means (averaged over replicates)
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                v = h[0, pos, :].numpy().copy()
                if normalize:
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-8:
                        continue
                    v = v / nrm
                vecs.append(v)
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Average over pairs to get (token, role) means
        tr_means = defaultdict(list)
        for (token, role, pair), mean_v in cell_means.items():
            tr_means[(token, role)].append(mean_v)
        tr_cell_means = {k: np.mean(vs, axis=0) for k, vs in tr_means.items()}

        # Grand mean
        all_means = list(tr_cell_means.values())
        if len(all_means) < 5:
            continue
        mu = np.mean(all_means, axis=0)
        N = len(all_means)

        # Token means
        token_vecs = defaultdict(list)
        for (token, role), mean_v in tr_cell_means.items():
            token_vecs[token].append(mean_v)
        token_means = {t: np.mean(vs, axis=0) for t, vs in token_vecs.items()}

        # Role means
        role_vecs = defaultdict(list)
        for (token, role), mean_v in tr_cell_means.items():
            role_vecs[role].append(mean_v)
        role_means = {r: np.mean(vs, axis=0) for r, vs in role_vecs.items()}

        # SS (pair-averaged, so no pair factor)
        total_ss = sum(np.sum((v - mu) ** 2) for v in all_means)
        identity_ss = sum(len(token_vecs[t]) * np.sum((token_means[t] - mu) ** 2) for t in token_means)
        role_ss = sum(len(role_vecs[r]) * np.sum((role_means[r] - mu) ** 2) for r in role_means)
        interaction_ss = max(total_ss - identity_ss - role_ss, 0)

        r_squared = (identity_ss + role_ss) / max(total_ss, 1e-10)

        # Orthogonality
        ortho_cosines = []
        for t, alpha_v in token_means.items():
            for r, beta_v in role_means.items():
                a = alpha_v - mu
                b = beta_v - mu
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    ortho_cosines.append(float(np.dot(a, b) / (na * nb)))
        avg_ortho = float(np.mean(ortho_cosines)) if ortho_cosines else 0.0

        results[li] = {
            "identity_ratio": float(identity_ss / max(total_ss, 1e-10)),
            "role_ratio": float(role_ss / max(total_ss, 1e-10)),
            "interaction_ratio": float(interaction_ss / max(total_ss, 1e-10)),
            "r_squared": float(r_squared),
            "ortho_ir": round(avg_ortho, 4),
            "n_cells": N,
            "n_tokens": len(token_means),
            "n_roles": len(role_means),
        }

    return results


# =====================================================================
# MEASUREMENT 4: Shared vs pair-specific role increment
# =====================================================================
def role_increment_decomposition(captures, stimuli, n_layers):
    """
    For each dual-role token:
      1. Compute shared role increment: avg delta across all pairs
      2. Compute pair-specific role increment: deviation from shared
      3. Shared/(shared+specific) ratio = how frame-independent is the role effect?
    """
    obs = defaultdict(list)
    for stim in stimuli:
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
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {
        "shared_role_ratio": defaultdict(dict),
        "cross_pair_cos": defaultdict(dict),
    }

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

        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list

            # For each pair, compute role delta
            deltas = []
            for p in sorted(token_pairs[token]):
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    deltas.append(m1 - m2)

            if len(deltas) < 2:
                continue

            # Shared role delta
            shared = np.mean(deltas, axis=0)
            shared_norm = np.linalg.norm(shared)

            # Pair-specific deviation
            specific_norms = [np.linalg.norm(d - shared) for d in deltas]
            avg_specific = float(np.mean(specific_norms))

            # Shared ratio
            if shared_norm + avg_specific > 1e-8:
                shared_ratio = shared_norm / (shared_norm + avg_specific)
                results["shared_role_ratio"][li][token] = round(float(shared_ratio), 4)

            # Cross-pair cosine
            cos_vals = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    n1, n2 = np.linalg.norm(deltas[i]), np.linalg.norm(deltas[j])
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_vals.append(float(np.dot(deltas[i], deltas[j]) / (n1 * n2)))
            if cos_vals:
                results["cross_pair_cos"][li][token] = round(float(np.mean(cos_vals)), 4)

    return results


# =====================================================================
# MEASUREMENT 5: Identity preservation (pair-controlled)
# =====================================================================
def identity_preservation_controlled(captures, stimuli, n_layers):
    """
    Same token, different role:
      - Same pair (controlled): cos(h(r1, p), h(r2, p))
      - Different pair: cos(h(r1, p1), h(r2, p2))

    Same token, same role, different pair:
      - cos(h(r, p1), h(r, p2))
    """
    obs = defaultdict(list)
    for stim in stimuli:
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
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {
        "same_pair_id_pres": defaultdict(dict),     # same pair, diff role
        "diff_pair_id_pres": defaultdict(dict),      # diff pair, diff role
        "same_role_pair_pres": defaultdict(dict),    # same role, diff pair
    }

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

        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            pairs_list = sorted(token_pairs[token])

            # Same pair, diff role
            same_pair_cos = []
            for p in pairs_list:
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        same_pair_cos.append(float(np.dot(m1, m2) / (n1 * n2)))
            if same_pair_cos:
                results["same_pair_id_pres"][li][token] = round(float(np.mean(same_pair_cos)), 4)

            # Different pair, diff role
            diff_pair_cos = []
            for p1 in pairs_list:
                for p2 in pairs_list:
                    if p1 == p2:
                        continue
                    m1 = cell_means.get((token, r1, p1))
                    m2 = cell_means.get((token, r2, p2))
                    if m1 is not None and m2 is not None:
                        n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                        if n1 > 1e-8 and n2 > 1e-8:
                            diff_pair_cos.append(float(np.dot(m1, m2) / (n1 * n2)))
            if diff_pair_cos:
                results["diff_pair_id_pres"][li][token] = round(float(np.mean(diff_pair_cos)), 4)

            # Same role, different pair
            same_role_cos = []
            for r in [r1, r2]:
                for pi in range(len(pairs_list)):
                    for pj in range(pi + 1, len(pairs_list)):
                        m_i = cell_means.get((token, r, pairs_list[pi]))
                        m_j = cell_means.get((token, r, pairs_list[pj]))
                        if m_i is not None and m_j is not None:
                            ni, nj = np.linalg.norm(m_i), np.linalg.norm(m_j)
                            if ni > 1e-8 and nj > 1e-8:
                                same_role_cos.append(float(np.dot(m_i, m_j) / (ni * nj)))
            if same_role_cos:
                results["same_role_pair_pres"][li][token] = round(float(np.mean(same_role_cos)), 4)

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase297_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 297: Orthogonal Identity-Role-Frame Decomposition (v2) — {model_name}")
    log(f"=" * 60)

    # ---- 1. Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")

    # ---- 2. Build and resolve stimuli ----
    stimuli = build_stimulus_set()
    stimuli = resolve_positions(stimuli, tok)

    valid_stimuli = [s for s in stimuli if s.get("target_pos") is not None]
    log(f"  Valid stimuli: {len(valid_stimuli)}")

    tokens = set(s["token_label"] for s in valid_stimuli)
    roles = set(s["role_label"] for s in valid_stimuli)
    pairs = set(s["pair_label"] for s in valid_stimuli)
    log(f"  Tokens: {len(tokens)}, Roles: {len(roles)}, Pairs: {len(pairs)}")

    cell_counts = defaultdict(int)
    for s in valid_stimuli:
        cell_counts[(s["token_label"], s["role_label"], s["pair_label"])] += 1
    log(f"  Cells: {len(cell_counts)}, avg per cell: {np.mean(list(cell_counts.values())):.1f}")

    all_sentences = []
    sent_to_idx = {}
    for s in valid_stimuli:
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

    # ---- 4. Measurement 1: ANOVA per role ----
    log(f"\n--- Measurement 1: ANOVA per Role (token × pair) ---")
    apr_raw = anova_per_role(captures, valid_stimuli, nl, normalize=False)
    apr_norm = anova_per_role(captures, valid_stimuli, nl, normalize=True)

    sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl]) & set(apr_raw.keys()))
    for role_label in ["adj", "verb", "noun"]:
        log(f"\n  Role={role_label} (RAW):")
        log(f"  Layer | Id% | Pair% | Interact% | Res% | R²")
        for li in sorted(apr_raw.keys()):
            r = apr_raw[li].get("roles", {}).get(role_label, {})
            if not r or li not in sample_layers:
                continue
            log(f"  L{li:3d} | {r['identity_ratio']*100:5.1f} | {r['pair_ratio']*100:5.1f} | "
                f"{r['interaction_ratio']*100:8.1f} | {r['residual_ratio']*100:4.1f} | {r['r_squared']:.3f}")

    # ---- 5. Measurement 2: Matched-pair role gap ----
    log(f"\n--- Measurement 2: Matched-Pair Role Gap ---")
    mprg = matched_pair_role_gap(captures, valid_stimuli, nl, normalize=False)

    log("  Layer | RawGap | MatchedGap | CrossPairCos | P1_gap | P2_gap | P3_gap | P4_gap")
    for li in sorted(mprg["raw_role_gap"].keys()):
        if li in sample_layers or li >= nl - 2:
            raw_gaps = list(mprg["raw_role_gap"][li].values())
            matched_gaps = list(mprg["matched_pair_role_gap"][li].values())
            cross_cos = list(mprg["cross_pair_consistency"][li].values())

            pair_gaps = {}
            for p, gaps in mprg["pair_role_gap_per_pair"].get(li, {}).items():
                pair_gaps[p] = np.mean(gaps)

            log(f"  L{li:3d} | {np.mean(raw_gaps):.2f} | {np.mean(matched_gaps):.2f} | "
                f"{np.mean(cross_cos):.4f} | "
                f"{pair_gaps.get('P1',0):.2f} | {pair_gaps.get('P2',0):.2f} | "
                f"{pair_gaps.get('P3',0):.2f} | {pair_gaps.get('P4',0):.2f}")

    # ---- 6. Measurement 3: Cross-role ANOVA (pair-averaged) ----
    log(f"\n--- Measurement 3: Cross-Role ANOVA (pair-averaged) ---")
    cra_raw = cross_role_anova(captures, valid_stimuli, nl, normalize=False)
    cra_norm = cross_role_anova(captures, valid_stimuli, nl, normalize=True)

    log("  Layer | Id% | Role% | Interact% | R² | I-R ortho (RAW)")
    for li in sorted(cra_raw.keys()):
        if li in sample_layers or li >= nl - 2:
            r = cra_raw[li]
            log(f"  L{li:3d} | {r['identity_ratio']*100:5.1f} | {r['role_ratio']*100:5.1f} | "
                f"{r['interaction_ratio']*100:8.1f} | {r['r_squared']:.3f} | {r['ortho_ir']:+.4f}")

    log("\n  (NORMALIZED):")
    log("  Layer | Id% | Role% | Interact% | R² | I-R ortho")
    for li in sorted(cra_norm.keys()):
        if li in sample_layers or li >= nl - 2:
            r = cra_norm[li]
            log(f"  L{li:3d} | {r['identity_ratio']*100:5.1f} | {r['role_ratio']*100:5.1f} | "
                f"{r['interaction_ratio']*100:8.1f} | {r['r_squared']:.3f} | {r['ortho_ir']:+.4f}")

    # ---- 7. Measurement 4: Role increment decomposition ----
    log(f"\n--- Measurement 4: Role Increment Decomposition ---")
    rid = role_increment_decomposition(captures, valid_stimuli, nl)

    log("  Layer | SharedRatio | CrossPairCos")
    for li in sorted(rid["shared_role_ratio"].keys()):
        if li in sample_layers or li >= nl - 2:
            sr = list(rid["shared_role_ratio"][li].values())
            cc = list(rid["cross_pair_cos"][li].values())
            log(f"  L{li:3d} | {np.mean(sr):.4f} | {np.mean(cc):.4f}")

    # ---- 8. Measurement 5: Identity preservation ----
    log(f"\n--- Measurement 5: Identity Preservation (pair-controlled) ---")
    ipc = identity_preservation_controlled(captures, valid_stimuli, nl)

    log("  Layer | SamePairId | DiffPairId | SameRoleDiffPair")
    for li in sorted(ipc["same_pair_id_pres"].keys()):
        if li in sample_layers or li >= nl - 2:
            sp = list(ipc["same_pair_id_pres"][li].values())
            dp = list(ipc["diff_pair_id_pres"][li].values())
            sr = list(ipc["same_role_pair_pres"][li].values())
            log(f"  L{li:3d} | {np.mean(sp):.4f} | {np.mean(dp):.4f} | {np.mean(sr):.4f}")

    # ---- 9. Save results ----
    log(f"\n--- Saving results ---")

    def convert_keys(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        if isinstance(d, dict):
            return {str(k): convert_keys(v) for k, v in d.items()}
        return d

    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "n_stimuli": len(valid_stimuli),
        "n_tokens": len(tokens),
        "n_roles": len(roles),
        "n_pairs": len(pairs),
        "anova_per_role_raw": convert_keys(apr_raw),
        "anova_per_role_normalized": convert_keys(apr_norm),
        "matched_pair_role_gap": convert_keys(mprg),
        "cross_role_anova_raw": convert_keys(cra_raw),
        "cross_role_anova_normalized": convert_keys(cra_norm),
        "role_increment_decomposition": convert_keys(rid),
        "identity_preservation_controlled": convert_keys(ipc),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULT_DIR / f"{model_name}_role_frame.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ---- 10. Summary ----
    log(f"\n{'='*60}")
    log(f"PHASE 297 SUMMARY — {model_name}")
    log(f"{'='*60}")

    log("\n  [A] Frame Variance within Role (RAW, mid-layer):")
    mid = nl // 2
    if mid in apr_raw:
        for role_label, r in apr_raw[mid].get("roles", {}).items():
            log(f"    {role_label}: Id={r['identity_ratio']*100:.1f}% Pair={r['pair_ratio']*100:.1f}% "
                f"Interact={r['interaction_ratio']*100:.1f}% R²={r['r_squared']:.3f}")

    log("\n  [B] Pair-Averaged Cross-Role ANOVA (vs Phase 296):")
    for li in sorted(cra_raw.keys()):
        if li in sample_layers or li >= nl - 2:
            r = cra_raw[li]
            log(f"    L{li:3d}: Id={r['identity_ratio']*100:.1f}% Role={r['role_ratio']*100:.1f}% "
                f"Interact={r['interaction_ratio']*100:.1f}% R²={r['r_squared']:.3f}")

    log("\n  [C] Role Gap: Raw vs Matched-Pair:")
    for li in sorted(mprg["raw_role_gap"].keys()):
        if li in sample_layers or li >= nl - 2:
            raw_gaps = list(mprg["raw_role_gap"][li].values())
            matched_gaps = list(mprg["matched_pair_role_gap"][li].values())
            cross_cos = list(mprg["cross_pair_consistency"][li].values())
            log(f"    L{li:3d}: Raw={np.mean(raw_gaps):.2f} Matched={np.mean(matched_gaps):.2f} "
                f"CrossPairCos={np.mean(cross_cos):.4f}")

    log("\n  [D] Shared vs Pair-Specific Role Increment:")
    for li in sorted(rid["shared_role_ratio"].keys()):
        if li in sample_layers or li >= nl - 2:
            sr = list(rid["shared_role_ratio"][li].values())
            cc = list(rid["cross_pair_cos"][li].values())
            log(f"    L{li:3d}: SharedRatio={np.mean(sr):.4f} CrossPairCos={np.mean(cc):.4f}")

    log("\n  [E] Identity Preservation:")
    for li in sorted(ipc["same_pair_id_pres"].keys()):
        if li in sample_layers or li >= nl - 2:
            sp = list(ipc["same_pair_id_pres"][li].values())
            dp = list(ipc["diff_pair_id_pres"][li].values())
            sr = list(ipc["same_role_pair_pres"][li].values())
            log(f"    L{li:3d}: SamePair={np.mean(sp):.4f} DiffPair={np.mean(dp):.4f} "
                f"SameRoleDiffPair={np.mean(sr):.4f}")

    # Release
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    log(f"Phase 297 complete for {model_name}!")


if __name__ == "__main__":
    main()
