"""
Phase 296: Identity-Role Residual Decomposition
================================================
Goal: Directly decompose each layer's residual stream h_l(token, role) into:
  h = μ + I(token) + R(role) + Interaction(token, role) + ε

Key Questions:
  1. What fraction of variance is explained by token identity vs role?
  2. Does the additive model h ≈ μ + I + R hold? (R²)
  3. How do identity/role/interaction ratios evolve across layers?
  4. Are identity and role subspaces orthogonal?
  5. Is the decomposition robust after norm normalization?

Stimulus Design:
  16 dual-role tokens (same word in 2 grammatical roles)
  Each (token, role) pair: 5 sentence frames
  10 single-role control tokens (always adjective/operand)
  Total: ~210 sentences

Usage:
  python tests/glm5/phase296_identity_role_decomposition.py qwen3
  python tests/glm5/phase296_identity_role_decomposition.py glm4
  python tests/glm5/phase296_identity_role_decomposition.py deepseek7b
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

RESULT_DIR = Path("results/phase296_residual_decomposition")
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
# STIMULUS DESIGN
# =====================================================================
def build_stimulus_set():
    """
    Build stimuli for Identity-Role Residual Decomposition.

    Design:
    - 16 dual-role tokens (adjective<->verb, adjective<->noun, noun<->verb)
    - Each (token, role) pair has 5 sentence frames
    - 10 single-role control tokens (always adjective)
    - Total: 16*2*5 + 10*5 = 210 sentences

    Returns list of dicts: {sentence, target_word, token_label, role_label, group}
    """
    stimuli = []

    # ---- Dual-role tokens: Adjective <-> Verb (6 tokens) ----
    adj_verb = {
        "open": {
            "adjective": ["the door is open", "the window is open", "the shop is open",
                          "the gate is open", "the road is open"],
            "verb": ["they open the door", "they open the window", "they open the shop",
                     "they open the gate", "they open the road"],
        },
        "clear": {
            "adjective": ["the table is clear", "the room is clear", "the desk is clear",
                          "the path is clear", "the area is clear"],
            "verb": ["they clear the table", "they clear the room", "they clear the desk",
                     "they clear the path", "they clear the area"],
        },
        "free": {
            "adjective": ["the bird is free", "the man is free", "the press is free",
                          "the people are free", "the spirit is free"],
            "verb": ["they free the bird", "they free the man", "they free the press",
                     "they free the people", "they free the spirit"],
        },
        "warm": {
            "adjective": ["the room is warm", "the water is warm", "the food is warm",
                          "the house is warm", "the air is warm"],
            "verb": ["they warm the room", "they warm the water", "they warm the food",
                     "they warm the house", "they warm the air"],
        },
        "clean": {
            "adjective": ["the house is clean", "the room is clean", "the floor is clean",
                          "the table is clean", "the water is clean"],
            "verb": ["they clean the house", "they clean the room", "they clean the floor",
                     "they clean the table", "they clean the water"],
        },
        "dry": {
            "adjective": ["the cloth is dry", "the ground is dry", "the wood is dry",
                          "the hair is dry", "the paint is dry"],
            "verb": ["they dry the cloth", "they dry the ground", "they dry the wood",
                     "they dry the hair", "they dry the paint"],
        },
    }

    # ---- Dual-role tokens: Adjective <-> Noun (4 tokens) ----
    adj_noun = {
        "fair": {
            "adjective": ["the game is fair", "the price is fair", "the rule is fair",
                          "the deal is fair", "the system is fair"],
            "noun": ["the fair is big", "the fair is fun", "the fair is open",
                     "the fair is popular", "the fair is annual"],
        },
        "right": {
            "adjective": ["the answer is right", "the choice is right", "the direction is right",
                          "the method is right", "the approach is right"],
            "noun": ["the right is clear", "the right is important", "the right is protected",
                     "the right is universal", "the right is fundamental"],
        },
        "light": {
            "adjective": ["the bag is light", "the box is light", "the load is light",
                          "the dress is light", "the material is light"],
            "noun": ["the light is bright", "the light is warm", "the light is soft",
                     "the light is natural", "the light is artificial"],
        },
        "cold": {
            "adjective": ["the water is cold", "the weather is cold", "the room is cold",
                          "the wind is cold", "the air is cold"],
            "noun": ["the cold is severe", "the cold is bitter", "the cold is intense",
                     "the cold is extreme", "the cold is unusual"],
        },
    }

    # ---- Dual-role tokens: Noun <-> Verb (6 tokens) ----
    noun_verb = {
        "fire": {
            "noun": ["the fire is hot", "the fire is big", "the fire is dangerous",
                     "the fire is bright", "the fire is spreading"],
            "verb": ["they fire the worker", "they fire the gun", "they fire the engine",
                     "they fire the missile", "they fire the employee"],
        },
        "run": {
            "noun": ["the run is long", "the run is fast", "the run is daily",
                     "the run is tiring", "the run is enjoyable"],
            "verb": ["they run fast", "they run daily", "they run together",
                     "they run outside", "they run home"],
        },
        "play": {
            "noun": ["the play is long", "the play is good", "the play is famous",
                     "the play is popular", "the play is dramatic"],
            "verb": ["they play music", "they play games", "they play sports",
                     "they play together", "they play outside"],
        },
        "record": {
            "noun": ["the record is old", "the record is broken", "the record is impressive",
                     "the record is long", "the record is official"],
            "verb": ["they record music", "they record meetings", "they record data",
                     "they record videos", "they record conversations"],
        },
        "sign": {
            "noun": ["the sign is clear", "the sign is big", "the sign is important",
                     "the sign is visible", "the sign is helpful"],
            "verb": ["they sign papers", "they sign contracts", "they sign documents",
                     "they sign forms", "they sign letters"],
        },
        "state": {
            "noun": ["the state is large", "the state is rich", "the state is powerful",
                     "the state is small", "the state is independent"],
            "verb": ["they state facts", "they state opinions", "they state rules",
                     "they state concerns", "they state positions"],
        },
    }

    # Add dual-role stimuli
    dual_role_tokens = {}
    dual_role_tokens.update(adj_verb)
    dual_role_tokens.update(adj_noun)
    dual_role_tokens.update(noun_verb)

    for token, roles in dual_role_tokens.items():
        for role, sentences in roles.items():
            for sent in sentences:
                stimuli.append({
                    "sentence": sent,
                    "target_word": token,
                    "token_label": token,
                    "role_label": role,
                    "group": "dual_role",
                    "role_pair": "adj_verb" if token in adj_verb else
                                 ("adj_noun" if token in adj_noun else "noun_verb"),
                })

    # ---- Single-role control tokens (always adjective/operand) ----
    control_adj = {
        "happy": ["she is happy", "they are happy", "he seems happy",
                  "it looks happy", "we feel happy"],
        "sad": ["she is sad", "they are sad", "he seems sad",
                "it looks sad", "we feel sad"],
        "good": ["the food is good", "the result is good", "she is good",
                 "they look good", "it seems good"],
        "bad": ["the food is bad", "the result is bad", "she is bad",
                "they look bad", "it seems bad"],
        "tall": ["she is tall", "he is tall", "the building is tall",
                 "the tree is tall", "they are tall"],
        "short": ["she is short", "he is short", "the story is short",
                  "the time is short", "they are short"],
        "strong": ["she is strong", "he is strong", "the bridge is strong",
                   "the wind is strong", "they are strong"],
        "weak": ["she is weak", "he is weak", "the argument is weak",
                 "the signal is weak", "they are weak"],
        "fast": ["she is fast", "he is fast", "the car is fast",
                 "the train is fast", "they are fast"],
        "slow": ["she is slow", "he is slow", "the bus is slow",
                 "the process is slow", "they are slow"],
    }

    for token, sentences in control_adj.items():
        for sent in sentences:
            stimuli.append({
                "sentence": sent,
                "target_word": token,
                "token_label": token,
                "role_label": "adjective",
                "group": "control",
                "role_pair": "control",
            })

    return stimuli


# =====================================================================
# MODEL LOADING (BF16 + device_map=auto + flash)
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

    # Show layer distribution
    layers = get_layers(model)
    nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU layers: {len(gpu_l)}{' (' + str(gpu_l[0]) + '-' + str(gpu_l[-1]) + ')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' (' + str(cpu_l[0]) + '-' + str(cpu_l[-1]) + ')' if cpu_l else ''}")
    return model, tok


# =====================================================================
# CAPTURE HIDDEN STATES
# =====================================================================
def _capture_single(model, tokenizer, sent, n_layers, max_len=64):
    """Capture hidden states for a single sentence."""
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


# =====================================================================
# RESOLVE TOKEN POSITIONS
# =====================================================================
def _find_token_pos(decoded_tokens, target):
    """Find position of target token in decoded token list."""
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
    """Resolve token positions for all stimuli."""
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
# MEASUREMENT 1: VARIANCE DECOMPOSITION (CORE)
# =====================================================================
def variance_decomposition(captures, stimuli, n_layers, normalize=False):
    """
    Two-way ANOVA decomposition of hidden states:
      h_{ijk} = μ + α_i + β_j + (αβ)_{ij} + ε_{ijk}

    where:
      i = token, j = role, k = frame (replicate)
      μ = grand mean
      α_i = identity effect (token deviation from grand mean)
      β_j = role effect (role deviation from grand mean)
      (αβ)_{ij} = interaction effect
      ε_{ijk} = residual (within-cell variation)

    If normalize=True, each vector is divided by its norm first
    (removes norm effects, isolates directional effects).

    Returns dict of per-layer results.
    """
    # Only use dual-role tokens for the ANOVA
    dual_stimuli = [s for s in stimuli if s.get("group") == "dual_role"]

    # Group observations by (token, role)
    obs = defaultdict(list)  # (token, role) -> list of (capture_idx, pos)

    for stim in dual_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role)].append((idx, pos))

    # Identify dual-role tokens (appear in >=2 roles)
    token_roles = defaultdict(set)
    for (token, role) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    log(f"  Variance decomposition: {len(dual_tokens)} dual-role tokens, "
        f"{len(obs)} (token,role) cells, normalize={normalize}")

    results = {}
    for li in range(n_layers + 1):
        # Collect all vectors for this layer
        all_entries = []  # (token, role, vec)
        for (token, role), entries in obs.items():
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
                all_entries.append((token, role, v))

        if len(all_entries) < 10:
            continue

        # Compute grand mean
        all_v = np.array([v for _, _, v in all_entries])
        mu = np.mean(all_v, axis=0)

        # Compute per-token means
        token_vecs = defaultdict(list)
        for token, role, v in all_entries:
            token_vecs[token].append(v)
        token_means = {t: np.mean(vs, axis=0) for t, vs in token_vecs.items()}

        # Compute per-role means
        role_vecs = defaultdict(list)
        for token, role, v in all_entries:
            role_vecs[role].append(v)
        role_means = {r: np.mean(vs, axis=0) for r, vs in role_vecs.items()}

        # Compute per-cell means
        cell_vecs = defaultdict(list)
        for token, role, v in all_entries:
            cell_vecs[(token, role)].append(v)
        cell_means = {k: np.mean(vs, axis=0) for k, vs in cell_vecs.items()}

        # Compute SS
        total_ss = sum(np.sum((v - mu) ** 2) for _, _, v in all_entries)

        identity_ss = 0
        for token, mean_v in token_means.items():
            alpha = mean_v - mu
            n_t = len(token_vecs[token])
            identity_ss += n_t * np.sum(alpha ** 2)

        role_ss = 0
        for role, mean_v in role_means.items():
            beta = mean_v - mu
            n_r = len(role_vecs[role])
            role_ss += n_r * np.sum(beta ** 2)

        interaction_ss = 0
        for (token, role), mean_v in cell_means.items():
            alpha_beta = mean_v - mu - (token_means[token] - mu) - (role_means[role] - mu)
            n_tr = len(cell_vecs[(token, role)])
            interaction_ss += n_tr * np.sum(alpha_beta ** 2)

        residual_ss = total_ss - identity_ss - role_ss - interaction_ss
        residual_ss = max(residual_ss, 0)

        r_squared = (identity_ss + role_ss) / max(total_ss, 1e-10)

        # Identity-Role orthogonality: avg cosine between identity and role deviations
        ortho_cosines = []
        for token, alpha_v in token_means.items():
            for role, beta_v in role_means.items():
                a = alpha_v - mu
                b = beta_v - mu
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    ortho_cosines.append(float(np.dot(a, b) / (na * nb)))
        avg_ortho = float(np.mean(ortho_cosines)) if ortho_cosines else 0.0

        results[li] = {
            "total_ss": float(total_ss),
            "identity_ss": float(identity_ss),
            "role_ss": float(role_ss),
            "interaction_ss": float(interaction_ss),
            "residual_ss": float(residual_ss),
            "identity_ratio": float(identity_ss / max(total_ss, 1e-10)),
            "role_ratio": float(role_ss / max(total_ss, 1e-10)),
            "interaction_ratio": float(interaction_ss / max(total_ss, 1e-10)),
            "residual_ratio": float(residual_ss / max(total_ss, 1e-10)),
            "r_squared": float(r_squared),
            "identity_role_orthogonality": round(avg_ortho, 4),
            "n_observations": len(all_entries),
            "n_tokens": len(token_means),
            "n_roles": len(role_means),
        }

    return results


# =====================================================================
# MEASUREMENT 2: NORM EVOLUTION
# =====================================================================
def norm_evolution(captures, stimuli, n_layers):
    """
    Track norms of identity and role components across layers.

    For each dual-role token:
      - Identity norm: ||I_l(token)|| = ||mean_role(h_l) - μ_l||
      - Role norm per role: ||R_l(role)|| = ||mean_token(h_l) - μ_l||
    """
    dual_stimuli = [s for s in stimuli if s.get("group") == "dual_role"]

    obs = defaultdict(list)
    for stim in dual_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    identity_norms = defaultdict(dict)  # layer -> {token: norm}
    role_norms = defaultdict(dict)      # layer -> {role: norm}
    raw_norms = defaultdict(dict)       # layer -> {(token,role): mean_norm}

    for li in range(n_layers + 1):
        all_vecs = []
        for (token, role), entries in obs.items():
            if token not in dual_tokens:
                continue
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                v = h[0, pos, :].numpy().copy()
                all_vecs.append((token, role, v))

        if len(all_vecs) < 5:
            continue

        mu = np.mean([v for _, _, v in all_vecs], axis=0)

        # Token means
        token_vecs = defaultdict(list)
        for token, role, v in all_vecs:
            token_vecs[token].append(v)
        for token, vs in token_vecs.items():
            mean_v = np.mean(vs, axis=0)
            identity_norms[li][token] = float(np.linalg.norm(mean_v - mu))

        # Role means
        role_vecs = defaultdict(list)
        for token, role, v in all_vecs:
            role_vecs[role].append(v)
        for role, vs in role_vecs.items():
            mean_v = np.mean(vs, axis=0)
            role_norms[li][role] = float(np.linalg.norm(mean_v - mu))

        # Raw norms per (token, role)
        cell_vecs = defaultdict(list)
        for token, role, v in all_vecs:
            cell_vecs[(token, role)].append(v)
        for (token, role), vs in cell_vecs.items():
            mean_v = np.mean(vs, axis=0)
            raw_norms[li][(token, role)] = float(np.linalg.norm(mean_v))

    return identity_norms, role_norms, raw_norms


# =====================================================================
# MEASUREMENT 3: ROLE INCREMENT ANALYSIS
# =====================================================================
def role_increment_analysis(captures, stimuli, n_layers):
    """
    For each dual-role token, compute role increment vectors:
      Δ_l(t, r) = mean(h_l(t, r)) - I_l(t)

    where I_l(t) = mean across all roles for token t.

    Analyze:
    - Cross-token consistency: are role increments aligned across different tokens?
    - Norm of role increments vs identity norms
    """
    dual_stimuli = [s for s in stimuli if s.get("group") == "dual_role"]

    obs = defaultdict(list)
    for stim in dual_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {
        "increment_norms": defaultdict(dict),     # layer -> {token: {role: norm}}
        "increment_cosines": defaultdict(dict),    # layer -> {token: cos between role increments}
        "cross_token_cosines": defaultdict(dict),  # layer -> {role_pair_type: avg_cos}
    }

    for li in range(n_layers + 1):
        # Collect cell means
        cell_means = {}
        for (token, role), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 2:
                cell_means[(token, role)] = np.mean(vecs, axis=0)

        if len(cell_means) < 5:
            continue

        # Compute identity basis for each token
        token_identity = {}
        for token in dual_tokens:
            token_cell_vecs = [cell_means[(token, role)]
                               for role in token_roles[token]
                               if (token, role) in cell_means]
            if len(token_cell_vecs) >= 2:
                token_identity[token] = np.mean(token_cell_vecs, axis=0)

        # Compute role increments
        increments = {}  # (token, role) -> delta
        for (token, role), mean_v in cell_means.items():
            if token in token_identity:
                delta = mean_v - token_identity[token]
                increments[(token, role)] = delta
                nrm = float(np.linalg.norm(delta))
                if token not in results["increment_norms"][li]:
                    results["increment_norms"][li][token] = {}
                results["increment_norms"][li][token][role] = round(nrm, 4)

        # Per-token: cosine between role increments for the same token
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) == 2:
                r1, r2 = roles_list
                d1 = increments.get((token, r1))
                d2 = increments.get((token, r2))
                if d1 is not None and d2 is not None:
                    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_val = float(np.dot(d1, d2) / (n1 * n2))
                        results["increment_cosines"][li][token] = round(cos_val, 4)

        # Cross-token: cosine between role increments for different tokens
        # Group by role pair type
        role_pair_groups = defaultdict(list)  # (role1, role2) -> list of (delta1, delta2)
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) == 2:
                r1, r2 = roles_list
                d1 = increments.get((token, r1))
                d2 = increments.get((token, r2))
                if d1 is not None and d2 is not None:
                    role_pair_groups[(r1, r2)].append((d1, d2))

        for (r1, r2), deltas in role_pair_groups.items():
            if len(deltas) < 2:
                continue
            # Compute pairwise cosine between same-role increments across tokens
            cos_r1 = []
            cos_r2 = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    d1_i, d2_i = deltas[i]
                    d1_j, d2_j = deltas[j]
                    n1i, n1j = np.linalg.norm(d1_i), np.linalg.norm(d1_j)
                    n2i, n2j = np.linalg.norm(d2_i), np.linalg.norm(d2_j)
                    if n1i > 1e-8 and n1j > 1e-8:
                        cos_r1.append(float(np.dot(d1_i, d1_j) / (n1i * n1j)))
                    if n2i > 1e-8 and n2j > 1e-8:
                        cos_r2.append(float(np.dot(d2_i, d2_j) / (n2i * n2j)))

            key = f"{r1}_vs_{r2}"
            if cos_r1:
                results["cross_token_cosines"][li][f"{key}_{r1}"] = round(float(np.mean(cos_r1)), 4)
            if cos_r2:
                results["cross_token_cosines"][li][f"{key}_{r2}"] = round(float(np.mean(cos_r2)), 4)

    return results


# =====================================================================
# MEASUREMENT 4: IDENTITY PRESERVATION
# =====================================================================
def identity_preservation_analysis(captures, stimuli, n_layers):
    """
    For same token in different roles:
    - Cosine similarity of cell means
    - How much identity information is preserved across roles

    For different tokens in same role:
    - Cosine similarity of cell means
    - Baseline for comparison
    """
    dual_stimuli = [s for s in stimuli if s.get("group") == "dual_role"]

    obs = defaultdict(list)
    for stim in dual_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {
        "same_token_diff_role": defaultdict(dict),  # layer -> {token: cosine}
        "diff_token_same_role": defaultdict(dict),  # layer -> {role: avg_cosine}
    }

    for li in range(n_layers + 1):
        # Compute cell means
        cell_means = {}
        for (token, role), entries in obs.items():
            if token not in dual_tokens:
                continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 2:
                cell_means[(token, role)] = np.mean(vecs, axis=0)

        # Same token, different role
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) == 2:
                r1, r2 = roles_list
                m1 = cell_means.get((token, r1))
                m2 = cell_means.get((token, r2))
                if m1 is not None and m2 is not None:
                    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_val = float(np.dot(m1, m2) / (n1 * n2))
                        results["same_token_diff_role"][li][token] = round(cos_val, 4)

        # Different token, same role
        for role in set(r for _, r in cell_means.keys()):
            role_tokens = [t for (t, r) in cell_means if r == role]
            if len(role_tokens) < 2:
                continue
            cos_vals = []
            for i in range(len(role_tokens)):
                for j in range(i + 1, len(role_tokens)):
                    m1 = cell_means[(role_tokens[i], role)]
                    m2 = cell_means[(role_tokens[j], role)]
                    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_vals.append(float(np.dot(m1, m2) / (n1 * n2)))
            if cos_vals:
                results["diff_token_same_role"][li][role] = round(float(np.mean(cos_vals)), 4)

    return results


# =====================================================================
# MEASUREMENT 5: SUBSPACE OVERLAP
# =====================================================================
def subspace_overlap_analysis(captures, stimuli, n_layers, k_components=5):
    """
    Compute the overlap between identity subspace and role subspace.

    Method:
    1. Collect identity deviation vectors: α_i = mean_token(h) - μ
    2. Collect role deviation vectors: β_j = mean_role(h) - μ
    3. PCA on each set to get principal components
    4. Compute subspace overlap: how much of identity subspace lies in role subspace

    Returns per-layer overlap metrics.
    """
    dual_stimuli = [s for s in stimuli if s.get("group") == "dual_role"]

    obs = defaultdict(list)
    for stim in dual_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}

    for li in range(n_layers + 1):
        all_vecs = []
        for (token, role), entries in obs.items():
            if token not in dual_tokens:
                continue
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                v = h[0, pos, :].numpy().copy()
                all_vecs.append((token, role, v))

        if len(all_vecs) < 10:
            continue

        mu = np.mean([v for _, _, v in all_vecs], axis=0)

        # Identity deviation vectors
        token_vecs = defaultdict(list)
        for token, role, v in all_vecs:
            token_vecs[token].append(v)
        alpha_vecs = []
        for token, vs in token_vecs.items():
            mean_v = np.mean(vs, axis=0)
            alpha_vecs.append(mean_v - mu)

        # Role deviation vectors
        role_vecs = defaultdict(list)
        for token, role, v in all_vecs:
            role_vecs[role].append(v)
        beta_vecs = []
        for role, vs in role_vecs.items():
            mean_v = np.mean(vs, axis=0)
            beta_vecs.append(mean_v - mu)

        if len(alpha_vecs) < 2 or len(beta_vecs) < 2:
            continue

        alpha_mat = np.array(alpha_vecs)  # [n_tokens, d_model]
        beta_mat = np.array(beta_vecs)    # [n_roles, d_model]

        # PCA via SVD
        k_a = min(k_components, alpha_mat.shape[0] - 1, alpha_mat.shape[1])
        k_b = min(k_components, beta_mat.shape[0] - 1, beta_mat.shape[1])

        if k_a < 1 or k_b < 1:
            continue

        try:
            U_a, s_a, _ = np.linalg.svd(alpha_mat, full_matrices=False)
            U_b, s_b, _ = np.linalg.svd(beta_mat, full_matrices=False)

            # Subspace overlap: ||U_a^T @ U_b||_F^2 / min(k_a, k_b)
            # This measures how aligned the two subspaces are
            overlap_mat = U_a[:k_a, :] @ U_b[:k_b, :].T  # [k_a, k_b]
            overlap = float(np.sum(overlap_mat ** 2) / min(k_a, k_b))

            # Principal angles
            svd_overlap = np.linalg.svd(overlap_mat, compute_uv=False)
            principal_angles = np.arccos(np.clip(svd_overlap, 0, 1))

            results[li] = {
                "subspace_overlap": round(overlap, 4),
                "principal_angles_deg": [round(float(a * 180 / np.pi), 1) for a in principal_angles],
                "identity_singular_values": [round(float(s), 4) for s in s_a[:k_a]],
                "role_singular_values": [round(float(s), 4) for s in s_b[:k_b]],
                "identity_explained_ratio": round(float(np.sum(s_a[:k_a]**2) / max(np.sum(s_a**2), 1e-10)), 4),
                "role_explained_ratio": round(float(np.sum(s_b[:k_b]**2) / max(np.sum(s_b**2), 1e-10)), 4),
            }
        except Exception as e:
            results[li] = {"error": str(e)[:80]}

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase296_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 296: Identity-Role Residual Decomposition — {model_name}")
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

    # Filter stimuli with valid positions
    valid_stimuli = [s for s in stimuli if s.get("target_pos") is not None]
    log(f"  Valid stimuli: {len(valid_stimuli)}")

    # Count by group
    for group in ["dual_role", "control"]:
        n = sum(1 for s in valid_stimuli if s.get("group") == group)
        log(f"    {group}: {n}")

    # Count dual-role tokens
    dual_tokens = set()
    for s in valid_stimuli:
        if s.get("group") == "dual_role":
            dual_tokens.add(s["token_label"])
    log(f"    dual_role tokens: {len(dual_tokens)}")

    # Collect unique sentences
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

    # ---- 4. Measurement 1: Variance Decomposition (RAW) ----
    log(f"\n--- Measurement 1a: Variance Decomposition (RAW) ---")
    t0 = time.time()
    vd_raw = variance_decomposition(captures, valid_stimuli, nl, normalize=False)
    log(f"  Done in {time.time()-t0:.0f}s, {len(vd_raw)} layers")

    # Print summary
    sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl]) & set(vd_raw.keys()))
    log("  Layer | Identity% | Role% | Interact% | Residual% | R² | I-R Ortho")
    for li in sorted(vd_raw.keys()):
        r = vd_raw[li]
        if li in sample_layers or li >= nl - 2:
            log(f"  L{li:3d} | {r['identity_ratio']*100:7.2f}% | {r['role_ratio']*100:5.2f}% | "
                f"{r['interaction_ratio']*100:8.2f}% | {r['residual_ratio']*100:8.2f}% | "
                f"{r['r_squared']:.4f} | {r['identity_role_orthogonality']:+.4f}")

    # ---- 5. Measurement 1b: Variance Decomposition (NORMALIZED) ----
    log(f"\n--- Measurement 1b: Variance Decomposition (NORMALIZED) ---")
    t0 = time.time()
    vd_norm = variance_decomposition(captures, valid_stimuli, nl, normalize=True)
    log(f"  Done in {time.time()-t0:.0f}s, {len(vd_norm)} layers")

    log("  Layer | Identity% | Role% | Interact% | Residual% | R² | I-R Ortho")
    for li in sorted(vd_norm.keys()):
        r = vd_norm[li]
        if li in sample_layers or li >= nl - 2:
            log(f"  L{li:3d} | {r['identity_ratio']*100:7.2f}% | {r['role_ratio']*100:5.2f}% | "
                f"{r['interaction_ratio']*100:8.2f}% | {r['residual_ratio']*100:8.2f}% | "
                f"{r['r_squared']:.4f} | {r['identity_role_orthogonality']:+.4f}")

    # ---- 6. Measurement 2: Norm Evolution ----
    log(f"\n--- Measurement 2: Norm Evolution ---")
    id_norms, role_norms, raw_norms = norm_evolution(captures, valid_stimuli, nl)

    log("  Layer | AvgIdNorm | AvgRoleNorm | Id/Role Ratio")
    for li in sorted(id_norms.keys()):
        if li in sample_layers or li >= nl - 2:
            avg_id = np.mean(list(id_norms[li].values())) if id_norms[li] else 0
            avg_role = np.mean(list(role_norms[li].values())) if role_norms[li] else 0
            ratio = avg_id / max(avg_role, 1e-8)
            log(f"  L{li:3d} | {avg_id:9.4f} | {avg_role:11.4f} | {ratio:8.2f}x")

    # ---- 7. Measurement 3: Role Increment Analysis ----
    log(f"\n--- Measurement 3: Role Increment Analysis ---")
    ri_results = role_increment_analysis(captures, valid_stimuli, nl)

    log("  Layer | AvgIncrCos | CrossTokenCos")
    for li in sorted(ri_results["increment_cosines"].keys()):
        if li in sample_layers or li >= nl - 2:
            cos_vals = list(ri_results["increment_cosines"][li].values())
            avg_cos = np.mean(cos_vals) if cos_vals else 0

            cross_cos = list(ri_results["cross_token_cosines"][li].values())
            avg_cross = np.mean(cross_cos) if cross_cos else 0

            log(f"  L{li:3d} | {avg_cos:+.4f} | {avg_cross:+.4f}")

    # ---- 8. Measurement 4: Identity Preservation ----
    log(f"\n--- Measurement 4: Identity Preservation ---")
    ip_results = identity_preservation_analysis(captures, valid_stimuli, nl)

    log("  Layer | SameTokenDiffRole | DiffTokenSameRole")
    for li in sorted(ip_results["same_token_diff_role"].keys()):
        if li in sample_layers or li >= nl - 2:
            st_vals = list(ip_results["same_token_diff_role"][li].values())
            dt_vals = list(ip_results["diff_token_same_role"][li].values())
            avg_st = np.mean(st_vals) if st_vals else 0
            avg_dt = np.mean(dt_vals) if dt_vals else 0
            log(f"  L{li:3d} | {avg_st:.4f} | {avg_dt:.4f}")

    # ---- 9. Measurement 5: Subspace Overlap ----
    log(f"\n--- Measurement 5: Subspace Overlap ---")
    so_results = subspace_overlap_analysis(captures, valid_stimuli, nl)

    log("  Layer | Overlap | MinAngle | IdExplRatio | RoleExplRatio")
    for li in sorted(so_results.keys()):
        r = so_results[li]
        if "error" in r:
            continue
        if li in sample_layers or li >= nl - 2:
            min_angle = min(r["principal_angles_deg"]) if r["principal_angles_deg"] else 0
            log(f"  L{li:3d} | {r['subspace_overlap']:.4f} | {min_angle:7.1f}° | "
                f"{r['identity_explained_ratio']:.4f} | {r['role_explained_ratio']:.4f}")

    # ---- 10. Save results ----
    log(f"\n--- Saving results ---")

    # Convert defaultdicts to regular dicts for JSON
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
        "n_dual_role_tokens": len(dual_tokens),
        "variance_decomposition_raw": {str(li): r for li, r in vd_raw.items()},
        "variance_decomposition_normalized": {str(li): r for li, r in vd_norm.items()},
        "identity_norms": convert_keys(id_norms),
        "role_norms": convert_keys(role_norms),
        "role_increment": convert_keys(ri_results),
        "identity_preservation": convert_keys(ip_results),
        "subspace_overlap": {str(li): r for li, r in so_results.items()},
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULT_DIR / f"{model_name}_residual_decomposition.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ---- 11. Print final summary ----
    log(f"\n{'='*60}")
    log(f"PHASE 296 SUMMARY — {model_name}")
    log(f"{'='*60}")

    log("\n  1. Variance Decomposition (RAW vs NORMALIZED):")
    for li in sorted(vd_raw.keys()):
        if li in sample_layers or li >= nl - 2:
            rr = vd_raw[li]
            rn = vd_norm.get(li, {})
            log(f"    L{li:3d}: RAW id={rr['identity_ratio']*100:.1f}% role={rr['role_ratio']*100:.1f}% "
                f"| NORM id={rn.get('identity_ratio',0)*100:.1f}% role={rn.get('role_ratio',0)*100:.1f}% "
                f"| R²={rr['r_squared']:.3f}")

    log("\n  2. Identity-Role Orthogonality:")
    for li in sorted(vd_raw.keys()):
        if li in sample_layers or li >= nl - 2:
            rr = vd_raw[li]
            rn = vd_norm.get(li, {})
            log(f"    L{li:3d}: RAW ortho={rr['identity_role_orthogonality']:+.4f} "
                f"| NORM ortho={rn.get('identity_role_orthogonality',0):+.4f}")

    log("\n  3. Identity Preservation (Same Token, Diff Role):")
    for li in sorted(ip_results["same_token_diff_role"].keys()):
        if li in sample_layers or li >= nl - 2:
            vals = list(ip_results["same_token_diff_role"][li].values())
            log(f"    L{li:3d}: avg_cos={np.mean(vals):.4f} (n={len(vals)} tokens)")

    log("\n  4. Role Increment Consistency (Cross-Token):")
    for li in sorted(ri_results["cross_token_cosines"].keys()):
        if li in sample_layers or li >= nl - 2:
            vals = ri_results["cross_token_cosines"][li]
            log(f"    L{li:3d}: {dict(vals)}")

    # ---- 12. Release model ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    log(f"Phase 296 complete for {model_name}!")


if __name__ == "__main__":
    main()
