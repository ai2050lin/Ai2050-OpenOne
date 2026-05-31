"""
Phase 304: Construction Identification + Gap Decomposition
===========================================================
Fixes two critical issues from Phase 303:
1. Frame (Construction) identifiability: within-role frame PCA instead of ANOVA marginal F
2. DS7B full_delta ≈ 0: Gap = full_delta - R decomposition

Key innovations:
- Within-role frame PCA: compute frame variation within each role, not across roles
  → Avoids ANOVA unbalanced design artifacts (F≈0, RF=-R)
- Gap decomposition: Gap = full_delta - R_only
  → What cancels R in DS7B? Position? Construction? Norm?
- Cross-role frame subspace comparison: Is C(construction) role-conditioned?
- Causal test of Construction direction and Gap components

Theory update: I + R + C + U (Construction + Unresolved)

Usage:
  python tests/glm5/phase304_construction_gap.py qwen3
  python tests/glm5/phase304_construction_gap.py glm4
  python tests/glm5/phase304_construction_gap.py deepseek7b
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

RESULT_DIR = Path("results/phase304_construction_gap")
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
# TOKEN DEFINITIONS — reuse Phase 303's 60 tokens
# =====================================================================
ADJ_VERB_TOKENS = {
    "open":   (["door", "gate"],   ["door", "gate"]),
    "clear":  (["path", "sky"],    ["path", "desk"]),
    "warm":   (["room", "water"],  ["room", "water"]),
    "clean":  (["floor", "table"], ["floor", "room"]),
    "dry":    (["ground", "cloth"],["cloth", "clothes"]),
    "close":  (["store", "school"],["door", "gate"]),
    "free":   (["bird", "person"], ["bird", "person"]),
    "quiet":  (["room", "house"],  ["room", "crowd"]),
    "cool":   (["water", "air"],   ["water", "room"]),
    "smooth": (["surface", "road"],["surface", "fabric"]),
    "empty":  (["room", "box"],    ["room", "box"]),
    "slow":   (["car", "train"],   ["car", "process"]),
    "dim":    (["light", "room"],  ["light", "screen"]),
    "double": (["bed", "door"],    ["amount", "size"]),
    "narrow": (["road", "gap"],    ["gap", "search"]),
    "level":  (["ground", "surface"],["field", "ground"]),
    "thin":   (["cloth", "ice"],   ["paint", "soup"]),
    "lower":  (["floor", "price"], ["price", "flag"]),
    "alert":  (["guard", "dog"],   ["guard", "team"]),
    "blunt":  (["knife", "edge"],  ["knife", "edge"]),
}

ADJ_NOUN_TOKENS = {
    "light":  (["bag", "box"],     ["bright", "warm"]),
    "cold":   (["water", "wind"],  ["severe", "bitter"]),
    "right":  (["answer", "choice"],["clear", "important"]),
    "fair":   (["price", "game"],  ["large", "popular"]),
    "round":  (["table", "ball"],  ["final", "last"]),
    "solid":  (["ground", "wall"], ["hard", "dense"]),
    "dark":   (["room", "night"],  ["deep", "cold"]),
    "plain":  (["food", "style"],  ["flat", "vast"]),
    "fine":   (["art", "weather"], ["heavy", "small"]),
    "grave":  (["matter", "risk"], ["deep", "old"]),
    "sweet":  (["taste", "fruit"], ["pure", "fresh"]),
    "green":  (["grass", "field"], ["bright", "dark"]),
    "flat":   (["surface", "tire"],["empty", "wide"]),
    "square": (["table", "room"],  ["central", "busy"]),
    "prime":  (["time", "spot"],   ["young", "golden"]),
    "waste":  (["land", "time"],   ["total", "industrial"]),
    "deal":   (["table", "board"], ["great", "fair"]),
    "match":  (["wood", "box"],    ["close", "perfect"]),
    "blue":   (["sky", "dress"],   ["deep", "clear"]),
    "mean":   (["person", "look"], ["average", "middle"]),
}

NOUN_VERB_TOKENS = {
    "fire":   (["hot", "big"],     ["gun", "worker"],   None, True),
    "record": (["old", "broken"],  ["music", "data"],   None, True),
    "run":    (["long", "hard"],   ["program", "company"], None, True),
    "play":   (["good", "long"],   ["music", "tennis"], None, True),
    "sign":   (["clear", "large"], ["paper", "contract"],None, True),
    "state":  (["large", "rich"],  ["facts", "rules"],  None, True),
    "book":   (["new", "long"],    ["room", "ticket"],  None, True),
    "paint":  (["fresh", "bright"],["wall", "house"],    None, True),
    "plant":  (["green", "large"], ["seed", "tree"],     None, True),
    "walk":   (["long", "short"],  ["dog", "path"],      None, True),
    "drink":  (["cold", "hot"],    ["water", "wine"],    None, True),
    "dream":  (["strange", "vivid"],None, ["child", "person"], False),
    "hope":   (["new", "faint"],   None, ["person", "group"], False),
    "love":   (["true", "deep"],   ["person", "child"],  None, True),
    "fear":   (["deep", "old"],    ["dark", "change"],   None, True),
    "doubt":  (["deep", "serious"],["claim", "story"],   None, True),
    "trust":  (["complete", "mutual"],["person", "process"],None, True),
    "face":   (["pale", "bright"], ["problem", "fact"],  None, True),
    "hand":   (["left", "small"],  ["paper", "note"],    None, True),
    "mark":   (["clear", "deep"],  ["paper", "target"],  None, True),
}

# =====================================================================
# FRAME TEMPLATES — same as Phase 303
# =====================================================================
ADJ_FRAMES = [
    ("F1_copula",  "the {obj} is {token}"),
    ("F2_remain",  "the {obj} remains {token}"),
    ("F3_attrib",  "the {token} {obj}"),
    ("F4_seem",    "the {obj} seemed {token}"),
    ("F5_become",  "the {obj} became {token}"),
    ("F6_feel",    "the {obj} felt {token}"),
    ("F7_look",    "the {obj} looked {token}"),
]

VERB_FRAMES_TRANSITIVE = [
    ("F1_transitive",  "they {token} the {obj}"),
    ("F2_intransitive","the {obj} will {token}"),
    ("F3_begin",       "they began to {token} the {obj}"),
    ("F4_modal",       "they can {token} the {obj}"),
    ("F5_causative",   "they made the {obj} {token}"),
    ("F6_try",         "they tried to {token} the {obj}"),
    ("F7_want",        "they wanted to {token} the {obj}"),
]

VERB_FRAMES_INTRANSITIVE = [
    ("F1_will",      "the {subj} will {token}"),
    ("F2_causative", "they made the {subj} {token}"),
    ("F3_begin",     "the {subj} began to {token}"),
    ("F4_modal",     "the {subj} can {token}"),
    ("F5_try",       "the {subj} tried to {token}"),
    ("F6_start",     "the {subj} started to {token}"),
    ("F7_continue",  "the {subj} continued to {token}"),
]

NOUN_FRAMES = [
    ("F1_copula",       "the {token} is {adj}"),
    ("F2_exist",        "that {token} is {adj}"),
    ("F3_locative_a",   "near the {token}"),
    ("F3_locative_b",   "by the {token}"),
    ("F4_action_a",     "they saw the {token}"),
    ("F4_action_b",     "they found the {token}"),
    ("F5_possessive",   "her {token} is {adj}"),
]


# =====================================================================
# STIMULUS GENERATION
# =====================================================================
def build_stimuli():
    """Generate observation stimulus set (same as Phase 303)."""
    stimuli = []
    
    for token, (adj_objs, verb_objs) in ADJ_VERB_TOKENS.items():
        for frame_name, template in ADJ_FRAMES:
            for obj in adj_objs:
                stimuli.append({
                    "sentence": template.format(token=token, obj=obj),
                    "target_word": token, "token_label": token, "role_label": "adj",
                    "frame_label": frame_name, "role_pair": "adj_verb",
                })
        for frame_name, template in VERB_FRAMES_TRANSITIVE:
            for obj in verb_objs:
                stimuli.append({
                    "sentence": template.format(token=token, obj=obj),
                    "target_word": token, "token_label": token, "role_label": "verb",
                    "frame_label": frame_name, "role_pair": "adj_verb",
                })
    
    for token, (adj_objs, noun_adjs) in ADJ_NOUN_TOKENS.items():
        for frame_name, template in ADJ_FRAMES:
            for obj in adj_objs:
                stimuli.append({
                    "sentence": template.format(token=token, obj=obj),
                    "target_word": token, "token_label": token, "role_label": "adj",
                    "frame_label": frame_name, "role_pair": "adj_noun",
                })
        for frame_name, template in NOUN_FRAMES:
            if "{adj}" in template:
                for adj in noun_adjs:
                    stimuli.append({
                        "sentence": template.format(token=token, adj=adj),
                        "target_word": token, "token_label": token, "role_label": "noun",
                        "frame_label": frame_name, "role_pair": "adj_noun",
                    })
            else:
                stimuli.append({
                    "sentence": template.format(token=token),
                    "target_word": token, "token_label": token, "role_label": "noun",
                    "frame_label": frame_name, "role_pair": "adj_noun",
                })
    
    for token, (noun_adjs, verb_objs, verb_subjs, is_trans) in NOUN_VERB_TOKENS.items():
        for frame_name, template in NOUN_FRAMES:
            if "{adj}" in template:
                for adj in noun_adjs:
                    stimuli.append({
                        "sentence": template.format(token=token, adj=adj),
                        "target_word": token, "token_label": token, "role_label": "noun",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
            else:
                stimuli.append({
                    "sentence": template.format(token=token),
                    "target_word": token, "token_label": token, "role_label": "noun",
                    "frame_label": frame_name, "role_pair": "noun_verb",
                })
        if is_trans:
            for frame_name, template in VERB_FRAMES_TRANSITIVE:
                for obj in verb_objs:
                    stimuli.append({
                        "sentence": template.format(token=token, obj=obj),
                        "target_word": token, "token_label": token, "role_label": "verb",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
        else:
            for frame_name, template in VERB_FRAMES_INTRANSITIVE:
                for subj in verb_subjs:
                    stimuli.append({
                        "sentence": template.format(token=token, subj=subj),
                        "target_word": token, "token_label": token, "role_label": "verb",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
    
    return stimuli


def build_causal_stimuli():
    """Causal test pairs: one sentence per (token, role)."""
    test_pairs = []
    
    for token, (adj_objs, verb_objs) in ADJ_VERB_TOKENS.items():
        test_pairs.append(("the {obj} is {token}".format(token=token, obj=adj_objs[0]),
                          token, "adj", "adj_verb"))
        test_pairs.append(("they {token} the {obj}".format(token=token, obj=verb_objs[0]),
                          token, "verb", "adj_verb"))
    
    for token, (adj_objs, noun_adjs) in ADJ_NOUN_TOKENS.items():
        test_pairs.append(("the {obj} is {token}".format(token=token, obj=adj_objs[0]),
                          token, "adj", "adj_noun"))
        test_pairs.append(("the {token} is {adj}".format(token=token, adj=noun_adjs[0]),
                          token, "noun", "adj_noun"))
    
    for token, (noun_adjs, verb_objs, verb_subjs, is_trans) in NOUN_VERB_TOKENS.items():
        test_pairs.append(("the {token} is {adj}".format(token=token, adj=noun_adjs[0]),
                          token, "noun", "noun_verb"))
        if is_trans:
            test_pairs.append(("they {token} the {obj}".format(token=token, obj=verb_objs[0]),
                              token, "verb", "noun_verb"))
        else:
            test_pairs.append(("the {subj} will {token}".format(token=token, subj=verb_subjs[0]),
                              token, "verb", "noun_verb"))
    
    stimuli = []
    for sent, target, role, rp in test_pairs:
        stimuli.append({
            "sentence": sent, "target_word": target,
            "token_label": target, "role_label": role,
            "frame_label": "causal_test", "role_pair": rp,
            "group": "causal_test",
        })
    return stimuli


# =====================================================================
# MODEL LOADING — BF16 + device_map="auto" + flash_attn
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto + flash_attn)...")
    
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                        local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl)
            log(f"  attn_implementation={attn_impl} succeeded")
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:120]}")
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    # Verify all layers are accessible
    layers = get_layers(model)
    log(f"  n_layers={len(layers)}, class={type(model).__name__}")
    
    # Check layer distribution for device_map="auto" models
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in str(v))
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in str(v))
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU (total {len(layers)})")
        # Check for deep layer gaps
        gpu_layer_ids = [int(lid) for lid, dev in layer_devices.items() if 'cuda' in str(dev)]
        if gpu_layer_ids and gpu_layers < len(layers):
            last_gpu_layer = max(gpu_layer_ids)
            log(f"  Last GPU layer: {last_gpu_layer}, first CPU layer: {last_gpu_layer + 1}")
    
    return model, tok


# =====================================================================
# CAPTURE & POSITION UTILITIES
# =====================================================================
def _capture_single(model, tokenizer, sent, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    logits = out.logits.detach().cpu().float()
    return {"hidden": hs, "logits": logits, "input_ids": inputs["input_ids"].cpu()}


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
    for stim in stimuli:
        toks = tokenizer.encode(stim["sentence"], add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, stim["target_word"])
        if pos is not None:
            new_stim = dict(stim)
            new_stim["target_pos"] = pos
            new_stim["n_tokens"] = len(toks)
            resolved.append(new_stim)
    return resolved


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def project_onto(v, direction):
    """Project v onto direction (unit vector or not)."""
    dn = np.linalg.norm(direction)
    if dn < 1e-10:
        return np.zeros_like(v)
    d_hat = direction / dn
    return np.dot(v, d_hat) * d_hat


# =====================================================================
# ACTIVATION PATCHING
# =====================================================================
def run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, patch_vec, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    layers = get_layers(model)
    injection_done = [False]
    patch_tensor = torch.tensor(patch_vec, dtype=torch.bfloat16, device=input_device)
    
    def inject_hook(module, input, output):
        if not injection_done[0]:
            out_tuple = list(output)
            out_tuple[0] = out_tuple[0].clone()
            out_tuple[0][0, pos, :] += patch_tensor.to(out_tuple[0].dtype)
            injection_done[0] = True
            return tuple(out_tuple)
        return output
    
    handle = layers[layer_idx].register_forward_hook(inject_hook)
    
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, output_hidden_states=False)
            patched_logits = out.logits.detach().cpu().float()
        except Exception as e:
            patched_logits = None
    
    handle.remove()
    return patched_logits


# =====================================================================
# WITHIN-ROLE FRAME PCA (Construction Identification)
# =====================================================================
def within_role_frame_pca(cell_means_by_role_frame, d_model, n_components=5):
    """
    Within-role frame PCA: identify frame variation within each role.
    
    For each role (adj, verb, noun):
    1. For each (token, role), compute mean across frames
    2. Compute frame deviation = h(token, role, frame) - mean(token, role)
    3. Stack frame deviations for all tokens within that role
    4. PCA on deviation matrix → get construction subspace
    
    Args:
        cell_means_by_role_frame: {(token, role, frame): vector}
        d_model: model dimension
        n_components: number of PCA components
    
    Returns:
        {role: {"pca_components": [n_comp, d_model], "explained_var": [...], 
                "mean_construction": vector, "n_observations": int}}
    """
    from sklearn.decomposition import PCA
    
    # Group by role
    role_data = defaultdict(list)  # {role: [(token, frame, deviation_vector)]}
    
    # First, compute per-(token, role) means
    token_role_vecs = defaultdict(list)
    for (token, role, frame), vec in cell_means_by_role_frame.items():
        token_role_vecs[(token, role)].append(vec)
    
    token_role_means = {}
    for (token, role), vecs in token_role_vecs.items():
        token_role_means[(token, role)] = np.mean(vecs, axis=0)
    
    # Compute frame deviations
    for (token, role, frame), vec in cell_means_by_role_frame.items():
        tr_mean = token_role_means.get((token, role))
        if tr_mean is not None:
            deviation = vec - tr_mean
            role_data[role].append((token, frame, deviation))
    
    results = {}
    for role, data in role_data.items():
        if len(data) < n_components:
            log(f"  Within-role PCA: {role} has only {len(data)} observations, skipping")
            continue
        
        # Stack deviations into matrix
        dev_matrix = np.array([d[2] for d in data])  # [n_obs, d_model]
        
        # PCA
        n_comp = min(n_components, dev_matrix.shape[0] - 1, dev_matrix.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(dev_matrix)
        
        results[role] = {
            "pca_components": pca.components_.tolist(),  # [n_comp, d_model]
            "explained_var_ratio": pca.explained_variance_ratio_.tolist(),
            "explained_var": pca.explained_variance_.tolist(),
            "mean_construction": np.mean(dev_matrix, axis=0).tolist(),
            "n_observations": len(data),
            "n_tokens": len(set(d[0] for d in data)),
        }
        
        log(f"  Within-role PCA [{role}]: {len(data)} obs from {results[role]['n_tokens']} tokens, "
            f"top-5 var ratio: {pca.explained_variance_ratio_[:5]}")
    
    return results


def compute_cross_role_subspace_angles(pca_results):
    """Compare construction subspaces across roles via subspace angles."""
    from scipy.linalg import subspace_angles
    
    roles = sorted(pca_results.keys())
    angles = {}
    
    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):
            if i >= j:
                continue
            comp1 = np.array(pca_results[r1]["pca_components"])  # [k1, d]
            comp2 = np.array(pca_results[r2]["pca_components"])  # [k2, d]
            
            if comp1.shape[0] == 0 or comp2.shape[0] == 0:
                continue
            
            # Subspace angles (in degrees)
            try:
                sa = subspace_angles(comp1.T, comp2.T)
                angles[f"{r1}_vs_{r2}"] = {
                    "angles_deg": [float(a * 180 / np.pi) for a in sa],
                    "min_angle_deg": float(min(sa) * 180 / np.pi),
                    "mean_angle_deg": float(np.mean(sa) * 180 / np.pi),
                }
                log(f"  Subspace angle {r1} vs {r2}: "
                    f"min={angles[f'{r1}_vs_{r2}']['min_angle_deg']:.1f}° "
                    f"mean={angles[f'{r1}_vs_{r2}']['mean_angle_deg']:.1f}°")
            except Exception as e:
                log(f"  Subspace angle {r1} vs {r2} failed: {e}")
    
    return angles


# =====================================================================
# GAP DECOMPOSITION
# =====================================================================
def decompose_gap(full_delta, R_direction, construction_pca, position_directions, d_model):
    """
    Decompose Gap = full_delta - R into components:
    1. C_construction: projection onto within-role frame PCA subspace
    2. P_position: projection onto position direction  
    3. N_norm: projection onto norm direction
    4. U_unresolved: residual
    
    Args:
        full_delta: full difference vector [d_model]
        R_direction: role direction vector [d_model]
        construction_pca: dict from within_role_frame_pca for the relevant role
        position_directions: dict {role: position_direction}
        d_model: model dimension
    
    Returns:
        dict with Gap, C, P, N, U vectors and their norms/cosines
    """
    gap = full_delta - R_direction
    gap_norm = np.linalg.norm(gap)
    R_norm = np.linalg.norm(R_direction)
    fd_norm = np.linalg.norm(full_delta)
    
    result = {
        "gap_norm": float(gap_norm),
        "R_norm": float(R_norm),
        "full_delta_norm": float(fd_norm),
        "cos_gap_R": float(cosine_sim(gap, R_direction)),
        "cos_gap_full": float(cosine_sim(gap, full_delta)),
    }
    
    # 1. Construction component: project gap onto PCA subspace
    if construction_pca and "pca_components" in construction_pca:
        comps = np.array(construction_pca["pca_components"])  # [k, d]
        # Project gap onto each PCA component
        proj_coeffs = comps @ gap  # [k]
        # Reconstruction from PCA subspace
        C_vec = proj_coeffs @ comps  # [d_model]
        result["C_vec_norm"] = float(np.linalg.norm(C_vec))
        result["C_proj_energy"] = float(np.sum(proj_coeffs**2) / max(gap_norm**2, 1e-20))
        result["cos_gap_C"] = float(cosine_sim(gap, C_vec))
        # Per-component projection
        for ci in range(min(3, len(proj_coeffs))):
            result[f"C_pc{ci+1}_coeff"] = float(proj_coeffs[ci])
            result[f"C_pc{ci+1}_var_ratio"] = float(construction_pca["explained_var_ratio"][ci])
    else:
        C_vec = np.zeros(d_model)
        result["C_vec_norm"] = 0.0
        result["C_proj_energy"] = 0.0
        result["cos_gap_C"] = 0.0
    
    # 2. Position component
    # position_directions is {role: vec}, use the target role's position direction
    P_vec = np.zeros(d_model)
    if position_directions:
        # Average position direction
        pos_dirs = [v for v in position_directions.values() if v is not None and np.linalg.norm(v) > 1e-10]
        if pos_dirs:
            avg_pos = np.mean(pos_dirs, axis=0)
            P_vec = project_onto(gap, avg_pos)
    result["P_vec_norm"] = float(np.linalg.norm(P_vec))
    result["cos_gap_P"] = float(cosine_sim(gap, P_vec))
    
    # 3. Norm direction (along the mean direction)
    N_vec = project_onto(gap, full_delta)  # project gap onto full_delta direction
    result["N_vec_norm"] = float(np.linalg.norm(N_vec))
    result["cos_gap_N"] = float(cosine_sim(gap, N_vec))
    
    # 4. Unresolved residual
    U_vec = gap - C_vec - P_vec
    result["U_vec_norm"] = float(np.linalg.norm(U_vec))
    result["U_norm_pct"] = float(np.linalg.norm(U_vec) / max(gap_norm, 1e-10) * 100)
    
    return result, gap, C_vec, P_vec


# =====================================================================
# MAKE SERIALIZABLE
# =====================================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase304_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 304: Construction Identification + Gap Decomposition -- {model_name}")
    
    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")
    
    # Verify deep layer accessibility
    layers = get_layers(model)
    test_layers = [0, nl // 2, nl - 2]
    for tl in test_layers:
        try:
            _ = layers[tl]
            log(f"  Layer {tl}: accessible")
        except Exception as e:
            log(f"  Layer {tl}: FAILED - {e}")
    
    # ---- Build stimuli ----
    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)
    log(f"  Observation stimuli resolved: {len(sub_stimuli)}, Causal test stimuli: {len(causal_stimuli)}")
    
    # Count tokens and roles
    token_roles = defaultdict(set)
    token_frames = defaultdict(set)
    token_rp = {}
    for stim in sub_stimuli:
        token_roles[stim["token_label"]].add(stim["role_label"])
        token_frames[stim["token_label"]].add(stim["frame_label"])
        token_rp[stim["token_label"]] = stim.get("role_pair", "")
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])
    log(f"  Dual-role tokens: {len(dual_tokens)}")
    
    # Per-role-pair counts
    rp_counts = defaultdict(int)
    for t in dual_tokens:
        rp_counts[token_rp.get(t, "")] += 1
    for rp, cnt in sorted(rp_counts.items()):
        log(f"    {rp}: {cnt} tokens")
    
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
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} unique sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # Organize observation data
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        frame = stim.get("frame_label", "")
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, frame)].append((idx, pos))
    
    # Organize causal test pairs
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim
    dual_test = [(t, sorted(rs.keys())) for t, rs in test_pairs.items() if len(rs) >= 2]
    log(f"  Causal test pairs: {len(dual_test)} tokens with both roles")
    
    # ---- Layer selection: 8 layers ----
    sample_layers = sorted(set([
        max(1, nl // 8), max(1, nl // 4), max(1, 3 * nl // 8),
        nl // 2, 5 * nl // 8, 3 * nl // 4, 7 * nl // 8, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")
    
    # =====================================================================
    # PART A: WITHIN-ROLE FRAME PCA (CONSTRUCTION IDENTIFICATION)
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: WITHIN-ROLE FRAME PCA (CONSTRUCTION IDENTIFICATION)")
    log(f"{'='*60}")
    
    all_pca_results = {}
    all_subspace_angles = {}
    all_construction_cosines = {}  # {(token, role): cos(h, C_direction)}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Within-role frame PCA ---")
        
        # Compute cell means
        cell_means = {}
        for (token, role, frame), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, frame)] = np.mean(vecs, axis=0)
        
        # Within-role frame PCA
        pca_results = within_role_frame_pca(cell_means, d_model, n_components=5)
        
        # Cross-role subspace comparison
        if len(pca_results) >= 2:
            subspace_angles = compute_cross_role_subspace_angles(pca_results)
        else:
            subspace_angles = {}
        
        # Compute per-token construction alignment
        for (token, role, frame), vec in cell_means.items():
            if role in pca_results and "pca_components" in pca_results[role]:
                comps = np.array(pca_results[role]["pca_components"])  # [k, d]
                # Project onto first component
                c1 = comps[0]
                cos_c1 = cosine_sim(vec, c1)
                if (token, role) not in all_construction_cosines:
                    all_construction_cosines[(token, role)] = {}
                all_construction_cosines[(token, role)][str(li)] = float(cos_c1)
        
        all_pca_results[str(li)] = make_serializable(pca_results)
        all_subspace_angles[str(li)] = make_serializable(subspace_angles)
    
    # =====================================================================
    # PART B: FACTORIAL DECOMPOSITION + GAP DECOMPOSITION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART B: FACTORIAL DECOMPOSITION + GAP DECOMPOSITION")
    log(f"{'='*60}")
    
    causal_results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Factorial + Gap ---")
        
        # Compute cell means
        cell_means = {}
        for (token, role, frame), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, frame)] = np.mean(vecs, axis=0)
        
        # Factorial decomposition
        decomp = {}
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            frames_list = sorted(token_frames[token])
            
            cells = {}
            for role in roles_list:
                for frame in frames_list:
                    key = (token, role, frame)
                    if key in cell_means:
                        cells[(role, frame)] = cell_means[key]
            
            if len(cells) < 4:
                continue
            
            all_vecs = list(cells.values())
            grand_mean = np.mean(all_vecs, axis=0)
            
            role_means = {}
            for role in roles_list:
                r_vecs = [cells[(role, f)] for f in frames_list if (role, f) in cells]
                if r_vecs:
                    role_means[role] = np.mean(r_vecs, axis=0)
            
            if r1 in role_means and r2 in role_means:
                R_direction = role_means[r2] - role_means[r1]
            else:
                continue
            
            decomp[token] = {
                "R_direction": R_direction,
                "r1": r1, "r2": r2,
                "grand_mean": grand_mean,
                "role_means": {r: role_means[r] for r in roles_list if r in role_means},
            }
        
        # Get PCA results for this layer
        pca_this_layer = {}
        li_str = str(li)
        if li_str in all_pca_results:
            for role, pdata in all_pca_results[li_str].items():
                if "pca_components" in pdata:
                    pca_this_layer[role] = pdata
        
        # ---- Causal test + Gap decomposition ----
        layer_results = {}
        
        for ti, (token, roles_list) in enumerate(dual_test):
            if len(roles_list) != 2:
                continue
            r1, r2 = roles_list
            s1 = test_pairs[token][r1]
            s2 = test_pairs[token][r2]
            
            idx1 = s1.get("_idx")
            pos1 = s1.get("target_pos")
            idx2 = s2.get("_idx")
            pos2 = s2.get("target_pos")
            if idx1 is None or idx2 is None:
                continue
            
            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None:
                continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]:
                continue
            
            logits1 = captures[idx1]["logits"][0, -1, :].numpy().copy()
            logits2 = captures[idx2]["logits"][0, -1, :].numpy().copy()
            target_shift = logits2 - logits1
            
            v1 = h1[0, pos1, :].numpy().copy()
            v2 = h2[0, pos2, :].numpy().copy()
            full_delta = v2 - v1
            
            d = decomp.get(token)
            if d is None:
                continue
            
            R_dir = d["R_direction"]
            
            # ---- Gap decomposition ----
            # Position direction: difference between positions in the two sentences
            # Use a simple proxy: the mean activation difference at the same position across sentences
            n_toks_1 = captures[idx1]["input_ids"].shape[1]
            n_toks_2 = captures[idx2]["input_ids"].shape[1]
            position_delta = float(n_toks_2 - n_toks_1)
            
            # Construction PCA for the relevant roles
            pca_r1 = pca_this_layer.get(r1)
            pca_r2 = pca_this_layer.get(r2)
            
            gap_result, gap_vec, C_vec, P_vec = decompose_gap(
                full_delta, R_dir, pca_r2, {}, d_model
            )
            
            # ---- Define patch conditions ----
            conditions = {
                "R_only": R_dir,
                "full_delta": full_delta,
                "Gap_only": gap_vec,
                "C_only": C_vec,
                "R+C": R_dir + C_vec,
                "R_neg_Gap": R_dir - gap_vec,  # = full_delta (should reconstruct)
            }
            
            # Only add meaningful conditions (non-zero vectors)
            final_conditions = {}
            for cname, cvec in conditions.items():
                cn = np.linalg.norm(cvec)
                if cn > 1e-10:
                    final_conditions[cname] = cvec
            
            key = f"{token}_{r1}->{r2}"
            layer_results[key] = {
                "token": token, "r1": r1, "r2": r2,
                "role_pair": token_rp.get(token, ""),
                "full_delta_norm": float(np.linalg.norm(full_delta)),
                "R_norm": float(np.linalg.norm(R_dir)),
                "gap_norm": gap_result["gap_norm"],
                "C_vec_norm": gap_result.get("C_vec_norm", 0),
                "U_vec_norm": gap_result.get("U_vec_norm", 0),
                "cos_gap_R": gap_result["cos_gap_R"],
                "cos_gap_C": gap_result.get("cos_gap_C", 0),
                "C_proj_energy": gap_result.get("C_proj_energy", 0),
                "U_norm_pct": gap_result.get("U_norm_pct", 0),
                "position_delta": position_delta,
                # Sign agreement
                "R_fd_same_sign": int(np.sign(np.sum(R_dir * full_delta)) > 0),
            }
            
            # ---- Run causal tests ----
            for cond_name, patch_vec in final_conditions.items():
                pnorm = np.linalg.norm(patch_vec)
                patched_logits = run_with_patched_hidden(model, tok, s1["sentence"],
                                                          li, pos1, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - logits1, target_shift)
                    layer_results[key][f"{cond_name}_cos_shift"] = float(cos_shift)
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
                else:
                    layer_results[key][f"{cond_name}_cos_shift"] = None
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
            
            # Random controls (3 directions)
            rand_shifts = []
            for ri in range(3):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model)
                rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * np.linalg.norm(full_delta)
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            layer_results[key]["avg_random_shift"] = float(np.mean(rand_shifts)) if rand_shifts else 0.0
            
            n_done = ti + 1
            if n_done % 10 == 0 or n_done == len(dual_test):
                log(f"  {n_done}/{len(dual_test)} test pairs done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect()
                torch.cuda.empty_cache()
        
        causal_results[str(li)] = layer_results
        log(f"  Layer {li}: {len(layer_results)} test pairs completed")
        
        # Print layer summary
        if layer_results:
            for metric in ["R_only_cos_shift", "full_delta_cos_shift", "Gap_only_cos_shift",
                          "C_only_cos_shift", "R+C_cos_shift", "avg_random_shift"]:
                cs = [v.get(metric) for v in layer_results.values() if v.get(metric) is not None]
                if cs:
                    log(f"    {metric}: avg={np.mean(cs):+.4f} pos={sum(1 for c in cs if c>0)}/{len(cs)}")
            
            # Gap decomposition summary
            gap_norms = [v.get("gap_norm", 0) for v in layer_results.values()]
            C_norms = [v.get("C_vec_norm", 0) for v in layer_results.values()]
            U_norms = [v.get("U_vec_norm", 0) for v in layer_results.values()]
            cos_gap_R = [v.get("cos_gap_R", 0) for v in layer_results.values()]
            cos_gap_C = [v.get("cos_gap_C", 0) for v in layer_results.values()]
            sign_agree = [v.get("R_fd_same_sign", 0) for v in layer_results.values()]
            
            log(f"    Gap norm: {np.mean(gap_norms):.4f}")
            log(f"    C(construction) norm: {np.mean(C_norms):.4f}")
            log(f"    U(unresolved) norm: {np.mean(U_norms):.4f} ({np.mean([v.get('U_norm_pct',0) for v in layer_results.values()]):.1f}%)")
            log(f"    cos(Gap, R): {np.mean(cos_gap_R):+.4f}")
            log(f"    cos(Gap, C): {np.mean(cos_gap_C):+.4f}")
            log(f"    R-FD sign agreement: {sum(sign_agree)}/{len(sign_agree)} ({sum(sign_agree)/max(len(sign_agree),1)*100:.0f}%)")
    
    # =====================================================================
    # PART C: PER-ROLE-PAIR BREAKDOWN + BOOTSTRAP
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART C: PER-ROLE-PAIR BREAKDOWN + BOOTSTRAP")
    log(f"{'='*60}")
    
    mid_li = str(nl // 2)
    bootstrap_layer = mid_li if mid_li in causal_results else str(max(1, nl // 4))
    log(f"  Using layer {bootstrap_layer} for detailed analysis")
    
    if bootstrap_layer in causal_results:
        lr = causal_results[bootstrap_layer]
        
        # Per-role-pair breakdown
        rp_groups = defaultdict(list)
        for key, v in lr.items():
            rp_groups[v.get("role_pair", "")].append(v)
        
        for rp, items in sorted(rp_groups.items()):
            log(f"\n  {rp} ({len(items)} tokens):")
            for metric in ["R_only_cos_shift", "full_delta_cos_shift", "Gap_only_cos_shift",
                          "C_only_cos_shift", "R+C_cos_shift", "avg_random_shift"]:
                cs = [v.get(metric) for v in items if v.get(metric) is not None]
                if cs:
                    log(f"    {metric}: {np.mean(cs):+.4f} ± {np.std(cs):.4f} "
                        f"pos={sum(1 for c in cs if c>0)}/{len(cs)}")
            
            # Gap decomposition
            cos_gR = [v.get("cos_gap_R", 0) for v in items if v.get("cos_gap_R") is not None]
            cos_gC = [v.get("cos_gap_C", 0) for v in items if v.get("cos_gap_C") is not None]
            C_energy = [v.get("C_proj_energy", 0) for v in items if v.get("C_proj_energy") is not None]
            U_pct = [v.get("U_norm_pct", 0) for v in items if v.get("U_norm_pct") is not None]
            sign_agree = [v.get("R_fd_same_sign", 0) for v in items]
            
            if cos_gR:
                log(f"    cos(Gap, R): {np.mean(cos_gR):+.4f}")
            if cos_gC:
                log(f"    cos(Gap, C): {np.mean(cos_gC):+.4f}")
            if C_energy:
                log(f"    C_proj_energy: {np.mean(C_energy):.4f}")
            if U_pct:
                log(f"    U(unresolved): {np.mean(U_pct):.1f}%")
            if sign_agree:
                log(f"    R-FD sign agreement: {sum(sign_agree)}/{len(sign_agree)} "
                    f"({sum(sign_agree)/len(sign_agree)*100:.0f}%)")
            
            # DS7B-style analysis: sign disagreement tokens
            disagree = [v for v in items if v.get("R_fd_same_sign", 1) == 0]
            if disagree:
                log(f"    R-FD disagree tokens ({len(disagree)}):")
                for v in disagree[:5]:
                    log(f"      {v['token']}: R_shift={v.get('R_only_cos_shift',0):+.3f} "
                        f"FD_shift={v.get('full_delta_cos_shift',0):+.3f} "
                        f"cos_gap_R={v.get('cos_gap_R',0):+.3f}")
        
        # Bootstrap stability
        per_token_data = {}
        for key, v in lr.items():
            token = v["token"]
            per_token_data[token] = dict(v)
        
        metric_keys = [
            "R_only_cos_shift", "full_delta_cos_shift", "Gap_only_cos_shift",
            "C_only_cos_shift", "R+C_cos_shift", "avg_random_shift",
        ]
        
        rng = np.random.RandomState(42)
        boot_results = {}
        rp_tokens_map = defaultdict(list)
        for token, data in per_token_data.items():
            rp = data.get("role_pair", "all")
            rp_tokens_map[rp].append(token)
        rp_tokens_map["all"] = list(per_token_data.keys())
        
        for rp, tokens in rp_tokens_map.items():
            n = len(tokens)
            if n < 3:
                continue
            for metric in metric_keys:
                vals = [per_token_data[t].get(metric) for t in tokens]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
                if len(vals) < 3:
                    continue
                vals_arr = np.array(vals)
                bootstrap_means = []
                for _ in range(1000):
                    sample = rng.choice(vals_arr, size=n, replace=True)
                    bootstrap_means.append(np.mean(sample))
                bootstrap_means = np.array(bootstrap_means)
                boot_results[f"{rp}::{metric}"] = {
                    "n_tokens": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci_low": float(np.percentile(bootstrap_means, 2.5)),
                    "ci_high": float(np.percentile(bootstrap_means, 97.5)),
                }
        
        # Print bootstrap results
        log(f"\n  Bootstrap 95% CI (layer {bootstrap_layer}):")
        for key, br in sorted(boot_results.items()):
            if "::" in key:
                rp, metric = key.split("::", 1)
                if rp == "all" and "cos_shift" in metric:
                    log(f"    [{rp}] {metric}: {br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] "
                        f"n={br['n_tokens']}")
    else:
        boot_results = {}
    
    # =====================================================================
    # PART D: PER-TOKEN DETAIL (for DS7B analysis)
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART D: PER-TOKEN GAP ANALYSIS (Layer {bootstrap_layer})")
    log(f"{'='*60}")
    
    if bootstrap_layer in causal_results:
        lr = causal_results[bootstrap_layer]
        
        # Sort by |cos(Gap, R)| to find anti-R gap tokens
        token_data = [(v["token"], v.get("role_pair", ""), v.get("cos_gap_R", 0),
                       v.get("cos_gap_C", 0), v.get("C_proj_energy", 0),
                       v.get("R_only_cos_shift", 0), v.get("full_delta_cos_shift", 0),
                       v.get("Gap_only_cos_shift", 0), v.get("R_fd_same_sign", 0))
                      for v in lr.values()]
        
        # Most anti-R gap tokens (cos_gap_R most negative)
        token_data.sort(key=lambda x: x[2])
        log(f"\n  Tokens with most anti-R Gap (cos(Gap,R) most negative):")
        for t, rp, cgR, cgC, Cen, Rsh, FDsh, Gsh, sign in token_data[:10]:
            log(f"    {t:10s} [{rp:10s}] cos_gR={cgR:+.3f} cos_gC={cgC:+.3f} "
                f"R_shift={Rsh:+.3f} FD_shift={FDsh:+.3f} Gap_shift={Gsh:+.3f} sign={sign}")
        
        # Most pro-R gap tokens
        token_data.sort(key=lambda x: -x[2])
        log(f"\n  Tokens with most pro-R Gap (cos(Gap,R) most positive):")
        for t, rp, cgR, cgC, Cen, Rsh, FDsh, Gsh, sign in token_data[:10]:
            log(f"    {t:10s} [{rp:10s}] cos_gR={cgR:+.3f} cos_gC={cgC:+.3f} "
                f"R_shift={Rsh:+.3f} FD_shift={FDsh:+.3f} Gap_shift={Gsh:+.3f} sign={sign}")
        
        # Sign disagreement tokens specifically
        disagree = [x for x in token_data if x[8] == 0]
        if disagree:
            log(f"\n  R-FD sign disagreement tokens ({len(disagree)}):")
            for t, rp, cgR, cgC, Cen, Rsh, FDsh, Gsh, sign in disagree[:10]:
                log(f"    {t:10s} [{rp:10s}] cos_gR={cgR:+.3f} cos_gC={cgC:+.3f} "
                    f"R_shift={Rsh:+.3f} FD_shift={FDsh:+.3f}")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    log(f"\nSaving results...")
    
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "n_dual_tokens": len(dual_tokens),
        "pca_results": make_serializable(all_pca_results),
        "subspace_angles": make_serializable(all_subspace_angles),
        "causal_results": make_serializable(causal_results),
        "bootstrap": make_serializable(boot_results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_construction_gap.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  Saved to {out_path}")
    
    # Release model
    release_model(model)
    log(f"Phase 304 complete for {model_name}!")


if __name__ == "__main__":
    main()
