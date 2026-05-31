"""
Phase 303: Large-Scale Factorial I/R/F Decomposition with Bootstrap Stability
==============================================================================
Goal: Fix the "sample size too small" issue from Phase 302.
- Expand from 22 to 60 dual-role tokens (20 adj_verb + 20 adj_noun + 20 noun_verb)
- Each role has 7 frames with 2 variants → 14 obs per (token, role)
- Bootstrap stability analysis (1000 resamples, 95% CI)
- Full layer coverage (8 layers)
- Per-role-pair breakdown with confidence intervals

Key questions:
1. Are Phase 302's conclusions stable with 3x larger sample?
2. Is DS7B's cos(R,F) extreme bimodality real or token-specific?
3. What are the confidence intervals for R, F, RF causal effects?
4. Is the RF binding term consistently negative across larger samples?
5. Is adj_noun's independent F channel robust across 20 tokens?

Usage:
  python tests/glm5/phase303_large_scale_factorial.py qwen3
  python tests/glm5/phase303_large_scale_factorial.py glm4
  python tests/glm5/phase303_large_scale_factorial.py deepseek7b
"""
import sys, os, gc, time, json, math, random
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

RESULT_DIR = Path("results/phase303_large_scale_factorial")
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
# TOKEN DEFINITIONS — 60 dual-role tokens (20 per category)
# =====================================================================
# adj_verb: {token: (adj_objects, verb_objects)}
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

# adj_noun: {token: (adj_objects, noun_adjectives)}
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

# noun_verb: {token: (noun_adjectives, verb_objects_or_None, verb_subjects_or_None, is_transitive)}
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
# FRAME TEMPLATES — 7 frames per role type
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
# STIMULUS GENERATION (template-based)
# =====================================================================
def build_stimuli():
    """Generate balanced factorial stimulus set from templates."""
    stimuli = []
    
    # ---- adj_verb tokens ----
    for token, (adj_objs, verb_objs) in ADJ_VERB_TOKENS.items():
        # Adj frames
        for frame_name, template in ADJ_FRAMES:
            for obj in adj_objs:
                sent = template.format(token=token, obj=obj)
                stimuli.append({
                    "sentence": sent, "target_word": token,
                    "token_label": token, "role_label": "adj",
                    "frame_label": frame_name, "role_pair": "adj_verb",
                })
        # Verb frames (transitive)
        for frame_name, template in VERB_FRAMES_TRANSITIVE:
            for obj in verb_objs:
                sent = template.format(token=token, obj=obj)
                stimuli.append({
                    "sentence": sent, "target_word": token,
                    "token_label": token, "role_label": "verb",
                    "frame_label": frame_name, "role_pair": "adj_verb",
                })
    
    # ---- adj_noun tokens ----
    for token, (adj_objs, noun_adjs) in ADJ_NOUN_TOKENS.items():
        # Adj frames
        for frame_name, template in ADJ_FRAMES:
            for obj in adj_objs:
                sent = template.format(token=token, obj=obj)
                stimuli.append({
                    "sentence": sent, "target_word": token,
                    "token_label": token, "role_label": "adj",
                    "frame_label": frame_name, "role_pair": "adj_noun",
                })
        # Noun frames
        for frame_name, template in NOUN_FRAMES:
            if "{adj}" in template:
                for adj in noun_adjs:
                    sent = template.format(token=token, adj=adj)
                    stimuli.append({
                        "sentence": sent, "target_word": token,
                        "token_label": token, "role_label": "noun",
                        "frame_label": frame_name, "role_pair": "adj_noun",
                    })
            else:
                # locative/action frames (no adj needed)
                sent = template.format(token=token)
                stimuli.append({
                    "sentence": sent, "target_word": token,
                    "token_label": token, "role_label": "noun",
                    "frame_label": frame_name, "role_pair": "adj_noun",
                })
    
    # ---- noun_verb tokens ----
    for token, (noun_adjs, verb_objs, verb_subjs, is_trans) in NOUN_VERB_TOKENS.items():
        # Noun frames
        for frame_name, template in NOUN_FRAMES:
            if "{adj}" in template:
                for adj in noun_adjs:
                    sent = template.format(token=token, adj=adj)
                    stimuli.append({
                        "sentence": sent, "target_word": token,
                        "token_label": token, "role_label": "noun",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
            else:
                sent = template.format(token=token)
                stimuli.append({
                    "sentence": sent, "target_word": token,
                    "token_label": token, "role_label": "noun",
                    "frame_label": frame_name, "role_pair": "noun_verb",
                })
        # Verb frames
        if is_trans:
            for frame_name, template in VERB_FRAMES_TRANSITIVE:
                for obj in verb_objs:
                    sent = template.format(token=token, obj=obj)
                    stimuli.append({
                        "sentence": sent, "target_word": token,
                        "token_label": token, "role_label": "verb",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
        else:
            for frame_name, template in VERB_FRAMES_INTRANSITIVE:
                for subj in verb_subjs:
                    sent = template.format(token=token, subj=subj)
                    stimuli.append({
                        "sentence": sent, "target_word": token,
                        "token_label": token, "role_label": "verb",
                        "frame_label": frame_name, "role_pair": "noun_verb",
                    })
    
    return stimuli


def build_causal_stimuli():
    """Causal test pairs: one sentence per (token, role) using the most natural frame."""
    test_pairs = []
    
    # adj_verb: adj_copula vs verb_transitive
    for token, (adj_objs, verb_objs) in ADJ_VERB_TOKENS.items():
        test_pairs.append(("the {obj} is {token}".format(token=token, obj=adj_objs[0]),
                          token, "adj", "adj_verb"))
        test_pairs.append(("they {token} the {obj}".format(token=token, obj=verb_objs[0]),
                          token, "verb", "adj_verb"))
    
    # adj_noun: adj_copula vs noun_copula
    for token, (adj_objs, noun_adjs) in ADJ_NOUN_TOKENS.items():
        test_pairs.append(("the {obj} is {token}".format(token=token, obj=adj_objs[0]),
                          token, "adj", "adj_noun"))
        test_pairs.append(("the {token} is {adj}".format(token=token, adj=noun_adjs[0]),
                          token, "noun", "adj_noun"))
    
    # noun_verb: noun_copula vs verb_transitive/intransitive
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
# MODEL LOADING — BF16 + device_map="auto" + flash_attn priority
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto + flash_attn)...")
    
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            log(f"  attn_implementation={attn_impl} succeeded")
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:100]}")
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
    
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
    return {"hidden": hs, "logits": logits}

def _find_token_pos(decoded_tokens, target):
    target_lower = target.lower()
    for i, t in enumerate(decoded_tokens):
        if t == target_lower: return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower: return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower: return i
    return None

def resolve_positions(stimuli, tokenizer):
    resolved = []
    for stim in stimuli:
        toks = tokenizer.encode(stim["sentence"], add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, stim["target_word"])
        if pos is not None:
            new_stim = dict(stim); new_stim["target_pos"] = pos; resolved.append(new_stim)
    return resolved

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


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
# FACTORIAL DECOMPOSITION (same as Phase 302)
# =====================================================================
def factorial_decomposition(cell_means, dual_tokens, token_roles, token_frames, d_model):
    decomp = {}
    for token in dual_tokens:
        roles_list = sorted(token_roles[token])
        if len(roles_list) != 2: continue
        r1, r2 = roles_list
        frames_list = sorted(token_frames[token])
        
        cells = {}
        for role in roles_list:
            for frame in frames_list:
                key = (token, role, frame)
                if key in cell_means:
                    cells[(role, frame)] = cell_means[key]
        
        if len(cells) < 4: continue
        
        all_vecs = list(cells.values())
        grand_mean = np.mean(all_vecs, axis=0)
        
        role_means = {}
        for role in roles_list:
            r_vecs = [cells[(role, f)] for f in frames_list if (role, f) in cells]
            if r_vecs:
                role_means[role] = np.mean(r_vecs, axis=0)
        
        frame_means = {}
        for frame in frames_list:
            f_vecs = [cells[(r, frame)] for r in roles_list if (r, frame) in cells]
            if f_vecs:
                frame_means[frame] = np.mean(f_vecs, axis=0)
        
        R_effect = {}
        for role in roles_list:
            if role in role_means:
                R_effect[role] = role_means[role] - grand_mean
        
        F_effect = {}
        for frame in frames_list:
            if frame in frame_means:
                F_effect[frame] = frame_means[frame] - grand_mean
        
        if r1 in R_effect and r2 in R_effect:
            R_direction = R_effect[r2] - R_effect[r1]
        else:
            continue
        
        F_direction_avg = np.mean(list(F_effect.values()), axis=0) if F_effect else np.zeros(d_model)
        F_directions = {f: F_effect[f] for f in frames_list if f in F_effect}
        
        RF_interaction = {}
        for role in roles_list:
            for frame in frames_list:
                if (role, frame) in cells and role in R_effect and frame in F_effect:
                    residual = cells[(role, frame)] - grand_mean - R_effect[role] - F_effect[frame]
                    RF_interaction[(role, frame)] = residual
        
        RF_r2_avg = np.mean([RF_interaction[(r2, f)] for f in frames_list 
                            if (r2, f) in RF_interaction], axis=0) if any((r2, f) in RF_interaction for f in frames_list) else np.zeros(d_model)
        RF_r1_avg = np.mean([RF_interaction[(r1, f)] for f in frames_list 
                            if (r1, f) in RF_interaction], axis=0) if any((r1, f) in RF_interaction for f in frames_list) else np.zeros(d_model)
        RF_direction = RF_r2_avg - RF_r1_avg
        
        decomp[token] = {
            "grand_mean": grand_mean,
            "R_direction": R_direction,
            "F_direction_avg": F_direction_avg,
            "F_directions": F_directions,
            "RF_direction": RF_direction,
            "r1": r1, "r2": r2,
            "frames": frames_list,
            "n_cells": len(cells),
        }
    
    return decomp


# =====================================================================
# BOOTSTRAP STABILITY ANALYSIS
# =====================================================================
def bootstrap_analysis(per_token_data, metric_keys, n_bootstrap=1000, seed=42):
    """
    Bootstrap resampling for stability analysis.
    
    per_token_data: dict {token: {metric: value, "role_pair": str, ...}}
    metric_keys: list of metric names to bootstrap
    n_bootstrap: number of resamples
    
    Returns: dict with CI for each (metric, group) combination
    """
    rng = np.random.RandomState(seed)
    
    # Group tokens by role_pair
    rp_tokens = defaultdict(list)
    for token, data in per_token_data.items():
        rp = data.get("role_pair", "all")
        rp_tokens[rp].append(token)
    rp_tokens["all"] = list(per_token_data.keys())
    
    results = {}
    for rp, tokens in rp_tokens.items():
        n = len(tokens)
        if n < 3: continue
        
        for metric in metric_keys:
            vals = [per_token_data[t].get(metric) for t in tokens]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
            if len(vals) < 3: continue
            
            vals_arr = np.array(vals)
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(vals_arr, size=n, replace=True)
                bootstrap_means.append(np.mean(sample))
            
            bootstrap_means = np.array(bootstrap_means)
            results[(rp, metric)] = {
                "n_tokens": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_low": float(np.percentile(bootstrap_means, 2.5)),
                "ci_high": float(np.percentile(bootstrap_means, 97.5)),
                "ci_width": float(np.percentile(bootstrap_means, 97.5) - np.percentile(bootstrap_means, 2.5)),
                "positive_pct": float(np.mean(vals_arr > 0) * 100),
            }
    
    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase303_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 303: Large-Scale Factorial I/R/F + Bootstrap -- {model_name}")

    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers; d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")
    
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
    all_sentences = []; sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences); all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]
    log(f"  Unique sentences: {len(all_sentences)}")
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} unique sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 100 == 0:
            el = time.time() - t0; rate = (i + 1) / max(el, 1)
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={(len(all_sentences)-i-1)/rate:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # Organize observation data
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]; role = stim["role_label"]; frame = stim.get("frame_label", "")
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, frame)].append((idx, pos))
    
    # Organize causal test pairs
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]; role = stim["role_label"]
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
    # FACTORIAL DECOMPOSITION + CAUSAL TEST
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"FACTORIAL I/R/F DECOMPOSITION + CAUSAL TEST")
    log(f"{'='*60}")
    
    results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        
        # ---- Compute cell means ----
        cell_means = {}
        for (token, role, frame), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, frame)] = np.mean(vecs, axis=0)
        
        # ---- Factorial decomposition ----
        decomp = factorial_decomposition(cell_means, dual_tokens, token_roles, token_frames, d_model)
        log(f"  Factorial decomposition: {len(decomp)} tokens")
        
        # ---- Compute LOO R direction ----
        token_R_loo = {}
        for token in dual_tokens:
            if token not in decomp: continue
            other_R = {t: decomp[t]["R_direction"] for t in decomp if t != token}
            if other_R:
                token_R_loo[token] = np.mean(list(other_R.values()), axis=0)
        
        # ---- Causal test ----
        layer_results = {}
        
        for ti, (token, roles_list) in enumerate(dual_test):
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            s1 = test_pairs[token][r1]; s2 = test_pairs[token][r2]
            
            idx1 = s1.get("_idx"); pos1 = s1.get("target_pos")
            idx2 = s2.get("_idx"); pos2 = s2.get("target_pos")
            if idx1 is None or idx2 is None: continue
            
            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None: continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]: continue
            
            logits1 = captures[idx1]["logits"][0, -1, :].numpy().copy()
            logits2 = captures[idx2]["logits"][0, -1, :].numpy().copy()
            target_shift = logits2 - logits1
            
            v1 = h1[0, pos1, :].numpy().copy()
            v2 = h2[0, pos2, :].numpy().copy()
            full_delta = v2 - v1
            
            d = decomp.get(token)
            if d is None: continue
            
            R_dir = d["R_direction"]
            F_dir = d["F_direction_avg"]
            RF_dir = d["RF_direction"]
            R_loo = token_R_loo.get(token, R_dir)
            F_residual = full_delta - R_dir
            
            R_norm = float(np.linalg.norm(R_dir))
            F_norm = float(np.linalg.norm(F_dir))
            RF_norm = float(np.linalg.norm(RF_dir))
            full_norm = float(np.linalg.norm(full_delta))
            
            cos_RF_factorial = cosine_sim(R_dir, F_dir)
            cos_RF_residual = cosine_sim(R_dir, F_residual)
            cos_F_fact_resid = cosine_sim(F_dir, F_residual)
            
            key = f"{token}_{r1}->{r2}"
            layer_results[key] = {
                "token": token, "r1": r1, "r2": r2, "role_pair": token_rp.get(token, ""),
                "R_norm": R_norm, "F_norm": F_norm, "RF_norm": RF_norm, "full_norm": full_norm,
                "cos_RF_factorial": cos_RF_factorial,
                "cos_RF_residual": cos_RF_residual,
                "cos_F_fact_resid": cos_F_fact_resid,
                "n_cells": d["n_cells"],
            }
            
            # ---- Define all patch conditions ----
            conditions = {
                "R_only": R_dir,
                "F_only": F_dir,
                "R+F": R_dir + F_dir,
                "R+F+RF": R_dir + F_dir + RF_dir,
                "RF_only": RF_dir,
                "full_delta": full_delta,
                "R_loo": R_loo,
                "R_loo+F": R_loo + F_dir,
                "F_residual": F_residual,
                "R+F_residual": R_dir + F_residual,
            }
            
            # ---- Run causal tests ----
            for cond_name, patch_vec in conditions.items():
                pnorm = np.linalg.norm(patch_vec)
                if pnorm < 1e-10:
                    layer_results[key][f"{cond_name}_cos_shift"] = 0.0
                    layer_results[key][f"{cond_name}_norm"] = 0.0
                    continue
                
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
            
            # ---- Random controls (3 directions) ----
            rand_shifts = []
            for ri in range(3):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model); rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * full_norm
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            layer_results[key]["avg_random_shift"] = float(np.mean(rand_shifts)) if rand_shifts else 0.0
            
            n_done = ti + 1
            if n_done % 10 == 0 or n_done == len(dual_test):
                log(f"  {n_done}/{len(dual_test)} test pairs done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect(); torch.cuda.empty_cache()
        
        results[str(li)] = layer_results
        log(f"  Layer {li}: {len(layer_results)} test pairs completed")
        
        # Print layer summary
        if layer_results:
            for metric in ["R_only", "F_only", "RF_only", "R+F", "R+F+RF", "full_delta", "avg_random_shift"]:
                cs = [v.get(f"{metric}_cos_shift", v.get(metric, None)) for v in layer_results.values()]
                cs = [c for c in cs if c is not None]
                if cs:
                    log(f"    {metric}: avg={np.mean(cs):+.4f} pos={sum(1 for c in cs if c>0)}/{len(cs)}")
            
            RpF_cs = [v.get("R+F_cos_shift") for v in layer_results.values() if v.get("R+F_cos_shift") is not None]
            RpFpRF_cs = [v.get("R+F+RF_cos_shift") for v in layer_results.values() if v.get("R+F+RF_cos_shift") is not None]
            if RpF_cs and RpFpRF_cs:
                rf_boost = np.mean(RpFpRF_cs) - np.mean(RpF_cs)
                log(f"    RF boost: {rf_boost:+.4f}")
    
    # =====================================================================
    # BOOTSTRAP STABILITY ANALYSIS
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"BOOTSTRAP STABILITY ANALYSIS (1000 resamples)")
    log(f"{'='*60}")
    
    # Collect per-token data at mid layer
    mid_li = str(nl // 2)
    mid_li_alt = str(max(1, nl // 4))  # alternative if mid not available
    
    bootstrap_layer = mid_li if mid_li in results else mid_li_alt
    log(f"  Using layer {bootstrap_layer} for bootstrap analysis")
    
    if bootstrap_layer in results:
        per_token_data = {}
        for key, v in results[bootstrap_layer].items():
            token = v["token"]
            per_token_data[token] = dict(v)
        
        metric_keys = [
            "R_only_cos_shift", "F_only_cos_shift", "RF_only_cos_shift",
            "R+F_cos_shift", "R+F+RF_cos_shift", "full_delta_cos_shift",
            "avg_random_shift", "cos_RF_factorial", "cos_RF_residual",
        ]
        
        boot_results = bootstrap_analysis(per_token_data, metric_keys, n_bootstrap=1000)
        
        # Print bootstrap results
        for (rp, metric), br in sorted(boot_results.items()):
            if rp == "all":
                log(f"  [{rp}] {metric}: mean={br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] "
                    f"width={br['ci_width']:.4f} pos={br['positive_pct']:.0f}% n={br['n_tokens']}")
        
        log(f"\n  --- Per-role-pair bootstrap ---")
        for (rp, metric), br in sorted(boot_results.items()):
            if rp != "all" and "cos_shift" in metric:
                log(f"  [{rp}] {metric}: mean={br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] "
                    f"pos={br['positive_pct']:.0f}% n={br['n_tokens']}")
    else:
        boot_results = {}
        log(f"  No data available for bootstrap at layer {bootstrap_layer}")
    
    # =====================================================================
    # KEY DIAGNOSTICS
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"KEY DIAGNOSTICS")
    log(f"{'='*60}")
    
    # 1. Cos(R,F) distribution: factorial vs residual
    log(f"\n--- Cos(R,F) Distribution ---")
    for li_str, layer_res in results.items():
        if not layer_res: continue
        cos_fact = [v.get("cos_RF_factorial", 0) for v in layer_res.values()
                   if v.get("cos_RF_factorial") is not None]
        cos_resid = [v.get("cos_RF_residual", 0) for v in layer_res.values()
                    if v.get("cos_RF_residual") is not None]
        
        if cos_fact:
            extreme_fact = sum(1 for v in cos_fact if abs(v) > 0.9)
            log(f"  L{li_str} factorial: mean={np.mean(cos_fact):+.4f} std={np.std(cos_fact):.4f} "
                f"|cos|>0.9: {extreme_fact}/{len(cos_fact)} ({extreme_fact/len(cos_fact)*100:.0f}%)")
        if cos_resid:
            extreme_resid = sum(1 for v in cos_resid if abs(v) > 0.9)
            log(f"  L{li_str} residual:  mean={np.mean(cos_resid):+.4f} std={np.std(cos_resid):.4f} "
                f"|cos|>0.9: {extreme_resid}/{len(cos_resid)} ({extreme_resid/len(cos_resid)*100:.0f}%)")
    
    # 2. Per-role-pair causal effects at mid layer
    log(f"\n--- Per-Role-Pair Causal Effects at Layer {bootstrap_layer} ---")
    if bootstrap_layer in results:
        rp_groups = defaultdict(list)
        for key, v in results[bootstrap_layer].items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp, items in sorted(rp_groups.items()):
            log(f"  {rp} ({len(items)} tokens):")
            for metric in ["R_only", "F_only", "RF_only", "R+F", "R+F+RF"]:
                cs = [v.get(f"{metric}_cos_shift", 0) for v in items 
                      if v.get(f"{metric}_cos_shift") is not None]
                if cs:
                    pos_pct = sum(1 for c in cs if c > 0) / len(cs) * 100
                    log(f"    {metric}: {np.mean(cs):+.4f} ± {np.std(cs):.4f} pos={pos_pct:.0f}%")
            
            # RF boost
            RpF = [v.get("R+F_cos_shift", 0) for v in items if v.get("R+F_cos_shift") is not None]
            RpFpRF = [v.get("R+F+RF_cos_shift", 0) for v in items if v.get("R+F+RF_cos_shift") is not None]
            if RpF and RpFpRF:
                boost = np.mean(RpFpRF) - np.mean(RpF)
                log(f"    RF boost: {boost:+.4f}")
    
    # 3. Per-token cos(R,F) extremes (for DS7B analysis)
    log(f"\n--- Per-Token cos(R,F) at Layer {bootstrap_layer} ---")
    if bootstrap_layer in results:
        cos_data = [(v["token"], v.get("role_pair",""), v.get("cos_RF_factorial",0), v.get("cos_RF_residual",0))
                    for v in results[bootstrap_layer].values()]
        cos_data.sort(key=lambda x: abs(x[2]), reverse=True)
        log(f"  Top 10 |cos(R,F)_factorial|:")
        for token, rp, cf, cr in cos_data[:10]:
            log(f"    {token} ({rp}): factorial={cf:+.4f} residual={cr:+.4f}")
        log(f"  Bottom 5 |cos(R,F)_factorial|:")
        for token, rp, cf, cr in cos_data[-5:]:
            log(f"    {token} ({rp}): factorial={cf:+.4f} residual={cr:+.4f}")
    
    # 4. R_loo / R_only ratio
    log(f"\n--- R_loo Generalization at Layer {bootstrap_layer} ---")
    if bootstrap_layer in results:
        rp_groups = defaultdict(list)
        for key, v in results[bootstrap_layer].items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp, items in sorted(rp_groups.items()):
            R_only = [v.get("R_only_cos_shift", 0) for v in items if v.get("R_only_cos_shift") is not None]
            R_loo = [v.get("R_loo_cos_shift", 0) for v in items if v.get("R_loo_cos_shift") is not None]
            if R_only and R_loo:
                ratio = np.mean(R_loo) / max(np.mean(R_only), 1e-6)
                log(f"  {rp}: R_loo/R_only = {ratio:.3f} (R_loo={np.mean(R_loo):+.4f}, R_only={np.mean(R_only):+.4f})")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    def make_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj
    
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "dual_tokens": dual_tokens,
        "n_dual_tokens": len(dual_tokens),
        "n_adj_verb": rp_counts.get("adj_verb", 0),
        "n_adj_noun": rp_counts.get("adj_noun", 0),
        "n_noun_verb": rp_counts.get("noun_verb", 0),
        "factorial_causal": make_serializable(results),
        "bootstrap": make_serializable({f"{rp}::{metric}": v for (rp, metric), v in boot_results.items()}),
    }
    
    out_path = RESULT_DIR / f"{model_name}_large_scale_factorial.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    release_model(model)
    log(f"Phase 303 complete for {model_name}")


if __name__ == "__main__":
    main()
