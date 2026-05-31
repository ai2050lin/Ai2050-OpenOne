"""
Phase 306: Normalized Construction PCA + Position/Norm Decomposition
=====================================================================
Resolves the biggest hard injury from Phase 304:
1. Is DS7B PC1>98% a real direction or norm artifact?
2. What is the position contribution to Gap?
3. O(not) vs R/C orthogonality check (using Phase 305 data)

Key tests:
A. Normalized vs Raw PCA comparison:
   - Raw delta PCA (original)
   - Unit delta PCA (normalize each deviation before PCA)
   - Cosine-only PCA (cosine distance matrix)
   → If raw high but unit low: 1D is norm artifact
   → If both high: 1D is real direction

B. R/C/P/N decomposition:
   full_delta = R + C + P + N + U
   R = role direction (as before)
   C = construction (from within-role PCA)
   P = position direction (same token at different positions)
   N = norm effect (hidden state norm difference)
   U = unresolved residual

C. O(not) orthogonality with R and C:
   Direct cos(O_not, R), cos(O_not, C) measurement

Usage:
  python tests/glm5/phase306_norm_position_decompose.py qwen3
  python tests/glm5/phase306_norm_position_decompose.py glm4
  python tests/glm5/phase306_norm_position_decompose.py deepseek7b
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

RESULT_DIR = Path("results/phase306_norm_position")
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
# TOKEN DEFINITIONS — same as Phase 304
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
# FRAME TEMPLATES
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
# POSITION CONTROL SENTENCES
# Same word at different syntactic positions
# =====================================================================
POSITION_SENTENCES = {
    # Token appears at position 2 (early), position 4 (mid), position 6+ (late)
    # These control for position effects independent of role/construction
    "open": [
        "the open door was",
        "they saw the open door",
        "they will open the door",
        "they began to open the gate",
    ],
    "clear": [
        "the clear path was",
        "they saw the clear sky",
        "they will clear the desk",
        "they began to clear the path",
    ],
    "warm": [
        "the warm room was",
        "they saw the warm water",
        "they will warm the room",
        "they began to warm the water",
    ],
    "clean": [
        "the clean floor was",
        "they saw the clean table",
        "they will clean the room",
        "they began to clean the floor",
    ],
    "cold": [
        "the cold water was",
        "they saw the cold wind",
        "a cold hit them",
        "they felt the cold room",
    ],
    "light": [
        "the light bag was",
        "they saw the light box",
        "a light shone through",
        "they felt the light touch",
    ],
    "fire": [
        "the hot fire was",
        "they saw the big fire",
        "they will fire the worker",
        "they began to fire the gun",
    ],
    "record": [
        "the old record was",
        "they saw the broken record",
        "they will record the data",
        "they began to record the music",
    ],
    "run": [
        "the long run was",
        "they saw the hard run",
        "they will run the program",
        "they began to run the company",
    ],
    "book": [
        "the new book was",
        "they saw the long book",
        "they will book the room",
        "they began to book the ticket",
    ],
}

# =====================================================================
# STIMULUS GENERATION
# =====================================================================
def build_stimuli():
    """Generate observation stimulus set."""
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


def build_position_stimuli():
    """Position control stimuli: same token at different positions."""
    stimuli = []
    for token, sentences in POSITION_SENTENCES.items():
        for i, sent in enumerate(sentences):
            stimuli.append({
                "sentence": sent, "target_word": token,
                "token_label": token, "role_label": "position_control",
                "frame_label": f"pos_{i}", "role_pair": "position",
                "group": "position_control",
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
        raise RuntimeError(f"Failed to load {model_name}")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    layers = get_layers(model)
    log(f"  n_layers={len(layers)}, class={type(model).__name__}")
    
    # Check layer distribution
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
        gpu_layer_ids = sorted([int(lid) for lid, dev in layer_devices.items() if 'cuda' in str(dev)])
        if gpu_layer_ids and gpu_layers < len(layers):
            last_gpu = max(gpu_layer_ids)
            log(f"  Last GPU layer: {last_gpu}, first CPU layer: {last_gpu + 1}")
            # Log all GPU layer IDs for verification
            log(f"  GPU layer IDs: {gpu_layer_ids}")
            # Verify deep layers accessible
            deep_layers = [li for li in range(len(layers)-4, len(layers))]
            for dl in deep_layers:
                layer_dev = layer_devices.get(str(dl), "unknown")
                log(f"  Layer {dl}: device={layer_dev}")
    
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
# PART A: NORMALIZED VS RAW PCA COMPARISON
# =====================================================================
def within_role_pca_normalized(cell_means_by_role_frame, d_model, n_components=5):
    """
    Compare raw vs normalized within-role frame PCA.
    
    4 versions:
    1. Raw PCA: PCA on raw deviations (includes norm)
    2. Unit PCA: PCA on L2-normalized deviations (direction only)
    3. Sign PCA: PCA on sign(deviation) (quantized direction)
    4. Norm-only: separate analysis of norm variation
    """
    from sklearn.decomposition import PCA
    
    # Group by role
    role_data = defaultdict(list)
    token_role_vecs = defaultdict(list)
    for (token, role, frame), vec in cell_means_by_role_frame.items():
        token_role_vecs[(token, role)].append(vec)
    
    token_role_means = {}
    for (token, role), vecs in token_role_vecs.items():
        token_role_means[(token, role)] = np.mean(vecs, axis=0)
    
    # Compute frame deviations with norms
    role_devs = defaultdict(list)  # {role: [(token, frame, raw_dev, unit_dev, norm)]}
    for (token, role, frame), vec in cell_means_by_role_frame.items():
        tr_mean = token_role_means.get((token, role))
        if tr_mean is not None:
            raw_dev = vec - tr_mean
            dev_norm = np.linalg.norm(raw_dev)
            unit_dev = raw_dev / max(dev_norm, 1e-10)
            role_devs[role].append({
                "token": token, "frame": frame,
                "raw_dev": raw_dev, "unit_dev": unit_dev,
                "dev_norm": dev_norm,
                "h_norm": np.linalg.norm(vec),
            })
    
    results = {}
    for role, data in role_devs.items():
        if len(data) < n_components:
            continue
        
        raw_matrix = np.array([d["raw_dev"] for d in data])
        unit_matrix = np.array([d["unit_dev"] for d in data])
        norms = np.array([d["dev_norm"] for d in data])
        h_norms = np.array([d["h_norm"] for d in data])
        
        # 1. Raw PCA
        n_comp = min(n_components, len(data) - 1, d_model)
        pca_raw = PCA(n_components=n_comp)
        pca_raw.fit(raw_matrix)
        
        # 2. Unit PCA (direction only)
        pca_unit = PCA(n_components=n_comp)
        pca_unit.fit(unit_matrix)
        
        # 3. Norm statistics
        norm_variation = np.std(norms) / max(np.mean(norms), 1e-10)  # CV
        h_norm_variation = np.std(h_norms) / max(np.mean(h_norms), 1e-10)
        
        # 4. Check: is PC1 aligned with norm direction?
        pc1_raw = pca_raw.components_[0]
        # Norm direction: direction of mean_vec (since norm increases along mean)
        mean_vec = np.mean(raw_matrix, axis=0)
        cos_pc1_mean = cosine_sim(pc1_raw, mean_vec)
        
        # 5. Check: do high-norm deviations dominate PC1?
        pc1_projections = raw_matrix @ pc1_raw  # [n_obs]
        corr_proj_norm = float(np.corrcoef(np.abs(pc1_projections), norms)[0, 1])
        
        results[role] = {
            "n_obs": len(data),
            "n_tokens": len(set(d["token"] for d in data)),
            # Raw PCA
            "raw_explained_var_ratio": pca_raw.explained_variance_ratio_.tolist(),
            "raw_pc1_var": float(pca_raw.explained_variance_ratio_[0]),
            # Unit PCA
            "unit_explained_var_ratio": pca_unit.explained_variance_ratio_.tolist(),
            "unit_pc1_var": float(pca_unit.explained_variance_ratio_[0]),
            # Norm statistics
            "dev_norm_cv": float(norm_variation),
            "h_norm_cv": float(h_norm_variation),
            "mean_dev_norm": float(np.mean(norms)),
            "std_dev_norm": float(np.std(norms)),
            "mean_h_norm": float(np.mean(h_norms)),
            "std_h_norm": float(np.std(h_norms)),
            # PC1-norm alignment
            "cos_pc1_mean_direction": float(cos_pc1_mean),
            "corr_pc1_proj_norm": float(corr_proj_norm),
            # Store components for later use
            "raw_pca_components": pca_raw.components_.tolist(),
            "unit_pca_components": pca_unit.components_.tolist(),
        }
        
        log(f"  [{role}] Raw PC1={pca_raw.explained_variance_ratio_[0]*100:.1f}%, "
            f"Unit PC1={pca_unit.explained_variance_ratio_[0]*100:.1f}%, "
            f"Norm CV={norm_variation:.3f}, "
            f"cos(PC1,mean)={cos_pc1_mean:+.3f}, "
            f"corr(|PC1|,norm)={corr_proj_norm:+.3f}")
    
    return results


# =====================================================================
# PART B: POSITION DIRECTION EXTRACTION
# =====================================================================
def extract_position_directions(position_data, d_model):
    """
    Extract position direction from position control sentences.
    
    For each token that appears at different positions:
    - Compute position delta = h(late_pos) - h(early_pos)
    - This captures pure position effects
    """
    pos_directions = {}  # {token: {layer: position_direction}}
    
    for token, entries in position_data.items():
        if len(entries) < 2:
            continue
        # Sort by position index (frame_label pos_0, pos_1, etc.)
        sorted_entries = sorted(entries, key=lambda e: e.get("frame_label", ""))
        pos_directions[token] = {}
        
        for li, vecs in [(li, [e["vecs"].get(li) for e in sorted_entries]) 
                         for li in sorted(sorted_entries[0]["vecs"].keys())]:
            valid_vecs = [v for v in vecs if v is not None]
            if len(valid_vecs) >= 2:
                # Position direction = late - early
                pos_dir = valid_vecs[-1] - valid_vecs[0]
                pos_directions[token][li] = pos_dir
    
    return pos_directions


# =====================================================================
# PART C: R/C/P/N DECOMPOSITION
# =====================================================================
def decompose_full_delta_rcpn(full_delta, R_direction, C_components, P_directions, 
                              d_model, h_norm_diff=None):
    """
    Decompose full_delta = R + C + P + N + U
    
    R = role direction (as before)
    C = construction (projection onto within-role PCA subspace)
    P = position (projection onto position direction)
    N = norm (scalar along hidden state norm difference)
    U = unresolved residual
    """
    gap = full_delta - R_direction
    gap_norm = np.linalg.norm(gap)
    fd_norm = np.linalg.norm(full_delta)
    R_norm = np.linalg.norm(R_direction)
    
    result = {
        "gap_norm": float(gap_norm),
        "R_norm": float(R_norm),
        "full_delta_norm": float(fd_norm),
        "cos_gap_R": float(cosine_sim(gap, R_direction)),
    }
    
    # 1. Construction component
    if C_components is not None and len(C_components) > 0:
        comps = np.array(C_components)  # [k, d]
        proj_coeffs = comps @ gap
        C_vec = proj_coeffs @ comps
        result["C_vec_norm"] = float(np.linalg.norm(C_vec))
        result["C_proj_energy"] = float(np.sum(proj_coeffs**2) / max(gap_norm**2, 1e-20))
        result["cos_gap_C"] = float(cosine_sim(gap, C_vec))
    else:
        C_vec = np.zeros(d_model)
        result["C_vec_norm"] = 0.0
        result["C_proj_energy"] = 0.0
        result["cos_gap_C"] = 0.0
    
    # 2. Position component
    P_vec = np.zeros(d_model)
    if P_directions:
        valid_p = [v for v in P_directions if v is not None and np.linalg.norm(v) > 1e-10]
        if valid_p:
            avg_pos = np.mean(valid_p, axis=0)
            P_vec = project_onto(gap, avg_pos)
    result["P_vec_norm"] = float(np.linalg.norm(P_vec))
    result["cos_gap_P"] = float(cosine_sim(gap, P_vec))
    
    # 3. Norm component (direction along R, capturing scale difference)
    # If R and gap share direction, norm is the scalar projection of gap onto R_hat
    R_hat = R_direction / max(np.linalg.norm(R_direction), 1e-10)
    N_scalar = np.dot(gap, R_hat)
    N_vec = N_scalar * R_hat
    result["N_vec_norm"] = float(np.linalg.norm(N_vec))
    result["cos_gap_N"] = float(cosine_sim(gap, N_vec))
    result["N_scalar"] = float(N_scalar)
    
    # 4. Unresolved residual (after removing C and P, but NOT N since N is along R)
    U_vec = gap - C_vec - P_vec
    result["U_vec_norm"] = float(np.linalg.norm(U_vec))
    result["U_norm_pct"] = float(np.linalg.norm(U_vec) / max(gap_norm, 1e-10) * 100)
    
    # Energy budget
    total_energy = gap_norm**2
    if total_energy > 1e-20:
        result["C_energy_pct"] = float(np.linalg.norm(C_vec)**2 / total_energy * 100)
        result["P_energy_pct"] = float(np.linalg.norm(P_vec)**2 / total_energy * 100)
        result["N_energy_pct"] = float(np.linalg.norm(N_vec)**2 / total_energy * 100)
        result["U_energy_pct"] = float(np.linalg.norm(U_vec)**2 / total_energy * 100)
    
    # Norm difference
    if h_norm_diff is not None:
        result["h_norm_diff"] = float(h_norm_diff)
    
    return result


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
    log_file = TMP_DIR / f"phase306_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 306: Normalized Construction PCA + Position/Norm Decomposition -- {model_name}")
    
    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    # Verify deep layers
    layers = get_layers(model)
    for tl in [0, nl // 2, nl - 2]:
        try:
            _ = layers[tl]
            log(f"  Layer {tl}: accessible")
        except Exception as e:
            log(f"  Layer {tl}: FAILED - {e}")
    
    # ---- Build stimuli ----
    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)
    pos_stimuli = resolve_positions(build_position_stimuli(), tok)
    log(f"  Observation stimuli: {len(sub_stimuli)}, Causal: {len(causal_stimuli)}, Position: {len(pos_stimuli)}")
    
    # Count tokens
    token_roles = defaultdict(set)
    token_rp = {}
    for stim in sub_stimuli:
        token_roles[stim["token_label"]].add(stim["role_label"])
        token_rp[stim["token_label"]] = stim.get("role_pair", "")
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])
    log(f"  Dual-role tokens: {len(dual_tokens)}")
    
    # Deduplicate sentences
    all_sentences = []
    sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli + pos_stimuli:
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
    
    # Organize data
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        frame = stim.get("frame_label", "")
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, frame)].append((idx, pos))
    
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]
        role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim
    dual_test = [(t, sorted(rs.keys())) for t, rs in test_pairs.items() if len(rs) >= 2]
    log(f"  Causal test pairs: {len(dual_test)} tokens")
    
    # Position data
    pos_data = defaultdict(list)
    for stim in pos_stimuli:
        token = stim["token_label"]
        idx = stim.get("_idx")
        pos = stim.get("target_pos")
        frame = stim.get("frame_label", "")
        if idx is not None and pos is not None:
            pos_data[token].append({"idx": idx, "pos": pos, "frame_label": frame})
    
    # Layer selection
    sample_layers = sorted(set([
        max(1, nl // 8), max(1, nl // 4), max(1, 3 * nl // 8),
        nl // 2, 5 * nl // 8, 3 * nl // 4, 7 * nl // 8, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")
    
    # =====================================================================
    # PART A: NORMALIZED VS RAW PCA COMPARISON
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: NORMALIZED VS RAW PCA COMPARISON")
    log(f"{'='*60}")
    
    all_norm_pca = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Normalized PCA ---")
        
        cell_means = {}
        for (token, role, frame), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, frame)] = np.mean(vecs, axis=0)
        
        norm_pca = within_role_pca_normalized(cell_means, d_model, n_components=5)
        all_norm_pca[str(li)] = norm_pca
    
    # =====================================================================
    # PART B: POSITION DIRECTION EXTRACTION & R/C/P/N DECOMPOSITION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART B: POSITION DIRECTION + R/C/P/N DECOMPOSITION")
    log(f"{'='*60}")
    
    all_decompose = {}
    all_position_dirs = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: R/C/P/N Decomposition ---")
        
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
        
        # Get PCA components from normalized analysis
        pca_results = all_norm_pca.get(str(li), {})
        
        # Position directions
        pos_dirs_layer = {}
        for token, entries in pos_data.items():
            vecs_with_pos = []
            for e in entries:
                h = captures[e["idx"]]["hidden"].get(li)
                if h is not None and e["pos"] < h.shape[1]:
                    vecs_with_pos.append((e["frame_label"], h[0, e["pos"], :].numpy().copy()))
            if len(vecs_with_pos) >= 2:
                sorted_vecs = sorted(vecs_with_pos, key=lambda x: x[0])
                # Position direction: late - early
                pos_dir = sorted_vecs[-1][1] - sorted_vecs[0][1]
                pos_dirs_layer[token] = pos_dir
        
        # Average position direction (for decomposition)
        avg_pos_dir = None
        if pos_dirs_layer:
            avg_pos_dir = np.mean(list(pos_dirs_layer.values()), axis=0)
            log(f"  Position direction: {len(pos_dirs_layer)} tokens, "
                f"norm={np.linalg.norm(avg_pos_dir):.2f}")
        
        all_position_dirs[str(li)] = {
            token: vec.tolist() for token, vec in pos_dirs_layer.items()
        }
        
        # Per-token R/C/P/N decomposition
        decompose_results = {}
        
        for token, roles in dual_test:
            if len(roles) < 2:
                continue
            
            role1, role2 = roles[0], roles[1]
            stim1 = test_pairs[token].get(role1)
            stim2 = test_pairs[token].get(role2)
            if stim1 is None or stim2 is None:
                continue
            
            idx1, pos1 = stim1.get("_idx"), stim1.get("target_pos")
            idx2, pos2 = stim2.get("_idx"), stim2.get("target_pos")
            if idx1 is None or idx2 is None:
                continue
            
            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None:
                continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]:
                continue
            
            v1 = h1[0, pos1, :].numpy().copy()
            v2 = h2[0, pos2, :].numpy().copy()
            
            full_delta = v2 - v1
            
            # R_direction = LOO role direction (same as Phase 304)
            # Compute from observation data: mean across all same-role tokens
            R_direction = np.zeros(d_model)
            n_contrib = 0
            for (t, r, f), entries in obs.items():
                if t == token or r not in [role1, role2]:
                    continue  # Leave this token out
                vecs_r = []
                for (idx, pos) in entries:
                    h = captures[idx]["hidden"].get(li)
                    if h is not None and pos < h.shape[1]:
                        vecs_r.append(h[0, pos, :].numpy().copy())
                if vecs_r:
                    R_direction += (np.mean(vecs_r, axis=0) * (1 if r == role2 else -1))
                    n_contrib += 1
            if n_contrib > 0:
                R_direction /= n_contrib
            else:
                R_direction = full_delta  # Fallback
            
            # Get construction components for this role
            C_comps = None
            for role in [role1, role2]:
                if role in pca_results and "raw_pca_components" in pca_results[role]:
                    C_comps = np.array(pca_results[role]["raw_pca_components"])
                    break
            
            # Position direction for this token (or average)
            P_dirs = [pos_dirs_layer.get(token)]
            if avg_pos_dir is not None:
                P_dirs.append(avg_pos_dir)
            
            # Norm difference
            h_norm_diff = np.linalg.norm(v2) - np.linalg.norm(v1)
            
            # Decompose
            decomp = decompose_full_delta_rcpn(
                full_delta, R_direction, C_comps, P_dirs, d_model, h_norm_diff
            )
            
            rp = token_rp.get(token, "")
            decompose_results[f"{token}_{role1}_{role2}"] = {
                "token": token, "role1": role1, "role2": role2,
                "role_pair": rp, **decomp,
            }
        
        all_decompose[str(li)] = decompose_results
        
        # Summary
        if decompose_results:
            C_pcts = [v.get("C_energy_pct", 0) for v in decompose_results.values()]
            P_pcts = [v.get("P_energy_pct", 0) for v in decompose_results.values()]
            N_pcts = [v.get("N_energy_pct", 0) for v in decompose_results.values()]
            U_pcts = [v.get("U_energy_pct", 0) for v in decompose_results.values()]
            log(f"  Energy budget: C={np.mean(C_pcts):.1f}%, P={np.mean(P_pcts):.1f}%, "
                f"N={np.mean(N_pcts):.1f}%, U={np.mean(U_pcts):.1f}%")
    
    # =====================================================================
    # PART C: O(not) ORTHOGONALITY WITH R AND C (using Phase 305 data)
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART C: O(not) ORTHOGONALITY CHECK (Phase 305 data)")
    log(f"{'='*60}")
    
    # Load Phase 305 data if available
    phase305_path = Path(f"results/phase305_operator_causal/{model_name}_operator_causal.json")
    orthogonality_results = {}
    
    if phase305_path.exists():
        p305 = json.load(open(phase305_path, "r", encoding="utf-8"))
        log(f"  Phase 305 data loaded: {len(p305.get('results', {}))} layers")
        
        # Extract O(not) directions from Phase 305
        for li_str, layer_results in p305.get("results", {}).items():
            li = int(li_str)
            if li not in sample_layers and str(li) not in [str(sl) for sl in sample_layers]:
                continue
            
            # Get per-role O(not) directions
            o_dirs = {}
            for key, val in layer_results.items():
                if "O_direction" in val and val.get("O_direction") is not None:
                    role = val.get("role", "unknown")
                    o_dirs[role] = np.array(val["O_direction"])
            
            # Compare with R and C directions
            pca_data = all_norm_pca.get(str(li), {})
            
            for role, o_dir in o_dirs.items():
                if role not in pca_data:
                    continue
                
                # O vs C (raw PCA PC1)
                raw_comps = pca_data[role].get("raw_pca_components", [])
                unit_comps = pca_data[role].get("unit_pca_components", [])
                
                if raw_comps:
                    pc1_raw = np.array(raw_comps[0])
                    cos_O_C_raw = cosine_sim(o_dir, pc1_raw)
                else:
                    cos_O_C_raw = 0.0
                
                if unit_comps:
                    pc1_unit = np.array(unit_comps[0])
                    cos_O_C_unit = cosine_sim(o_dir, pc1_unit)
                else:
                    cos_O_C_unit = 0.0
                
                orthogonality_results[f"L{li}_{role}"] = {
                    "layer": li, "role": role,
                    "cos_O_C_raw": float(cos_O_C_raw),
                    "cos_O_C_unit": float(cos_O_C_unit),
                }
                log(f"  L{li} {role}: cos(O,C_raw)={cos_O_C_raw:+.3f}, "
                    f"cos(O,C_unit)={cos_O_C_unit:+.3f}")
    else:
        log(f"  Phase 305 data not found at {phase305_path}")
    
    # =====================================================================
    # PART D: CAUSAL TEST WITH NORMALIZED CONSTRUCTION DIRECTION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART D: CAUSAL TEST WITH NORMALIZED C + POSITION")
    log(f"{'='*60}")
    
    causal_results = {}
    W_U_cache = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Causal test ---")
        
        # Get unembedding matrix
        if li not in W_U_cache:
            W_U_cache[li] = None
            try:
                if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                    W_U_cache[li] = model.lm_head.weight.detach().cpu().float().numpy().T
                elif hasattr(model, 'embed_out'):
                    W_U_cache[li] = model.embed_out.weight.detach().cpu().float().numpy().T
            except Exception as e:
                log(f"  W_U extraction failed: {e}")
        W_U = W_U_cache[li]
        
        pca_data = all_norm_pca.get(str(li), {})
        pos_dirs = all_position_dirs.get(str(li), {})
        avg_pos_dir = None
        if pos_dirs:
            avg_pos_dir = np.mean(list(pos_dirs.values()), axis=0)
        
        layer_causal = {}
        
        for token, roles in dual_test:
            if len(roles) < 2:
                continue
            
            role1, role2 = roles[0], roles[1]
            stim1 = test_pairs[token].get(role1)
            stim2 = test_pairs[token].get(role2)
            if stim1 is None or stim2 is None:
                continue
            
            sent1 = stim1["sentence"]
            sent2 = stim2["sentence"]
            idx1, pos1 = stim1.get("_idx"), stim1.get("target_pos")
            idx2, pos2 = stim2.get("_idx"), stim2.get("target_pos")
            
            if idx1 is None or idx2 is None:
                continue
            
            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None:
                continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]:
                continue
            
            v1 = h1[0, pos1, :].numpy().copy()
            v2 = h2[0, pos2, :].numpy().copy()
            
            full_delta = v2 - v1
            
            # Target token for causal readout
            target_tokens = [stim1["target_word"], stim2["target_word"]]
            target_ids = []
            for tw in target_tokens:
                ids = tok.encode(" " + tw, add_special_tokens=False)
                target_ids.extend(ids)
            if not target_ids:
                ids = tok.encode(target_tokens[0], add_special_tokens=False)
                target_ids.extend(ids)
            if not target_ids:
                continue
            
            # Baseline logit
            base_logits = captures[idx1]["logits"][0, pos1, :].numpy()
            base_logit_target = float(np.mean(base_logits[target_ids]))
            
            # --- R_only causal test ---
            # Use LOO role direction (computed from observation data, excluding this token)
            R_loo_dir = np.zeros(d_model)
            n_contrib = 0
            for (t, r, f), entries in obs.items():
                if t == token or r not in [role1, role2]:
                    continue
                vecs_r = []
                for (idx, pos) in entries:
                    h = captures[idx]["hidden"].get(li)
                    if h is not None and pos < h.shape[1]:
                        vecs_r.append(h[0, pos, :].numpy().copy())
                if vecs_r:
                    R_loo_dir += (np.mean(vecs_r, axis=0) * (1 if r == role2 else -1))
                    n_contrib += 1
            if n_contrib > 0:
                R_loo_dir /= n_contrib
            else:
                R_loo_dir = full_delta
            
            R_dir = R_loo_dir / max(np.linalg.norm(R_loo_dir), 1e-10) * np.linalg.norm(full_delta) * 0.1
            patched_R = run_with_patched_hidden(model, tok, sent1, li, pos1, R_dir)
            R_cos_shift = 0.0
            if patched_R is not None:
                patched_logit_target = float(np.mean(patched_R[0, pos1, target_ids].numpy()))
                R_cos_shift = patched_logit_target - base_logit_target
            
            # --- C_only causal test (raw PCA PC1) ---
            C_raw_dir = np.zeros(d_model)
            for role in [role1, role2]:
                if role in pca_data and "raw_pca_components" in pca_data[role]:
                    pc1 = np.array(pca_data[role]["raw_pca_components"][0])
                    C_raw_dir += pc1
            if np.linalg.norm(C_raw_dir) > 1e-10:
                C_raw_dir = C_raw_dir / np.linalg.norm(C_raw_dir) * np.linalg.norm(full_delta) * 0.1
                patched_C_raw = run_with_patched_hidden(model, tok, sent1, li, pos1, C_raw_dir)
                C_raw_cos_shift = 0.0
                if patched_C_raw is not None:
                    patched_logit_target = float(np.mean(patched_C_raw[0, pos1, target_ids].numpy()))
                    C_raw_cos_shift = patched_logit_target - base_logit_target
            else:
                C_raw_cos_shift = 0.0
            
            # --- C_only causal test (unit PCA PC1) ---
            C_unit_dir = np.zeros(d_model)
            for role in [role1, role2]:
                if role in pca_data and "unit_pca_components" in pca_data[role]:
                    pc1 = np.array(pca_data[role]["unit_pca_components"][0])
                    C_unit_dir += pc1
            if np.linalg.norm(C_unit_dir) > 1e-10:
                C_unit_dir = C_unit_dir / np.linalg.norm(C_unit_dir) * np.linalg.norm(full_delta) * 0.1
                patched_C_unit = run_with_patched_hidden(model, tok, sent1, li, pos1, C_unit_dir)
                C_unit_cos_shift = 0.0
                if patched_C_unit is not None:
                    patched_logit_target = float(np.mean(patched_C_unit[0, pos1, target_ids].numpy()))
                    C_unit_cos_shift = patched_logit_target - base_logit_target
            else:
                C_unit_cos_shift = 0.0
            
            # --- P_only causal test ---
            P_cos_shift = 0.0
            if avg_pos_dir is not None and np.linalg.norm(avg_pos_dir) > 1e-10:
                P_dir = avg_pos_dir / np.linalg.norm(avg_pos_dir) * np.linalg.norm(full_delta) * 0.1
                patched_P = run_with_patched_hidden(model, tok, sent1, li, pos1, P_dir)
                if patched_P is not None:
                    patched_logit_target = float(np.mean(patched_P[0, pos1, target_ids].numpy()))
                    P_cos_shift = patched_logit_target - base_logit_target
            
            # --- full_delta causal test ---
            fd_inject = full_delta * 0.1
            patched_fd = run_with_patched_hidden(model, tok, sent1, li, pos1, fd_inject)
            fd_cos_shift = 0.0
            if patched_fd is not None:
                patched_logit_target = float(np.mean(patched_fd[0, pos1, target_ids].numpy()))
                fd_cos_shift = patched_logit_target - base_logit_target
            
            rp = token_rp.get(token, "")
            key = f"{token}_{role1}_{role2}"
            layer_causal[key] = {
                "token": token, "role1": role1, "role2": role2,
                "role_pair": rp,
                "R_only_cos_shift": float(R_cos_shift),
                "C_raw_cos_shift": float(C_raw_cos_shift),
                "C_unit_cos_shift": float(C_unit_cos_shift),
                "P_only_cos_shift": float(P_cos_shift),
                "full_delta_cos_shift": float(fd_cos_shift),
                "base_logit_target": float(base_logit_target),
            }
        
        causal_results[str(li)] = layer_causal
        
        # Summary
        if layer_causal:
            R_shifts = [v["R_only_cos_shift"] for v in layer_causal.values()]
            C_raw_shifts = [v["C_raw_cos_shift"] for v in layer_causal.values()]
            C_unit_shifts = [v["C_unit_cos_shift"] for v in layer_causal.values()]
            P_shifts = [v["P_only_cos_shift"] for v in layer_causal.values()]
            FD_shifts = [v["full_delta_cos_shift"] for v in layer_causal.values()]
            log(f"  R_only={np.mean(R_shifts):+.4f}, C_raw={np.mean(C_raw_shifts):+.4f}, "
                f"C_unit={np.mean(C_unit_shifts):+.4f}, P_only={np.mean(P_shifts):+.4f}, "
                f"FD={np.mean(FD_shifts):+.4f}")
    
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
        "n_position_tokens": len(pos_data),
        
        # Part A: Normalized PCA
        "norm_pca_results": make_serializable(all_norm_pca),
        
        # Part B: R/C/P/N decomposition
        "decompose_results": make_serializable(all_decompose),
        "position_directions": make_serializable(all_position_dirs),
        
        # Part C: O(not) orthogonality
        "orthogonality_results": make_serializable(orthogonality_results),
        
        # Part D: Causal tests
        "causal_results": make_serializable(causal_results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_norm_position.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  Saved to {out_path}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY: Phase 306 -- {model_name}")
    log(f"{'='*60}")
    
    # Part A summary
    log(f"\n--- Normalized PCA comparison ---")
    for li_str, pca_data in all_norm_pca.items():
        log(f"  Layer {li_str}:")
        for role, data in pca_data.items():
            log(f"    {role}: Raw PC1={data['raw_pc1_var']*100:.1f}%, "
                f"Unit PC1={data['unit_pc1_var']*100:.1f}%, "
                f"Norm CV={data['dev_norm_cv']:.3f}, "
                f"cos(PC1,mean)={data['cos_pc1_mean_direction']:+.3f}, "
                f"corr(|PC1|,norm)={data['corr_pc1_proj_norm']:+.3f}")
    
    # Part B summary
    log(f"\n--- R/C/P/N energy budget ---")
    for li_str, decomp in all_decompose.items():
        if decomp:
            C_pcts = [v.get("C_energy_pct", 0) for v in decomp.values()]
            P_pcts = [v.get("P_energy_pct", 0) for v in decomp.values()]
            N_pcts = [v.get("N_energy_pct", 0) for v in decomp.values()]
            U_pcts = [v.get("U_energy_pct", 0) for v in decomp.values()]
            log(f"  Layer {li_str}: C={np.mean(C_pcts):.1f}%, P={np.mean(P_pcts):.1f}%, "
                f"N={np.mean(N_pcts):.1f}%, U={np.mean(U_pcts):.1f}%")
    
    # Part C summary
    log(f"\n--- O(not) orthogonality with C ---")
    for key, val in orthogonality_results.items():
        log(f"  {key}: cos(O,C_raw)={val['cos_O_C_raw']:+.3f}, "
            f"cos(O,C_unit)={val['cos_O_C_unit']:+.3f}")
    
    # Part D summary
    log(f"\n--- Causal test summary ---")
    for li_str, causal in causal_results.items():
        if causal:
            R = [v["R_only_cos_shift"] for v in causal.values()]
            C_r = [v["C_raw_cos_shift"] for v in causal.values()]
            C_u = [v["C_unit_cos_shift"] for v in causal.values()]
            P = [v["P_only_cos_shift"] for v in causal.values()]
            FD = [v["full_delta_cos_shift"] for v in causal.values()]
            log(f"  Layer {li_str}: R={np.mean(R):+.4f}, C_raw={np.mean(C_r):+.4f}, "
                f"C_unit={np.mean(C_u):+.4f}, P={np.mean(P):+.4f}, FD={np.mean(FD):+.4f}")
    
    # ---- Release model ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 306 complete for {model_name}")


if __name__ == "__main__":
    main()
