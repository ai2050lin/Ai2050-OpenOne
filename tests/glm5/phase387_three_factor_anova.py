"""
Phase 387: Three-Factor ANOVA Decomposition (I + A + V + I×A + I×V + A×V + I×A×V) + Causal Test
================================================================================================

Core Question:
  What is the causal hierarchy of language encoding factors?
  - I (object identity): high variance, but does it have independent causal effect?
  - A (category/relation): small variance centroid - confirmed causal (Phase 386)
  - V (value/attribute): never tested independently
  - I×A: interaction - never tested
  - I×V: interaction - never tested
  - A×V: interaction - never tested
  - I×A×V: triple interaction - the hypothesized core of language encoding

Key insight from Phase 386-386b:
  - A(category centroid) has significant causal effect despite tiny R²
  - I+A is NOT simply additive (nonlinear interaction)
  - epsilon (which contains V + interactions) is unexplained
  → We MUST decompose epsilon into V + I×A + I×V + A×V + I×A×V

Data Design:
  For proper three-factor ANOVA, we need:
  - Objects that appear in multiple categories
  - Both correct and incorrect values for each object-category pair
  - This gives us V variation within I×A cells

  Example:
    apple × color → red (correct), blue (incorrect)
    apple × taste → sweet (correct), sour (incorrect)
    fire × temperature → hot (correct), cold (incorrect)
    snow × color → white (correct), black (incorrect)

Method:
  Part 1: Three-way ANOVA on raw Δh
    Δh = μ + I + A + V + I×A + I×V + A×V + I×A×V + ε
    Using cell means approach for balanced/unbalanced designs

  Part 2: Centroid-based causal test for each component
    - For I: mean Δh for each object, residualized
    - For A: mean Δh for each category, residualized after I
    - For V: mean Δh for each value, residualized after I+A
    - For I×A: cell mean - I - A - μ (after removing main effects)
    - For I×V: cell mean - I - V - μ
    - For A×V: cell mean - A - V - μ
    - For I×A×V: residual after all main + two-way effects

  Part 3: Raw-space causal patch test (add to corrupt, remove from clean)
  Part 4: Cross-model comparison of factor hierarchy

Usage:
  python tests/glm5/phase387_three_factor_anova.py qwen3
  python tests/glm5/phase387_three_factor_anova.py deepseek7b
  python tests/glm5/phase387_three_factor_anova.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Extended Data: Objects with multiple categories + correct/incorrect values =====

# For three-factor ANOVA, we need objects that span multiple categories
# with both correct and incorrect attribute values

MULTI_CATEGORY_OBJECTS = {
    # object: {category: [correct_value, incorrect_values...]}
    "apple": {
        "color": ["red", "blue"],
        "taste": ["sweet", "sour"],
    },
    "snow": {
        "color": ["white", "black"],
        "temperature": ["cold", "hot"],
    },
    "fire": {
        "temperature": ["hot", "cold"],
        "brightness": ["bright", "dark"],
    },
    "ocean": {
        "color": ["blue", "red"],
        "moisture": ["wet", "dry"],
    },
    "desert": {
        "temperature": ["hot", "cold"],
        "moisture": ["dry", "wet"],
    },
    "elephant": {
        "size": ["big", "small"],
        "weight": ["heavy", "light"],
    },
    "feather": {
        "weight": ["light", "heavy"],
        "size": ["small", "big"],
    },
    "coal": {
        "color": ["black", "white"],
        "temperature": ["hot", "cold"],
    },
    "cheetah": {
        "speed": ["fast", "slow"],
        "size": ["big", "small"],
    },
    "star": {
        "brightness": ["bright", "dark"],
        "size": ["big", "small"],
    },
    "cloud": {
        "color": ["white", "black"],
        "weight": ["light", "heavy"],
    },
    "lava": {
        "temperature": ["hot", "cold"],
        "brightness": ["bright", "dark"],
    },
}

# Build experiment stimuli: for each (object, category, value) triplet
# Clean = "The {obj} is {value}"
# We test whether the model prefers correct_value over incorrect_value for each object-category pair

TEMPLATE = "The {obj} is {attr}."
CORRUPTED_BASELINE = "The item"

# Build the full stimulus list
# Each entry: (object, category, correct_value, incorrect_value, condition)
# condition = "correct" or "incorrect"
STIMULI = []
for obj, cats in MULTI_CATEGORY_OBJECTS.items():
    for cat, values in cats.items():
        correct_v = values[0]
        for inc_v in values[1:]:
            STIMULI.append({
                "object": obj,
                "category": cat,
                "value": correct_v,
                "value_type": "correct",
                "competitor": inc_v,
            })
            STIMULI.append({
                "object": obj,
                "category": cat,
                "value": inc_v,
                "value_type": "incorrect",
                "competitor": correct_v,
            })


# ===== Model Loading =====

def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash_attention_2 for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bfloat16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        log(f"  Loaded with flash_attention_2")
    except Exception as e:
        log(f"  flash_attention_2 failed ({str(e)[:60]}), falling back to eager...")
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
    log(f"  {model_name} loaded: GPU={gpu_mem:.2f}GB, device={device}")
    return model, tokenizer, device


def _load_ln_weight(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    for attr_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
        ln = getattr(layers[layer_idx], attr_name, None)
        if ln is not None:
            try:
                w = ln.weight.detach().cpu().float().numpy()
                if w is not None and len(w) > 0:
                    return w
            except (NotImplementedError, RuntimeError):
                pass
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    for ln_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
                        if f"layers.{layer_idx}.{ln_name}.weight" in key:
                            return sf.get_tensor(key).float().numpy()
        except:
            continue
    log(f"    WARNING: Could not load LN weight for layer {layer_idx}")
    return None


def rms_norm_single(x, weight=None, eps=1e-6):
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result


# ===== Forward pass utilities =====

def run_model_with_patch(model, tokenizer, device, prompt, layer_idx,
                         patch_delta, target_token_id, competitor_token_id):
    """Run model with delta added to residual at layer l (last token position)."""
    if target_token_id < 0 or competitor_token_id < 0:
        return None
    layers = get_layers(model)
    delta_tensor = torch.tensor(patch_delta, dtype=torch.bfloat16, device=device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h_patched = h.clone()
        h_patched[0, -1, :] += delta_tensor
        if isinstance(output, tuple):
            return (h_patched,) + output[1:]
        return h_patched

    hook = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            out = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )
            logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        log(f"    Forward failed: {str(e)[:80]}")
        hook.remove()
        return None
    hook.remove()
    return float(logits[target_token_id] - logits[competitor_token_id])


# ===== Three-Way ANOVA Decomposition =====

def three_way_anova(dh_raw, obj_labels, cat_labels, val_type_labels):
    """
    Three-way ANOVA decomposition of dh_raw using cell means approach.
    
    Factors:
      I: object identity
      A: category/relation
      V: value type (correct=1, incorrect=0)
    
    Returns dict of component vectors for each sample, plus R² stats.
    
    Cell means approach:
      μ_{i,j,k} = mean(dh | obj=i, cat=j, val_type=k)
      μ = grand mean
      α_i = mean(μ_{i,*,*}) - μ       [I main effect]
      β_j = mean(μ_{*,j,*}) - μ       [A main effect]  
      γ_k = mean(μ_{*,*,k}) - μ       [V main effect]
      (αβ)_{ij} = mean(μ_{i,j,*}) - μ - α_i - β_j  [I×A interaction]
      (αγ)_{ik} = mean(μ_{i,*,k}) - μ - α_i - γ_k  [I×V interaction]
      (βγ)_{jk} = mean(μ_{*,j,k}) - μ - β_j - γ_k  [A×V interaction]
      (αβγ)_{ijk} = μ_{ijk} - μ - α_i - β_j - γ_k - (αβ)_{ij} - (αγ)_{ik} - (βγ)_{jk}
    """
    n, d = dh_raw.shape
    mu = np.mean(dh_raw, axis=0)  # grand mean
    
    # Build factor index maps
    unique_objs = sorted(set(obj_labels))
    unique_cats = sorted(set(cat_labels))
    unique_vtypes = sorted(set(val_type_labels))
    
    obj_to_idx = {o: i for i, o in enumerate(unique_objs)}
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    vtype_to_idx = {v: i for i, v in enumerate(unique_vtypes)}
    
    n_obj = len(unique_objs)
    n_cat = len(unique_cats)
    n_vtype = len(unique_vtypes)
    
    # Cell means: μ_{i,j,k}
    cell_sums = np.zeros((n_obj, n_cat, n_vtype, d))
    cell_counts = np.zeros((n_obj, n_cat, n_vtype))
    
    for s in range(n):
        oi = obj_to_idx[obj_labels[s]]
        ci = cat_to_idx[cat_labels[s]]
        vi = vtype_to_idx[val_type_labels[s]]
        cell_sums[oi, ci, vi] += dh_raw[s]
        cell_counts[oi, ci, vi] += 1
    
    # Compute cell means (handle empty cells)
    cell_means = np.zeros((n_obj, n_cat, n_vtype, d))
    for i in range(n_obj):
        for j in range(n_cat):
            for k in range(n_vtype):
                if cell_counts[i, j, k] > 0:
                    cell_means[i, j, k] = cell_sums[i, j, k] / cell_counts[i, j, k]
                else:
                    cell_means[i, j, k] = mu  # fill empty cells with grand mean
    
    # Marginal means
    # μ_{i,*,*} = mean over j,k of μ_{i,j,k} (weighted by counts)
    obj_marginals = np.zeros((n_obj, d))
    for i in range(n_obj):
        total_w = 0
        for j in range(n_cat):
            for k in range(n_vtype):
                w = cell_counts[i, j, k]
                if w > 0:
                    obj_marginals[i] += w * cell_means[i, j, k]
                    total_w += w
        if total_w > 0:
            obj_marginals[i] /= total_w
        else:
            obj_marginals[i] = mu
    
    # μ_{*,j,*}
    cat_marginals = np.zeros((n_cat, d))
    for j in range(n_cat):
        total_w = 0
        for i in range(n_obj):
            for k in range(n_vtype):
                w = cell_counts[i, j, k]
                if w > 0:
                    cat_marginals[j] += w * cell_means[i, j, k]
                    total_w += w
        if total_w > 0:
            cat_marginals[j] /= total_w
        else:
            cat_marginals[j] = mu
    
    # μ_{*,*,k}
    vtype_marginals = np.zeros((n_vtype, d))
    for k in range(n_vtype):
        total_w = 0
        for i in range(n_obj):
            for j in range(n_cat):
                w = cell_counts[i, j, k]
                if w > 0:
                    vtype_marginals[k] += w * cell_means[i, j, k]
                    total_w += w
        if total_w > 0:
            vtype_marginals[k] /= total_w
        else:
            vtype_marginals[k] = mu
    
    # Main effects (centroids)
    I_effect = np.zeros((n_obj, d))  # α_i = μ_{i,*,*} - μ
    for i in range(n_obj):
        I_effect[i] = obj_marginals[i] - mu
    
    A_effect = np.zeros((n_cat, d))  # β_j = μ_{*,j,*} - μ
    for j in range(n_cat):
        A_effect[j] = cat_marginals[j] - mu
    
    V_effect = np.zeros((n_vtype, d))  # γ_k = μ_{*,*,k} - μ
    for k in range(n_vtype):
        V_effect[k] = vtype_marginals[k] - mu
    
    # Two-way interaction means
    # μ_{i,j,*}
    obj_cat_marginals = np.zeros((n_obj, n_cat, d))
    for i in range(n_obj):
        for j in range(n_cat):
            total_w = 0
            for k in range(n_vtype):
                w = cell_counts[i, j, k]
                if w > 0:
                    obj_cat_marginals[i, j] += w * cell_means[i, j, k]
                    total_w += w
            if total_w > 0:
                obj_cat_marginals[i, j] /= total_w
            else:
                obj_cat_marginals[i, j] = mu
    
    # (αβ)_{ij} = μ_{i,j,*} - μ - α_i - β_j
    IA_effect = np.zeros((n_obj, n_cat, d))
    for i in range(n_obj):
        for j in range(n_cat):
            IA_effect[i, j] = obj_cat_marginals[i, j] - mu - I_effect[i] - A_effect[j]
    
    # μ_{i,*,k}
    obj_vtype_marginals = np.zeros((n_obj, n_vtype, d))
    for i in range(n_obj):
        for k in range(n_vtype):
            total_w = 0
            for j in range(n_cat):
                w = cell_counts[i, j, k]
                if w > 0:
                    obj_vtype_marginals[i, k] += w * cell_means[i, j, k]
                    total_w += w
            if total_w > 0:
                obj_vtype_marginals[i, k] /= total_w
            else:
                obj_vtype_marginals[i, k] = mu
    
    # (αγ)_{ik} = μ_{i,*,k} - μ - α_i - γ_k
    IV_effect = np.zeros((n_obj, n_vtype, d))
    for i in range(n_obj):
        for k in range(n_vtype):
            IV_effect[i, k] = obj_vtype_marginals[i, k] - mu - I_effect[i] - V_effect[k]
    
    # μ_{*,j,k}
    cat_vtype_marginals = np.zeros((n_cat, n_vtype, d))
    for j in range(n_cat):
        for k in range(n_vtype):
            total_w = 0
            for i in range(n_obj):
                w = cell_counts[i, j, k]
                if w > 0:
                    cat_vtype_marginals[j, k] += w * cell_means[i, j, k]
                    total_w += w
            if total_w > 0:
                cat_vtype_marginals[j, k] /= total_w
            else:
                cat_vtype_marginals[j, k] = mu
    
    # (βγ)_{jk} = μ_{*,j,k} - μ - β_j - γ_k
    AV_effect = np.zeros((n_cat, n_vtype, d))
    for j in range(n_cat):
        for k in range(n_vtype):
            AV_effect[j, k] = cat_vtype_marginals[j, k] - mu - A_effect[j] - V_effect[k]
    
    # Three-way interaction
    # (αβγ)_{ijk} = μ_{ijk} - μ - α_i - β_j - γ_k - (αβ)_{ij} - (αγ)_{ik} - (βγ)_{jk}
    IAV_effect = np.zeros((n_obj, n_cat, n_vtype, d))
    for i in range(n_obj):
        for j in range(n_cat):
            for k in range(n_vtype):
                IAV_effect[i, j, k] = (cell_means[i, j, k] - mu 
                    - I_effect[i] - A_effect[j] - V_effect[k]
                    - IA_effect[i, j] - IV_effect[i, k] - AV_effect[j, k])
    
    # Assign components to each sample
    comp_I = np.zeros_like(dh_raw)
    comp_A = np.zeros_like(dh_raw)
    comp_V = np.zeros_like(dh_raw)
    comp_IA = np.zeros_like(dh_raw)
    comp_IV = np.zeros_like(dh_raw)
    comp_AV = np.zeros_like(dh_raw)
    comp_IAV = np.zeros_like(dh_raw)
    comp_eps = np.zeros_like(dh_raw)
    
    for s in range(n):
        oi = obj_to_idx[obj_labels[s]]
        ci = cat_to_idx[cat_labels[s]]
        vi = vtype_to_idx[val_type_labels[s]]
        comp_I[s] = I_effect[oi]
        comp_A[s] = A_effect[ci]
        comp_V[s] = V_effect[vi]
        comp_IA[s] = IA_effect[oi, ci]
        comp_IV[s] = IV_effect[oi, vi]
        comp_AV[s] = AV_effect[ci, vi]
        comp_IAV[s] = IAV_effect[oi, ci, vi]
        comp_eps[s] = (dh_raw[s] - mu - comp_I[s] - comp_A[s] - comp_V[s]
                       - comp_IA[s] - comp_IV[s] - comp_AV[s] - comp_IAV[s])
    
    # R² computation
    ss_total = np.sum((dh_raw - mu) ** 2)
    ss_I = np.sum(comp_I ** 2)
    ss_A = np.sum(comp_A ** 2)
    ss_V = np.sum(comp_V ** 2)
    ss_IA = np.sum(comp_IA ** 2)
    ss_IV = np.sum(comp_IV ** 2)
    ss_AV = np.sum(comp_AV ** 2)
    ss_IAV = np.sum(comp_IAV ** 2)
    ss_eps = np.sum(comp_eps ** 2)
    
    stats = {
        'r2_I': float(ss_I / ss_total) if ss_total > 0 else 0,
        'r2_A': float(ss_A / ss_total) if ss_total > 0 else 0,
        'r2_V': float(ss_V / ss_total) if ss_total > 0 else 0,
        'r2_IA': float(ss_IA / ss_total) if ss_total > 0 else 0,
        'r2_IV': float(ss_IV / ss_total) if ss_total > 0 else 0,
        'r2_AV': float(ss_AV / ss_total) if ss_total > 0 else 0,
        'r2_IAV': float(ss_IAV / ss_total) if ss_total > 0 else 0,
        'r2_eps': float(ss_eps / ss_total) if ss_total > 0 else 0,
        'ss_total': float(ss_total),
        'n_obj': n_obj, 'n_cat': n_cat, 'n_vtype': n_vtype,
        'cells_filled': int(np.sum(cell_counts > 0)),
        'cells_total': n_obj * n_cat * n_vtype,
    }
    
    components = {
        'I': comp_I,
        'A': comp_A,
        'V': comp_V,
        'IA': comp_IA,
        'IV': comp_IV,
        'AV': comp_AV,
        'IAV': comp_IAV,
        'eps': comp_eps,
        'full': dh_raw,
        'mu': np.tile(mu, (n, 1)),
    }
    
    return components, stats


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 387: Three-Factor ANOVA — {model_name}")
    log(f"=" * 70)
    
    # Stimuli summary
    n_stim = len(STIMULI)
    objects = sorted(set(s["object"] for s in STIMULI))
    categories = sorted(set(s["category"] for s in STIMULI))
    values = sorted(set(s["value"] for s in STIMULI))
    obj_labels = [s["object"] for s in STIMULI]
    cat_labels = [s["category"] for s in STIMULI]
    val_type_labels = [s["value_type"] for s in STIMULI]
    
    log(f"Stimuli: {n_stim} total")
    log(f"  Objects: {len(objects)} {objects}")
    log(f"  Categories: {len(categories)} {categories}")
    log(f"  Value types: {sorted(set(val_type_labels))}")
    log(f"  Correct: {sum(1 for v in val_type_labels if v=='correct')}")
    log(f"  Incorrect: {sum(1 for v in val_type_labels if v=='incorrect')}")
    
    # Object-category distribution
    obj_cats = defaultdict(set)
    for s in STIMULI:
        obj_cats[s["object"]].add(s["category"])
    for obj in sorted(obj_cats.keys()):
        log(f"  {obj}: categories={sorted(obj_cats[obj])}")
    
    # Target layers
    if model_name == "qwen3":
        target_layers = [4, 12, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 20, 30]
    elif model_name == "deepseek7b":
        target_layers = [4, 8, 12, 20, 24]
    
    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    results = {}
    
    for l in target_layers:
        log(f"\n{'='*70}")
        log(f"Layer {l}")
        log(f"{'='*70}")
        t_l = time.time()
        
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # ===== Step 1: Collect residual states =====
        log(f"  Step 1: Collecting residual states for {n_stim} stimuli...")
        
        h_clean_raw = []
        h_corrupt_raw = []
        clean_logits_list = []
        corrupt_logits_list = []
        target_token_ids = []
        competitor_token_ids = []
        valid_indices = []
        
        for sidx, stim in enumerate(STIMULI):
            if sidx % 10 == 0:
                log(f"    Stimulus {sidx+1}/{n_stim}")
            
            obj = stim["object"]
            value = stim["value"]
            competitor = stim["competitor"]
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=value)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=value)
            
            t_ids = tokenizer.encode(value, add_special_tokens=False)
            c_ids = tokenizer.encode(competitor, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else -1
            c_id = c_ids[0] if len(c_ids) > 0 else -1
            
            if t_id < 0 or c_id < 0:
                log(f"    SKIP: cannot tokenize '{value}' or '{competitor}'")
                continue
            
            target_token_ids.append(t_id)
            competitor_token_ids.append(c_id)
            valid_indices.append(sidx)
            
            # Clean forward
            with torch.no_grad():
                toks = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
                out = model(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            last_pos = toks["input_ids"].shape[1] - 1
            h_raw_c = out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
            h_clean_raw.append(h_raw_c)
            clean_logits_list.append(out.logits[0, -1].float().cpu().numpy())
            del out
            
            # Corrupt forward
            with torch.no_grad():
                toks = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
                out = model(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            last_pos_r = toks["input_ids"].shape[1] - 1
            h_raw_r = out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            h_corrupt_raw.append(h_raw_r)
            corrupt_logits_list.append(out.logits[0, -1].float().cpu().numpy())
            del out
            
            if sidx % 3 == 0:
                torch.cuda.empty_cache()
        
        h_clean_raw = np.array(h_clean_raw)
        h_corrupt_raw = np.array(h_corrupt_raw)
        dh_raw = h_clean_raw - h_corrupt_raw
        
        # Filter labels for valid samples only
        obj_labels_v = [obj_labels[i] for i in valid_indices]
        cat_labels_v = [cat_labels[i] for i in valid_indices]
        val_type_labels_v = [val_type_labels[i] for i in valid_indices]
        
        n_valid = len(valid_indices)
        log(f"  Collected {n_valid} valid samples")
        
        # Baseline logit_diff
        baseline_clean_ld = []
        baseline_corrupt_ld = []
        for i in range(n_valid):
            t_id, c_id = target_token_ids[i], competitor_token_ids[i]
            baseline_clean_ld.append(float(clean_logits_list[i][t_id] - clean_logits_list[i][c_id]))
            baseline_corrupt_ld.append(float(corrupt_logits_list[i][t_id] - corrupt_logits_list[i][c_id]))
        baseline_clean_ld = np.array(baseline_clean_ld)
        baseline_corrupt_ld = np.array(baseline_corrupt_ld)
        
        log(f"  Baseline: clean_ld={np.mean(baseline_clean_ld):.3f}±{np.std(baseline_clean_ld):.3f}, "
            f"corrupt_ld={np.mean(baseline_corrupt_ld):.3f}±{np.std(baseline_corrupt_ld):.3f}")
        
        # ===== Step 2: Three-Way ANOVA =====
        log(f"  Step 2: Three-way ANOVA decomposition...")
        
        components, anova_stats = three_way_anova(
            dh_raw, obj_labels_v, cat_labels_v, val_type_labels_v
        )
        
        log(f"  ANOVA R²: I={anova_stats['r2_I']:.4f}, A={anova_stats['r2_A']:.4f}, "
            f"V={anova_stats['r2_V']:.4f}")
        log(f"  Interact R²: I×A={anova_stats['r2_IA']:.4f}, I×V={anova_stats['r2_IV']:.4f}, "
            f"A×V={anova_stats['r2_AV']:.4f}, I×A×V={anova_stats['r2_IAV']:.4f}")
        log(f"  Residual R²: eps={anova_stats['r2_eps']:.4f}")
        log(f"  Cells: {anova_stats['cells_filled']}/{anova_stats['cells_total']} filled")
        
        # Component norms
        for cname in ['I', 'A', 'V', 'IA', 'IV', 'AV', 'IAV', 'eps', 'full']:
            norms = np.linalg.norm(components[cname], axis=1)
            log(f"  Norm({cname}): {np.mean(norms):.4f}±{np.std(norms):.4f}")
        
        # ===== Step 3: Causal Tests =====
        log(f"  Step 3: Causal tests (9 components)...")
        
        ca_results = {}
        comp_names = ['I', 'A', 'V', 'IA', 'IV', 'AV', 'IAV', 'eps', 'full']
        for cn in comp_names:
            ca_results[cn] = {'add': [], 'remove': []}
        
        for cnt in range(n_valid):
            if cnt % 10 == 0:
                log(f"    Sample {cnt+1}/{n_valid}")
            
            stim = STIMULI[valid_indices[cnt]]
            obj = stim["object"]
            value = stim["value"]
            t_id = target_token_ids[cnt]
            c_id = competitor_token_ids[cnt]
            if t_id < 0 or c_id < 0:
                continue
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=value)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=value)
            
            for cn in comp_names:
                delta_add = components[cn][cnt]
                
                # Add to corrupt
                ld = run_model_with_patch(model, tokenizer, device, corrupt_prompt, l,
                                          delta_add, t_id, c_id)
                ca_results[cn]['add'].append(ld)
                
                # Remove from clean
                ld = run_model_with_patch(model, tokenizer, device, clean_prompt, l,
                                          -delta_add, t_id, c_id)
                ca_results[cn]['remove'].append(ld)
            
            if cnt % 2 == 0:
                torch.cuda.empty_cache()
        
        # ===== Step 4: Compute Effects =====
        log(f"  Step 4: Computing effects...")
        
        layer_result = {
            "layer": l,
            "n_valid": n_valid,
            "anova": anova_stats,
        }
        
        for cn in comp_names:
            add_vals = [v for v in ca_results[cn]['add'] if v is not None]
            rem_vals = [v for v in ca_results[cn]['remove'] if v is not None]
            
            # Add effect: (patched_corrupt_ld - baseline_corrupt_ld)
            if len(add_vals) > 0:
                n_eff = min(len(add_vals), len(baseline_corrupt_ld))
                add_eff = np.array(add_vals[:n_eff]) - baseline_corrupt_ld[:n_eff]
                mean_add = float(np.mean(add_eff))
                std_add = float(np.std(add_eff))
                t_add = mean_add / (std_add / np.sqrt(n_eff) + 1e-10)
                layer_result[f"{cn}_add"] = {
                    "mean": mean_add, "std": std_add, "t": float(t_add), "n": n_eff
                }
            
            # Remove effect: (baseline_clean_ld - patched_clean_ld)
            if len(rem_vals) > 0:
                n_eff = min(len(rem_vals), len(baseline_clean_ld))
                rem_eff = baseline_clean_ld[:n_eff] - np.array(rem_vals[:n_eff])
                mean_rem = float(np.mean(rem_eff))
                std_rem = float(np.std(rem_eff))
                t_rem = mean_rem / (std_rem / np.sqrt(n_eff) + 1e-10)
                layer_result[f"{cn}_remove"] = {
                    "mean": mean_rem, "std": std_rem, "t": float(t_rem), "n": n_eff
                }
        
        # Print summary
        log(f"\n  Layer {l} Summary:")
        log(f"  {'Comp':>6s} | {'Add mean':>10s} {'t':>8s} | {'Rem mean':>10s} {'t':>8s} | {'R²':>8s}")
        log(f"  {'-'*60}")
        for cn in comp_names:
            add_info = layer_result.get(f"{cn}_add", {})
            rem_info = layer_result.get(f"{cn}_remove", {})
            r2_key = f"r2_{cn}" if cn != "full" else None
            r2_val = anova_stats.get(r2_key, None)
            
            add_str = f"{add_info.get('mean', 0):+.4f} {add_info.get('t', 0):+6.2f}" if add_info else "    N/A"
            rem_str = f"{rem_info.get('mean', 0):+.4f} {rem_info.get('t', 0):+6.2f}" if rem_info else "    N/A"
            r2_str = f"{r2_val:.4f}" if r2_val is not None else "    N/A"
            log(f"  {cn:>6s} | {add_str} | {rem_str} | {r2_str}")
        
        results[f"L{l}"] = layer_result
        log(f"  Layer {l} done in {time.time()-t_l:.1f}s")
    
    # ===== Save =====
    os.makedirs("results/phase387_three_factor_anova", exist_ok=True)
    out_path = f"results/phase387_three_factor_anova/{model_name}_phase387.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")
    
    # Release model
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 387 complete for {model_name}!")


if __name__ == "__main__":
    main()
