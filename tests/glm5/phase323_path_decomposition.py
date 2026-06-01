"""
Phase 323: Attribute Path Component Decomposition
==================================================

Core question: In GLM4 L2-L4, which component (attention, MLP, or residual)
transforms the attribute direction into an output-readable signal?

This experiment:
1. Extracts attribute direction at optimal layer
2. Ablates each component (attention output, MLP output) separately
3. Measures which component removal kills the causal effect
4. Extends readout from single token to target cluster + competitor cluster
5. Tests cross-model (Qwen3, GLM4, DS7B)

Design:
- For each model at its optimal attribute layer:
  a. Baseline: inject attribute direction, measure delta
  b. Ablate attention: zero out attention output for that layer
  c. Ablate MLP: zero out MLP output for that layer
  d. Replace only attention output from attribute sentence
  e. Replace only MLP output from attribute sentence
  f. Measure logit changes on: target word, target cluster, competitor cluster

Word clusters:
- Color cluster: red, blue, green, yellow, white, black, orange, purple
- Temperature cluster: hot, cold, warm, cool, freezing, boiling
- Taste cluster: sweet, sour, bitter, salty, spicy
- Texture cluster: smooth, rough, soft, hard, sharp
- Object cluster: apple, sky, fire, ice, lemon, honey

Usage:
  python tests/glm5/phase323_path_decomposition.py qwen3
  python tests/glm5/phase323_path_decomposition.py glm4
  python tests/glm5/phase323_path_decomposition.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase323_path_decomp")
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


# ===== Word Clusters for Readout =====
WORD_CLUSTERS = {
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"],
    "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling"],
    "taste": ["sweet", "sour", "bitter", "salty", "spicy"],
    "texture": ["smooth", "rough", "soft", "hard", "sharp"],
    "object": ["apple", "sky", "fire", "ice", "lemon", "honey", "knife", "silk"],
    "action": ["cut", "write", "open", "close", "run", "fly"],
    "negation": ["not", "never", "barely", "no", "neither", "without"],
    "positive": ["happy", "good", "great", "safe", "beautiful", "love"],
    "negative": ["sad", "bad", "terrible", "dangerous", "ugly", "hate"],
}

# Attribute test pairs: (noun, attribute_value, cluster_type)
ATTRIBUTE_PAIRS = [
    ("apple", "red", "color"),
    ("sky", "blue", "color"),
    ("fire", "hot", "temperature"),
    ("ice", "cold", "temperature"),
    ("lemon", "sour", "taste"),
    ("honey", "sweet", "taste"),
    ("silk", "smooth", "texture"),
    ("sandpaper", "rough", "texture"),
]

# Transfer pairs: same attribute type, different object
TRANSFER_PAIRS = [
    ("apple", "red", "strawberry", "red", "color"),
    ("sky", "blue", "ocean", "blue", "color"),
    ("fire", "hot", "stove", "hot", "temperature"),
    ("ice", "cold", "frost", "cold", "temperature"),
    ("lemon", "sour", "grapefruit", "sour", "taste"),
    ("honey", "sweet", "candy", "sweet", "taste"),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try attention implementations in order of preference
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  {impl} failed: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded {model_name}: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_cluster_token_ids(tokenizer, cluster_words):
    """Get token IDs for a cluster of words."""
    ids = []
    for w in cluster_words:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append((w, tok_ids[0]))
    return ids


def compute_cluster_logit_stats(logits, cluster_ids):
    """Compute mean, max, min logit for a cluster."""
    if not cluster_ids:
        return {"mean": 0, "max": 0, "min": 0, "sum": 0}
    vals = [float(logits[tid]) for _, tid in cluster_ids]
    return {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
            "min": float(np.min(vals)), "sum": float(np.sum(vals))}


def extract_component_outputs(model, tokenizer, device, sentence, layer_idx):
    """
    Extract attention output, MLP output, and full layer output at a specific layer.
    
    Returns:
        dict with keys: 'attn_out', 'mlp_out', 'layer_out', 'residual_before'
    """
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
    captured = {}

    def make_hook(key, is_layer=False):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
            # Also capture residual input for layer-level hooks
            if is_layer and isinstance(input, tuple) and len(input) > 0:
                captured[key + '_input'] = input[0].detach().float().cpu()
        return hook_fn

    hooks = []
    # Hook the attention sublayer
    if hasattr(layer, 'self_attn'):
        hooks.append(layer.self_attn.register_forward_hook(make_hook('attn_out')))
    # Hook the MLP sublayer
    if hasattr(layer, 'mlp'):
        hooks.append(layer.mlp.register_forward_hook(make_hook('mlp_out')))
    # Hook the full layer
    hooks.append(layer.register_forward_hook(make_hook('layer_out', is_layer=True)))

    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp)

    for h in hooks:
        h.remove()

    result = {}
    for key in ['attn_out', 'mlp_out', 'layer_out']:
        if key in captured:
            result[key] = captured[key][0, -1].numpy()  # last token position
    if 'layer_out_input' in captured:
        result['residual_before'] = captured['layer_out_input'][0, -1].numpy()

    return result


def inject_direction_at_layer(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    """Inject direction at a specific layer and return logits."""
    layers_list = get_layers(model)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified

    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    return logits


def ablate_component_at_layer(model, tokenizer, device, prompt, layer_idx, component, direction=None, alpha=0):
    """
    Ablate or replace a component (attention or MLP) at a specific layer.
    
    component: 'attn', 'mlp', or 'both'
    If direction is provided, inject it after ablation.
    """
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]

    hooks = []

    def make_ablation_hook(comp_name):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Zero out the component output
            zeroed = torch.zeros_like(hidden)
            if isinstance(output, tuple):
                return (zeroed,) + output[1:]
            return zeroed
        return hook_fn

    if component in ('attn', 'both') and hasattr(layer, 'self_attn'):
        hooks.append(layer.self_attn.register_forward_hook(make_ablation_hook('attn')))
    if component in ('mlp', 'both') and hasattr(layer, 'mlp'):
        hooks.append(layer.mlp.register_forward_hook(make_ablation_hook('mlp')))

    # Also inject direction if provided
    inject_hook = None
    if direction is not None:
        def inject_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
            hidden_modified = hidden.clone()
            hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_modified,) + output[1:]
            return hidden_modified
        inject_hook = layers_list[layer_idx].register_forward_hook(inject_fn)
        hooks.append(inject_hook)

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return logits


def replace_component_at_layer(model, tokenizer, device, prompt, layer_idx, 
                                component, replacement_output):
    """
    Replace a component's output at a specific layer with a replacement tensor.
    
    component: 'attn' or 'mlp'
    replacement_output: numpy array [d_model] or None (to zero it)
    """
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]

    if replacement_output is not None:
        repl_tensor = torch.tensor(replacement_output, dtype=torch.bfloat16, device=device)
    else:
        repl_tensor = None

    def make_replace_hook():
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if repl_tensor is not None:
                # Replace the last token position with the replacement
                replaced = hidden.clone()
                replaced[0, -1, :] = repl_tensor.to(hidden.dtype)
            else:
                replaced = torch.zeros_like(hidden)
            if isinstance(output, tuple):
                return (replaced,) + output[1:]
            return replaced
        return hook_fn

    target = layer.self_attn if component == 'attn' else layer.mlp
    hook = target.register_forward_hook(make_replace_hook())
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    return logits


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def extract_direction_at_layer(model, tokenizer, device, sent_pos, sent_neg, layer_idx):
    """Extract attribute direction at a specific layer."""
    h_pos = extract_component_outputs(model, tokenizer, device, sent_pos, layer_idx)
    h_neg = extract_component_outputs(model, tokenizer, device, sent_neg, layer_idx)

    result = {}
    for key in ['layer_out']:
        if key in h_pos and key in h_neg:
            d = h_pos[key] - h_neg[key]
            norm = np.linalg.norm(d)
            result['direction'] = d / norm if norm > 1e-10 else d * 0
            result['norm'] = norm
            result['h_pos'] = h_pos[key]
            result['h_neg'] = h_neg[key]

    # Also extract component-specific deltas
    for comp in ['attn_out', 'mlp_out']:
        if comp in h_pos and comp in h_neg:
            d = h_pos[comp] - h_neg[comp]
            norm = np.linalg.norm(d)
            result[f'{comp}_delta'] = d
            result[f'{comp}_delta_norm'] = norm
            result[f'{comp}_pos'] = h_pos[comp]
            result[f'{comp}_neg'] = h_neg[comp]

    if 'residual_before' in h_pos and 'residual_before' in h_neg:
        result['residual_pos'] = h_pos['residual_before']
        result['residual_neg'] = h_neg['residual_before']

    return result


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase323_{model_name}.log")

    log(f"=== Phase 323: Path Component Decomposition for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}, mlp_type={info.mlp_type}")

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    # Determine optimal layers
    if model_name == "glm4":
        opt_layers = [1, 2, 3, 4, 5, 6, 7, 8]
        opt_attr_layer = 3
    elif model_name == "qwen3":
        opt_layers = [0, 1, 2, 3, 4, 5, 6]
        opt_attr_layer = 0
    else:  # deepseek7b
        opt_layers = [3, 4, 5, 6, 7, 8, 9]
        opt_attr_layer = 6

    alpha = 2.0
    results = {}

    # ===================================================================
    # Test 1: Component Output Analysis
    # Extract attn_out and mlp_out for positive vs negative sentences
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Component Output Analysis at Optimal Layer")
    log("="*60)

    component_analysis = []

    for pair_idx, (noun, attr_val, cluster_type) in enumerate(ATTRIBUTE_PAIRS[:6]):
        sent_pos = f"the {noun} is {attr_val}"
        sent_neg = f"the {noun} is just an object"

        for li in opt_layers[:5]:  # Test at first 5 optimal layers
            log(f"  Pair {pair_idx}: {noun}→{attr_val} at L{li}...")
            ext = extract_direction_at_layer(model, tokenizer, device, sent_pos, sent_neg, li)

            if 'direction' not in ext:
                log(f"    Skipping - no direction extracted")
                continue

            d_attr = ext['direction']
            d_attr_norm = ext.get('norm', 0)

            # Readout alignment
            attr_ids = tokenizer.encode(attr_val, add_special_tokens=False)
            cos_wu = 0
            if attr_ids:
                w_tgt = W_U[attr_ids[0]]
                w_norm = np.linalg.norm(w_tgt)
                cos_wu = float(np.dot(d_attr, w_tgt / w_norm)) if w_norm > 0 else 0

            # Component delta norms
            attn_delta_norm = ext.get('attn_out_delta_norm', 0)
            mlp_delta_norm = ext.get('mlp_out_delta_norm', 0)

            # Component delta alignment with direction
            cos_attn_dir = 0
            cos_mlp_dir = 0
            if attn_delta_norm > 1e-10:
                cos_attn_dir = float(np.dot(ext['attn_out_delta'] / attn_delta_norm, d_attr))
            if mlp_delta_norm > 1e-10:
                cos_mlp_dir = float(np.dot(ext['mlp_out_delta'] / mlp_delta_norm, d_attr))

            # Component contribution ratio
            total_delta_norm = d_attr_norm
            attn_ratio = attn_delta_norm / total_delta_norm if total_delta_norm > 1e-10 else 0
            mlp_ratio = mlp_delta_norm / total_delta_norm if total_delta_norm > 1e-10 else 0

            # Component delta alignment with W_U
            cos_attn_wu = 0
            cos_mlp_wu = 0
            if attr_ids and attn_delta_norm > 1e-10:
                cos_attn_wu = float(np.dot(ext['attn_out_delta'] / attn_delta_norm, w_tgt / w_norm))
            if attr_ids and mlp_delta_norm > 1e-10:
                cos_mlp_wu = float(np.dot(ext['mlp_out_delta'] / mlp_delta_norm, w_tgt / w_norm))

            entry = {
                "pair": f"{noun}→{attr_val}",
                "layer": li,
                "direction_norm": round(d_attr_norm, 4),
                "cos_dir_wu": round(cos_wu, 4),
                "attn_delta_norm": round(attn_delta_norm, 4),
                "mlp_delta_norm": round(mlp_delta_norm, 4),
                "attn_ratio": round(attn_ratio, 4),
                "mlp_ratio": round(mlp_ratio, 4),
                "cos_attn_dir": round(cos_attn_dir, 4),
                "cos_mlp_dir": round(cos_mlp_dir, 4),
                "cos_attn_wu": round(cos_attn_wu, 4),
                "cos_mlp_wu": round(cos_mlp_wu, 4),
            }
            component_analysis.append(entry)

            if pair_idx == 0:
                log(f"    L{li}: dir_norm={d_attr_norm:.3f}, cos_wu={cos_wu:.4f}, "
                    f"attn_norm={attn_delta_norm:.3f}({attn_ratio:.2f}), mlp_norm={mlp_delta_norm:.3f}({mlp_ratio:.2f}), "
                    f"cos_attn_wu={cos_attn_wu:.4f}, cos_mlp_wu={cos_mlp_wu:.4f}")

        torch.cuda.empty_cache()

    results["component_analysis"] = component_analysis

    # Summary by layer
    log("\n  Component Analysis Summary (mean across pairs):")
    sum_keys = ['attn_delta_norm', 'mlp_delta_norm', 'attn_ratio', 'mlp_ratio', 'cos_attn_wu', 'cos_mlp_wu']
    layer_sums = defaultdict(lambda: defaultdict(list))
    for r in component_analysis:
        li = r["layer"]
        for key in sum_keys:
            if key in r:
                layer_sums[li][key].append(r[key])

    for li in sorted(layer_sums.keys()):
        d = layer_sums[li]
        log(f"    L{li}: attn_norm={np.mean(d.get('attn_delta_norm', [0])):.3f}, mlp_norm={np.mean(d.get('mlp_delta_norm', [0])):.3f}, "
            f"attn_ratio={np.mean(d.get('attn_ratio', [0])):.3f}, mlp_ratio={np.mean(d.get('mlp_ratio', [0])):.3f}, "
            f"cos_attn_wu={np.mean(d.get('cos_attn_wu', [0])):.4f}, cos_mlp_wu={np.mean(d.get('cos_mlp_wu', [0])):.4f}")

    # ===================================================================
    # Test 2: Component Ablation
    # At optimal layer, ablate attention or MLP and measure effect
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Component Ablation at Optimal Layer")
    log("="*60)

    ablation_results = []

    for pair_idx, (noun, attr_val, cluster_type) in enumerate(ATTRIBUTE_PAIRS[:6]):
        sent_pos = f"the {noun} is {attr_val}"
        sent_neg = f"the {noun} is just an object"
        target_prompt = f"The {noun} is"

        # Get target token ID
        attr_ids = tokenizer.encode(attr_val, add_special_tokens=False)
        if not attr_ids:
            continue
        tgt_id = attr_ids[0]

        # Get cluster token IDs
        cluster_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS.get(cluster_type, []))
        all_clusters = {}
        for cname, cwords in WORD_CLUSTERS.items():
            cids = get_cluster_token_ids(tokenizer, cwords)
            if cids:
                all_clusters[cname] = cids

        for li in [opt_attr_layer]:
            log(f"  Ablation: {noun}→{attr_val} at L{li}...")

            # Extract attribute direction
            ext = extract_direction_at_layer(model, tokenizer, device, sent_pos, sent_neg, li)
            if 'direction' not in ext:
                continue
            d_attr = ext['direction']

            # Baseline logits
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit_tgt = float(baseline_logits[tgt_id])

            # 1. Full injection (no ablation)
            inj_logits = inject_direction_at_layer(model, tokenizer, device, target_prompt, d_attr, li, alpha)
            full_delta = float(inj_logits[tgt_id] - baseline_logit_tgt)

            # 2. Ablate attention + inject direction
            ablate_attn_logits = ablate_component_at_layer(
                model, tokenizer, device, target_prompt, li, 'attn', d_attr, alpha)
            ablate_attn_delta = float(ablate_attn_logits[tgt_id] - baseline_logit_tgt)

            # 3. Ablate MLP + inject direction
            ablate_mlp_logits = ablate_component_at_layer(
                model, tokenizer, device, target_prompt, li, 'mlp', d_attr, alpha)
            ablate_mlp_delta = float(ablate_mlp_logits[tgt_id] - baseline_logit_tgt)

            # 4. Ablate both components + inject direction (should be near zero)
            ablate_both_logits = ablate_component_at_layer(
                model, tokenizer, device, target_prompt, li, 'both', d_attr, alpha)
            ablate_both_delta = float(ablate_both_logits[tgt_id] - baseline_logit_tgt)

            # Cluster readout
            cluster_deltas = {}
            for cname, cids in all_clusters.items():
                base_stats = compute_cluster_logit_stats(baseline_logits, cids)
                full_stats = compute_cluster_logit_stats(inj_logits, cids)
                attn_abl_stats = compute_cluster_logit_stats(ablate_attn_logits, cids)
                mlp_abl_stats = compute_cluster_logit_stats(ablate_mlp_logits, cids)
                cluster_deltas[cname] = {
                    "full_mean_delta": round(full_stats['mean'] - base_stats['mean'], 4),
                    "attn_abl_mean_delta": round(attn_abl_stats['mean'] - base_stats['mean'], 4),
                    "mlp_abl_mean_delta": round(mlp_abl_stats['mean'] - base_stats['mean'], 4),
                }

            entry = {
                "pair": f"{noun}→{attr_val}",
                "layer": li,
                "full_delta": round(full_delta, 4),
                "attn_abl_delta": round(ablate_attn_delta, 4),
                "mlp_abl_delta": round(ablate_mlp_delta, 4),
                "both_abl_delta": round(ablate_both_delta, 4),
                "attn_importance": round(full_delta - ablate_attn_delta, 4),
                "mlp_importance": round(full_delta - ablate_mlp_delta, 4),
                "cluster_type": cluster_type,
                "cluster_deltas": cluster_deltas,
            }
            ablation_results.append(entry)

            if pair_idx < 3:
                log(f"    full={full_delta:.4f}, attn_abl={ablate_attn_delta:.4f}, "
                    f"mlp_abl={ablate_mlp_delta:.4f}, both_abl={ablate_both_delta:.4f}")
                log(f"    attn_importance={full_delta - ablate_attn_delta:.4f}, "
                    f"mlp_importance={full_delta - ablate_mlp_delta:.4f}")

        torch.cuda.empty_cache()

    results["ablation"] = ablation_results

    # Summary
    log("\n  Ablation Summary (mean across pairs):")
    if ablation_results:
        full_mean = np.mean([r["full_delta"] for r in ablation_results])
        attn_imp_mean = np.mean([r["attn_importance"] for r in ablation_results])
        mlp_imp_mean = np.mean([r["mlp_importance"] for r in ablation_results])
        log(f"    full_delta_mean={full_mean:.4f}")
        log(f"    attn_importance_mean={attn_imp_mean:.4f}")
        log(f"    mlp_importance_mean={mlp_imp_mean:.4f}")

        if abs(attn_imp_mean) > abs(mlp_imp_mean):
            log(f"    → ATTENTION is more important for causal efficacy")
        else:
            log(f"    → MLP is more important for causal efficacy")

    # ===================================================================
    # Test 3: Component Replacement (Transfer attribute signal via component)
    # ===================================================================
    log("\n" + "="*60)
    log("Test 3: Component Replacement (Replace attn or MLP output)")
    log("="*60)

    replacement_results = []

    for pair_idx, (src_noun, src_val, tgt_noun, tgt_val, cluster_type) in enumerate(TRANSFER_PAIRS[:4]):
        sent_src_pos = f"the {src_noun} is {src_val}"
        sent_src_neg = f"the {src_noun} is just an object"
        target_prompt = f"The {tgt_noun} is"

        tgt_ids = tokenizer.encode(tgt_val, add_special_tokens=False)
        if not tgt_ids:
            continue
        tgt_id = tgt_ids[0]

        for li in [opt_attr_layer]:
            log(f"  Replacement: {src_noun}→{tgt_noun} ({src_val}/{tgt_val}) at L{li}...")

            # Extract component outputs for positive and negative source sentences
            src_pos_comp = extract_component_outputs(model, tokenizer, device, sent_src_pos, li)
            src_neg_comp = extract_component_outputs(model, tokenizer, device, sent_src_neg, li)

            # Baseline logits for target prompt
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit_tgt = float(baseline_logits[tgt_id])

            # 1. Replace attention output (positive - negative)
            if 'attn_out' in src_pos_comp and 'attn_out' in src_neg_comp:
                attn_delta = src_pos_comp['attn_out'] - src_neg_comp['attn_out']
                # Inject attention delta into target prompt
                try:
                    repl_logits = inject_direction_at_layer(
                        model, tokenizer, device, target_prompt, attn_delta / max(np.linalg.norm(attn_delta), 1e-10), li, alpha)
                    attn_repl_delta = float(repl_logits[tgt_id] - baseline_logit_tgt)
                except Exception as e:
                    log(f"    attn inject failed: {e}")
                    attn_repl_delta = 0
            else:
                attn_repl_delta = 0

            # 2. Replace MLP output (positive - negative)
            if 'mlp_out' in src_pos_comp and 'mlp_out' in src_neg_comp:
                mlp_delta = src_pos_comp['mlp_out'] - src_neg_comp['mlp_out']
                try:
                    repl_logits = inject_direction_at_layer(
                        model, tokenizer, device, target_prompt, mlp_delta / max(np.linalg.norm(mlp_delta), 1e-10), li, alpha)
                    mlp_repl_delta = float(repl_logits[tgt_id] - baseline_logit_tgt)
                except Exception as e:
                    log(f"    mlp inject failed: {e}")
                    mlp_repl_delta = 0
            else:
                mlp_repl_delta = 0

            # 3. Full direction injection for comparison
            ext = extract_direction_at_layer(model, tokenizer, device, sent_src_pos, sent_src_neg, li)
            if 'direction' in ext:
                full_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, ext['direction'], li, alpha)
                full_delta = float(full_logits[tgt_id] - baseline_logit_tgt)
            else:
                full_delta = 0

            entry = {
                "pair": f"{src_noun}→{tgt_noun}",
                "layer": li,
                "full_dir_delta": round(full_delta, 4),
                "attn_delta_inject": round(attn_repl_delta, 4),
                "mlp_delta_inject": round(mlp_repl_delta, 4),
            }
            replacement_results.append(entry)

            log(f"    full_dir={full_delta:.4f}, attn_inject={attn_repl_delta:.4f}, mlp_inject={mlp_repl_delta:.4f}")

        torch.cuda.empty_cache()

    results["replacement"] = replacement_results

    # ===================================================================
    # Test 4: Cluster Readout Analysis
    # Measure logit changes on target cluster, competitor cluster, and random cluster
    # ===================================================================
    log("\n" + "="*60)
    log("Test 4: Cluster Readout Analysis")
    log("="*60)

    cluster_readout = []

    for pair_idx, (noun, attr_val, cluster_type) in enumerate(ATTRIBUTE_PAIRS[:6]):
        sent_pos = f"the {noun} is {attr_val}"
        sent_neg = f"the {noun} is just an object"
        target_prompt = f"The {noun} is"

        # Identify clusters
        target_cluster = cluster_type
        # Find a competitor cluster (different type)
        competitor_clusters = [c for c in WORD_CLUSTERS if c != target_cluster and c not in ('object', 'action', 'negation', 'positive', 'negative')]
        competitor_cluster = competitor_clusters[0] if competitor_clusters else "object"

        target_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS.get(target_cluster, []))
        competitor_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS.get(competitor_cluster, []))
        object_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS.get('object', []))

        for li in [opt_attr_layer]:
            ext = extract_direction_at_layer(model, tokenizer, device, sent_pos, sent_neg, li)
            if 'direction' not in ext:
                continue

            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            inj_logits = inject_direction_at_layer(
                model, tokenizer, device, target_prompt, ext['direction'], li, alpha)

            # Target cluster stats
            tgt_base = compute_cluster_logit_stats(baseline_logits, target_ids)
            tgt_inj = compute_cluster_logit_stats(inj_logits, target_ids)

            # Competitor cluster stats
            comp_base = compute_cluster_logit_stats(baseline_logits, competitor_ids)
            comp_inj = compute_cluster_logit_stats(inj_logits, competitor_ids)

            # Object cluster stats
            obj_base = compute_cluster_logit_stats(baseline_logits, object_ids)
            obj_inj = compute_cluster_logit_stats(inj_logits, object_ids)

            entry = {
                "pair": f"{noun}→{attr_val}",
                "layer": li,
                "target_cluster": target_cluster,
                "competitor_cluster": competitor_cluster,
                "target_mean_delta": round(tgt_inj['mean'] - tgt_base['mean'], 4),
                "target_max_delta": round(tgt_inj['max'] - tgt_base['max'], 4),
                "competitor_mean_delta": round(comp_inj['mean'] - comp_base['mean'], 4),
                "object_mean_delta": round(obj_inj['mean'] - obj_base['mean'], 4),
                "specificity": round((tgt_inj['mean'] - tgt_base['mean']) - (comp_inj['mean'] - comp_base['mean']), 4),
            }
            cluster_readout.append(entry)

            if pair_idx < 3:
                log(f"    {noun}→{attr_val}: target_delta={tgt_inj['mean'] - tgt_base['mean']:.4f}, "
                    f"competitor_delta={comp_inj['mean'] - comp_base['mean']:.4f}, "
                    f"object_delta={obj_inj['mean'] - obj_base['mean']:.4f}, "
                    f"specificity={(tgt_inj['mean'] - tgt_base['mean']) - (comp_inj['mean'] - comp_base['mean']):.4f}")

    results["cluster_readout"] = cluster_readout

    # ===================================================================
    # Test 5: Layer-wise Jacobian-like Sensitivity
    # How much does each component change when we inject the attribute direction?
    # ===================================================================
    log("\n" + "="*60)
    log("Test 5: Component Sensitivity (How does injection affect attn/mlp?)")
    log("="*60)

    sensitivity_results = []

    for pair_idx, (noun, attr_val, cluster_type) in enumerate(ATTRIBUTE_PAIRS[:4]):
        sent_pos = f"the {noun} is {attr_val}"
        sent_neg = f"the {noun} is just an object"

        for li in opt_layers[:5]:
            ext = extract_direction_at_layer(model, tokenizer, device, sent_pos, sent_neg, li)
            if 'direction' not in ext:
                continue
            d_attr = ext['direction']

            # Get component outputs with injection
            # We need to inject the direction and then capture the next layer's component outputs
            if li + 1 >= info.n_layers:
                continue

            # Without injection
            target_prompt = f"The {noun} is"
            comp_base = extract_component_outputs(model, tokenizer, device, target_prompt, li + 1)

            # With injection at layer li
            layers_list = get_layers(model)

            def make_inject_hook(direction, layer_idx, alpha_val):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
                    hidden_modified = hidden.clone()
                    hidden_modified[0, -1, :] += (alpha_val * d_tensor).to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (hidden_modified,) + output[1:]
                    return hidden_modified
                return hook_fn

            hook = layers_list[li].register_forward_hook(make_inject_hook(d_attr, li, alpha))
            comp_inj = extract_component_outputs(model, tokenizer, device, target_prompt, li + 1)
            hook.remove()

            # Compute sensitivity
            sensitivity = {}
            for comp in ['attn_out', 'mlp_out']:
                if comp in comp_base and comp in comp_inj:
                    delta = comp_inj[comp] - comp_base[comp]
                    sensitivity[f'{comp}_sens_norm'] = round(float(np.linalg.norm(delta)), 4)
                    # Alignment with the attribute direction
                    d_norm = np.linalg.norm(delta)
                    cos_delta_dir = float(np.dot(delta / max(d_norm, 1e-10), d_attr))
                    sensitivity[f'{comp}_cos_delta_dir'] = round(cos_delta_dir, 4)
                    # Alignment with W_U
                    attr_ids = tokenizer.encode(attr_val, add_special_tokens=False)
                    if attr_ids:
                        w_tgt = W_U[attr_ids[0]]
                        w_norm = np.linalg.norm(w_tgt)
                        cos_delta_wu = float(np.dot(delta / max(d_norm, 1e-10), w_tgt / max(w_norm, 1e-10)))
                        sensitivity[f'{comp}_cos_delta_wu'] = round(cos_delta_wu, 4)

            sensitivity["pair"] = f"{noun}→{attr_val}"
            sensitivity["inject_layer"] = li
            sensitivity["read_layer"] = li + 1
            sensitivity_results.append(sensitivity)

            if pair_idx == 0:
                attn_sens = sensitivity.get('attn_out_sens_norm', 0)
                mlp_sens = sensitivity.get('mlp_out_sens_norm', 0)
                log(f"    L{li}→L{li+1}: attn_sens={attn_sens:.4f}, mlp_sens={mlp_sens:.4f}")

        torch.cuda.empty_cache()

    results["sensitivity"] = sensitivity_results

    # ===================================================================
    # Save results
    # ===================================================================
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_attr_layer": opt_attr_layer,
        "alpha": alpha,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase323.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # ===================================================================
    # Print Overall Summary
    # ===================================================================
    log("\n" + "="*60)
    log(f"PHASE 323 SUMMARY - {model_name}")
    log("="*60)

    # Component analysis summary
    if component_analysis:
        log("\n  Component Analysis:")
        layer_sums2 = defaultdict(lambda: defaultdict(list))
        for r in component_analysis:
            for key in ['attn_ratio', 'mlp_ratio', 'cos_attn_wu', 'cos_mlp_wu']:
                layer_sums2[r['layer']][key].append(r[key])
        for li in sorted(layer_sums2.keys()):
            d = layer_sums2[li]
            log(f"    L{li}: attn_ratio={np.mean(d['attn_ratio']):.3f}, mlp_ratio={np.mean(d['mlp_ratio']):.3f}, "
                f"cos_attn_wu={np.mean(d['cos_attn_wu']):.4f}, cos_mlp_wu={np.mean(d['cos_mlp_wu']):.4f}")

    # Ablation summary
    if ablation_results:
        log("\n  Ablation Results:")
        full_mean = np.mean([r["full_delta"] for r in ablation_results])
        attn_imp = np.mean([r["attn_importance"] for r in ablation_results])
        mlp_imp = np.mean([r["mlp_importance"] for r in ablation_results])
        log(f"    full_delta={full_mean:.4f}, attn_importance={attn_imp:.4f}, mlp_importance={mlp_imp:.4f}")

    # Cluster readout summary
    if cluster_readout:
        log("\n  Cluster Readout:")
        tgt_deltas = [r["target_mean_delta"] for r in cluster_readout]
        comp_deltas = [r["competitor_mean_delta"] for r in cluster_readout]
        obj_deltas = [r["object_mean_delta"] for r in cluster_readout]
        specs = [r["specificity"] for r in cluster_readout]
        log(f"    target_cluster_delta={np.mean(tgt_deltas):.4f}")
        log(f"    competitor_cluster_delta={np.mean(comp_deltas):.4f}")
        log(f"    object_cluster_delta={np.mean(obj_deltas):.4f}")
        log(f"    specificity={np.mean(specs):.4f}")

    # Cleanup
    del W_U
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released. Total time: {time.time()-t0:.1f}s")

    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
