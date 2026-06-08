"""
Phase 399: Norm Amplification Mechanism Attribution + Baseline Audit
====================================================================

Phase 398b proved: Even component ≈ norm effect (ortho/L1 ≈ 1.0)
But we still don't know:
1. WHICH component converts norm change into preference amplification?
2. WHAT is the "current preference" being amplified?

Three sub-experiments:

A. Baseline Logit Distribution Audit
   - Record full logit distribution for corrupt prompts
   - Identify top-10 tokens, target-competitor gap
   - This answers: "What preference is being amplified?"

B. RMSNorm Sign Preservation (Fixed from 398b)
   - Hook input_layernorm at layer l+1
   - Capture residual BEFORE and AFTER RMSNorm
   - Measure cos(delta_before_norm, delta_after_norm)
   - If cos ≈ 1: RMSNorm preserves sign → norm effect from elsewhere
   - If cos ≈ 0: RMSNorm destroys sign → RMSNorm is a key contributor

C. Attention vs MLP Attribution
   - Inject random orthogonal direction at layer l
   - At layers l+1..l+3, hook self_attn output, mlp output, full layer output
   - Compute delta_attn, delta_mlp per component
   - Project each onto preference direction (unembedding)
   - Identify: which component amplifies the preference?

Usage:
  python tests/glm5/phase399_norm_attribution.py qwen3
  python tests/glm5/phase399_norm_attribution.py deepseek7b
  python tests/glm5/phase399_norm_attribution.py glm4
"""
import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
    "This {obj} is {attr}.",
    "That {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
    "This item is {attr}.",
    "That item is {attr}.",
]

SIZE_DATA = {
    "size": {
        "objects": {
            "ant":     [("small","big"),("tiny","large")],
            "elephant":[("big","small"),("large","tiny")],
            "mountain":[("big","small"),("large","tiny")],
        },
    },
}

VALUE_ALIGNMENT = {
    "ant": "small", "elephant": "big", "mountain": "big",
}

# Multi-candidate tokens for baseline audit
CANDIDATE_TOKENS = {
    "size": ["tiny", "small", "medium", "large", "big", "huge", "massive", "miniature"],
}

LAYER_CONFIGS = {
    "qwen3": [4, 16],
    "deepseek7b": [4, 12],
    "glm4": [5, 15],
}

# Number of layers to track after injection
N_TRACK_LAYERS = 5

# Number of random orthogonal directions for attribution
N_RANDOM = 10  # increased from 5 for better statistics


def build_pairs():
    pairs = []
    for cat, cat_data in SIZE_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name,
                        'target': target, 'comp': comp,
                        'cat': cat, 'frame_idx': f_idx, 'value_idx': v_idx,
                    })
    return pairs


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            print(f"  Loaded with {impl}")
            break
        except Exception as e:
            print(f"  Failed with {impl}: {str(e)[:100]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
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
        print(f"  Layer allocation: {gpu_layers} GPU + {cpu_layers} CPU")
    return model, tokenizer


def get_logit_diff(logits_tensor, target_id, comp_id):
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return t_logit - c_logit, t_logit, c_logit


def make_orthogonal_directions(d, n, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    d_norm = np.linalg.norm(d)
    results = []
    for i in range(n):
        r = rng.randn(len(d)).astype(np.float32)
        r = r - (np.dot(r, d) / (np.dot(d, d) + 1e-10)) * d
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-10:
            r = r * (d_norm / r_norm)
        results.append(r)
    return results


def find_norm_module(layer):
    """Find the input RMSNorm module in a transformer layer."""
    # Try common names
    for name in ['input_layernorm', 'ln_1', 'ln_before_attn']:
        if hasattr(layer, name):
            mod = getattr(layer, name)
            return mod, name
    # Search submodules
    for name, mod in layer.named_modules():
        if 'input_layernorm' in name or 'ln_1' in name:
            return mod, name
    return None, None


def find_attn_mlp_modules(layer):
    """Find attention and MLP output modules in a transformer layer."""
    attn_mod = None
    mlp_mod = None
    
    # Attention output: typically self_attn.o_proj or self_attn.out_proj
    if hasattr(layer, 'self_attn'):
        sa = layer.self_attn
        for name in ['o_proj', 'out_proj', 'dense']:
            if hasattr(sa, name):
                attn_mod = getattr(sa, name)
                break
    
    # MLP output: typically mlp.down_proj or mlp.c_proj
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        for name in ['down_proj', 'c_proj', 'dense_4h_to_h']:
            if hasattr(mlp, name):
                mlp_mod = getattr(mlp, name)
                break
    
    return attn_mod, mlp_mod


# ==============================================================================
# Sub-Experiment A: Baseline Logit Distribution Audit
# ==============================================================================
def baseline_audit(model, tokenizer, pairs, token_ids):
    """Audit the baseline logit distribution for corrupt prompts."""
    device = next(model.parameters()).device
    results = {}
    
    # Get all candidate token IDs
    candidate_ids = {}
    for cat, tokens in CANDIDATE_TOKENS.items():
        for tok in tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            candidate_ids[tok] = ids[0] if ids else None
    
    # Per-object audit
    for obj in ["ant", "elephant", "mountain"]:
        obj_pairs = [p for p in pairs if p['obj'] == obj]
        if not obj_pairs:
            continue
        
        # Use first value pair for baseline
        p0 = obj_pairs[0]
        ctpl = CORRUPT_FRAMES[p0['frame_idx']]
        prompt = ctpl.format(attr=p0['target'])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device))
        
        logits = out.logits[0, -1].float().cpu().numpy()
        
        # Top-20 tokens
        top20_idx = np.argsort(logits)[-20:][::-1]
        top20 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top20_idx]
        
        # Candidate token logits
        cand_logits = {}
        for tok, tid in candidate_ids.items():
            if tid is not None:
                cand_logits[tok] = float(logits[tid])
        
        # Target vs competitor
        target_id = token_ids.get(p0['target'])
        comp_id = token_ids.get(p0['comp'])
        target_logit = float(logits[target_id]) if target_id else None
        comp_logit = float(logits[comp_id]) if comp_id else None
        
        results[obj] = {
            'prompt': prompt,
            'top20': top20,
            'candidate_logits': cand_logits,
            'target_token': p0['target'],
            'target_logit': target_logit,
            'comp_token': p0['comp'],
            'comp_logit': comp_logit,
            'target_comp_gap': (target_logit or 0) - (comp_logit or 0),
        }
        
        print(f"  {obj}: target={p0['target']}({target_logit:.3f}) vs comp={p0['comp']}({comp_logit:.3f}) "
              f"gap={results[obj]['target_comp_gap']:+.3f}")
        print(f"    Top5: {top20[:5]}")
        print(f"    Candidates: {dict(sorted(cand_logits.items(), key=lambda x: -x[1]))}")
    
    return results


# ==============================================================================
# Sub-Experiment B: RMSNorm Sign Preservation
# ==============================================================================
def rmsnorm_sign_test(model, tokenizer, layers_list, device, li, direction, prompt):
    """Test how much sign information is preserved through RMSNorm."""
    alpha = 1.0
    scaled = alpha * direction
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
    
    # Find the norm module at layer li+1
    next_li = min(li + 1, len(layers_list) - 1)
    if next_li == li:
        return None
    
    norm_mod, norm_name = find_norm_module(layers_list[next_li])
    if norm_mod is None:
        print(f"    L{next_li}: No norm module found")
        return None
    
    # --- Run with injection ---
    captured = {}
    def make_capture_hook(key):
        def hook_fn(module, input, output):
            # Capture input (before norm) and output (after norm)
            inp = input[0] if isinstance(input, tuple) else input
            out_val = output[0] if isinstance(output, tuple) else output
            captured[key + '_input'] = inp.detach().float().cpu()
            captured[key + '_output'] = out_val.detach().float().cpu()
        return hook_fn
    
    # Hook: inject at layer li
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    
    # Run with +d injection
    h_inject = layers_list[li].register_forward_hook(make_add_hook(delta))
    h_norm = norm_mod.register_forward_hook(make_capture_hook('plus'))
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h_inject.remove()
        h_norm.remove()
    
    # Run with -d injection
    neg_delta = torch.tensor(-scaled, dtype=torch.bfloat16, device=device)
    h_inject2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
    h_norm2 = norm_mod.register_forward_hook(make_capture_hook('minus'))
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h_inject2.remove()
        h_norm2.remove()
    
    # Run baseline (no injection)
    h_norm3 = norm_mod.register_forward_hook(make_capture_hook('base'))
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h_norm3.remove()
    
    # Compute deltas
    required_keys = ['plus_input', 'plus_output', 'minus_input', 'minus_output', 
                     'base_input', 'base_output']
    if not all(k in captured for k in required_keys):
        missing = [k for k in required_keys if k not in captured]
        print(f"    Missing keys: {missing}")
        return None
    
    # Delta before norm: (plus_input - base_input) and (minus_input - base_input)
    delta_plus_before = (captured['plus_input'][0, -1] - captured['base_input'][0, -1]).numpy()
    delta_minus_before = (captured['minus_input'][0, -1] - captured['base_input'][0, -1]).numpy()
    
    # Delta after norm: (plus_output - base_output) and (minus_output - base_output)
    delta_plus_after = (captured['plus_output'][0, -1] - captured['base_output'][0, -1]).numpy()
    delta_minus_after = (captured['minus_output'][0, -1] - captured['base_output'][0, -1]).numpy()
    
    # Measure sign preservation
    # cos(+delta_before, +delta_after) tells us if RMSNorm preserves the +d direction
    # cos(+delta_before, -delta_after) tells us if RMSNorm flips the direction
    norm_pb = np.linalg.norm(delta_plus_before)
    norm_pa = np.linalg.norm(delta_plus_after)
    norm_mb = np.linalg.norm(delta_minus_before)
    norm_ma = np.linalg.norm(delta_minus_after)
    
    if norm_pb < 1e-10 or norm_pa < 1e-10 or norm_mb < 1e-10 or norm_ma < 1e-10:
        return None
    
    cos_plus = float(np.dot(delta_plus_before, delta_plus_after) / (norm_pb * norm_pa))
    cos_minus = float(np.dot(delta_minus_before, delta_minus_after) / (norm_mb * norm_ma))
    cos_cross = float(np.dot(delta_plus_before, delta_minus_after) / (norm_pb * norm_ma))
    
    # Even/Odd decomposition of the norm effect
    # Even_before = (delta_plus_before + delta_minus_before) / 2
    # Even_after = (delta_plus_after + delta_minus_after) / 2
    even_before = (delta_plus_before + delta_minus_before) / 2
    even_after = (delta_plus_after + delta_minus_after) / 2
    odd_before = (delta_plus_before - delta_minus_before) / 2
    odd_after = (delta_plus_after - delta_minus_after) / 2
    
    # Norm amplification ratio
    norm_ratio = (norm_pa + norm_ma) / (norm_pb + norm_mb + 1e-10)
    
    return {
        'norm_module': norm_name,
        'cos_plus': cos_plus,       # sign preservation for +d
        'cos_minus': cos_minus,     # sign preservation for -d
        'cos_cross': cos_cross,     # cross-correlation
        'avg_cos_preserved': (cos_plus + cos_minus) / 2,
        'norm_ratio': norm_ratio,   # how much norm changes through RMSNorm
        'even_norm_ratio': float(np.linalg.norm(even_after) / (np.linalg.norm(even_before) + 1e-10)),
        'odd_norm_ratio': float(np.linalg.norm(odd_after) / (np.linalg.norm(odd_before) + 1e-10)),
        'delta_plus_before_norm': float(norm_pb),
        'delta_plus_after_norm': float(norm_pa),
    }


# ==============================================================================
# Sub-Experiment C: Attention vs MLP Attribution
# ==============================================================================
def component_attribution(model, tokenizer, layers_list, device, li, direction, 
                          prompt, tid, cid, W_U_np, n_track=N_TRACK_LAYERS):
    """
    Decompose the injection effect into attention and MLP contributions.
    
    At each layer l+1..l+K:
    - Hook attention output (o_proj output)
    - Hook MLP output (down_proj output)
    - Hook full layer output
    
    Compare with/without injection to get delta_attn, delta_mlp, delta_layer.
    """
    alpha = 1.0
    scaled = alpha * direction
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
    
    # Determine which layers to track
    max_li = len(layers_list) - 1
    track_layers = [min(li + k, max_li) for k in range(1, n_track + 1)]
    track_layers = sorted(set(track_layers))
    
    # Compute preference direction from unembedding
    # W_U_np shape: [vocab_size, d_model], so W_U_np[tid] is the embedding of token tid
    if tid is not None and cid is not None:
        pref_dir = (W_U_np[tid] - W_U_np[cid])
        pref_norm = np.linalg.norm(pref_dir)
        if pref_norm > 1e-10:
            pref_dir = pref_dir / pref_norm
    else:
        pref_dir = None
    
    results = {}
    
    for tl in track_layers:
        layer = layers_list[tl]
        attn_mod, mlp_mod = find_attn_mlp_modules(layer)
        
        captured_with = {}
        captured_without = {}
        
        def make_capture(key, store):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    store[key] = output[0].detach().float().cpu()
                else:
                    store[key] = output.detach().float().cpu()
            return hook_fn
        
        # Hooks for WITH injection
        hooks_with = []
        hooks_with.append(layer.register_forward_hook(make_capture('layer_out', captured_with)))
        if attn_mod is not None:
            hooks_with.append(attn_mod.register_forward_hook(make_capture('attn_out', captured_with)))
        if mlp_mod is not None:
            hooks_with.append(mlp_mod.register_forward_hook(make_capture('mlp_out', captured_with)))
        
        # Inject at layer li
        def make_add_hook(dv):
            def hook_fn(module, input, output):
                hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                hs[0, -1, :] += dv
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            return hook_fn
        
        h_inject = layers_list[li].register_forward_hook(make_add_hook(delta))
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            logit_diff_with, t_with, c_with = get_logit_diff(out.logits[0, -1], tid, cid)
        finally:
            h_inject.remove()
            for h in hooks_with:
                h.remove()
        
        # Hooks for WITHOUT injection (baseline)
        hooks_without = []
        hooks_without.append(layer.register_forward_hook(make_capture('layer_out', captured_without)))
        if attn_mod is not None:
            hooks_without.append(attn_mod.register_forward_hook(make_capture('attn_out', captured_without)))
        if mlp_mod is not None:
            hooks_without.append(mlp_mod.register_forward_hook(make_capture('mlp_out', captured_without)))
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            logit_diff_without, t_without, c_without = get_logit_diff(out.logits[0, -1], tid, cid)
        finally:
            for h in hooks_without:
                h.remove()
        
        # Compute deltas
        layer_delta = None
        attn_delta = None
        mlp_delta = None
        
        if 'layer_out' in captured_with and 'layer_out' in captured_without:
            layer_delta = (captured_with['layer_out'][0, -1] - captured_without['layer_out'][0, -1]).numpy()
        
        if 'attn_out' in captured_with and 'attn_out' in captured_without:
            attn_delta = (captured_with['attn_out'][0, -1] - captured_without['attn_out'][0, -1]).numpy()
        
        if 'mlp_out' in captured_with and 'mlp_out' in captured_without:
            mlp_delta = (captured_with['mlp_out'][0, -1] - captured_without['mlp_out'][0, -1]).numpy()
        
        # Projection onto preference direction
        pref_proj_layer = float(np.dot(layer_delta, pref_dir)) if (layer_delta is not None and pref_dir is not None) else None
        pref_proj_attn = float(np.dot(attn_delta, pref_dir)) if (attn_delta is not None and pref_dir is not None) else None
        pref_proj_mlp = float(np.dot(mlp_delta, pref_dir)) if (mlp_delta is not None and pref_dir is not None) else None
        
        # Norms
        layer_norm = float(np.linalg.norm(layer_delta)) if layer_delta is not None else None
        attn_norm = float(np.linalg.norm(attn_delta)) if attn_delta is not None else None
        mlp_norm = float(np.linalg.norm(mlp_delta)) if mlp_delta is not None else None
        
        # Logit change
        delta_diff = logit_diff_with - logit_diff_without
        delta_t = t_with - t_without
        delta_c = c_with - c_without
        
        result = {
            'layer_delta_norm': layer_norm,
            'attn_delta_norm': attn_norm,
            'mlp_delta_norm': mlp_norm,
            'pref_proj_layer': pref_proj_layer,
            'pref_proj_attn': pref_proj_attn,
            'pref_proj_mlp': pref_proj_mlp,
            'delta_diff': float(delta_diff),
            'delta_t': float(delta_t),
            'delta_c': float(delta_c),
            'attn_found': attn_mod is not None,
            'mlp_found': mlp_mod is not None,
        }
        
        results[str(tl)] = result
        
        print(f"    L{tl}: layer_norm={layer_norm:.4f} attn_norm={attn_norm:.4f} mlp_norm={mlp_norm:.4f} "
              f"pref_layer={pref_proj_layer:+.4f} pref_attn={pref_proj_attn:+.4f} pref_mlp={pref_proj_mlp:+.4f} "
              f"Δdiff={delta_diff:+.4f}")
    
    return results


def run_phase399(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 399: Norm Amplification Attribution ({model_name}) [{timestamp}] ===")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    pairs = build_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs")
    print(f"  Layers: {layer_indices}")
    print(f"  Track layers after injection: {N_TRACK_LAYERS}")
    print(f"  Random orthogonal dirs: {N_RANDOM}")
    
    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # Get unembedding matrix
    W_U = get_W_U(model, model_name)
    W_U_np = W_U.astype(np.float32)
    
    # Resolve token IDs
    token_ids = {}
    for cat_data in SIZE_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None
    
    # Also resolve candidate tokens
    for cat, tokens in CANDIDATE_TOKENS.items():
        for tok in tokens:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    all_results = {
        'model': model_name, 'timestamp': timestamp,
        'layer_indices': layer_indices,
        'baseline_audit': {},
        'per_layer': {},
    }
    
    # ========================================================================
    # Sub-Experiment A: Baseline Logit Distribution Audit
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"=== A. Baseline Logit Distribution Audit ===")
    baseline = baseline_audit(model, tokenizer, pairs, token_ids)
    all_results['baseline_audit'] = baseline
    
    # ========================================================================
    # Per-layer experiments
    # ========================================================================
    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        # Collect activations for computing directions
        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn
        
        handle = layers_list[li].register_forward_hook(make_hook('h'))
        
        h_correct = np.zeros((N, d_model), dtype=np.float32)
        h_correct_corrupt = np.zeros((N, d_model), dtype=np.float32)
        baseline_diffs = np.zeros(N, dtype=np.float32)
        baseline_t = np.zeros(N, dtype=np.float32)
        baseline_c = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            p = pairs[i]
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])
            
            tpl = FRAMES[p['frame_idx']]
            ctpl = CORRUPT_FRAMES[p['frame_idx']]
            correct_clean = tpl.format(obj=p['obj'], attr=p['target'])
            correct_corrupt = ctpl.format(attr=p['target'])
            
            captured.clear()
            inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_correct[i] = captured['h'][0, -1].numpy()
            
            captured.clear()
            inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_correct_corrupt[i] = captured['h'][0, -1].numpy()
            diff, tl, cl = get_logit_diff(out.logits[0, -1], tid, cid)
            baseline_diffs[i] = diff
            baseline_t[i] = tl
            baseline_c[i] = cl
        
        handle.remove()
        
        # Compute delta_h
        dh_correct = h_correct - h_correct_corrupt
        
        # Per-object directions
        obj_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            obj_groups[p['obj']].append(i)
        
        per_obj_dirs = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs[obj] = np.mean(dh_correct[indices], axis=0)
        
        layer_result = {}
        
        for obj in ["ant", "elephant", "mountain"]:
            obj_indices = obj_groups.get(obj, [])
            if len(obj_indices) == 0:
                continue
            
            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])
            val_align = VALUE_ALIGNMENT.get(obj, "?")
            dir_l1 = per_obj_dirs[obj]
            
            print(f"\n  --- {obj} (align={val_align}) ---")
            
            # Generate random orthogonal directions
            ortho_dirs = make_orthogonal_directions(dir_l1, N_RANDOM)
            
            # Use first prompt as representative
            ctpl = CORRUPT_FRAMES[p0['frame_idx']]
            prompt = ctpl.format(attr=p0['target'])
            
            obj_result = {
                'value_align': val_align,
                'prompt': prompt,
            }
            
            # ================================================================
            # Sub-Experiment B: RMSNorm Sign Preservation
            # ================================================================
            print(f"\n  === B. RMSNorm Sign Preservation ({obj}) ===")
            
            # Test with L1 direction
            rms_l1 = rmsnorm_sign_test(model, tokenizer, layers_list, device, li, dir_l1, prompt)
            if rms_l1:
                print(f"    L1 dir: cos_avg={rms_l1['avg_cos_preserved']:.4f} "
                      f"norm_ratio={rms_l1['norm_ratio']:.4f} "
                      f"even_ratio={rms_l1['even_norm_ratio']:.4f} "
                      f"odd_ratio={rms_l1['odd_norm_ratio']:.4f} "
                      f"module={rms_l1['norm_module']}")
            else:
                print(f"    L1 dir: RMSNorm test failed")
            
            # Test with random orthogonal direction
            rms_ortho = rmsnorm_sign_test(model, tokenizer, layers_list, device, li, ortho_dirs[0], prompt)
            if rms_ortho:
                print(f"    Ortho: cos_avg={rms_ortho['avg_cos_preserved']:.4f} "
                      f"norm_ratio={rms_ortho['norm_ratio']:.4f} "
                      f"module={rms_ortho['norm_module']}")
            else:
                print(f"    Ortho: RMSNorm test failed")
            
            obj_result['rmsnorm_l1'] = rms_l1
            obj_result['rmsnorm_ortho'] = rms_ortho
            
            # ================================================================
            # Sub-Experiment C: Component Attribution
            # ================================================================
            print(f"\n  === C. Attention vs MLP Attribution ({obj}) ===")
            
            # Test with random orthogonal direction (since Phase 398b showed ortho ≈ L1)
            comp_result = component_attribution(
                model, tokenizer, layers_list, device, li,
                ortho_dirs[0], prompt, tid, cid, W_U_np, N_TRACK_LAYERS)
            
            obj_result['component_attribution'] = comp_result
            
            # Also test with L1 direction for comparison
            print(f"\n  === C2. L1 Direction Attribution ({obj}) ===")
            comp_l1 = component_attribution(
                model, tokenizer, layers_list, device, li,
                dir_l1, prompt, tid, cid, W_U_np, N_TRACK_LAYERS)
            
            obj_result['component_attribution_l1'] = comp_l1
            
            layer_result[obj] = obj_result
        
        # Layer summary
        print(f"\n  === Layer {li} Summary ===")
        print(f"  {'Object':10s} {'RMSNorm cos':>12s} {'norm_ratio':>11s} "
              f"{'Attn pref':>10s} {'MLP pref':>10s} {'Δdiff':>8s}")
        
        for obj in ["ant", "elephant", "mountain"]:
            r = layer_result.get(obj, {})
            if not r:
                continue
            
            rms = r.get('rmsnorm_l1') or {}
            cos = rms.get('avg_cos_preserved', 'N/A')
            nr = rms.get('norm_ratio', 'N/A')
            
            comp = r.get('component_attribution', {})
            # Get the first tracked layer's results
            first_track = list(comp.keys())[0] if comp else None
            attn_pref = comp.get(first_track, {}).get('pref_proj_attn', 'N/A') if first_track else 'N/A'
            mlp_pref = comp.get(first_track, {}).get('pref_proj_mlp', 'N/A') if first_track else 'N/A'
            dd = comp.get(first_track, {}).get('delta_diff', 'N/A') if first_track else 'N/A'
            
            cos_str = f"{cos:.4f}" if isinstance(cos, float) else cos
            nr_str = f"{nr:.4f}" if isinstance(nr, float) else nr
            ap_str = f"{attn_pref:+.4f}" if isinstance(attn_pref, float) else attn_pref
            mp_str = f"{mlp_pref:+.4f}" if isinstance(mlp_pref, float) else mlp_pref
            dd_str = f"{dd:+.4f}" if isinstance(dd, float) else dd
            
            print(f"  {obj:10s} {cos_str:>12s} {nr_str:>11s} {ap_str:>10s} {mp_str:>10s} {dd_str:>8s}")
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")
    
    # Save results
    out_dir = ROOT / "results" / "phase399_norm_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase399.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    print(f"\nBaseline Logit Gaps:")
    for obj, b in all_results['baseline_audit'].items():
        print(f"  {obj}: target={b['target_token']}({b['target_logit']:.3f}) "
              f"comp={b['comp_token']}({b['comp_logit']:.3f}) "
              f"gap={b['target_comp_gap']:+.3f}")
    
    print(f"\nRMSNorm Sign Preservation:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            rms = r.get('rmsnorm_l1')
            if rms:
                print(f"  L{li} {obj}: cos={rms['avg_cos_preserved']:.4f} "
                      f"norm_ratio={rms['norm_ratio']:.4f} "
                      f"even/odd={rms['even_norm_ratio']:.3f}/{rms['odd_norm_ratio']:.3f}")
            else:
                print(f"  L{li} {obj}: RMSNorm test failed")
    
    print(f"\nComponent Attribution (first tracked layer):")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            comp = r.get('component_attribution', {})
            first_track = list(comp.keys())[0] if comp else None
            if first_track:
                c = comp[first_track]
                print(f"  L{li}→L{first_track} {obj}: "
                      f"attn_norm={c['attn_delta_norm']:.4f} mlp_norm={c['mlp_delta_norm']:.4f} "
                      f"pref_attn={c['pref_proj_attn']:+.4f} pref_mlp={c['pref_proj_mlp']:+.4f} "
                      f"Δdiff={c['delta_diff']:+.4f}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase399(model_name)
