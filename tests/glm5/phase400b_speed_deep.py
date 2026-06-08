"""
Phase 400b: Speed Category Deep Analysis (Round 2)
===================================================

Phase 400 discovered that speed is the ONLY category where direction > norm
(MIXED/ATTRACTOR_DOM instead of NORM_DOM).

This Round 2 digs deeper into speed to understand:
1. What is the semantic direction for speed?
2. Does the speed direction generalize across objects?
3. Is the speed direction interpretable (e.g., fast-slow axis in W_U)?
4. How does speed compare to size at the component level (RMSNorm, MLP)?

Additional categories for comparison: temperature (which showed strong norm effects)

Key tests:
A. Speed direction generalization: train direction on cheetah, test on turtle/rocket
B. Speed direction interpretability: project onto W_U, check fast/slow loading
C. Speed vs Size component comparison: RMSNorm sign preservation, MLP attribution
D. Odd component analysis: for speed, Odd should be larger than for size

Usage:
  python tests/glm5/phase400b_speed_deep.py qwen3
  python tests/glm5/phase400b_speed_deep.py deepseek7b
  python tests/glm5/phase400b_speed_deep.py glm4
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

# Speed and Size data for comparison
CATEGORY_DATA = {
    "speed": {
        "objects": {
            "cheetah": {"target": "fast", "comp": "slow", "align": "fast"},
            "turtle":  {"target": "slow", "comp": "fast", "align": "slow"},
            "rocket":  {"target": "fast", "comp": "slow", "align": "fast"},
        },
        "candidates": ["fast", "slow", "rapid", "sluggish", "quick", "swift", "leisurely", "speedy"],
    },
    "size": {
        "objects": {
            "ant":     {"target": "small", "comp": "big", "align": "small"},
            "elephant":{"target": "big",   "comp": "small", "align": "big"},
            "mountain":{"target": "big",   "comp": "small", "align": "big"},
        },
        "candidates": ["tiny", "small", "medium", "large", "big", "huge", "massive", "miniature"],
    },
    "temperature": {
        "objects": {
            "fire":    {"target": "hot",  "comp": "cold", "align": "hot"},
            "ice":     {"target": "cold", "comp": "hot",  "align": "cold"},
            "volcano": {"target": "hot",  "comp": "cold", "align": "hot"},
        },
        "candidates": ["hot", "cold", "warm", "cool", "freezing", "boiling", "lukewarm", "frigid"],
    },
}

LAYER_CONFIGS = {
    "qwen3": [4, 16, 28],
    "deepseek7b": [4, 12, 20],
    "glm4": [5, 15, 25, 35],
}

N_RANDOM = 30  # More directions for better statistics


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
    for name in ['input_layernorm', 'ln_1', 'ln_before_attn']:
        if hasattr(layer, name):
            return getattr(layer, name), name
    for name, mod in layer.named_modules():
        if 'input_layernorm' in name or 'ln_1' in name:
            return mod, name
    return None, None


def rmsnorm_sign_test(model, tokenizer, layers_list, device, li, direction, prompt):
    """Test RMSNorm sign preservation."""
    alpha = 1.0
    scaled = alpha * direction
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
    
    next_li = min(li + 1, len(layers_list) - 1)
    if next_li == li:
        return None
    
    norm_mod, norm_name = find_norm_module(layers_list[next_li])
    if norm_mod is None:
        return None
    
    captured = {}
    def make_capture_hook(key):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out_val = output[0] if isinstance(output, tuple) else output
            captured[key + '_input'] = inp.detach().float().cpu()
            captured[key + '_output'] = out_val.detach().float().cpu()
        return hook_fn
    
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    
    # Run with +d
    h1 = layers_list[li].register_forward_hook(make_add_hook(delta))
    h2 = norm_mod.register_forward_hook(make_capture_hook('plus'))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h1.remove(); h2.remove()
    
    # Run with -d
    neg_delta = torch.tensor(-scaled, dtype=torch.bfloat16, device=device)
    h3 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
    h4 = norm_mod.register_forward_hook(make_capture_hook('minus'))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h3.remove(); h4.remove()
    
    # Baseline
    h5 = norm_mod.register_forward_hook(make_capture_hook('base'))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h5.remove()
    
    required = ['plus_input', 'plus_output', 'minus_input', 'minus_output', 'base_input', 'base_output']
    if not all(k in captured for k in required):
        return None
    
    delta_plus_before = (captured['plus_input'][0, -1] - captured['base_input'][0, -1]).numpy()
    delta_minus_before = (captured['minus_input'][0, -1] - captured['base_input'][0, -1]).numpy()
    delta_plus_after = (captured['plus_output'][0, -1] - captured['base_output'][0, -1]).numpy()
    delta_minus_after = (captured['minus_output'][0, -1] - captured['base_output'][0, -1]).numpy()
    
    norm_pb = np.linalg.norm(delta_plus_before)
    norm_pa = np.linalg.norm(delta_plus_after)
    norm_mb = np.linalg.norm(delta_minus_before)
    norm_ma = np.linalg.norm(delta_minus_after)
    
    if norm_pb < 1e-10 or norm_pa < 1e-10 or norm_mb < 1e-10 or norm_ma < 1e-10:
        return None
    
    cos_plus = float(np.dot(delta_plus_before, delta_plus_after) / (norm_pb * norm_pa))
    cos_minus = float(np.dot(delta_minus_before, delta_minus_after) / (norm_mb * norm_ma))
    
    even_before = (delta_plus_before + delta_minus_before) / 2
    even_after = (delta_plus_after + delta_minus_after) / 2
    odd_before = (delta_plus_before - delta_minus_before) / 2
    odd_after = (delta_plus_after - delta_minus_after) / 2
    
    norm_ratio = (norm_pa + norm_ma) / (norm_pb + norm_mb + 1e-10)
    
    return {
        'norm_module': norm_name,
        'cos_plus': cos_plus,
        'cos_minus': cos_minus,
        'avg_cos_preserved': (cos_plus + cos_minus) / 2,
        'norm_ratio': norm_ratio,
        'even_norm_before': float(np.linalg.norm(even_before)),
        'even_norm_after': float(np.linalg.norm(even_after)),
        'odd_norm_before': float(np.linalg.norm(odd_before)),
        'odd_norm_after': float(np.linalg.norm(odd_after)),
        'even_ratio': float(np.linalg.norm(even_after) / (np.linalg.norm(even_before) + 1e-10)),
        'odd_ratio': float(np.linalg.norm(odd_after) / (np.linalg.norm(odd_before) + 1e-10)),
    }


def run_phase400b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 400b: Speed Deep Analysis ({model_name}) [{timestamp}] ===")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    print(f"  Layers: {layer_indices}")
    print(f"  Categories: speed, size, temperature")
    print(f"  N_RANDOM: {N_RANDOM}")
    
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
    for cat_data in CATEGORY_DATA.values():
        for obj_data in cat_data["objects"].values():
            for tok in [obj_data["target"], obj_data["comp"]]:
                if tok not in token_ids:
                    ids = tokenizer.encode(tok, add_special_tokens=False)
                    token_ids[tok] = ids[0] if ids else None
        for tok in cat_data["candidates"]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    all_results = {
        'model': model_name, 'timestamp': timestamp,
        'per_layer': {},
    }
    
    for li in layer_indices:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        layer_result = {}
        
        for cat_name, cat_data in CATEGORY_DATA.items():
            print(f"\n  === Category: {cat_name} ===")
            
            # Compute per-object directions
            captured = {}
            def make_hook(key):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook_fn
            
            handle = layers_list[li].register_forward_hook(make_hook('h'))
            
            obj_dirs = {}
            obj_baselines = {}
            
            for obj_name, obj_data in cat_data["objects"].items():
                target = obj_data["target"]
                comp = obj_data["comp"]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                
                h_correct_list = []
                h_corrupt_list = []
                baseline_diffs = []
                
                for f_idx in range(len(FRAMES)):
                    correct_clean = FRAMES[f_idx].format(obj=obj_name, attr=target)
                    correct_corrupt = CORRUPT_FRAMES[f_idx].format(attr=target)
                    
                    captured.clear()
                    inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        model(input_ids=inputs["input_ids"].to(device),
                              attention_mask=inputs["attention_mask"].to(device))
                    h_correct_list.append(captured['h'][0, -1].numpy())
                    
                    captured.clear()
                    inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out = model(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device))
                    h_corrupt_list.append(captured['h'][0, -1].numpy())
                    diff, _, _ = get_logit_diff(out.logits[0, -1], tid, cid)
                    baseline_diffs.append(diff)
                
                dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
                obj_dirs[obj_name] = dh
                obj_baselines[obj_name] = float(np.mean(baseline_diffs))
            
            handle.remove()
            
            # Category-level direction (average of all objects)
            cat_dir = np.mean(list(obj_dirs.values()), axis=0)
            
            cat_result = {}
            
            for obj_name, obj_data in cat_data["objects"].items():
                target = obj_data["target"]
                comp = obj_data["comp"]
                align = obj_data["align"]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                dir_l1 = obj_dirs[obj_name]
                baseline_diff = obj_baselines[obj_name]
                
                if tid is None or cid is None:
                    continue
                
                prompt = CORRUPT_FRAMES[0].format(attr=target)
                
                print(f"\n    --- {obj_name}(align={align}) ---")
                
                obj_result = {
                    'align': align,
                    'baseline_gap': baseline_diff,
                }
                
                # ============================================================
                # Test A: Odd/Even decomposition with 30 random directions
                # ============================================================
                ortho_dirs = make_orthogonal_directions(dir_l1, N_RANDOM)
                
                l1_effects = {}
                for alpha in [-1.0, 1.0]:
                    scaled = alpha * dir_l1
                    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
                    diff_list = []
                    for f_idx in range(len(FRAMES)):
                        p = CORRUPT_FRAMES[f_idx].format(attr=target)
                        def make_add_hook(dv):
                            def hook_fn(module, input, output):
                                hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                                hs[0, -1, :] += dv
                                return (hs,) + output[1:] if isinstance(output, tuple) else hs
                            return hook_fn
                        h = layers_list[li].register_forward_hook(make_add_hook(delta))
                        try:
                            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
                            with torch.no_grad():
                                out2 = model(input_ids=inputs["input_ids"].to(device),
                                            attention_mask=inputs["attention_mask"].to(device))
                            d2, _, _ = get_logit_diff(out2.logits[0, -1], tid, cid)
                        finally:
                            h.remove()
                        diff_list.append(d2 - obj_baselines.get(obj_name, 0))
                    l1_effects[alpha] = float(np.mean(diff_list))
                
                l1_even = (l1_effects[1.0] + l1_effects[-1.0]) / 2
                l1_odd = (l1_effects[1.0] - l1_effects[-1.0]) / 2
                
                ortho_even_list = []
                ortho_odd_list = []
                for ortho_dir in ortho_dirs:
                    r_effects = {}
                    for alpha in [-1.0, 1.0]:
                        scaled = alpha * ortho_dir
                        delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
                        diff_list = []
                        for f_idx in range(len(FRAMES)):
                            p = CORRUPT_FRAMES[f_idx].format(attr=target)
                            def make_add_hook(dv):
                                def hook_fn(module, input, output):
                                    hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                                    hs[0, -1, :] += dv
                                    return (hs,) + output[1:] if isinstance(output, tuple) else hs
                                return hook_fn
                            h = layers_list[li].register_forward_hook(make_add_hook(delta))
                            try:
                                inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
                                with torch.no_grad():
                                    out2 = model(input_ids=inputs["input_ids"].to(device),
                                                attention_mask=inputs["attention_mask"].to(device))
                                d2, _, _ = get_logit_diff(out2.logits[0, -1], tid, cid)
                            finally:
                                h.remove()
                            diff_list.append(d2 - obj_baselines.get(obj_name, 0))
                        r_effects[alpha] = float(np.mean(diff_list))
                    ortho_even_list.append((r_effects[1.0] + r_effects[-1.0]) / 2)
                    ortho_odd_list.append((r_effects[1.0] - r_effects[-1.0]) / 2)
                
                avg_ortho_even = float(np.mean(ortho_even_list))
                std_ortho_even = float(np.std(ortho_even_list))
                avg_ortho_odd = float(np.mean(ortho_odd_list))
                
                even_ratio = avg_ortho_even / (l1_even + 1e-10)
                source = "NORM_DOM" if abs(even_ratio) > 0.7 else ("ATTRACTOR_DOM" if abs(even_ratio) < 0.3 else "MIXED")
                
                # Odd% = |l1_odd| / (|l1_even| + |l1_odd|)
                odd_pct = abs(l1_odd) / (abs(l1_even) + abs(l1_odd) + 1e-10)
                
                obj_result['l1_even'] = l1_even
                obj_result['l1_odd'] = l1_odd
                obj_result['avg_ortho_even'] = avg_ortho_even
                obj_result['std_ortho_even'] = std_ortho_even
                obj_result['even_ratio'] = even_ratio
                obj_result['even_source'] = source
                obj_result['odd_pct'] = float(odd_pct)
                
                print(f"      L1_even={l1_even:+.4f} L1_odd={l1_odd:+.4f} "
                      f"ortho_even={avg_ortho_even:+.4f}+/-{std_ortho_even:.4f} "
                      f"ratio={even_ratio:.3f} source={source} odd%={odd_pct:.1%}")
                
                # ============================================================
                # Test B: Direction interpretability (W_U projection)
                # ============================================================
                # Project direction onto all candidate token embeddings
                dir_norm = np.linalg.norm(dir_l1)
                if dir_norm > 1e-10:
                    dir_normalized = dir_l1 / dir_norm
                else:
                    dir_normalized = dir_l1
                
                cand_projections = {}
                for tok in cat_data["candidates"]:
                    c_id = token_ids.get(tok)
                    if c_id is not None and c_id < W_U_np.shape[0]:
                        w = W_U_np[c_id]
                        w_norm = np.linalg.norm(w)
                        if w_norm > 1e-10:
                            cos = float(np.dot(dir_normalized, w / w_norm))
                        else:
                            cos = 0.0
                        cand_projections[tok] = cos
                
                # Sort by projection
                sorted_proj = sorted(cand_projections.items(), key=lambda x: -x[1])
                obj_result['wu_projections'] = sorted_proj
                
                print(f"      W_U proj: {[(t, f'{v:.3f}') for t, v in sorted_proj[:4]]}")
                
                # ============================================================
                # Test C: Cross-object generalization
                # ============================================================
                # Test if cheetah's direction helps turtle and vice versa
                cross_obj_results = {}
                for other_obj, other_dir in obj_dirs.items():
                    if other_obj == obj_name:
                        continue
                    other_data = cat_data["objects"][other_obj]
                    other_tid = token_ids.get(other_data["target"])
                    other_cid = token_ids.get(other_data["comp"])
                    if other_tid is None or other_cid is None:
                        continue
                    
                    # Inject obj's direction on other's corrupt prompt
                    other_prompt = CORRUPT_FRAMES[0].format(attr=other_data["target"])
                    scaled = dir_l1  # Use obj's direction
                    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
                    
                    def make_add_hook(dv):
                        def hook_fn(module, input, output):
                            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                            hs[0, -1, :] += dv
                            return (hs,) + output[1:] if isinstance(output, tuple) else hs
                        return hook_fn
                    
                    h = layers_list[li].register_forward_hook(make_add_hook(delta))
                    try:
                        inputs = tokenizer(other_prompt, return_tensors="pt", truncation=True, max_length=64)
                        with torch.no_grad():
                            out2 = model(input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device))
                        cross_diff, _, _ = get_logit_diff(out2.logits[0, -1], other_tid, other_cid)
                    finally:
                        h.remove()
                    
                    # Also test with -direction
                    neg_delta = torch.tensor(-scaled, dtype=torch.bfloat16, device=device)
                    h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
                    try:
                        inputs = tokenizer(other_prompt, return_tensors="pt", truncation=True, max_length=64)
                        with torch.no_grad():
                            out3 = model(input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device))
                        cross_diff_neg, _, _ = get_logit_diff(out3.logits[0, -1], other_tid, other_cid)
                    finally:
                        h2.remove()
                    
                    # Baseline for other object
                    inputs = tokenizer(other_prompt, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out_base = model(input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device))
                    base_diff, _, _ = get_logit_diff(out_base.logits[0, -1], other_tid, other_cid)
                    
                    cross_even = ((cross_diff - base_diff) + (cross_diff_neg - base_diff)) / 2
                    cross_odd = ((cross_diff - base_diff) - (cross_diff_neg - base_diff)) / 2
                    
                    cross_obj_results[other_obj] = {
                        'plus_delta_diff': float(cross_diff - base_diff),
                        'minus_delta_diff': float(cross_diff_neg - base_diff),
                        'cross_even': float(cross_even),
                        'cross_odd': float(cross_odd),
                    }
                    
                    print(f"      Cross {obj_name}->{other_obj}: "
                          f"even={cross_even:+.4f} odd={cross_odd:+.4f} "
                          f"+d={cross_diff-base_diff:+.4f} -d={cross_diff_neg-base_diff:+.4f}")
                
                obj_result['cross_generalization'] = cross_obj_results
                
                # ============================================================
                # Test D: RMSNorm sign preservation
                # ============================================================
                rms_result = rmsnorm_sign_test(model, tokenizer, layers_list, device, li, dir_l1, prompt)
                obj_result['rmsnorm'] = rms_result
                if rms_result:
                    print(f"      RMSNorm: cos={rms_result['avg_cos_preserved']:.4f} "
                          f"norm_ratio={rms_result['norm_ratio']:.4f} "
                          f"even/odd={rms_result['even_ratio']:.2f}/{rms_result['odd_ratio']:.2f}")
                
                cat_result[obj_name] = obj_result
            
            # Category summary: average odd% across objects
            avg_odd_pct = np.mean([r.get('odd_pct', 0) for r in cat_result.values()])
            cat_result['_category_odd_pct'] = float(avg_odd_pct)
            
            layer_result[cat_name] = cat_result
        
        # Layer summary: compare speed vs size vs temperature
        print(f"\n  === Layer {li} Category Comparison ===")
        print(f"  {'Category':12s} {'avg odd%':>9s} {'NORM_DOM':>8s} {'MIXED':>6s} {'ATTRACTOR':>9s}")
        for cat_name in ["speed", "size", "temperature"]:
            cr = layer_result.get(cat_name, {})
            n_norm = sum(1 for k, v in cr.items() if isinstance(v, dict) and v.get('even_source') == 'NORM_DOM')
            n_mixed = sum(1 for k, v in cr.items() if isinstance(v, dict) and v.get('even_source') == 'MIXED')
            n_attr = sum(1 for k, v in cr.items() if isinstance(v, dict) and v.get('even_source') == 'ATTRACTOR_DOM')
            avg_odd = cr.get('_category_odd_pct', 0)
            print(f"  {cat_name:12s} {avg_odd:9.1%} {n_norm:8d} {n_mixed:6d} {n_attr:9d}")
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0:.0f}s")
    
    # Save
    out_dir = ROOT / "results" / "phase400b_speed_deep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase400b.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    
    print(f"\nOdd% by Category and Layer:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for cat_name in ["speed", "size", "temperature"]:
            cr = lr.get(cat_name, {})
            avg_odd = cr.get('_category_odd_pct', 0)
            print(f"  L{li} {cat_name}: avg odd%={avg_odd:.1%}")
    
    print(f"\nW_U Projections (speed category, L4/L5):")
    first_li = layer_indices[0]
    lr = all_results['per_layer'].get(str(first_li), {})
    speed_cr = lr.get('speed', {})
    for obj_name in ["cheetah", "turtle", "rocket"]:
        obj_r = speed_cr.get(obj_name, {})
        proj = obj_r.get('wu_projections', [])
        if proj:
            print(f"  {obj_name}: {[(t, f'{v:.3f}') for t, v in proj[:4]]}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase400b(model_name)
