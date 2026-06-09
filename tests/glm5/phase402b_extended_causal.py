"""
Phase 402b Extended: Multi-layer Causal Decomposition
=====================================================

扩展测试：更多对象、更多层、更详细分析
用于验证和确认Phase 402的结果
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

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model

# Extended objects for comprehensive testing
SPEED_OBJECTS = {
    # Animals
    "snail":      {"type": "animal", "speed_level": 1, "target": "slow", "comp": "fast"},
    "turtle":     {"type": "animal", "speed_level": 2, "target": "slow", "comp": "fast"},
    "horse":      {"type": "animal", "speed_level": 4, "target": "fast", "comp": "slow"},
    "cheetah":    {"type": "animal", "speed_level": 5, "target": "fast", "comp": "slow"},
    "falcon":     {"type": "animal", "speed_level": 5, "target": "fast", "comp": "slow"},
    
    # Vehicles
    "bicycle":    {"type": "vehicle", "speed_level": 2, "target": "slow", "comp": "fast"},
    "car":        {"type": "vehicle", "speed_level": 3, "target": "fast", "comp": "slow"},
    "train":      {"type": "vehicle", "speed_level": 4, "target": "fast", "comp": "slow"},
    "rocket":     {"type": "vehicle", "speed_level": 5, "target": "fast", "comp": "slow"},
    
    # Phenomena
    "glacier":    {"type": "phenomenon", "speed_level": 1, "target": "slow", "comp": "fast"},
    "wind":       {"type": "phenomenon", "speed_level": 4, "target": "fast", "comp": "slow"},
    "lightning":  {"type": "phenomenon", "speed_level": 5, "target": "fast", "comp": "slow"},
}

# Multi-candidate speed scale
SPEED_CANDIDATES = [
    "sluggish", "slow", "steady", "moderate", "quick", "fast", "rapid", "swift",
]

FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
]

# Layer configurations for extended testing
LAYER_CONFIGS = {
    "qwen3": [4, 12, 20, 28],
    "deepseek7b": [4, 10, 16, 20],
    "glm4": [5, 15, 25, 35],
}


def log_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    """Safe BF16 loading with fallback to eager attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], 
        trust_remote_code=True, 
        local_files_only=True, 
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try multiple attention implementations
    model = None
    for impl in ["eager", "sdpa"]:
        try:
            print(f"  Trying {impl} attention...")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl
            )
            print(f"  Success with {impl}")
            break
        except Exception as e:
            print(f"  Failed with {impl}: {str(e)[:100]}")
            continue
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_direction_batch(model, tokenizer, layers_list, device, li, obj_name, obj_data, token_ids):
    """Compute speed direction efficiently."""
    target = obj_data["target"]
    comp = obj_data["comp"]
    tid = token_ids.get(target)
    cid = token_ids.get(comp)
    
    h_correct_list = []
    h_corrupt_list = []
    baseline_diffs = []
    
    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn
    
    handle = layers_list[li].register_forward_hook(make_hook('h'))
    
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
        diff = float(out.logits[0, -1, tid].float().cpu() - out.logits[0, -1, cid].float().cpu()) if tid is not None and cid is not None else 0.0
        baseline_diffs.append(diff)
    
    handle.remove()
    
    dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
    baseline = float(np.mean(baseline_diffs))
    return dh, baseline


def orthogonalize(v, basis_list):
    """Remove components of v along each basis vector in basis_list."""
    result = v.copy()
    for b in basis_list:
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            continue
        proj = np.dot(result, b) / (np.dot(b, b) + 1e-10) * b
        result = result - proj
    return result


def run_cross_patch(model, tokenizer, layers_list, device, li, direction, 
                    target_obj, token_ids, obj_baselines):
    """Run cross-patch test efficiently."""
    tgt_data = SPEED_OBJECTS[target_obj]
    tgt_tid = token_ids.get(tgt_data["target"])
    tgt_cid = token_ids.get(tgt_data["comp"])
    
    if tgt_tid is None or tgt_cid is None:
        return None
    
    tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt_data["target"])
    tgt_baseline = obj_baselines[target_obj]
    
    # +direction
    delta = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    
    h = layers_list[li].register_forward_hook(make_add_hook(delta))
    try:
        inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_plus = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
        diff_plus = float(out_plus.logits[0, -1, tgt_tid].float().cpu() - out_plus.logits[0, -1, tgt_cid].float().cpu())
    finally:
        h.remove()
    
    # -direction
    neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
    h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
    try:
        inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_minus = model(input_ids=inputs["input_ids"].to(device),
                             attention_mask=inputs["attention_mask"].to(device))
        diff_minus = float(out_minus.logits[0, -1, tgt_tid].float().cpu() - out_minus.logits[0, -1, tgt_cid].float().cpu())
    finally:
        h2.remove()
    
    plus_eff = diff_plus - tgt_baseline
    minus_eff = diff_minus - tgt_baseline
    even = (plus_eff + minus_eff) / 2
    odd = (plus_eff - minus_eff) / 2
    
    return {
        'odd': float(odd), 'even': float(even),
        'plus': float(plus_eff), 'minus': float(minus_eff),
        'same_type': False,  # Will be set by caller
        'same_speed': False,  # Will be set by caller
    }


def run_phase402b_extended(model_name):
    """Extended causal decomposition with more layers and objects."""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 402b Extended: Multi-layer Causal Decomposition ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4, 20])
    obj_names = sorted(SPEED_OBJECTS.keys())
    n_obj = len(obj_names)
    print(f"  Layers: {layer_indices}")
    print(f"  Objects: {n_obj} ({', '.join(obj_names)})")
    
    # Load model
    t0_load = time.time()
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    print(f"  Model loaded in {time.time()-t0_load:.1f}s. {log_memory()}")
    
    # Resolve token IDs
    token_ids = {}
    for obj_name, obj_data in SPEED_OBJECTS.items():
        for tok in [obj_data["target"], obj_data["comp"]]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'objects': {k: v for k, v in SPEED_OBJECTS.items()},
        'per_layer': {},
    }
    
    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        # Step 1: Compute speed directions for all objects
        print(f"\n  Step 1: Computing speed directions...")
        obj_dirs = {}
        obj_baselines = {}
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            dh, baseline = compute_direction_batch(
                model, tokenizer, layers_list, device, li,
                obj_name, obj_data, token_ids)
            obj_dirs[obj_name] = dh
            obj_baselines[obj_name] = baseline
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.2f} baseline={baseline:.4f}")
        
        # Step 2: Compute TYPE and SPEED basis directions
        print(f"\n  Step 2: Computing TYPE and SPEED basis directions...")
        
        # TYPE directions: mean direction per type
        type_groups = defaultdict(list)
        for obj_name in obj_names:
            type_groups[SPEED_OBJECTS[obj_name]["type"]].append(obj_name)
        
        type_dirs = {}
        for type_name, members in type_groups.items():
            type_dir = np.mean([obj_dirs[m] for m in members], axis=0)
            type_dirs[type_name] = type_dir
            print(f"    TYPE {type_name}: |dir|={np.linalg.norm(type_dir):.2f} members={len(members)}")
        
        # SPEED direction: contrast fast vs slow objects
        fast_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] >= 4]
        slow_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] <= 2]
        dir_fast = np.mean([obj_dirs[n] for n in fast_objs], axis=0)
        dir_slow = np.mean([obj_dirs[n] for n in slow_objs], axis=0)
        speed_axis = dir_fast - dir_slow  # fast - slow direction
        print(f"    SPEED axis: |dir|={np.linalg.norm(speed_axis):.2f} fast={len(fast_objs)} slow={len(slow_objs)}")
        
        # Compute TYPE mean (grand mean of all objects)
        grand_mean = np.mean(list(obj_dirs.values()), axis=0)
        
        # Step 3: Orthogonal decomposition
        print(f"\n  Step 3: Orthogonal decomposition...")
        
        # Build orthogonal TYPE basis
        type_basis = {}
        type_names_sorted = sorted(type_dirs.keys())
        orthogonal_basis = []
        for tn in type_names_sorted:
            d = type_dirs[tn] - grand_mean
            d_orth = orthogonalize(d, orthogonal_basis)
            d_norm = np.linalg.norm(d_orth)
            if d_norm > 1e-10:
                orthogonal_basis.append(d_orth / d_norm)
                type_basis[tn] = d_orth / d_norm
            else:
                type_basis[tn] = np.zeros_like(d_orth)
        
        # Speed axis orthogonal to TYPE
        speed_orth = orthogonalize(speed_axis, orthogonal_basis)
        speed_orth_norm = np.linalg.norm(speed_orth)
        if speed_orth_norm > 1e-10:
            speed_basis = speed_orth / speed_orth_norm
        else:
            speed_basis = speed_axis / (np.linalg.norm(speed_axis) + 1e-10)
        
        # Decompose each object
        obj_decomp = {}
        for obj_name in obj_names:
            d = obj_dirs[obj_name]
            d_centered = d - grand_mean
            
            # TYPE component
            obj_type = SPEED_OBJECTS[obj_name]["type"]
            type_comp = np.dot(d_centered, type_basis.get(obj_type, np.zeros_like(d_centered))) * \
                        type_basis.get(obj_type, np.zeros_like(d_centered))
            
            # SPEED component
            speed_proj = np.dot(d_centered, speed_basis) * speed_basis
            
            # Residual
            residual = d_centered - type_comp - speed_proj
            
            obj_decomp[obj_name] = {
                'type_comp': type_comp,
                'speed_comp': speed_proj,
                'residual': residual,
                'type_comp_norm': float(np.linalg.norm(type_comp)),
                'speed_comp_norm': float(np.linalg.norm(speed_proj)),
                'residual_norm': float(np.linalg.norm(residual)),
                'total_norm': float(np.linalg.norm(d_centered)),
            }
        
        # Step 4: Causal patching tests (only cross-type for efficiency)
        print(f"\n  Step 4: Causal patching tests (cross-type only)...")
        
        cross_results = {}
        type_cross_results = {}
        speed_cross_results = {}
        
        # Select representative objects: one from each type
        rep_objects = ["cheetah", "rocket", "lightning"]  # fast objects from each type
        
        for src_name in rep_objects:
            src_data = SPEED_OBJECTS[src_name]
            src_dir = obj_dirs[src_name]
            src_type_comp = obj_decomp[src_name]['type_comp'] + grand_mean
            src_speed_comp = obj_decomp[src_name]['speed_comp'] + grand_mean
            
            src_cross = {}
            src_type_cross = {}
            src_speed_cross = {}
            
            for tgt_name in obj_names:
                if tgt_name == src_name:
                    continue
                
                # Full direction
                result = run_cross_patch(model, tokenizer, layers_list, device, li,
                                        src_dir, tgt_name, token_ids, obj_baselines)
                if result:
                    result['same_type'] = SPEED_OBJECTS[src_name]["type"] == SPEED_OBJECTS[tgt_name]["type"]
                    result['same_speed'] = abs(SPEED_OBJECTS[src_name]["speed_level"] - SPEED_OBJECTS[tgt_name]["speed_level"]) <= 1
                    src_cross[tgt_name] = result
                
                # TYPE-only
                result_type = run_cross_patch(model, tokenizer, layers_list, device, li,
                                            src_type_comp, tgt_name, token_ids, obj_baselines)
                if result_type:
                    result_type['same_type'] = SPEED_OBJECTS[src_name]["type"] == SPEED_OBJECTS[tgt_name]["type"]
                    src_type_cross[tgt_name] = result_type
                
                # SPEED-only
                result_speed = run_cross_patch(model, tokenizer, layers_list, device, li,
                                            src_speed_comp, tgt_name, token_ids, obj_baselines)
                if result_speed:
                    result_speed['same_type'] = SPEED_OBJECTS[src_name]["type"] == SPEED_OBJECTS[tgt_name]["type"]
                    result_speed['same_speed'] = abs(SPEED_OBJECTS[src_name]["speed_level"] - SPEED_OBJECTS[tgt_name]["speed_level"]) <= 1
                    src_speed_cross[tgt_name] = result_speed
            
            cross_results[src_name] = src_cross
            type_cross_results[src_name] = src_type_cross
            speed_cross_results[src_name] = src_speed_cross
            
            print(f"    {src_name}: full={len(src_cross)}, type={len(src_type_cross)}, speed={len(src_speed_cross)}")
        
        # Step 5: Aggregate causal analysis
        print(f"\n  Step 5: Aggregate causal analysis...")
        
        # Aggregate results
        full_within = []
        full_across = []
        type_within = []
        type_across = []
        speed_within = []
        speed_across = []
        speed_samespeed = []
        speed_diffspeed = []
        
        for src, tgts in cross_results.items():
            for tgt, vals in tgts.items():
                if vals['same_type']:
                    full_within.append(vals['odd'])
                else:
                    full_across.append(vals['odd'])
        
        for src, tgts in type_cross_results.items():
            for tgt, vals in tgts.items():
                if vals['same_type']:
                    type_within.append(vals['odd'])
                else:
                    type_across.append(vals['odd'])
        
        for src, tgts in speed_cross_results.items():
            for tgt, vals in tgts.items():
                if vals['same_type']:
                    speed_within.append(vals['odd'])
                else:
                    speed_across.append(vals['odd'])
                if vals.get('same_speed', False):
                    speed_samespeed.append(vals['odd'])
                else:
                    speed_diffspeed.append(vals['odd'])
        
        print(f"\n    === Causal Decomposition Results ===")
        print(f"    {'Component':>12s} {'Within-type odd':>16s} {'Across-type odd':>16s} {'Diff':>8s}")
        for label, within, across in [
            ("Full", full_within, full_across),
            ("TYPE-only", type_within, type_across),
            ("SPEED-only", speed_within, speed_across),
        ]:
            mw = np.mean(within) if within else 0
            ma = np.mean(across) if across else 0
            print(f"    {label:>12s} {mw:>+16.4f} {ma:>+16.4f} {mw-ma:>+8.4f}")
        
        print(f"\n    SPEED-only by speed-level similarity:")
        ms = np.mean(speed_samespeed) if speed_samespeed else 0
        md = np.mean(speed_diffspeed) if speed_diffspeed else 0
        print(f"    Same-speed: {ms:+.4f} Different-speed: {md:+.4f} Diff: {ms-md:+.4f}")
        
        # Store layer results
        layer_result = {
            'obj_dirs_norm': {n: float(np.linalg.norm(obj_dirs[n])) for n in obj_names},
            'obj_baselines': obj_baselines,
            'decomposition_norms': {
                n: {k: v for k, v in d.items() if 'norm' in k}
                for n, d in obj_decomp.items()
            },
            'aggregate': {
                'full_within_odd': float(np.mean(full_within)) if full_within else 0,
                'full_across_odd': float(np.mean(full_across)) if full_across else 0,
                'type_within_odd': float(np.mean(type_within)) if type_within else 0,
                'type_across_odd': float(np.mean(type_across)) if type_across else 0,
                'speed_within_odd': float(np.mean(speed_within)) if speed_within else 0,
                'speed_across_odd': float(np.mean(speed_across)) if speed_across else 0,
                'speed_samespeed_odd': float(np.mean(speed_samespeed)) if speed_samespeed else 0,
                'speed_diffspeed_odd': float(np.mean(speed_diffspeed)) if speed_diffspeed else 0,
            }
        }
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s. {log_memory()}")
    
    # Save results
    out_dir = ROOT / "results" / "phase402b_extended_causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase402b.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cross-layer summary
    print(f"\n{'='*80}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    print(f"{'='*80}")
    
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        agg = lr.get('aggregate', {})
        print(f"\n  L{li}:")
        print(f"    Full: within={agg.get('full_within_odd',0):+.4f}, across={agg.get('full_across_odd',0):+.4f}, diff={agg.get('full_within_odd',0)-agg.get('full_across_odd',0):+.4f}")
        print(f"    TYPE: within={agg.get('type_within_odd',0):+.4f}, across={agg.get('type_across_odd',0):+.4f}, diff={agg.get('type_within_odd',0)-agg.get('type_across_odd',0):+.4f}")
        print(f"    SPEED: within={agg.get('speed_within_odd',0):+.4f}, across={agg.get('speed_across_odd',0):+.4f}, diff={agg.get('speed_within_odd',0)-agg.get('speed_across_odd',0):+.4f}")
    
    # Cleanup
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nReleased. {log_memory()}")
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase402b_extended(model_name)