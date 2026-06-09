"""
Phase 403b: Confirmation Test - Sign Alignment & Speed-Level Gradient
======================================================================

重点验证:
1. 方向符号是否正确对齐 (fast候选odd应该为正)
2. speed-level gradient是否随深度变化
3. SPEED-only patch对慢/快对象的不同影响
4. 范数控制是否真的只产生压缩效应

关键修正:
- 明确追踪"提高目标方向"的效果
- 区分fast-target对象和slow-target对象
- 检查speed-level gradient的符号一致性

Usage:
  python tests/glm5/phase403b_confirm.py qwen3
  python tests/glm5/phase403b_confirm.py deepseek7b
  python tests/glm5/phase403b_confirm.py glm4
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

# 8-level speed scale
SPEED_CANDIDATES = {
    "sluggish": 1,
    "slow":     2,
    "steady":   3,
    "moderate": 4,
    "quick":    5,
    "fast":     6,
    "rapid":    7,
    "swift":    8,
}

# 6 objects with clear speed assignments
SPEED_OBJECTS = {
    "snail":      {"type": "animal",     "speed_level": 1, "target": "slow",   "comp": "fast"},
    "cheetah":    {"type": "animal",     "speed_level": 5, "target": "fast",   "comp": "slow"},
    "bicycle":    {"type": "vehicle",    "speed_level": 2, "target": "slow",   "comp": "fast"},
    "rocket":     {"type": "vehicle",    "speed_level": 5, "target": "fast",   "comp": "slow"},
    "glacier":    {"type": "phenomenon", "speed_level": 1, "target": "slow",   "comp": "fast"},
    "lightning":  {"type": "phenomenon", "speed_level": 5, "target": "fast",   "comp": "slow"},
}

FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
]

LAYER_CONFIGS = {
    "qwen3": [4, 16, 28],
    "deepseek7b": [4, 12, 20],
    "glm4": [5, 20, 35],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["eager", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            break
        except Exception as e:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_speed_direction(model, tokenizer, layers_list, device, li, obj_name, obj_data, token_ids):
    """Compute speed direction for an object."""
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
    
    for f_idx in range(2):
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
    result = v.copy()
    for b in basis_list:
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            continue
        proj = np.dot(result, b) / (np.dot(b, b) + 1e-10) * b
        result = result - proj
    return result


def get_full_candidate_logits(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device))
    return out.logits[0, -1, :].float().cpu().numpy()


def patch_and_get_distribution(model, tokenizer, layers_list, device, li, direction, obj_name, obj_data):
    """Apply direction patch and get full distribution."""
    tgt_prompt = CORRUPT_FRAMES[0].format(attr=obj_data["target"])
    
    baseline_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
    
    delta = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    
    h = layers_list[li].register_forward_hook(make_add_hook(delta))
    try:
        plus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
    finally:
        h.remove()
    
    neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
    h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
    try:
        minus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
    finally:
        h2.remove()
    
    return baseline_logits, plus_logits, minus_logits


def run_phase403b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 403b: Sign Alignment & Speed-Level Gradient ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    obj_names = sorted(SPEED_OBJECTS.keys())
    print(f"  Layers: {layer_indices}")
    print(f"  Objects: {len(obj_names)}")
    
    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    
    # Resolve token IDs
    token_ids = {}
    for cand_name in SPEED_CANDIDATES:
        ids = tokenizer.encode(cand_name, add_special_tokens=False)
        token_ids[cand_name] = ids[0] if ids else None
    for obj_name, obj_data in SPEED_OBJECTS.items():
        for tok in [obj_data["target"], obj_data["comp"]]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    print(f"  Token IDs: fast={token_ids.get('fast')}, slow={token_ids.get('slow')}")
    
    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'per_layer': {},
    }
    
    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        # Step 1: Compute speed directions
        print(f"\n  Step 1: Computing speed directions...")
        obj_dirs = {}
        obj_baselines = {}
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            dh, baseline = compute_speed_direction(model, tokenizer, layers_list, device, li,
                                                    obj_name, obj_data, token_ids)
            obj_dirs[obj_name] = dh
            obj_baselines[obj_name] = baseline
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.4f} baseline={baseline:.4f}")
        
        # Step 2: Orthogonal decomposition
        print(f"\n  Step 2: Orthogonal decomposition...")
        type_groups = defaultdict(list)
        for obj_name in obj_names:
            type_groups[SPEED_OBJECTS[obj_name]["type"]].append(obj_name)
        
        type_dirs = {}
        for type_name, members in type_groups.items():
            type_dirs[type_name] = np.mean([obj_dirs[m] for m in members], axis=0)
        
        fast_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] >= 4]
        slow_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] <= 2]
        speed_axis = np.mean([obj_dirs[n] for n in fast_objs], axis=0) - \
                     np.mean([obj_dirs[n] for n in slow_objs], axis=0)
        
        grand_mean = np.mean(list(obj_dirs.values()), axis=0)
        
        type_basis = {}
        orthogonal_basis = []
        for tn in sorted(type_dirs.keys()):
            d = type_dirs[tn] - grand_mean
            d_orth = orthogonalize(d, orthogonal_basis)
            d_norm = np.linalg.norm(d_orth)
            if d_norm > 1e-10:
                orthogonal_basis.append(d_orth / d_norm)
                type_basis[tn] = d_orth / d_norm
        
        speed_orth = orthogonalize(speed_axis, orthogonal_basis)
        if np.linalg.norm(speed_orth) > 1e-10:
            speed_basis = speed_orth / np.linalg.norm(speed_orth)
        else:
            speed_basis = speed_axis / (np.linalg.norm(speed_axis) + 1e-10)
        
        # Decompose each object
        obj_decomp = {}
        for obj_name in obj_names:
            d = obj_dirs[obj_name]
            d_centered = d - grand_mean
            obj_type = SPEED_OBJECTS[obj_name]["type"]
            type_comp = np.dot(d_centered, type_basis.get(obj_type, np.zeros_like(d_centered))) * \
                        type_basis.get(obj_type, np.zeros_like(d_centered))
            speed_proj = np.dot(d_centered, speed_basis) * speed_basis
            residual = d_centered - type_comp - speed_proj
            obj_decomp[obj_name] = {
                'type_comp': type_comp,
                'speed_comp': speed_proj,
                'residual': residual,
            }
        
        # Step 3: KEY TEST - Self-patch with full direction
        # For each object, patch its OWN direction to a corrupt prompt
        # This should INCREASE the target-competitor gap
        print(f"\n  Step 3: Self-patch verification (sign alignment check)...")
        
        self_patch_results = {}
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            tgt = obj_data["target"]
            comp = obj_data["comp"]
            tid = token_ids.get(tgt)
            cid = token_ids.get(comp)
            
            # Get baseline distribution for corrupt prompt
            tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt)
            baseline_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
            baseline_diff = float(baseline_logits[tid] - baseline_logits[cid]) if tid is not None and cid is not None else 0.0
            
            # Patch with +direction
            direction = obj_dirs[obj_name]
            delta = torch.tensor(direction, dtype=torch.bfloat16, device=device)
            def make_add_hook(dv):
                def hook_fn(module, input, output):
                    hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                    hs[0, -1, :] += dv
                    return (hs,) + output[1:] if isinstance(output, tuple) else hs
                return hook_fn
            
            h = layers_list[li].register_forward_hook(make_add_hook(delta))
            try:
                plus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
            finally:
                h.remove()
            plus_diff = float(plus_logits[tid] - plus_logits[cid]) if tid is not None and cid is not None else 0.0
            
            # Patch with -direction
            neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
            h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
            try:
                minus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
            finally:
                h2.remove()
            minus_diff = float(minus_logits[tid] - minus_logits[cid]) if tid is not None and cid is not None else 0.0
            
            plus_eff = plus_diff - baseline_diff
            minus_eff = minus_diff - baseline_diff
            odd = (plus_eff - minus_eff) / 2
            even = (plus_eff + minus_eff) / 2
            
            # Per-candidate effects
            cand_effects = {}
            for cand_name, speed_level in SPEED_CANDIDATES.items():
                cand_id = token_ids.get(cand_name)
                if cand_id is None:
                    continue
                b_val = float(baseline_logits[cand_id])
                p_val = float(plus_logits[cand_id])
                m_val = float(minus_logits[cand_id])
                p_eff = p_val - b_val
                m_eff = m_val - b_val
                cand_odd = (p_eff - m_eff) / 2
                cand_even = (p_eff + m_eff) / 2
                cand_effects[cand_name] = {
                    'odd': float(cand_odd),
                    'even': float(cand_even),
                    'speed_level': speed_level,
                }
            
            self_patch_results[obj_name] = {
                'baseline_diff': baseline_diff,
                'plus_diff': plus_diff,
                'minus_diff': minus_diff,
                'odd': float(odd),
                'even': float(even),
                'candidate_effects': cand_effects,
                'target': tgt,
                'competitor': comp,
            }
            
            # Print candidate-level odd effects
            fast_cand_odd = [v['odd'] for k, v in cand_effects.items() if v['speed_level'] >= 6]
            slow_cand_odd = [v['odd'] for k, v in cand_effects.items() if v['speed_level'] <= 2]
            mid_cand_odd = [v['odd'] for k, v in cand_effects.items() if 3 <= v['speed_level'] <= 5]
            print(f"    {obj_name} (target={tgt}): odd={odd:+.4f}, "
                  f"fast_cand_odd={np.mean(fast_cand_odd):+.4f}, "
                  f"mid_cand_odd={np.mean(mid_cand_odd):+.4f}, "
                  f"slow_cand_odd={np.mean(slow_cand_odd):+.4f}")
        
        # Step 4: Cross-object SPEED-only patch with per-candidate analysis
        print(f"\n  Step 4: Cross-object SPEED-only patch (per-candidate)...")
        
        cross_speed_results = {}
        for src_name in ['cheetah', 'rocket', 'lightning']:  # fast objects only
            src_speed_comp = obj_decomp[src_name]['speed_comp'] + grand_mean
            src_data = SPEED_OBJECTS[src_name]
            
            src_cross = {}
            for tgt_name in obj_names:
                if tgt_name == src_name:
                    continue
                tgt_data = SPEED_OBJECTS[tgt_name]
                tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt_data["target"])
                
                # Get baseline
                baseline_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
                
                # Patch with speed component
                delta = torch.tensor(src_speed_comp, dtype=torch.bfloat16, device=device)
                def make_add_hook(dv):
                    def hook_fn(module, input, output):
                        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                        hs[0, -1, :] += dv
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook_fn
                
                h = layers_list[li].register_forward_hook(make_add_hook(delta))
                try:
                    plus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
                finally:
                    h.remove()
                
                neg_delta = torch.tensor(-src_speed_comp, dtype=torch.bfloat16, device=device)
                h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
                try:
                    minus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
                finally:
                    h2.remove()
                
                # Per-candidate effects
                cand_effects = {}
                for cand_name, speed_level in SPEED_CANDIDATES.items():
                    cand_id = token_ids.get(cand_name)
                    if cand_id is None:
                        continue
                    b_val = float(baseline_logits[cand_id])
                    p_val = float(plus_logits[cand_id])
                    m_val = float(minus_logits[cand_id])
                    cand_odd = (p_val - m_val) / 2
                    cand_effects[cand_name] = {
                        'odd': float(cand_odd),
                        'speed_level': speed_level,
                    }
                
                # Compute speed-level gradient
                from scipy.stats import spearmanr
                levels = [v['speed_level'] for v in cand_effects.values()]
                odds = [v['odd'] for v in cand_effects.values()]
                grad_corr, _ = spearmanr(levels, odds) if len(levels) >= 3 else (0, 1)
                
                is_same_type = src_data['type'] == tgt_data['type']
                
                fast_odd = np.mean([v['odd'] for v in cand_effects.values() if v['speed_level'] >= 6])
                slow_odd = np.mean([v['odd'] for v in cand_effects.values() if v['speed_level'] <= 2])
                
                src_cross[tgt_name] = {
                    'same_type': is_same_type,
                    'grad_corr': float(grad_corr) if not np.isnan(grad_corr) else 0,
                    'fast_cand_odd': float(fast_odd),
                    'slow_cand_odd': float(slow_odd),
                    'candidate_effects': cand_effects,
                }
                
                print(f"    {src_name}->{tgt_name}: same_type={is_same_type}, "
                      f"grad={grad_corr:+.4f}, fast_odd={fast_odd:+.4f}, slow_odd={slow_odd:+.4f}")
            
            cross_speed_results[src_name] = src_cross
        
        # Step 5: Aggregate analysis
        print(f"\n  Step 5: Aggregate analysis...")
        
        # Self-patch: check if odd is positive (correct sign)
        self_odds = {n: r['odd'] for n, r in self_patch_results.items()}
        self_fast_odd = {}
        self_slow_odd = {}
        for obj_name, result in self_patch_results.items():
            ce = result['candidate_effects']
            fast_vals = [v['odd'] for v in ce.values() if v['speed_level'] >= 6]
            slow_vals = [v['odd'] for v in ce.values() if v['speed_level'] <= 2]
            self_fast_odd[obj_name] = np.mean(fast_vals) if fast_vals else 0
            self_slow_odd[obj_name] = np.mean(slow_vals) if slow_vals else 0
        
        print(f"\n    Self-patch sign alignment:")
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            tgt = obj_data['target']
            print(f"      {obj_name} (target={tgt}): self_odd={self_odds[obj_name]:+.4f}, "
                  f"fast_cand_odd={self_fast_odd[obj_name]:+.4f}, slow_cand_odd={self_slow_odd[obj_name]:+.4f}")
        
        # Cross-speed: gradient by type relation
        cross_within_grads = []
        cross_across_grads = []
        cross_within_fast = []
        cross_across_fast = []
        cross_within_slow = []
        cross_across_slow = []
        
        for src, tgts in cross_speed_results.items():
            for tgt, result in tgts.items():
                if result.get('same_type', False):
                    cross_within_grads.append(result['grad_corr'])
                    cross_within_fast.append(result['fast_cand_odd'])
                    cross_within_slow.append(result['slow_cand_odd'])
                else:
                    cross_across_grads.append(result['grad_corr'])
                    cross_across_fast.append(result['fast_cand_odd'])
                    cross_across_slow.append(result['slow_cand_odd'])
        
        print(f"\n    Cross SPEED-only gradient by type relation:")
        print(f"      Within-type: grad={np.mean(cross_within_grads):+.4f}, "
              f"fast_odd={np.mean(cross_within_fast):+.4f}, slow_odd={np.mean(cross_within_slow):+.4f}")
        print(f"      Across-type: grad={np.mean(cross_across_grads):+.4f}, "
              f"fast_odd={np.mean(cross_across_fast):+.4f}, slow_odd={np.mean(cross_across_slow):+.4f}")
        
        # Store layer results
        layer_result = {
            'self_patch': {k: {kk: vv for kk, vv in v.items() if kk != 'candidate_effects'} 
                          for k, v in self_patch_results.items()},
            'self_patch_fast_odd': {k: float(v) for k, v in self_fast_odd.items()},
            'self_patch_slow_odd': {k: float(v) for k, v in self_slow_odd.items()},
            'cross_speed': {
                'within_grad_mean': float(np.mean(cross_within_grads)) if cross_within_grads else 0,
                'across_grad_mean': float(np.mean(cross_across_grads)) if cross_across_grads else 0,
                'within_fast_odd': float(np.mean(cross_within_fast)) if cross_within_fast else 0,
                'across_fast_odd': float(np.mean(cross_across_fast)) if cross_across_fast else 0,
                'within_slow_odd': float(np.mean(cross_within_slow)) if cross_within_slow else 0,
                'across_slow_odd': float(np.mean(cross_across_slow)) if cross_across_slow else 0,
            },
            'aggregate': {
                'self_odd_mean': float(np.mean(list(self_odds.values()))),
                'self_fast_odd_mean': float(np.mean(list(self_fast_odd.values()))),
                'self_slow_odd_mean': float(np.mean(list(self_slow_odd.values()))),
            }
        }
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s. {log_memory()}")
    
    # Cross-layer summary
    print(f"\n{'='*80}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    print(f"{'='*80}")
    
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        agg = lr.get('aggregate', {})
        cross = lr.get('cross_speed', {})
        print(f"  L{li}: self_odd={agg.get('self_odd_mean',0):+.4f}, "
              f"self_fast_odd={agg.get('self_fast_odd_mean',0):+.4f}, "
              f"self_slow_odd={agg.get('self_slow_odd_mean',0):+.4f}")
        print(f"        cross_within_grad={cross.get('within_grad_mean',0):+.4f}, "
              f"cross_across_grad={cross.get('across_grad_mean',0):+.4f}")
        print(f"        cross_within_fast={cross.get('within_fast_odd',0):+.4f}, "
              f"cross_across_fast={cross.get('across_fast_odd',0):+.4f}")
        print(f"        cross_within_slow={cross.get('within_slow_odd',0):+.4f}, "
              f"cross_across_slow={cross.get('across_slow_odd',0):+.4f}")
    
    # Save results
    out_dir = ROOT / "results" / "phase403_multi_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase403b.json"
    
    import copy
    results_to_save = copy.deepcopy(all_results)
    with open(out_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cleanup
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. {log_memory()}")
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase403b(model_name)
