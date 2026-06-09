"""
Phase 403: Multi-Candidate Speed Distribution Dynamics
======================================================

目标: 验证TYPE × SPEED是否真的改变完整速度候选排序

核心问题:
1. 速度语义几何是否反映完整候选分布?
2. 内部状态是否改变速度等级排序?
3. TYPE/SPEED成分对多候选的影响是否一致?

测试设计:
- 8个速度候选词: sluggish/slow/steady/moderate/quick/fast/rapid/swift
- 6个对象: snail/cheetah/bicycle/rocket/glacier/lightning (2×3类型)
- 4种patch: full/TYPE-only/SPEED-only/norm-control
- 2层: 早期+深层

关键指标:
- candidate_rank_correlation: 候选排序相关
- distribution_entropy: 分布熵
- speed_level_monotonicity: 速度等级单调性
- norm_vs_semantic_residual: 范数vs语义残差

Usage:
  python tests/glm5/phase403_multi_candidate.py qwen3
  python tests/glm5/phase403_multi_candidate.py deepseek7b
  python tests/glm5/phase403_multi_candidate.py glm4
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

# ===== 配置 =====

# 8-level speed scale with explicit ranking
SPEED_CANDIDATES = {
    "sluggish": 1,    # level 1 - slowest
    "slow":     2,    # level 2
    "steady":   3,    # level 3
    "moderate": 4,    # level 4
    "quick":    5,    # level 5
    "fast":     6,    # level 6
    "rapid":    7,    # level 7
    "swift":    8,    # level 8 - fastest
}

# 6 objects: 2 per type (fast + slow)
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

# Layer configs (early + late)
LAYER_CONFIGS = {
    "qwen3": [4, 28],
    "deepseek7b": [4, 20],
    "glm4": [5, 35],
}


def log_memory():
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


def get_full_candidate_logits(model, tokenizer, device, prompt):
    """Get logits for all speed candidate tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device))
    return out.logits[0, -1, :].float().cpu().numpy()


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
    """Remove components of v along each basis vector in basis_list."""
    result = v.copy()
    for b in basis_list:
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            continue
        proj = np.dot(result, b) / (np.dot(b, b) + 1e-10) * b
        result = result - proj
    return result


def patch_and_get_distribution(model, tokenizer, layers_list, device, li, direction, obj_name, obj_data, token_ids):
    """Apply direction patch and get full candidate distribution."""
    tgt_data = SPEED_OBJECTS[obj_name]
    tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt_data["target"])
    
    # Get baseline distribution (no patch)
    baseline_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
    
    # Get patched distribution (+direction)
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
    
    # Get patched distribution (-direction)
    neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
    h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
    try:
        minus_logits = get_full_candidate_logits(model, tokenizer, device, tgt_prompt)
    finally:
        h2.remove()
    
    return baseline_logits, plus_logits, minus_logits


def extract_speed_logits(full_logits, token_ids):
    """Extract logit values for speed candidates."""
    result = {}
    for cand_name, speed_level in SPEED_CANDIDATES.items():
        tid = token_ids.get(cand_name)
        if tid is not None:
            result[cand_name] = {
                'logit': float(full_logits[tid]),
                'speed_level': speed_level,
            }
    return result


def compute_rank_correlation(speed_logits_1, speed_logits_2):
    """Compute Spearman rank correlation between two speed logit distributions."""
    # Sort by candidate name to ensure consistent ordering
    common_keys = sorted(set(speed_logits_1.keys()) & set(speed_logits_2.keys()))
    if len(common_keys) < 2:
        return 0.0
    
    levels_1 = [SPEED_CANDIDATES[k] for k in common_keys]
    logit_vals_1 = [speed_logits_1[k]['logit'] for k in common_keys]
    logit_vals_2 = [speed_logits_2[k]['logit'] for k in common_keys]
    
    # Rank the logit values
    from scipy.stats import spearmanr
    corr, _ = spearmanr(logit_vals_1, logit_vals_2)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_speed_monotonicity(speed_logits):
    """Compute how monotonically logit values increase with speed level.
    Returns correlation between speed_level and logit value."""
    from scipy.stats import spearmanr
    levels = [v['speed_level'] for v in speed_logits.values()]
    logits = [v['logit'] for v in speed_logits.values()]
    if len(levels) < 3:
        return 0.0
    corr, _ = spearmanr(levels, logits)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_distribution_entropy(speed_logits, temperature=1.0):
    """Compute entropy of softmax distribution over speed candidates."""
    logits = np.array([v['logit'] for v in speed_logits.values()])
    logits_scaled = logits / temperature
    logits_scaled = logits_scaled - np.max(logits_scaled)  # stability
    probs = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def run_phase403(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 403: Multi-Candidate Speed Distribution Dynamics ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    obj_names = sorted(SPEED_OBJECTS.keys())
    n_obj = len(obj_names)
    cand_names = sorted(SPEED_CANDIDATES.keys())
    print(f"  Layers: {layer_indices}")
    print(f"  Objects: {n_obj} ({', '.join(obj_names)})")
    print(f"  Speed candidates: {len(cand_names)} ({', '.join(cand_names)})")
    
    # Load model
    t0_load = time.time()
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  Model loaded in {time.time()-t0_load:.1f}s. {log_memory()}")
    
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
    
    print(f"  Token IDs resolved: fast={token_ids.get('fast')}, slow={token_ids.get('slow')}, "
          f"rapid={token_ids.get('rapid')}, swift={token_ids.get('swift')}")
    
    # Check which speed candidates have valid token IDs
    valid_candidates = {k: v for k, v in SPEED_CANDIDATES.items() if token_ids.get(k) is not None}
    print(f"  Valid speed candidates: {len(valid_candidates)}/{len(SPEED_CANDIDATES)}")
    
    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'objects': {k: v for k, v in SPEED_OBJECTS.items()},
        'speed_candidates': {k: v for k, v in SPEED_CANDIDATES.items()},
        'valid_candidates': list(valid_candidates.keys()),
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
            dh, baseline = compute_speed_direction(
                model, tokenizer, layers_list, device, li,
                obj_name, obj_data, token_ids)
            obj_dirs[obj_name] = dh
            obj_baselines[obj_name] = baseline
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.4f} baseline={baseline:.4f}")
        
        # Step 2: Compute TYPE and SPEED basis directions
        print(f"\n  Step 2: Computing TYPE and SPEED basis directions...")
        
        type_groups = defaultdict(list)
        for obj_name in obj_names:
            type_groups[SPEED_OBJECTS[obj_name]["type"]].append(obj_name)
        
        type_dirs = {}
        for type_name, members in type_groups.items():
            type_dir = np.mean([obj_dirs[m] for m in members], axis=0)
            type_dirs[type_name] = type_dir
            print(f"    TYPE {type_name}: |dir|={np.linalg.norm(type_dir):.4f}")
        
        fast_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] >= 4]
        slow_objs = [n for n in obj_names if SPEED_OBJECTS[n]["speed_level"] <= 2]
        dir_fast = np.mean([obj_dirs[n] for n in fast_objs], axis=0)
        dir_slow = np.mean([obj_dirs[n] for n in slow_objs], axis=0)
        speed_axis = dir_fast - dir_slow
        print(f"    SPEED axis: |dir|={np.linalg.norm(speed_axis):.4f}")
        
        grand_mean = np.mean(list(obj_dirs.values()), axis=0)
        
        # Norm of speed direction (for norm control)
        speed_norm = np.linalg.norm(speed_axis)
        norm_control_dir = speed_axis / (speed_norm + 1e-10) * np.linalg.norm(np.mean(list(obj_dirs.values()), axis=0))
        # Actually use the mean norm of all directions for norm control
        mean_dir_norm = np.mean([np.linalg.norm(obj_dirs[n]) for n in obj_names])
        norm_control = np.random.randn(d_model).astype(np.float32)
        norm_control = norm_control / np.linalg.norm(norm_control) * mean_dir_norm
        
        # Step 3: Orthogonal decomposition
        print(f"\n  Step 3: Orthogonal decomposition...")
        
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
        
        # Step 4: Multi-candidate distribution testing
        print(f"\n  Step 4: Multi-candidate distribution testing...")
        
        # For each source object, patch to each target and measure full distribution
        patch_types = ['full', 'type_only', 'speed_only', 'norm_control']
        
        distribution_results = {}
        
        for src_name in obj_names:
            src_data = SPEED_OBJECTS[src_name]
            src_dir = obj_dirs[src_name]
            src_type_comp = obj_decomp[src_name]['type_comp'] + grand_mean
            src_speed_comp = obj_decomp[src_name]['speed_comp'] + grand_mean
            
            src_results = {}
            
            for tgt_name in obj_names:
                if tgt_name == src_name:
                    continue
                
                tgt_data = SPEED_OBJECTS[tgt_name]
                print(f"    {src_name} -> {tgt_name} ({src_data['type']}->{tgt_data['type']})")
                
                tgt_result = {}
                
                for patch_type in patch_types:
                    if patch_type == 'full':
                        direction = src_dir
                    elif patch_type == 'type_only':
                        direction = src_type_comp
                    elif patch_type == 'speed_only':
                        direction = src_speed_comp
                    elif patch_type == 'norm_control':
                        direction = norm_control
                    else:
                        continue
                    
                    try:
                        baseline_logits, plus_logits, minus_logits = patch_and_get_distribution(
                            model, tokenizer, layers_list, device, li,
                            direction, tgt_name, tgt_data, token_ids)
                        
                        # Extract speed candidate logits
                        base_speed = extract_speed_logits(baseline_logits, token_ids)
                        plus_speed = extract_speed_logits(plus_logits, token_ids)
                        minus_speed = extract_speed_logits(minus_logits, token_ids)
                        
                        # Compute metrics
                        # 1. Rank correlation between baseline and patched
                        rank_corr_plus = compute_rank_correlation(base_speed, plus_speed)
                        rank_corr_minus = compute_rank_correlation(base_speed, minus_speed)
                        
                        # 2. Speed monotonicity change
                        mono_base = compute_speed_monotonicity(base_speed)
                        mono_plus = compute_speed_monotonicity(plus_speed)
                        mono_minus = compute_speed_monotonicity(minus_speed)
                        
                        # 3. Distribution entropy change
                        ent_base = compute_distribution_entropy(base_speed)
                        ent_plus = compute_distribution_entropy(plus_speed)
                        ent_minus = compute_distribution_entropy(minus_speed)
                        
                        # 4. Odd/even decomposition for each candidate
                        candidate_effects = {}
                        for cand_name in valid_candidates:
                            tid = token_ids[cand_name]
                            base_val = float(baseline_logits[tid])
                            plus_val = float(plus_logits[tid])
                            minus_val = float(minus_logits[tid])
                            plus_eff = plus_val - base_val
                            minus_eff = minus_val - base_val
                            even = (plus_eff + minus_eff) / 2
                            odd = (plus_eff - minus_eff) / 2
                            candidate_effects[cand_name] = {
                                'odd': float(odd),
                                'even': float(even),
                                'plus_eff': float(plus_eff),
                                'minus_eff': float(minus_eff),
                                'speed_level': SPEED_CANDIDATES[cand_name],
                            }
                        
                        tgt_result[patch_type] = {
                            'rank_corr_plus': rank_corr_plus,
                            'rank_corr_minus': rank_corr_minus,
                            'mono_base': mono_base,
                            'mono_plus': mono_plus,
                            'mono_minus': mono_minus,
                            'mono_change_plus': mono_plus - mono_base,
                            'mono_change_minus': mono_minus - mono_base,
                            'ent_base': ent_base,
                            'ent_plus': ent_plus,
                            'ent_minus': ent_minus,
                            'ent_change_plus': ent_plus - ent_base,
                            'ent_change_minus': ent_minus - ent_base,
                            'candidate_effects': candidate_effects,
                            'same_type': src_data['type'] == tgt_data['type'],
                            'src_speed': src_data['speed_level'],
                            'tgt_speed': tgt_data['speed_level'],
                        }
                        
                    except Exception as e:
                        print(f"      {patch_type} failed: {str(e)[:80]}")
                        tgt_result[patch_type] = {'error': str(e)}
                
                src_results[tgt_name] = tgt_result
            
            distribution_results[src_name] = src_results
            print(f"    {src_name}: {len(src_results)} targets tested")
        
        # Step 5: Aggregate analysis
        print(f"\n  Step 5: Aggregate multi-candidate analysis...")
        
        # Key analysis: for each patch type, how does speed monotonicity change?
        # by same_type vs across_type
        aggregate = {}
        for patch_type in patch_types:
            mono_changes_same = []
            mono_changes_diff = []
            ent_changes_same = []
            ent_changes_diff = []
            rank_corrs = []
            
            # Speed-level dependent odd effects
            fast_cand_odd_same = []  # fast candidates (level >= 6)
            slow_cand_odd_same = []  # slow candidates (level <= 2)
            fast_cand_odd_diff = []
            slow_cand_odd_diff = []
            
            for src_name, tgts in distribution_results.items():
                for tgt_name, tgt_data in tgts.items():
                    if patch_type not in tgt_data or 'error' in tgt_data[patch_type]:
                        continue
                    
                    result = tgt_data[patch_type]
                    is_same_type = result.get('same_type', False)
                    
                    if is_same_type:
                        mono_changes_same.append(result['mono_change_plus'])
                        ent_changes_same.append(result['ent_change_plus'])
                    else:
                        mono_changes_diff.append(result['mono_change_plus'])
                        ent_changes_diff.append(result['ent_change_plus'])
                    
                    rank_corrs.append(result['rank_corr_plus'])
                    
                    # Per-candidate analysis
                    cand_effs = result.get('candidate_effects', {})
                    for cand_name, eff in cand_effs.items():
                        level = eff['speed_level']
                        odd_val = eff['odd']
                        if is_same_type:
                            if level >= 6:
                                fast_cand_odd_same.append(odd_val)
                            elif level <= 2:
                                slow_cand_odd_same.append(odd_val)
                        else:
                            if level >= 6:
                                fast_cand_odd_diff.append(odd_val)
                            elif level <= 2:
                                slow_cand_odd_diff.append(odd_val)
            
            aggregate[patch_type] = {
                'mono_change_within': float(np.mean(mono_changes_same)) if mono_changes_same else 0,
                'mono_change_across': float(np.mean(mono_changes_diff)) if mono_changes_diff else 0,
                'ent_change_within': float(np.mean(ent_changes_same)) if ent_changes_same else 0,
                'ent_change_across': float(np.mean(ent_changes_diff)) if ent_changes_diff else 0,
                'rank_corr_mean': float(np.mean(rank_corrs)) if rank_corrs else 0,
                'fast_cand_odd_within': float(np.mean(fast_cand_odd_same)) if fast_cand_odd_same else 0,
                'slow_cand_odd_within': float(np.mean(slow_cand_odd_same)) if slow_cand_odd_same else 0,
                'fast_cand_odd_across': float(np.mean(fast_cand_odd_diff)) if fast_cand_odd_diff else 0,
                'slow_cand_odd_across': float(np.mean(slow_cand_odd_diff)) if slow_cand_odd_diff else 0,
                'n_within': len(mono_changes_same),
                'n_across': len(mono_changes_diff),
            }
        
        # Print results
        print(f"\n    === Aggregate Results ===")
        print(f"    {'Patch':>15s} {'MonoΔ_within':>14s} {'MonoΔ_across':>14s} {'EntΔ_within':>14s} {'EntΔ_across':>14s} {'RankCorr':>10s}")
        for pt in patch_types:
            agg = aggregate[pt]
            print(f"    {pt:>15s} {agg['mono_change_within']:>+14.4f} {agg['mono_change_across']:>+14.4f} "
                  f"{agg['ent_change_within']:>+14.4f} {agg['ent_change_across']:>+14.4f} {agg['rank_corr_mean']:>+10.4f}")
        
        print(f"\n    Fast/Slow candidate odd by type relation:")
        print(f"    {'Patch':>15s} {'Fast_within':>13s} {'Fast_across':>13s} {'Slow_within':>13s} {'Slow_across':>13s}")
        for pt in patch_types:
            agg = aggregate[pt]
            print(f"    {pt:>15s} {agg['fast_cand_odd_within']:>+13.4f} {agg['fast_cand_odd_across']:>+13.4f} "
                  f"{agg['slow_cand_odd_within']:>+13.4f} {agg['slow_cand_odd_across']:>+13.4f}")
        
        # Speed-level gradient analysis: does odd effect increase with speed level?
        print(f"\n    Speed-level gradient (full direction patch):")
        if 'full' in aggregate:
            level_grads = defaultdict(list)
            for src_name, tgts in distribution_results.items():
                for tgt_name, tgt_data in tgts.items():
                    if 'full' not in tgt_data or 'error' in tgt_data.get('full', {}):
                        continue
                    cand_effs = tgt_data['full'].get('candidate_effects', {})
                    for cand_name, eff in cand_effs.items():
                        level_grads[eff['speed_level']].append(eff['odd'])
            
            print(f"    {'Level':>6s} {'Mean odd':>10s} {'N':>5s}")
            for level in sorted(level_grads.keys()):
                vals = level_grads[level]
                print(f"    {level:>6d} {np.mean(vals):>+10.4f} {len(vals):>5d}")
            
            # Check if odd effect increases with speed level
            levels_sorted = sorted(level_grads.keys())
            means = [np.mean(level_grads[l]) for l in levels_sorted]
            from scipy.stats import spearmanr
            grad_corr, _ = spearmanr(levels_sorted, means) if len(levels_sorted) >= 3 else (0, 1)
            print(f"    Speed-level gradient correlation: {grad_corr:+.4f}")
        
        # Store layer results
        layer_result = {
            'obj_dirs_norm': {n: float(np.linalg.norm(obj_dirs[n])) for n in obj_names},
            'obj_baselines': obj_baselines,
            'aggregate': aggregate,
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
        for pt in ['full', 'type_only', 'speed_only', 'norm_control']:
            if pt in agg:
                a = agg[pt]
                print(f"  L{li} {pt:>15s}: mono_Δ={a['mono_change_within']:+.4f}/{a['mono_change_across']:+.4f}, "
                      f"ent_Δ={a['ent_change_within']:+.4f}/{a['ent_change_across']:+.4f}")
    
    # Save results
    out_dir = ROOT / "results" / "phase403_multi_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase403.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Deep convert
    import copy
    results_to_save = copy.deepcopy(all_results)
    
    with open(out_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=convert_numpy)
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
    run_phase403(model_name)
