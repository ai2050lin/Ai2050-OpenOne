"""
Phase 395: Denoised L2 with Rich Dataset + Distribution Damage Metrics
======================================================================

Key improvements over Phase 394b:
1. Rich dataset: 6-8 samples per (obj,cat) using diverse frames + value combos
2. Denoised L2: L2_denoised = L1 + lambda * ObjectOffset (shrinkage)
3. Distribution damage: candidate_mean, non_candidate_mean, logit_norm
4. Frame as explicit factor in ANOVA decomposition

Data design:
- 3 categories: moisture (positive control), color (complex), size (comparison)
- 6 objects per category
- Per (obj,cat): 6-8 samples from:
  * 4 frames: "The/An/This/That {obj} is {attr}."
  * 2-3 value combos: e.g., apple-red-blue, apple-red-green (different competitors)
  → Total ~6-8 samples per (obj,cat)

Cross-fitting: 5-fold (train on 80%, test on 20%)
Shrinkage: sweep lambda in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
  lambda=0 → L1_category (pure shared)
  lambda=1 → L2_original (no shrinkage)
  lambda>1 → amplify obj-cat offset (risky)
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
from itertools import product

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model

# ============================================================
# Rich dataset: 6-8 samples per (obj, cat)
# ============================================================
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

# For each (obj, cat), we create multiple value combos
# Each combo: (target, competitor) — the correct value vs the wrong value
# This gives different delta_h directions for the same (obj, cat)
RICH_DATA = {
    "moisture": {
        "objects": {
            "ocean": [("wet","dry"),("wet","arid")],
            "rain":  [("wet","dry"),("wet","arid")],
            "river": [("wet","dry"),("wet","arid")],
            "desert":[("dry","wet"),("dry","moist")],
            "sand":  [("dry","wet"),("dry","moist")],
            "dust":  [("dry","wet"),("dry","moist")],
        },
    },
    "color": {
        "objects": {
            "apple":   [("red","blue"),("red","green")],
            "cherry":  [("red","blue"),("red","green")],
            "sky":     [("blue","red"),("blue","green")],
            "ocean_c": [("blue","red"),("blue","green")],  # ocean_c to avoid clash
            "snow":    [("white","black"),("white","gray")],
            "grass":   [("green","blue"),("green","red")],
        },
    },
    "size": {
        "objects": {
            "elephant":[("big","small"),("large","tiny")],
            "mountain":[("big","small"),("large","tiny")],
            "whale":   [("big","small"),("large","tiny")],
            "ant":     [("small","big"),("tiny","large")],
            "grain":   [("small","big"),("tiny","large")],
            "pin":     [("small","big"),("tiny","large")],
        },
    },
}

# Map display names back to actual object names for prompts
DISPLAY_TO_PROMPT = {
    "ocean_c": "ocean",
}


def build_extended_pairs():
    """Build all (obj, target, comp, cat, frame_idx, value_idx) tuples.
    
    For each (obj, cat): 
      - 2 value combos × 4 frames = 8 samples per (obj,cat)
    """
    pairs = []
    for cat, cat_data in RICH_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            prompt_obj = DISPLAY_TO_PROMPT.get(obj_name, obj_name)
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name,           # group key
                        'prompt_obj': prompt_obj,  # for template
                        'target': target,
                        'comp': comp,
                        'cat': cat,
                        'frame_idx': f_idx,
                        'value_idx': v_idx,
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


def classify_mechanism(td, cd):
    if td > 0 and cd < 0:
        return "IDEAL"
    elif td > 0 and cd > 0:
        if td > cd:
            return "DOM_BOOST"
        else:
            return "BOOST_C"
    elif td < 0 and cd > 0:
        return "REVERSED"
    elif td < 0 and cd < 0:
        if abs(td) > abs(cd):
            return "SUPP_T"
        else:
            return "SUPP_C"
    else:
        return "MIXED"


def get_logit_stats(logits_tensor, target_id, comp_id, top_k=20):
    """Compute target/competitor logits + distribution damage metrics."""
    logits = logits_tensor.float().cpu().numpy()
    
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    
    # Distribution damage metrics
    logit_norm = float(np.linalg.norm(logits))
    logit_mean = float(np.mean(logits))
    logit_std = float(np.std(logits))
    
    # Top-k stats
    sorted_ids = np.argsort(logits)[::-1]
    top_k_logits = logits[sorted_ids[:top_k]]
    top_k_mean = float(np.mean(top_k_logits))
    
    # Exclude target and competitor
    exclude_ids = set()
    if target_id is not None:
        exclude_ids.add(target_id)
    if comp_id is not None:
        exclude_ids.add(comp_id)
    
    other_logits = np.delete(logits, list(exclude_ids))
    other_mean = float(np.mean(other_logits))
    other_std = float(np.std(other_logits))
    
    return {
        't_logit': t_logit,
        'c_logit': c_logit,
        'logit_norm': logit_norm,
        'logit_mean': logit_mean,
        'logit_std': logit_std,
        'top_k_mean': top_k_mean,
        'other_mean': other_mean,
        'other_std': other_std,
    }


def test_direction_at_layer_with_stats(model, tokenizer, layers_list, device, li, 
                                         delta_np, prompt, tid, cid):
    """Patch with direction and return full logit statistics."""
    delta = torch.tensor(delta_np, dtype=torch.bfloat16, device=device)

    def make_add_hook(delta_vec):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
            else:
                hs = output.clone()
            hs[0, -1, :] += delta_vec
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        return hook_fn

    handle = layers_list[li].register_forward_hook(make_add_hook(delta))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device))
        stats = get_logit_stats(out.logits[0, -1], tid, cid)
    finally:
        handle.remove()
    return stats


def run_phase395(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 395: Denoised L2 ({model_name}) [{timestamp}] ===")

    # Focus layers
    LAYER_CONFIGS = {
        "qwen3": [4, 12, 20],
        "deepseek7b": [4, 12, 20],
        "glm4": [4, 20, 30],
    }
    layer_indices = LAYER_CONFIGS.get(model_name, [4, 20])

    # Build pairs
    extended_pairs = build_extended_pairs()
    N = len(extended_pairs)
    
    # Count samples per (obj, cat)
    oc_groups = defaultdict(list)
    for i, p in enumerate(extended_pairs):
        oc_groups[(p['obj'], p['cat'])].append(i)
    
    min_samples = min(len(v) for v in oc_groups.values())
    max_samples = max(len(v) for v in oc_groups.values())
    print(f"  Total pairs: {N}")
    print(f"  (obj,cat) groups: {len(oc_groups)}, samples/group: {min_samples}-{max_samples}")
    print(f"  Categories: {list(RICH_DATA.keys())}")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Build prompts and get token IDs
    prompts_clean = []
    prompts_corrupt = []
    for p in extended_pairs:
        tpl = FRAMES[p['frame_idx']]
        prompts_clean.append(tpl.format(obj=p['prompt_obj'], attr=p['target']))
        ctpl = CORRUPT_FRAMES[p['frame_idx']]
        prompts_corrupt.append(ctpl.format(attr=p['target']))

    # Get token IDs for all target/comp words
    token_ids = {}
    for cat_data in RICH_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None

    # Shrinkage lambdas to test
    LAMBDAS = [0.0, 0.2, 0.5, 1.0, 2.0]

    results = {
        'model': model_name, 'timestamp': timestamp,
        'n_pairs': N, 'layers': layer_indices,
        'n_oc_groups': len(oc_groups),
        'samples_per_group': f"{min_samples}-{max_samples}",
        'lambdas': LAMBDAS,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li}/{n_layers-1} ---")
        print(f"{'='*70}")

        # ---- Collect activations ----
        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_hook('h'))
        h_clean = np.zeros((N, d_model), dtype=np.float32)
        h_corrupt = np.zeros((N, d_model), dtype=np.float32)

        for i in range(N):
            captured.clear()
            inputs = tokenizer(prompts_clean[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_clean[i] = captured['h'][0, -1].numpy()

            captured.clear()
            inputs = tokenizer(prompts_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_corrupt[i] = captured['h'][0, -1].numpy()

            if (i+1) % 30 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()
        print(f"  Activations collected ({time.time()-t0_layer:.0f}s)")

        # ---- Compute delta_h ----
        dh = h_clean - h_corrupt
        obj_labels = [p['obj'] for p in extended_pairs]
        cat_labels = [p['cat'] for p in extended_pairs]
        frame_labels = [p['frame_idx'] for p in extended_pairs]
        value_labels = [p['value_idx'] for p in extended_pairs]

        # ---- ANOVA decomposition: Cat + ObjCat + Frame + Value ----
        mu = dh.mean(axis=0)

        # Category effect
        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        cat_centroids = np.zeros((len(unique_cats), d_model))
        cat_counts = np.zeros(len(unique_cats))
        for i, c in enumerate(cat_labels):
            cat_centroids[cat_to_idx[c]] += dh[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                cat_centroids[j] /= cat_counts[j]
        A_cat = np.zeros_like(dh)
        for i, c in enumerate(cat_labels):
            A_cat[i] = cat_centroids[cat_to_idx[c]] - mu

        # Object-category effect (residual after category)
        dh_resid_cat = dh - mu - A_cat
        obj_cat_keys = [(obj_labels[i], cat_labels[i]) for i in range(N)]
        unique_obj_cats = sorted(set(obj_cat_keys))
        oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
        oc_centroids = np.zeros((len(unique_obj_cats), d_model))
        oc_counts = np.zeros(len(unique_obj_cats))
        for i, oc in enumerate(obj_cat_keys):
            oc_centroids[oc_to_idx[oc]] += dh_resid_cat[i]
            oc_counts[oc_to_idx[oc]] += 1
        for j in range(len(unique_obj_cats)):
            if oc_counts[j] > 0:
                oc_centroids[j] /= oc_counts[j]
        A_obj_cat = np.zeros_like(dh)
        for i, oc in enumerate(obj_cat_keys):
            A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

        # Frame effect (residual after cat + obj-cat)
        dh_resid_oc = dh - mu - A_cat - A_obj_cat
        unique_frames = sorted(set(frame_labels))
        frame_centroids = np.zeros((len(unique_frames), d_model))
        frame_counts = np.zeros(len(unique_frames))
        for i, f in enumerate(frame_labels):
            frame_centroids[f] += dh_resid_oc[i]
            frame_counts[f] += 1
        for j in range(len(unique_frames)):
            if frame_counts[j] > 0:
                frame_centroids[j] /= frame_counts[j]
        A_frame = np.zeros_like(dh)
        for i, f in enumerate(frame_labels):
            A_frame[i] = frame_centroids[f]

        print(f"  ANOVA: Cat={len(unique_cats)}, ObjCat={len(unique_obj_cats)}, "
              f"Frame={len(unique_frames)}")

        # ---- Build cross-fitted directions ----
        # For each (obj,cat) group, do LOPO: estimate from N-1, test on 1
        # Then apply shrinkage: L2_denoised = mu + A_cat + lambda * A_obj_cat_crossfit
        
        # Pre-compute cross-fitted A_obj_cat (LOPO within each group)
        A_obj_cat_crossfit = np.zeros_like(dh)
        for oc_key in unique_obj_cats:
            group_indices = oc_groups[oc_key]
            for test_i in group_indices:
                train_indices = [j for j in group_indices if j != test_i]
                if len(train_indices) == 0:
                    # Only 1 sample in group → fallback to L1
                    A_obj_cat_crossfit[test_i] = np.zeros(d_model)
                else:
                    # Average residual from training samples
                    train_resids = np.mean([dh_resid_cat[j] for j in train_indices], axis=0)
                    A_obj_cat_crossfit[test_i] = train_resids - oc_centroids[oc_to_idx[oc_key]] + oc_centroids[oc_to_idx[oc_key]]
                    # Actually, cross-fitted = mean of training residuals
                    A_obj_cat_crossfit[test_i] = np.mean([dh_resid_cat[j] for j in train_indices], axis=0)

        # ---- Baseline logits with distribution damage ----
        print(f"  Computing baseline logits + distribution stats...")
        baseline_stats = []
        for i, p in enumerate(extended_pairs):
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])
            inputs = tokenizer(prompts_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            stats = get_logit_stats(out.logits[0, -1], tid, cid)
            baseline_stats.append(stats)
            if (i+1) % 30 == 0:
                print(f"    Baseline: {i+1}/{N}")
        print(f"  Baseline done ({time.time()-t0_layer:.0f}s)")

        # ---- Test shrinkage versions ----
        # For each lambda, compute direction and test
        # Also test L1_category and L2_original for reference
        
        test_versions = {}
        # L1: category direction only
        test_versions["L1_category"] = lambda i, lam=None: mu + A_cat[i]
        # L2_original: no cross-fitting, no shrinkage
        test_versions["L2_original"] = lambda i, lam=None: mu + A_cat[i] + A_obj_cat[i]
        # L2_crossfit: cross-fitted, no shrinkage
        test_versions["L2_crossfit"] = lambda i, lam=None: mu + A_cat[i] + A_obj_cat_crossfit[i]
        
        # Shrinkage versions: L2_denoised(lambda) = mu + A_cat + lambda * A_obj_cat_crossfit
        for lam in LAMBDAS:
            if lam == 0.0:
                continue  # same as L1
            if lam == 1.0:
                continue  # same as L2_crossfit
            lam_val = lam
            test_versions[f"L2_cf_lam{lam}"] = lambda i, lam=lam_val: mu + A_cat[i] + lam * A_obj_cat_crossfit[i]

        version_results = {}
        for ver_name, direction_fn in test_versions.items():
            print(f"\n  Testing {ver_name}...")
            t0_v = time.time()

            patched_stats = []
            for i in range(N):
                p = extended_pairs[i]
                tid = token_ids.get(p['target'])
                cid = token_ids.get(p['comp'])
                if tid is None or cid is None:
                    patched_stats.append(None)
                    continue

                delta_np = direction_fn(i)
                stats = test_direction_at_layer_with_stats(
                    model, tokenizer, layers_list, device, li,
                    delta_np, prompts_corrupt[i], tid, cid
                )
                patched_stats.append(stats)

                if (i+1) % 30 == 0:
                    print(f"    {ver_name}: {i+1}/{N} ({time.time()-t0_v:.0f}s)")

            # Aggregate results
            td_arr = np.zeros(N)
            cd_arr = np.zeros(N)
            logit_norm_delta = np.zeros(N)
            other_mean_delta = np.zeros(N)
            valid = 0
            for i in range(N):
                if patched_stats[i] is None:
                    continue
                bs = baseline_stats[i]
                ps = patched_stats[i]
                td_arr[i] = ps['t_logit'] - bs['t_logit']
                cd_arr[i] = ps['c_logit'] - bs['c_logit']
                logit_norm_delta[i] = ps['logit_norm'] - bs['logit_norm']
                other_mean_delta[i] = ps['other_mean'] - bs['other_mean']
                valid += 1

            add_arr = td_arr - cd_arr

            # Category breakdown
            cat_breakdown = {}
            for cat in unique_cats:
                cat_mask = np.array(cat_labels) == cat
                if cat_mask.sum() == 0:
                    continue
                cat_td = td_arr[cat_mask]
                cat_cd = cd_arr[cat_mask]
                cat_add = add_arr[cat_mask]
                cat_norm_delta = logit_norm_delta[cat_mask]
                cat_other_delta = other_mean_delta[cat_mask]
                mech = classify_mechanism(float(np.mean(cat_td)), float(np.mean(cat_cd)))
                
                # Distribution damage score: how much did non-target/comp logits change?
                # High |other_mean_delta| relative to |target_delta| = damage
                damage_ratio = 0.0
                if abs(np.mean(cat_td)) > 1e-6:
                    damage_ratio = abs(np.mean(cat_other_delta)) / abs(np.mean(cat_td))
                
                cat_breakdown[cat] = {
                    'add_mean': float(np.mean(cat_add)),
                    'target_delta_mean': float(np.mean(cat_td)),
                    'competitor_delta_mean': float(np.mean(cat_cd)),
                    'mechanism': mech,
                    'n': int(cat_mask.sum()),
                    'logit_norm_delta_mean': float(np.mean(cat_norm_delta)),
                    'other_mean_delta': float(np.mean(cat_other_delta)),
                    'damage_ratio': float(damage_ratio),
                }

            overall_mech = classify_mechanism(float(np.mean(td_arr)), float(np.mean(cd_arr)))
            ideal_count = sum(1 for c in cat_breakdown.values() if c['mechanism'] == 'IDEAL')

            version_results[ver_name] = {
                'add_mean': float(np.mean(add_arr)),
                'target_delta_mean': float(np.mean(td_arr)),
                'competitor_delta_mean': float(np.mean(cd_arr)),
                'mechanism': overall_mech,
                'ideal_count': ideal_count,
                'total_cats': len(cat_breakdown),
                'category_breakdown': cat_breakdown,
                'logit_norm_delta_mean': float(np.mean(logit_norm_delta)),
                'other_mean_delta': float(np.mean(other_mean_delta)),
            }

            print(f"    {ver_name}: add={version_results[ver_name]['add_mean']:+.4f}, "
                  f"T={version_results[ver_name]['target_delta_mean']:+.4f}, "
                  f"C={version_results[ver_name]['competitor_delta_mean']:+.4f}, "
                  f"IDEAL={ideal_count}/{len(cat_breakdown)}, "
                  f"mech={overall_mech}, "
                  f"norm_d={version_results[ver_name]['logit_norm_delta_mean']:+.2f}, "
                  f"other_d={version_results[ver_name]['other_mean_delta']:+.4f} "
                  f"({time.time()-t0_v:.0f}s)")

        # ---- Category comparison: L1 vs L2_original vs L2_crossfit ----
        print(f"\n  === Category: L1 vs L2_orig vs L2_crossfit ===")
        for cat in RICH_DATA.keys():
            l1 = version_results.get('L1_category', {}).get('category_breakdown', {}).get(cat, {})
            orig = version_results.get('L2_original', {}).get('category_breakdown', {}).get(cat, {})
            cf = version_results.get('L2_crossfit', {}).get('category_breakdown', {}).get(cat, {})
            
            l1_mech = l1.get('mechanism', '?')
            orig_mech = orig.get('mechanism', '?')
            cf_mech = cf.get('mechanism', '?')
            
            leak_flag = " ← LEAK!" if orig_mech == "IDEAL" and cf_mech != "IDEAL" else ""
            
            l1_td = l1.get('target_delta_mean', 0)
            l1_cd = l1.get('competitor_delta_mean', 0)
            orig_td = orig.get('target_delta_mean', 0)
            orig_cd = orig.get('competitor_delta_mean', 0)
            cf_td = cf.get('target_delta_mean', 0)
            cf_cd = cf.get('competitor_delta_mean', 0)
            cf_dmg = cf.get('damage_ratio', 0)
            l1_dmg = l1.get('damage_ratio', 0)
            
            print(f"    {cat:12s}: L1={l1_mech}(T{l1_td:+.3f}C{l1_cd:+.3f}dmg{l1_dmg:.2f}) "
                  f"L2_orig={orig_mech}(T{orig_td:+.3f}C{orig_cd:+.3f}) "
                  f"L2_cf={cf_mech}(T{cf_td:+.3f}C{cf_cd:+.3f}dmg{cf_dmg:.2f}){leak_flag}")

        # ---- Shrinkage sweep: find best lambda per category ----
        print(f"\n  === Shrinkage Sweep (per category) ===")
        shrinkage_versions = [v for v in test_versions.keys() if v.startswith("L2_cf_lam")]
        # Add L1 and L2_crossfit for reference
        shrinkage_versions = ["L1_category", "L2_crossfit"] + sorted(shrinkage_versions)
        
        for cat in RICH_DATA.keys():
            print(f"\n    {cat}:")
            best_lambda = None
            best_score = -999
            for ver in shrinkage_versions:
                cb = version_results.get(ver, {}).get('category_breakdown', {}).get(cat, {})
                if not cb:
                    continue
                mech = cb.get('mechanism', '?')
                td = cb.get('target_delta_mean', 0)
                cd = cb.get('competitor_delta_mean', 0)
                dmg = cb.get('damage_ratio', 0)
                # Score: IDEAL=2, DOM_BOOST=1, others=0; tiebreak by add-damage
                score = 0
                if mech == "IDEAL":
                    score = 2 + td - 0.1 * dmg
                elif mech == "DOM_BOOST":
                    score = 1 + td - 0.1 * dmg
                else:
                    score = td - 0.5 * abs(cd) - 0.2 * dmg
                
                print(f"      {ver:18s}: {mech:10s} T={td:+.4f} C={cd:+.4f} dmg={dmg:.2f} score={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_lambda = ver
            print(f"      → Best: {best_lambda}")

        results['per_layer'][str(li)] = version_results
        elapsed = time.time() - t0_layer
        print(f"\n  L{li} done in {elapsed:.0f}s")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"--- Phase 395 Summary ---")
    print(f"{'='*70}")
    for li in layer_indices:
        vr = results['per_layer'][str(li)]
        print(f"\n  Layer {li}:")
        for ver in ["L1_category", "L2_original", "L2_crossfit"]:
            if ver in vr:
                r = vr[ver]
                print(f"    {ver:14s}: add={r['add_mean']:+.4f}, T={r['target_delta_mean']:+.4f}, "
                      f"C={r['competitor_delta_mean']:+.4f}, IDEAL={r['ideal_count']}/{r['total_cats']}, "
                      f"mech={r['mechanism']}, norm_d={r['logit_norm_delta_mean']:+.2f}, "
                      f"other_d={r['other_mean_delta']:+.4f}")
        # Show shrinkage results
        for ver in sorted(vr.keys()):
            if ver.startswith("L2_cf_lam"):
                r = vr[ver]
                print(f"    {ver:18s}: add={r['add_mean']:+.4f}, T={r['target_delta_mean']:+.4f}, "
                      f"C={r['competitor_delta_mean']:+.4f}, IDEAL={r['ideal_count']}/{r['total_cats']}, "
                      f"mech={r['mechanism']}")

    # Save
    out_dir = ROOT / "results" / "phase395_denoised_l2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase395.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase395(model_name)
