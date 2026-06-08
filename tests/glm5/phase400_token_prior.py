"""
Phase 400: Token Prior Audit + Cross-Category Norm Effect + Multi-Candidate Ranking
====================================================================================

Three sub-experiments to address the three most critical gaps:

A. No-Context Token Prior Test
   - Measure logit distribution with minimal/no context
   - Templates: empty, "The", "It is", "___"
   - Categories: size, moisture, color, speed, temperature
   - Record full candidate rankings
   → Quantifies "default token priority" hypothesis

B. Cross-Category Norm Effect Test
   - Phase 399b protocol extended to moisture, color, speed
   - Compare Even/Odd ratios and norm boost effects across categories
   - Test if NORM_DOM holds for all categories
   → Verifies generality of norm dominance

C. Multi-Candidate Ranking Under Norm Injection
   - Record FULL candidate distribution before/after norm injection
   - Not just target vs competitor, but all 6-8 candidates per category
   - Measure temperature change, rank reordering, distribution shift
   → Shows how norm injection affects entire candidate landscape

Usage:
  python tests/glm5/phase400_token_prior.py qwen3
  python tests/glm5/phase400_token_prior.py deepseek7b
  python tests/glm5/phase400_token_prior.py glm4
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

# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================
CATEGORIES = {
    "size": {
        "objects": {
            "ant":     {"target": "small", "comp": "big", "align": "small"},
            "elephant":{"target": "big",   "comp": "small", "align": "big"},
            "mountain":{"target": "big",   "comp": "small", "align": "big"},
        },
        "candidates": ["tiny", "small", "medium", "large", "big", "huge", "massive", "miniature"],
    },
    "moisture": {
        "objects": {
            "ocean":  {"target": "wet",  "comp": "dry", "align": "wet"},
            "desert": {"target": "dry",  "comp": "wet", "align": "dry"},
            "rainforest":{"target": "wet", "comp": "dry", "align": "wet"},
        },
        "candidates": ["wet", "damp", "moist", "dry", "arid", "humid", "soaked", "parched"],
    },
    "color": {
        "objects": {
            "apple": {"target": "red",  "comp": "green", "align": "red"},
            "sky":   {"target": "blue", "comp": "yellow", "align": "blue"},
            "grass": {"target": "green","comp": "red",   "align": "green"},
        },
        "candidates": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"],
    },
    "speed": {
        "objects": {
            "cheetah": {"target": "fast", "comp": "slow", "align": "fast"},
            "turtle":  {"target": "slow", "comp": "fast", "align": "slow"},
            "rocket":  {"target": "fast", "comp": "slow", "align": "fast"},
        },
        "candidates": ["fast", "slow", "rapid", "sluggish", "quick", "swift", "leisurely", "speedy"],
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

# Minimal context templates for token prior test
PRIOR_TEMPLATES = [
    "",
    "The",
    "It is",
    "The item is",
    "___",
]

# Corrupt prompt templates
CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
    "This item is {attr}.",
    "That item is {attr}.",
]

# Clean prompt templates
FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
    "This {obj} is {attr}.",
    "That {obj} is {attr}.",
]

LAYER_CONFIGS = {
    "qwen3": [4, 28],
    "deepseek7b": [4, 20],
    "glm4": [5, 35],
}

N_RANDOM = 20  # Random orthogonal directions (fewer than 399b for speed)
NORM_BOOST = 0.1  # Fixed norm boost factor


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


# ==============================================================================
# Sub-Experiment A: No-Context Token Prior
# ==============================================================================
def test_token_prior(model, tokenizer, categories_data):
    """Measure model's default logit distribution with minimal context."""
    device = next(model.parameters()).device
    results = {}
    
    # Resolve token IDs for all candidates
    all_tokens = set()
    for cat_data in categories_data.values():
        for tok in cat_data["candidates"]:
            all_tokens.add(tok)
        for obj_data in cat_data["objects"].values():
            all_tokens.add(obj_data["target"])
            all_tokens.add(obj_data["comp"])
    
    token_ids = {}
    for tok in all_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        token_ids[tok] = ids[0] if ids else None
    
    for template in PRIOR_TEMPLATES:
        tmpl_key = template if template else "<empty>"
        results[tmpl_key] = {}
        
        if template == "":
            # Empty: use BOS token only
            bos_id = tokenizer.bos_token_id
            if bos_id is not None:
                input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                attn_mask = torch.tensor([[1]], dtype=torch.long, device=device)
            else:
                # Fallback: use a space
                toks = tokenizer(" ", return_tensors="pt", truncation=True, max_length=64)
                input_ids = toks["input_ids"].to(device)
                attn_mask = toks["attention_mask"].to(device)
        elif template == "___":
            toks = tokenizer("___", return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks["input_ids"].to(device)
            attn_mask = toks["attention_mask"].to(device)
        else:
            toks = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks["input_ids"].to(device)
            attn_mask = toks["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        
        logits = out.logits[0, -1].float().cpu().numpy()
        
        for cat_name, cat_data in categories_data.items():
            cand_logits = {}
            for tok in cat_data["candidates"]:
                tid = token_ids.get(tok)
                if tid is not None:
                    cand_logits[tok] = float(logits[tid])
            
            # Sort by logit
            sorted_cands = sorted(cand_logits.items(), key=lambda x: -x[1])
            
            # Target vs comp gaps per object
            obj_gaps = {}
            for obj_name, obj_data in cat_data["objects"].items():
                t_id = token_ids.get(obj_data["target"])
                c_id = token_ids.get(obj_data["comp"])
                t_logit = float(logits[t_id]) if t_id is not None else 0
                c_logit = float(logits[c_id]) if c_id is not None else 0
                obj_gaps[obj_name] = {
                    "target": obj_data["target"],
                    "comp": obj_data["comp"],
                    "target_logit": t_logit,
                    "comp_logit": c_logit,
                    "gap": t_logit - c_logit,
                }
            
            results[tmpl_key][cat_name] = {
                "candidate_ranking": sorted_cands,
                "obj_gaps": obj_gaps,
            }
    
    return results, token_ids


# ==============================================================================
# Sub-Experiment B: Cross-Category Norm Effect
# ==============================================================================
def test_cross_category_norm(model, tokenizer, layers_list, device, li,
                              categories_data, token_ids, d_model):
    """Test Even/Odd decomposition and norm boost across categories."""
    
    results = {}
    
    for cat_name, cat_data in categories_data.items():
        print(f"\n    === Category: {cat_name} ===")
        cat_results = {}
        
        for obj_name, obj_data in cat_data["objects"].items():
            target = obj_data["target"]
            comp = obj_data["comp"]
            align = obj_data["align"]
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            
            if tid is None or cid is None:
                print(f"      {obj_name}: token not found, skip")
                continue
            
            # Collect activations for direction computation
            captured = {}
            def make_hook(key):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook_fn
            
            handle = layers_list[li].register_forward_hook(make_hook('h'))
            
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
            
            handle.remove()
            
            # Compute direction
            dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
            baseline_diff = float(np.mean(baseline_diffs))
            
            # Generate orthogonal directions
            ortho_dirs = make_orthogonal_directions(dh, N_RANDOM)
            
            # Use first corrupt prompt
            prompt = CORRUPT_FRAMES[0].format(attr=target)
            
            # ---- Test 1: L1 direction Even/Odd ----
            l1_effects = {}
            for alpha in [-1.0, 1.0]:
                scaled = alpha * dh
                delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
                diff_list = []
                for f_idx in range(len(FRAMES)):
                    p = CORRUPT_FRAMES[f_idx].format(attr=target)
                    # Inject
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
                    diff_list.append(d2 - baseline_diffs[f_idx])
                l1_effects[alpha] = float(np.mean(diff_list))
            
            l1_even = (l1_effects[1.0] + l1_effects[-1.0]) / 2
            l1_odd = (l1_effects[1.0] - l1_effects[-1.0]) / 2
            
            # ---- Test 2: Orthogonal direction Even (average over N_RANDOM) ----
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
                        diff_list.append(d2 - baseline_diffs[f_idx])
                    r_effects[alpha] = float(np.mean(diff_list))
                ortho_even_list.append((r_effects[1.0] + r_effects[-1.0]) / 2)
                ortho_odd_list.append((r_effects[1.0] - r_effects[-1.0]) / 2)
            
            avg_ortho_even = float(np.mean(ortho_even_list))
            std_ortho_even = float(np.std(ortho_even_list))
            avg_ortho_odd = float(np.mean(ortho_odd_list))
            
            even_ratio = avg_ortho_even / (l1_even + 1e-10)
            source = "NORM_DOM" if abs(even_ratio) > 0.7 else ("ATTRACTOR_DOM" if abs(even_ratio) < 0.3 else "MIXED")
            
            # ---- Test 3: Pure norm boost ----
            factor = 1.0 + NORM_BOOST
            norm_boost_diffs = []
            for f_idx in range(len(FRAMES)):
                p = CORRUPT_FRAMES[f_idx].format(attr=target)
                def make_norm_hook(f):
                    def hook_fn(module, input, output):
                        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                        hs[0, -1, :] = hs[0, -1, :] * f
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook_fn
                bf16_factor = torch.tensor(factor, dtype=torch.bfloat16, device=device)
                h = layers_list[li].register_forward_hook(make_norm_hook(bf16_factor))
                try:
                    inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out2 = model(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device))
                    d2, _, _ = get_logit_diff(out2.logits[0, -1], tid, cid)
                finally:
                    h.remove()
                norm_boost_diffs.append(d2 - baseline_diffs[f_idx])
            
            norm_boost_delta_diff = float(np.mean(norm_boost_diffs))
            
            print(f"      {obj_name}(align={align}): baseline_gap={baseline_diff:+.3f} "
                  f"L1_even={l1_even:+.4f} L1_odd={l1_odd:+.4f} "
                  f"ortho_even={avg_ortho_even:+.4f}±{std_ortho_even:.4f} "
                  f"ratio={even_ratio:.3f} source={source} "
                  f"norm_boost={norm_boost_delta_diff:+.4f}")
            
            cat_results[obj_name] = {
                "align": align,
                "baseline_gap": baseline_diff,
                "l1_even": l1_even,
                "l1_odd": l1_odd,
                "avg_ortho_even": avg_ortho_even,
                "std_ortho_even": std_ortho_even,
                "avg_ortho_odd": avg_ortho_odd,
                "even_ratio": even_ratio,
                "even_source": source,
                "n_ortho": N_RANDOM,
                "norm_boost_delta_diff": norm_boost_delta_diff,
            }
        
        results[cat_name] = cat_results
    
    return results


# ==============================================================================
# Sub-Experiment C: Multi-Candidate Ranking Under Norm Injection
# ==============================================================================
def test_multi_candidate_ranking(model, tokenizer, layers_list, device, li,
                                  categories_data, token_ids):
    """Record full candidate distribution before and after norm injection."""
    results = {}
    
    for cat_name, cat_data in categories_data.items():
        print(f"\n    === Multi-Candidate: {cat_name} ===")
        cat_results = {}
        
        for obj_name, obj_data in cat_data["objects"].items():
            target = obj_data["target"]
            comp = obj_data["comp"]
            align = obj_data["align"]
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            
            if tid is None or cid is None:
                continue
            
            # Use corrupt prompt
            prompt = CORRUPT_FRAMES[0].format(attr=target)
            candidates = cat_data["candidates"]
            cand_ids = {}
            for tok in candidates:
                cand_ids[tok] = token_ids.get(tok)
            
            # Baseline logits
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            baseline_logits = out.logits[0, -1].float().cpu().numpy()
            
            # Baseline candidate ranking
            baseline_cand = {}
            for tok, c_id in cand_ids.items():
                if c_id is not None:
                    baseline_cand[tok] = float(baseline_logits[c_id])
            baseline_sorted = sorted(baseline_cand.items(), key=lambda x: -x[1])
            
            # Norm boost
            factor = 1.0 + NORM_BOOST
            def make_norm_hook(f):
                def hook_fn(module, input, output):
                    hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                    hs[0, -1, :] = hs[0, -1, :] * f
                    return (hs,) + output[1:] if isinstance(output, tuple) else hs
                return hook_fn
            bf16_factor = torch.tensor(factor, dtype=torch.bfloat16, device=device)
            h = layers_list[li].register_forward_hook(make_norm_hook(bf16_factor))
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out2 = model(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device))
            finally:
                h.remove()
            boosted_logits = out2.logits[0, -1].float().cpu().numpy()
            
            # Boosted candidate ranking
            boosted_cand = {}
            for tok, c_id in cand_ids.items():
                if c_id is not None:
                    boosted_cand[tok] = float(boosted_logits[c_id])
            boosted_sorted = sorted(boosted_cand.items(), key=lambda x: -x[1])
            
            # Compute changes
            cand_changes = {}
            for tok in baseline_cand:
                cand_changes[tok] = {
                    "baseline": baseline_cand[tok],
                    "boosted": boosted_cand[tok],
                    "delta": boosted_cand[tok] - baseline_cand[tok],
                }
            
            # Rank changes
            baseline_rank = {tok: i for i, (tok, _) in enumerate(baseline_sorted)}
            boosted_rank = {tok: i for i, (tok, _) in enumerate(boosted_sorted)}
            rank_changes = {}
            for tok in baseline_rank:
                rank_changes[tok] = baseline_rank[tok] - boosted_rank[tok]  # positive = moved up
            
            # Temperature change: std of logits
            baseline_std = float(np.std(list(baseline_cand.values())))
            boosted_std = float(np.std(list(boosted_cand.values())))
            
            print(f"      {obj_name}(align={align}):")
            print(f"        Baseline: {[(t, f'{v:.2f}') for t, v in baseline_sorted[:5]]}")
            print(f"        Boosted:  {[(t, f'{v:.2f}') for t, v in boosted_sorted[:5]]}")
            print(f"        Std: {baseline_std:.3f} → {boosted_std:.3f} (Δ={boosted_std-baseline_std:+.3f})")
            
            # Key metric: does norm boost push toward default prior or away?
            target_delta = cand_changes.get(target, {}).get("delta", 0)
            comp_delta = cand_changes.get(comp, {}).get("delta", 0)
            
            cat_results[obj_name] = {
                "align": align,
                "target": target,
                "comp": comp,
                "baseline_ranking": baseline_sorted,
                "boosted_ranking": boosted_sorted,
                "cand_changes": cand_changes,
                "rank_changes": rank_changes,
                "baseline_std": baseline_std,
                "boosted_std": boosted_std,
                "target_delta": target_delta,
                "comp_delta": comp_delta,
                "target_comp_delta_diff": target_delta - comp_delta,
            }
        
        results[cat_name] = cat_results
    
    return results


def run_phase400(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 400: Token Prior + Cross-Category + Multi-Candidate ({model_name}) [{timestamp}] ===")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    print(f"  Layers: {layer_indices}")
    print(f"  Categories: {list(CATEGORIES.keys())}")
    print(f"  N_RANDOM: {N_RANDOM}")
    print(f"  NORM_BOOST: {NORM_BOOST}")
    
    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device
    
    all_results = {
        'model': model_name, 'timestamp': timestamp,
        'layer_indices': layer_indices,
        'categories': list(CATEGORIES.keys()),
        'token_prior': {},
        'cross_category_norm': {},
        'multi_candidate': {},
    }
    
    # ========================================================================
    # Sub-Experiment A: Token Prior (no context)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"=== A. No-Context Token Prior ===")
    t0 = time.time()
    prior_results, token_ids = test_token_prior(model, tokenizer, CATEGORIES)
    all_results['token_prior'] = prior_results
    
    # Print summary
    print(f"\n  --- Token Prior Summary ---")
    for tmpl_key in PRIOR_TEMPLATES:
        tmpl_disp = tmpl_key if tmpl_key else "<empty>"
        print(f"\n  Template: '{tmpl_disp}'")
        for cat_name in CATEGORIES:
            cat_prior = prior_results.get(tmpl_key, {}).get(cat_name, {})
            if not cat_prior:
                continue
            obj_gaps = cat_prior.get('obj_gaps', {})
            ranking = cat_prior.get('candidate_ranking', [])
            top3 = [(t, f'{v:.2f}') for t, v in ranking[:3]]
            print(f"    {cat_name}: top3={top3}")
            for obj_name, gap_info in obj_gaps.items():
                print(f"      {obj_name}: {gap_info['target']}({gap_info['target_logit']:.2f}) "
                      f"vs {gap_info['comp']}({gap_info['comp_logit']:.2f}) "
                      f"gap={gap_info['gap']:+.2f}")
    
    print(f"  Token prior done in {time.time()-t0:.0f}s")
    
    # ========================================================================
    # Per-layer experiments
    # ========================================================================
    for li in layer_indices:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        # B. Cross-Category Norm Effect
        print(f"\n  === B. Cross-Category Norm Effect (L{li}) ===")
        cc_result = test_cross_category_norm(
            model, tokenizer, layers_list, device, li,
            CATEGORIES, token_ids, d_model)
        all_results['cross_category_norm'][str(li)] = cc_result
        
        # C. Multi-Candidate Ranking
        print(f"\n  === C. Multi-Candidate Ranking (L{li}) ===")
        mc_result = test_multi_candidate_ranking(
            model, tokenizer, layers_list, device, li,
            CATEGORIES, token_ids)
        all_results['multi_candidate'][str(li)] = mc_result
        
        print(f"\n  L{li} done in {time.time()-t0:.0f}s")
    
    # Save
    out_dir = ROOT / "results" / "phase400_token_prior"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase400.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # ========================================================================
    # Cross-Category Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"=== Cross-Category Summary ({model_name}) ===")
    
    # Token Prior Summary (using "The item is" template as reference)
    ref_tmpl = "The item is"
    print(f"\nToken Prior (template='{ref_tmpl}'):")
    for cat_name in CATEGORIES:
        cat_prior = prior_results.get(ref_tmpl, {}).get(cat_name, {})
        if not cat_prior:
            continue
        obj_gaps = cat_prior.get('obj_gaps', {})
        for obj_name, gap_info in obj_gaps.items():
            align = CATEGORIES[cat_name]["objects"][obj_name]["align"]
            gap = gap_info['gap']
            sign = "OK" if (align == gap_info['target'] and gap > 0) or \
                          (align != gap_info['target'] and gap < 0) else "X"
            print(f"  {cat_name}/{obj_name}: prior gap={gap:+.3f} "
                  f"({gap_info['target']} vs {gap_info['comp']}) {sign}")
    
    # Norm Effect Summary
    print(f"\nNorm Effect (ortho/L1 ratio, n={N_RANDOM}):")
    for li in layer_indices:
        cc = all_results['cross_category_norm'].get(str(li), {})
        for cat_name in CATEGORIES:
            cat_data = cc.get(cat_name, {})
            for obj_name, obj_data in cat_data.items():
                print(f"  L{li} {cat_name}/{obj_name}: "
                      f"ratio={obj_data.get('even_ratio',0):.3f} "
                      f"source={obj_data.get('even_source','?')} "
                      f"norm_boost={obj_data.get('norm_boost_delta_diff',0):+.4f} "
                      f"L1_even={obj_data.get('l1_even',0):+.4f} "
                      f"L1_odd={obj_data.get('l1_odd',0):+.4f}")
    
    # Multi-Candidate Summary
    print(f"\nMulti-Candidate Distribution Shift:")
    for li in layer_indices:
        mc = all_results['multi_candidate'].get(str(li), {})
        for cat_name in CATEGORIES:
            cat_data = mc.get(cat_name, {})
            for obj_name, obj_data in cat_data.items():
                print(f"  L{li} {cat_name}/{obj_name}: "
                      f"std {obj_data.get('baseline_std',0):.3f}→{obj_data.get('boosted_std',0):.3f} "
                      f"target_Δ={obj_data.get('target_delta',0):+.3f} "
                      f"comp_Δ={obj_data.get('comp_delta',0):+.3f} "
                      f"Δdiff={obj_data.get('target_comp_delta_diff',0):+.3f}")
    
    # Key question: does token prior predict norm boost direction?
    print(f"\n=== Token Prior vs Norm Boost Correlation ===")
    for li in layer_indices:
        cc = all_results['cross_category_norm'].get(str(li), {})
        mc = all_results['multi_candidate'].get(str(li), {})
        prior_ref = prior_results.get(ref_tmpl, {})
        
        prior_gaps = []
        norm_boost_deltas = []
        
        for cat_name in CATEGORIES:
            cat_prior = prior_ref.get(cat_name, {})
            cat_cc = cc.get(cat_name, {})
            
            for obj_name in CATEGORIES[cat_name]["objects"]:
                gap_info = cat_prior.get('obj_gaps', {}).get(obj_name, {})
                obj_cc = cat_cc.get(obj_name, {})
                
                if gap_info and obj_cc:
                    prior_gaps.append(gap_info['gap'])
                    norm_boost_deltas.append(obj_cc.get('norm_boost_delta_diff', 0))
        
        if len(prior_gaps) >= 3:
            # Correlation: does norm boost direction align with prior?
            same_sign = sum(1 for p, n in zip(prior_gaps, norm_boost_deltas) 
                          if (p > 0 and n > 0) or (p < 0 and n < 0))
            total = len(prior_gaps)
            corr = np.corrcoef(prior_gaps, norm_boost_deltas)[0, 1] if total >= 3 else 0
            print(f"  L{li}: prior-norm alignment = {same_sign}/{total} "
                  f"({100*same_sign/total:.0f}%), correlation={corr:.3f}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase400(model_name)
