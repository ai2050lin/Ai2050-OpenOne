"""
Phase 399b: Round 2 Confirmation — Norm Amplification Verification
===================================================================

Critical findings from Round 1 to verify:

1. GLM4 RMSNorm amplifies delta norm by 7-19x (others compress)
2. DS7B RMSNorm destroys sign for ant (cos=0.17)
3. GLM4 Even/Odd ratio through RMSNorm is 100-1000x

Round 2 tests:

A. Pure norm injection: multiply residual by (1+epsilon) instead of adding a direction
   - If pure norm boost gives same Even effect → norm is truly the mechanism
   - This is the most direct test of "norm amplification"

B. More orthogonal directions (30 instead of 10) for better statistics
   - Focus on GLM4 L5 and DS7B L4 where results were most surprising

C. Baseline preference → Even direction correlation
   - Measure: does the sign of Even correlate with baseline logit gap?
   - If yes → norm amplification truly amplifies existing preference

D. RMSNorm amplification verification for GLM4
   - Test at more layers (L5, L10, L15, L20, L25, L30, L35)
   - Is norm_ratio > 1 at ALL layers or just specific ones?

Usage:
  python tests/glm5/phase399b_norm_verify.py qwen3
  python tests/glm5/phase399b_norm_verify.py deepseek7b
  python tests/glm5/phase399b_norm_verify.py glm4
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

CANDIDATE_TOKENS = {
    "size": ["tiny", "small", "medium", "large", "big", "huge", "massive", "miniature"],
}

LAYER_CONFIGS = {
    "qwen3": [4, 16, 28],
    "deepseek7b": [4, 12, 20],
    "glm4": [5, 15, 25, 35],
}

N_RANDOM = 30  # More directions for better statistics

# Norm boost factors for pure norm injection
NORM_BOOST_FACTORS = [0.05, 0.1, 0.2, 0.5]


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
    for name in ['input_layernorm', 'ln_1', 'ln_before_attn']:
        if hasattr(layer, name):
            return getattr(layer, name), name
    for name, mod in layer.named_modules():
        if 'input_layernorm' in name or 'ln_1' in name:
            return mod, name
    return None, None


def test_direction_injection(model, tokenizer, layers_list, device, li,
                              delta_np, alpha, prompt, tid, cid):
    """Inject alpha * delta at layer li, return logit diff."""
    scaled = alpha * delta_np
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    handle = layers_list[li].register_forward_hook(make_add_hook(delta))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device))
        logit_diff, t_logit, c_logit = get_logit_diff(out.logits[0, -1], tid, cid)
    finally:
        handle.remove()
    return logit_diff, t_logit, c_logit


def test_norm_boost(model, tokenizer, layers_list, device, li,
                     boost_factor, prompt, tid, cid):
    """Multiply residual by (1 + boost_factor) at layer li."""
    factor = 1.0 + boost_factor
    def make_norm_hook(f):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] = hs[0, -1, :] * f
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn
    bf16_factor = torch.tensor(factor, dtype=torch.bfloat16, device=device)
    handle = layers_list[li].register_forward_hook(make_norm_hook(bf16_factor))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device))
        logit_diff, t_logit, c_logit = get_logit_diff(out.logits[0, -1], tid, cid)
    finally:
        handle.remove()
    return logit_diff, t_logit, c_logit


def rmsnorm_sign_test(model, tokenizer, layers_list, device, li, direction, prompt):
    """Test RMSNorm sign preservation (same as Phase 399 but streamlined)."""
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


def run_phase399b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 399b: Norm Amplification Verification ({model_name}) [{timestamp}] ===")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    pairs = build_pairs()
    N = len(pairs)
    print(f"  Layers: {layer_indices}")
    print(f"  N_RANDOM: {N_RANDOM}")
    print(f"  Norm boost factors: {NORM_BOOST_FACTORS}")
    
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # Resolve token IDs
    token_ids = {}
    for cat_data in SIZE_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None
    for cat, tokens in CANDIDATE_TOKENS.items():
        for tok in tokens:
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
        
        # Collect activations
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
        dh_correct = h_correct - h_correct_corrupt
        
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
            ctpl = CORRUPT_FRAMES[p0['frame_idx']]
            prompt = ctpl.format(attr=p0['target'])
            
            print(f"\n  --- {obj} (align={val_align}) ---")
            
            obj_result = {'value_align': val_align, 'baseline_gap': float(baseline_diffs[obj_indices[0]])}
            
            # ============================================================
            # Test A: Pure norm boost (multiply by 1+epsilon)
            # ============================================================
            print(f"\n  === A. Pure Norm Boost ===")
            norm_boost_results = {}
            for bf in NORM_BOOST_FACTORS:
                diff_list = []
                td_list = []
                cd_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl_p = CORRUPT_FRAMES[p['frame_idx']]
                    prompt_p = ctpl_p.format(attr=p['target'])
                    diff, tl, cl = test_norm_boost(
                        model, tokenizer, layers_list, device, li,
                        bf, prompt_p, tid, cid)
                    diff_list.append(diff - baseline_diffs[idx])
                    td_list.append(tl - baseline_t[idx])
                    cd_list.append(cl - baseline_c[idx])
                
                avg_diff = float(np.mean(diff_list))
                avg_td = float(np.mean(td_list))
                avg_cd = float(np.mean(cd_list))
                norm_boost_results[str(bf)] = {
                    'delta_diff': avg_diff,
                    'delta_t': avg_td,
                    'delta_c': avg_cd,
                }
                print(f"    boost={bf:+.2f}: Δdiff={avg_diff:+.4f} Δt={avg_td:+.4f} Δc={avg_cd:+.4f}")
            
            obj_result['norm_boost'] = norm_boost_results
            
            # ============================================================
            # Test B: Orthogonal directions with more samples
            # ============================================================
            print(f"\n  === B. Orthogonal Directions ({N_RANDOM} samples) ===")
            ortho_dirs = make_orthogonal_directions(dir_l1, N_RANDOM)
            
            ortho_even_list = []
            ortho_odd_list = []
            ortho_diff_list = []
            
            for r_idx, ortho_dir in enumerate(ortho_dirs):
                r_effects = {}
                for alpha in [-1.0, 1.0]:
                    diff_list = []
                    td_list = []
                    cd_list = []
                    for idx in obj_indices:
                        p = pairs[idx]
                        ctpl_p = CORRUPT_FRAMES[p['frame_idx']]
                        prompt_p = ctpl_p.format(attr=p['target'])
                        diff, tl, cl = test_direction_injection(
                            model, tokenizer, layers_list, device, li,
                            ortho_dir, alpha, prompt_p, tid, cid)
                        diff_list.append(diff - baseline_diffs[idx])
                        td_list.append(tl - baseline_t[idx])
                        cd_list.append(cl - baseline_c[idx])
                    r_effects[alpha] = {
                        'delta_diff': float(np.mean(diff_list)),
                        'delta_t': float(np.mean(td_list)),
                        'delta_c': float(np.mean(cd_list)),
                    }
                
                ep = r_effects[1.0]
                en = r_effects[-1.0]
                r_even = (ep['delta_diff'] + en['delta_diff']) / 2
                r_odd = (ep['delta_diff'] - en['delta_diff']) / 2
                
                ortho_even_list.append(r_even)
                ortho_odd_list.append(r_odd)
                ortho_diff_list.append(ep['delta_diff'])
            
            # Also test L1 direction for comparison
            l1_effects = {}
            for alpha in [-1.0, 1.0]:
                diff_list = []
                td_list = []
                cd_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl_p = CORRUPT_FRAMES[p['frame_idx']]
                    prompt_p = ctpl_p.format(attr=p['target'])
                    diff, tl, cl = test_direction_injection(
                        model, tokenizer, layers_list, device, li,
                        dir_l1, alpha, prompt_p, tid, cid)
                    diff_list.append(diff - baseline_diffs[idx])
                    td_list.append(tl - baseline_t[idx])
                    cd_list.append(cl - baseline_c[idx])
                l1_effects[alpha] = {
                    'delta_diff': float(np.mean(diff_list)),
                    'delta_t': float(np.mean(td_list)),
                    'delta_c': float(np.mean(cd_list)),
                }
            
            l1_even = (l1_effects[1.0]['delta_diff'] + l1_effects[-1.0]['delta_diff']) / 2
            l1_odd = (l1_effects[1.0]['delta_diff'] - l1_effects[-1.0]['delta_diff']) / 2
            
            avg_ortho_even = float(np.mean(ortho_even_list))
            std_ortho_even = float(np.std(ortho_even_list))
            avg_ortho_odd = float(np.mean(ortho_odd_list))
            std_ortho_odd = float(np.std(ortho_odd_list))
            
            even_ratio = avg_ortho_even / (l1_even + 1e-10)
            
            print(f"    L1 Even={l1_even:+.4f} L1 Odd={l1_odd:+.4f}")
            print(f"    Ortho Even: avg={avg_ortho_even:+.4f} std={std_ortho_even:.4f} (n={N_RANDOM})")
            print(f"    Ortho Odd:  avg={avg_ortho_odd:+.4f} std={std_ortho_odd:.4f}")
            print(f"    Even ratio (ortho/L1) = {even_ratio:.3f}")
            
            if abs(even_ratio) > 0.7:
                source = "NORM_DOM"
            elif abs(even_ratio) < 0.3:
                source = "ATTRACTOR_DOM"
            else:
                source = "MIXED"
            print(f"    → Even source: {source}")
            
            obj_result['ortho_stats'] = {
                'l1_even': l1_even,
                'l1_odd': l1_odd,
                'avg_ortho_even': avg_ortho_even,
                'std_ortho_even': std_ortho_even,
                'avg_ortho_odd': avg_ortho_odd,
                'std_ortho_odd': std_ortho_odd,
                'even_ratio': even_ratio,
                'even_source': source,
                'n_samples': N_RANDOM,
            }
            
            # ============================================================
            # Test C: RMSNorm sign preservation
            # ============================================================
            print(f"\n  === C. RMSNorm Sign Preservation ===")
            rms_l1 = rmsnorm_sign_test(model, tokenizer, layers_list, device, li, dir_l1, prompt)
            if rms_l1:
                print(f"    cos_avg={rms_l1['avg_cos_preserved']:.4f} "
                      f"norm_ratio={rms_l1['norm_ratio']:.4f} "
                      f"even_ratio={rms_l1['even_ratio']:.4f} "
                      f"odd_ratio={rms_l1['odd_ratio']:.4f} "
                      f"module={rms_l1['norm_module']}")
            else:
                print(f"    RMSNorm test failed")
            
            rms_ortho = rmsnorm_sign_test(model, tokenizer, layers_list, device, li, ortho_dirs[0], prompt)
            if rms_ortho:
                print(f"    Ortho: cos_avg={rms_ortho['avg_cos_preserved']:.4f} "
                      f"norm_ratio={rms_ortho['norm_ratio']:.4f}")
            else:
                print(f"    Ortho: RMSNorm test failed")
            
            obj_result['rmsnorm_l1'] = rms_l1
            obj_result['rmsnorm_ortho'] = rms_ortho
            
            layer_result[obj] = obj_result
        
        # Layer summary
        print(f"\n  === Layer {li} Summary ===")
        print(f"  {'Object':10s} {'Baseline':>8s} {'L1 Even':>9s} {'Ortho Even':>11s} "
              f"{'Even/L1':>8s} {'Source':>14s} {'RMSNorm cos':>12s} {'norm_ratio':>11s}")
        for obj in ["ant", "elephant", "mountain"]:
            r = layer_result.get(obj, {})
            if not r:
                continue
            bg = r.get('baseline_gap', 0)
            os = r.get('ortho_stats', {})
            rms = r.get('rmsnorm_l1') or {}
            cos = rms.get('avg_cos_preserved')
            nr = rms.get('norm_ratio')
            cos_s = f"{cos:.4f}" if isinstance(cos, (int, float)) else str(cos or 'N/A')
            nr_s = f"{nr:.4f}" if isinstance(nr, (int, float)) else str(nr or 'N/A')
            print(f"  {obj:10s} {bg:+8.3f} {os.get('l1_even',0):+9.4f} "
                  f"{os.get('avg_ortho_even',0):+11.4f} {os.get('even_ratio',0):8.3f} "
                  f"{os.get('even_source','?'):>14s} "
                  f"{cos_s:>12s} {nr_s:>11s}")
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0:.0f}s")
    
    # Save
    out_dir = ROOT / "results" / "phase399b_norm_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase399b.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    
    print(f"\nNorm Boost Effects:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            nb = r.get('norm_boost', {})
            if nb:
                # Show the effect at boost=0.1
                nb01 = nb.get('0.1', {})
                print(f"  L{li} {obj}: boost=0.1 → Δdiff={nb01.get('delta_diff',0):+.4f}")
    
    print(f"\nEven Source (ortho/L1 ratio, n={N_RANDOM}):")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            os = r.get('ortho_stats', {})
            print(f"  L{li} {obj}: ratio={os.get('even_ratio',0):.3f} "
                  f"source={os.get('even_source','?')} "
                  f"ortho_even={os.get('avg_ortho_even',0):+.4f}±{os.get('std_ortho_even',0):.4f}")
    
    print(f"\nRMSNorm Behavior:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            rms = r.get('rmsnorm_l1') or {}
            cos = rms.get('avg_cos_preserved', 'N/A')
            nr = rms.get('norm_ratio', 'N/A')
            er = rms.get('even_ratio', 'N/A')
            or_ = rms.get('odd_ratio', 'N/A')
            cos_s = f"{cos:.4f}" if isinstance(cos, float) else cos
            nr_s = f"{nr:.4f}" if isinstance(nr, float) else nr
            er_s = f"{er:.2f}" if isinstance(er, float) else er
            or_s = f"{or_:.2f}" if isinstance(or_, float) else or_
            print(f"  L{li} {obj}: cos={cos_s} norm_ratio={nr_s} even/odd={er_s}/{or_s}")
    
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase399b(model_name)
