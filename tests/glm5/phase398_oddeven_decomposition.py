"""
Phase 398: Odd-Even Decomposition of Direction Injection Effects
=================================================================

Core question: Is the effect of direction injection dominated by
linear (direction-sign-dependent) or nonlinear (sign-independent) components?

Method:
- For same direction d, test multiple strengths:
  alpha in {-2, -1, -0.5, 0, 0.5, 1, 2}
- Record Effect(alpha * d) for each alpha
- Decompose:
  Odd(alpha)  = [Effect(alpha*d) - Effect(-alpha*d)] / 2  (linear component)
  Even(alpha) = [Effect(alpha*d) + Effect(-alpha*d)] / 2  (nonlinear component)

Interpretation:
- If Even >> Odd: direction sign doesn't matter, nonlinear interpreter dominates
- If Odd >> Even: linear direction effect dominates
- If Odd ≈ Even: mixed regime

Test objects (size category - most dramatic from Phase 397):
- Small-compatible: ant, grain
- Big-compatible: elephant, mountain

Plus moisture and color for cross-category validation:
- Wet-compatible: ocean
- Dry-compatible: desert
- Red-compatible: apple
- Blue-compatible: sky

Layer config (4 layers per model):
- Qwen3: [4, 16, 28, 35]
- DS7B: [4, 12, 20, 27]
- GLM4: [5, 15, 30, 39]
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

# Comprehensive test data covering 3 categories
TEST_DATA = {
    "size": {
        "objects": {
            "ant":     [("small","big"),("tiny","large")],
            "grain":   [("small","big"),("tiny","large")],
            "elephant":[("big","small"),("large","tiny")],
            "mountain":[("big","small"),("large","tiny")],
        },
    },
    "moisture": {
        "objects": {
            "ocean": [("wet","dry"),("wet","arid")],
            "desert":[("dry","wet"),("dry","moist")],
        },
    },
    "color": {
        "objects": {
            "apple": [("red","blue"),("red","green")],
            "sky":   [("blue","red"),("blue","green")],
        },
    },
}

VALUE_ALIGNMENT = {
    "ant": "small", "grain": "small",
    "elephant": "big", "mountain": "big",
    "ocean": "wet", "desert": "dry",
    "apple": "red", "sky": "blue",
}

# Alpha values for odd-even decomposition
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

# Positive-only alphas for computing Odd/Even pairs
POS_ALPHAS = [0.5, 1.0, 2.0]

LAYER_CONFIGS = {
    "qwen3": [4, 16, 28, 35],
    "deepseek7b": [4, 12, 20, 27],
    "glm4": [5, 15, 30, 39],
}


def build_pairs():
    pairs = []
    for cat, cat_data in TEST_DATA.items():
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
    """Get target_logit - competitor_logit (the key metric)"""
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return t_logit - c_logit, t_logit, c_logit


def test_direction_alpha(model, tokenizer, layers_list, device, li,
                         delta_np, alpha, prompt, tid, cid):
    """Inject alpha * delta at layer li, return logit changes"""
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


def classify_mechanism(td, cd):
    if td > 0 and cd < 0: return "IDEAL"
    elif td > 0 and cd > 0: return "DOM_BOOST" if td > cd else "BOOST_C"
    elif td < 0 and cd > 0: return "REVERSED"
    elif td < 0 and cd < 0: return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    else: return "MIXED"


def run_phase398(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 398: Odd-Even Decomposition ({model_name}) [{timestamp}] ===")

    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    pairs = build_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs across 3 categories")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Layers: {layer_indices}")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device

    # Resolve token IDs
    token_ids = {}
    for cat_data in TEST_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'alphas': ALPHAS,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")

        # === Step 1: Collect activations ===
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

            if (i+1) % 16 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()

        # Compute delta_h (correct - corrupt)
        dh_correct = h_correct - h_correct_corrupt

        # === Step 2: Compute L1 category directions ===
        cat_labels = [p['cat'] for p in pairs]
        obj_labels = [p['obj'] for p in pairs]

        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        cat_centroids = np.zeros((len(unique_cats), d_model))
        cat_counts = np.zeros(len(unique_cats))
        for i, c in enumerate(cat_labels):
            cat_centroids[cat_to_idx[c]] += dh_correct[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                cat_centroids[j] /= cat_counts[j]

        # Per-object directions
        obj_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            obj_groups[p['obj']].append(i)

        per_obj_dirs = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs[obj] = np.mean(dh_correct[indices], axis=0)

        # === Step 3: Odd-Even Decomposition ===
        print(f"\n  === Odd-Even Decomposition ===")

        # For each object, test L1_cat direction at multiple alphas
        # Focus on correct-corrupt prompts
        oddeven_results = {}

        unique_objs = sorted(set(obj_labels))
        total_tests = len(unique_objs) * len(ALPHAS) * len(set(obj_labels))  # rough
        test_count = 0

        for obj in unique_objs:
            obj_mask = np.array(obj_labels) == obj
            obj_indices = np.where(obj_mask)[0]
            if len(obj_indices) == 0:
                continue

            cat = cat_labels[obj_indices[0]]
            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])
            val_align = VALUE_ALIGNMENT.get(obj, "?")

            # Use L1 category direction (same as Phase 397)
            dir_l1 = cat_centroids[cat_to_idx[cat]]

            # Also use per-object direction
            dir_pobj = per_obj_dirs[obj]

            for dir_name, direction in [("L1", dir_l1), ("POBJ", dir_pobj)]:
                alpha_effects = {}  # alpha -> (delta_diff, delta_t, delta_c)

                for alpha in ALPHAS:
                    td_list = []
                    cd_list = []
                    diff_list = []

                    for idx in obj_indices:
                        p = pairs[idx]
                        ctpl = CORRUPT_FRAMES[p['frame_idx']]
                        prompt = ctpl.format(attr=p['target'])

                        diff, tl, cl = test_direction_alpha(
                            model, tokenizer, layers_list, device, li,
                            direction, alpha, prompt, tid, cid)

                        delta_diff = diff - baseline_diffs[idx]
                        delta_t = tl - baseline_t[idx]
                        delta_c = cl - baseline_c[idx]

                        diff_list.append(delta_diff)
                        td_list.append(delta_t)
                        cd_list.append(delta_c)

                    alpha_effects[alpha] = {
                        'delta_diff': float(np.mean(diff_list)),
                        'delta_t': float(np.mean(td_list)),
                        'delta_c': float(np.mean(cd_list)),
                    }

                    test_count += 1

                # Compute Odd and Even components for positive alphas
                odd_components = {}
                even_components = {}

                for alpha in POS_ALPHAS:
                    eff_pos = alpha_effects[alpha]
                    eff_neg = alpha_effects[-alpha]

                    # For delta_diff (target - competitor)
                    odd_diff = (eff_pos['delta_diff'] - eff_neg['delta_diff']) / 2
                    even_diff = (eff_pos['delta_diff'] + eff_neg['delta_diff']) / 2

                    # For delta_t (target logit change)
                    odd_t = (eff_pos['delta_t'] - eff_neg['delta_t']) / 2
                    even_t = (eff_pos['delta_t'] + eff_neg['delta_t']) / 2

                    # For delta_c (competitor logit change)
                    odd_c = (eff_pos['delta_c'] - eff_neg['delta_c']) / 2
                    even_c = (eff_pos['delta_c'] + eff_neg['delta_c']) / 2

                    odd_components[alpha] = {
                        'diff': odd_diff, 't': odd_t, 'c': odd_c
                    }
                    even_components[alpha] = {
                        'diff': even_diff, 't': even_t, 'c': even_c
                    }

                # Classify dominant regime
                # Use alpha=1.0 as representative
                odd_rep = abs(odd_components[1.0]['diff'])
                even_rep = abs(even_components[1.0]['diff'])
                total_rep = odd_rep + even_rep + 1e-10
                odd_frac = odd_rep / total_rep
                even_frac = even_rep / total_rep

                if even_frac > 0.75:
                    regime = "NONLINEAR_DOM"
                elif odd_frac > 0.75:
                    regime = "LINEAR_DOM"
                else:
                    regime = "MIXED"

                key = f"{obj}_{dir_name}"
                oddeven_results[key] = {
                    'cat': cat,
                    'value_align': val_align,
                    'direction': dir_name,
                    'alpha_effects': alpha_effects,
                    'odd_components': odd_components,
                    'even_components': even_components,
                    'odd_frac_alpha1': float(odd_frac),
                    'even_frac_alpha1': float(even_frac),
                    'regime': regime,
                }

                # Print summary for this object+direction
                print(f"\n  {obj:10s} {dir_name} (align={val_align:5s}): {regime}")
                print(f"    alpha  delta_diff   odd_diff   even_diff  odd%    even%")
                for alpha in POS_ALPHAS:
                    ae = alpha_effects[alpha]
                    oc = odd_components[alpha]
                    ec = even_components[alpha]
                    odd_pct = abs(oc['diff']) / (abs(oc['diff']) + abs(ec['diff']) + 1e-10) * 100
                    even_pct = abs(ec['diff']) / (abs(oc['diff']) + abs(ec['diff']) + 1e-10) * 100
                    print(f"    {alpha:+5.1f}  {ae['delta_diff']:+8.4f}   {oc['diff']:+8.4f}   {ec['diff']:+8.4f}  {odd_pct:5.1f}%  {even_pct:5.1f}%")

                # Print full alpha sweep
                print(f"    Full alpha sweep (delta_diff):")
                for alpha in ALPHAS:
                    ae = alpha_effects[alpha]
                    print(f"      alpha={alpha:+5.1f}: delta_diff={ae['delta_diff']:+.4f}, "
                          f"delta_t={ae['delta_t']:+.4f}, delta_c={ae['delta_c']:+.4f}")

            if (unique_objs.index(obj) + 1) % 4 == 0:
                elapsed = time.time() - t0_layer
                print(f"  --- Progress: {unique_objs.index(obj)+1}/{len(unique_objs)} objects, "
                      f"{test_count} tests, {elapsed:.0f}s ---")

        # === Step 4: Cross-object summary ===
        print(f"\n  === Layer {li} Summary ===")

        for cat in unique_cats:
            cat_objs = [o for o in unique_objs if cat_labels[obj_groups[o][0]] == cat]
            print(f"\n  Category: {cat}")
            print(f"  {'Object':10s} {'Dir':5s} {'Align':6s} {'Odd%':>6s} {'Even%':>6s} {'Regime':15s}")
            for obj in cat_objs:
                for dir_name in ["L1", "POBJ"]:
                    key = f"{obj}_{dir_name}"
                    r = oddeven_results[key]
                    print(f"  {obj:10s} {dir_name:5s} {r['value_align']:6s} "
                          f"{r['odd_frac_alpha1']*100:5.1f}% {r['even_frac_alpha1']*100:5.1f}% "
                          f"{r['regime']:15s}")

        # Aggregate by value alignment
        print(f"\n  By value alignment:")
        for va in sorted(set(VALUE_ALIGNMENT.values())):
            va_keys = [k for k, v in oddeven_results.items() if v['value_align'] == va and v['direction'] == 'L1']
            if va_keys:
                avg_odd = np.mean([oddeven_results[k]['odd_frac_alpha1'] for k in va_keys])
                avg_even = np.mean([oddeven_results[k]['even_frac_alpha1'] for k in va_keys])
                regime = "NONLINEAR_DOM" if avg_even > 0.75 else "LINEAR_DOM" if avg_odd > 0.75 else "MIXED"
                print(f"    {va:6s} (n={len(va_keys)}): Odd={avg_odd*100:.1f}% Even={avg_even*100:.1f}% -> {regime}")

        results['per_layer'][str(li)] = oddeven_results
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")

    # Save
    out_dir = ROOT / "results" / "phase398_oddeven"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase398.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # === Cross-layer summary ===
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Odd-Even Summary ({model_name}) ===")
    for li in layer_indices:
        lr = results['per_layer'].get(str(li), {})
        l1_keys = [k for k in lr.keys() if k.endswith('_L1')]
        if not l1_keys:
            continue
        avg_odd = np.mean([lr[k]['odd_frac_alpha1'] for k in l1_keys])
        avg_even = np.mean([lr[k]['even_frac_alpha1'] for k in l1_keys])
        regime = "NONLINEAR_DOM" if avg_even > 0.75 else "LINEAR_DOM" if avg_odd > 0.75 else "MIXED"

        # Per-object detail
        obj_details = []
        for k in sorted(l1_keys):
            obj = k.replace('_L1', '')
            r = lr[k]
            obj_details.append(f"{obj}({r['value_align']}:{r['regime'][:3]})")

        print(f"  L{li}: Odd={avg_odd*100:.1f}% Even={avg_even*100:.1f}% -> {regime}")
        print(f"       {', '.join(obj_details)}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase398(model_name)
