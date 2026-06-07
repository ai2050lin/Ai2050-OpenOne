"""
Phase 395b: Confirmation Round — Add speed + intermediate layers
================================================================
Round 1 key findings to confirm:
1. DS7B L12: L2_crossfit=2/3 IDEAL (moisture+size) ← confirm with speed
2. Qwen3 L4: color=IDEAL ← confirm and check layer evolution
3. damage_ratio as quality metric ← verify across more data

This round adds:
- speed category (4 objects × 2 values × 4 frames = 32 more samples)
- intermediate layers: L8, L16 for DS7B/Qwen3
- Total: 4 categories × ~176 samples
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

# Full data with speed added
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
            "ocean_c": [("blue","red"),("blue","green")],
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
    "speed": {
        "objects": {
            "cheetah": [("fast","slow"),("quick","sluggish")],
            "rocket":  [("fast","slow"),("quick","sluggish")],
            "falcon":  [("fast","slow"),("quick","sluggish")],
            "turtle":  [("slow","fast"),("sluggish","quick")],
            "snail":   [("slow","fast"),("sluggish","quick")],
            "sloth":   [("slow","fast"),("sluggish","quick")],
        },
    },
}

DISPLAY_TO_PROMPT = {"ocean_c": "ocean"}


def build_extended_pairs():
    pairs = []
    for cat, cat_data in RICH_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            prompt_obj = DISPLAY_TO_PROMPT.get(obj_name, obj_name)
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name,
                        'prompt_obj': prompt_obj,
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
    if td > 0 and cd < 0: return "IDEAL"
    elif td > 0 and cd > 0: return "DOM_BOOST" if td > cd else "BOOST_C"
    elif td < 0 and cd > 0: return "REVERSED"
    elif td < 0 and cd < 0: return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    else: return "MIXED"


def get_logit_stats(logits_tensor, target_id, comp_id):
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    exclude_ids = set()
    if target_id is not None: exclude_ids.add(target_id)
    if comp_id is not None: exclude_ids.add(comp_id)
    other_logits = np.delete(logits, list(exclude_ids))
    return {
        't_logit': t_logit, 'c_logit': c_logit,
        'logit_norm': float(np.linalg.norm(logits)),
        'other_mean': float(np.mean(other_logits)),
    }


def test_direction_with_stats(model, tokenizer, layers_list, device, li,
                                delta_np, prompt, tid, cid):
    delta = torch.tensor(delta_np, dtype=torch.bfloat16, device=device)
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
        stats = get_logit_stats(out.logits[0, -1], tid, cid)
    finally:
        handle.remove()
    return stats


def run_phase395b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 395b: Confirmation ({model_name}) [{timestamp}] ===")

    # More layers for evolution tracking
    LAYER_CONFIGS = {
        "qwen3": [4, 8, 12, 16, 20],
        "deepseek7b": [4, 8, 12, 16, 20],
        "glm4": [4, 10, 20, 30],
    }
    layer_indices = LAYER_CONFIGS.get(model_name, [4, 20])

    extended_pairs = build_extended_pairs()
    N = len(extended_pairs)
    
    oc_groups = defaultdict(list)
    for i, p in enumerate(extended_pairs):
        oc_groups[(p['obj'], p['cat'])].append(i)
    
    min_s = min(len(v) for v in oc_groups.values())
    max_s = max(len(v) for v in oc_groups.values())
    print(f"  Total: {N} pairs, {len(oc_groups)} groups, {min_s}-{max_s} samples/group")

    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  n_layers={n_layers}, d_model={d_model}")

    prompts_clean = []
    prompts_corrupt = []
    for p in extended_pairs:
        tpl = FRAMES[p['frame_idx']]
        prompts_clean.append(tpl.format(obj=p['prompt_obj'], attr=p['target']))
        ctpl = CORRUPT_FRAMES[p['frame_idx']]
        prompts_corrupt.append(ctpl.format(attr=p['target']))

    token_ids = {}
    for cat_data in RICH_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None

    LAMBDAS = [0.0, 0.5, 1.0]

    results = {
        'model': model_name, 'timestamp': timestamp,
        'n_pairs': N, 'layers': layer_indices,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li}/{n_layers-1} ---")

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

            if (i+1) % 40 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()

        # ANOVA
        dh = h_clean - h_corrupt
        obj_labels = [p['obj'] for p in extended_pairs]
        cat_labels = [p['cat'] for p in extended_pairs]
        mu = dh.mean(axis=0)

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

        # Cross-fitted A_obj_cat (LOPO)
        A_obj_cat_cf = np.zeros_like(dh)
        for oc_key in unique_obj_cats:
            group_indices = oc_groups[oc_key]
            for test_i in group_indices:
                train_indices = [j for j in group_indices if j != test_i]
                if len(train_indices) == 0:
                    A_obj_cat_cf[test_i] = np.zeros(d_model)
                else:
                    A_obj_cat_cf[test_i] = np.mean([dh_resid_cat[j] for j in train_indices], axis=0)

        # Baseline
        print(f"  Baseline logits...")
        baseline_stats = []
        for i, p in enumerate(extended_pairs):
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])
            inputs = tokenizer(prompts_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            baseline_stats.append(get_logit_stats(out.logits[0, -1], tid, cid))
            if (i+1) % 40 == 0:
                print(f"    {i+1}/{N}")

        # Test versions (only key ones for speed)
        test_versions = {}
        test_versions["L1_category"] = lambda i: mu + A_cat[i]
        test_versions["L2_original"] = lambda i: mu + A_cat[i] + A_obj_cat[i]
        test_versions["L2_crossfit"] = lambda i: mu + A_cat[i] + A_obj_cat_cf[i]
        for lam in LAMBDAS:
            if lam in [0.0, 1.0]: continue
            lam_val = lam
            test_versions[f"L2_cf_lam{lam}"] = lambda i, l=lam_val: mu + A_cat[i] + l * A_obj_cat_cf[i]

        version_results = {}
        for ver_name, direction_fn in test_versions.items():
            t0_v = time.time()
            td_arr = np.zeros(N)
            cd_arr = np.zeros(N)
            norm_delta = np.zeros(N)
            other_delta = np.zeros(N)

            for i in range(N):
                p = extended_pairs[i]
                tid = token_ids.get(p['target'])
                cid = token_ids.get(p['comp'])
                if tid is None or cid is None:
                    continue
                delta_np = direction_fn(i)
                stats = test_direction_with_stats(
                    model, tokenizer, layers_list, device, li,
                    delta_np, prompts_corrupt[i], tid, cid)
                bs = baseline_stats[i]
                td_arr[i] = stats['t_logit'] - bs['t_logit']
                cd_arr[i] = stats['c_logit'] - bs['c_logit']
                norm_delta[i] = stats['logit_norm'] - bs['logit_norm']
                other_delta[i] = stats['other_mean'] - bs['other_mean']

                if (i+1) % 40 == 0:
                    print(f"    {ver_name}: {i+1}/{N} ({time.time()-t0_v:.0f}s)")

            add_arr = td_arr - cd_arr
            cat_breakdown = {}
            for cat in unique_cats:
                cat_mask = np.array(cat_labels) == cat
                if cat_mask.sum() == 0: continue
                cat_td = td_arr[cat_mask]
                cat_cd = cd_arr[cat_mask]
                cat_add = add_arr[cat_mask]
                cat_other = other_delta[cat_mask]
                mech = classify_mechanism(float(np.mean(cat_td)), float(np.mean(cat_cd)))
                dmg_ratio = 0.0
                if abs(np.mean(cat_td)) > 1e-6:
                    dmg_ratio = abs(np.mean(cat_other)) / abs(np.mean(cat_td))
                cat_breakdown[cat] = {
                    'add_mean': float(np.mean(cat_add)),
                    'target_delta_mean': float(np.mean(cat_td)),
                    'competitor_delta_mean': float(np.mean(cat_cd)),
                    'mechanism': mech,
                    'n': int(cat_mask.sum()),
                    'damage_ratio': float(dmg_ratio),
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
                'logit_norm_delta_mean': float(np.mean(norm_delta)),
                'other_mean_delta': float(np.mean(other_delta)),
            }
            print(f"    {ver_name}: add={version_results[ver_name]['add_mean']:+.4f}, "
                  f"T={version_results[ver_name]['target_delta_mean']:+.4f}, "
                  f"C={version_results[ver_name]['competitor_delta_mean']:+.4f}, "
                  f"IDEAL={ideal_count}/{len(cat_breakdown)}, mech={overall_mech} ({time.time()-t0_v:.0f}s)")

        results['per_layer'][str(li)] = version_results
        print(f"  L{li} done in {time.time()-t0_layer:.0f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"--- Phase 395b Summary ---")
    for li in layer_indices:
        vr = results['per_layer'][str(li)]
        print(f"\n  Layer {li}:")
        for ver in ["L1_category", "L2_original", "L2_crossfit"]:
            if ver in vr:
                r = vr[ver]
                cats_str = " | ".join([f"{c}={d['mechanism']}" for c,d in r['category_breakdown'].items()])
                print(f"    {ver:14s}: IDEAL={r['ideal_count']}/{r['total_cats']} | {cats_str}")

    out_dir = ROOT / "results" / "phase395b_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase395b.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase395b(model_name)
