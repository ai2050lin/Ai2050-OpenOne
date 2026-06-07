"""
Phase 394b: Extended Data for Cross-Fitted L2 Validation
========================================================
Phase 394 revealed: all (obj,cat) groups have only 1 sample,
so LOPO cross-fitting degrades to L1_category.

Phase 394b adds more samples per (obj,cat) by creating
multiple sentence frames for the same object-category pair.

For example, apple-color now has 3 variants:
  "The apple is red." vs "The apple is blue."
  "An apple is red." vs "An apple is blue."
  "This apple is red." vs "This apple is blue."

This creates 3 samples per (obj,cat), enabling real LOPO.

Focus: GLM4 L20 (where L2 was strongest) and DS7B L4/L20 (moisture IDEAL).
Only test the most important categories:
  - GLM4: size (was strongest L2 IDEAL), speed (stable IDEAL)
  - DS7B: moisture (stable IDEAL), size (was REVERSED)

Test: L1_category vs L2_original vs L2_crossfit(3-fold)
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

# ============================================================
# Extended data: 3 sentence frames per pair
# ============================================================
TEMPLATES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
    "This {obj} is {attr}.",
]

# Focus on key categories with most pairs
FOCUS_DATA = {
    "size": [
        ("elephant","big","small"),("mountain","big","small"),("whale","big","small"),
        ("building","big","small"),("planet","big","small"),("continent","big","small"),
        ("ant","small","big"),("grain","small","big"),("pin","small","big"),
        ("dot","small","big"),("speck","small","big"),
    ],
    "speed": [
        ("cheetah","fast","slow"),("rocket","fast","slow"),("lightning","fast","slow"),
        ("falcon","fast","slow"),("bullet","fast","slow"),("jet","fast","slow"),
        ("turtle","slow","fast"),("snail","slow","fast"),("sloth","slow","fast"),
    ],
    "moisture": [
        ("ocean","wet","dry"),("rain","wet","dry"),("river","wet","dry"),
        ("lake","wet","dry"),("swamp","wet","dry"),("sponge","wet","dry"),
        ("desert","dry","wet"),("sand","dry","wet"),("dust","dry","wet"),
    ],
    "color": [
        ("apple","red","blue"),("cherry","red","blue"),("tomato","red","blue"),
        ("sky","blue","red"),("ocean","blue","red"),("sapphire","blue","red"),
        ("snow","white","black"),("cloud","white","black"),
        ("grass","green","blue"),("emerald","green","blue"),
    ],
}

CORRUPT_TEMPLATES = [
    "The item is {attr}.",
    "An item is {attr}.",
    "This item is {attr}.",
]


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


def test_direction_at_layer(model, tokenizer, layers_list, device, li, delta_np, prompt, tid, cid):
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
        t_logit = float(out.logits[0, -1, tid]) if tid is not None else 0.0
        c_logit = float(out.logits[0, -1, cid]) if cid is not None else 0.0
    finally:
        handle.remove()
    return t_logit, c_logit


def run_phase394b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 394b: Extended Data Cross-Fit ({model_name}) [{timestamp}] ===")

    # Focus layers
    LAYER_CONFIGS = {
        "qwen3": [4, 20],
        "deepseek7b": [4, 20],
        "glm4": [4, 20],
    }
    layer_indices = LAYER_CONFIGS[model_name]

    # Focus categories per model
    FOCUS_CATS = {
        "qwen3": ["size", "moisture", "speed", "color"],
        "deepseek7b": ["moisture", "size", "speed", "color"],
        "glm4": ["size", "speed", "moisture", "color"],
    }
    focus_cats = FOCUS_CATS[model_name]

    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Build extended pairs: each (obj, cat, target, comp) × 3 templates
    # Each sample has a template_idx to identify its group
    extended_pairs = []  # (obj, target, comp, cat, template_idx)
    for cat in focus_cats:
        for obj, target, comp in FOCUS_DATA[cat]:
            for t_idx in range(3):
                extended_pairs.append((obj, target, comp, cat, t_idx))

    N = len(extended_pairs)
    print(f"  Extended pairs: {N} ({len(focus_cats)} cats × {3} templates)")

    # Build prompts
    prompts_clean = []
    prompts_corrupt = []
    for obj, target, comp, cat, t_idx in extended_pairs:
        tpl = TEMPLATES[t_idx]
        prompts_clean.append(tpl.format(obj=obj, attr=target))
        ctpl = CORRUPT_TEMPLATES[t_idx]
        prompts_corrupt.append(ctpl.format(attr=target))

    # Get token IDs
    token_ids = {}
    for cat in focus_cats:
        for obj, target, comp in FOCUS_DATA[cat]:
            for tok in [target, comp]:
                if tok not in token_ids:
                    ids = tokenizer.encode(tok, add_special_tokens=False)
                    token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'n_pairs': N, 'layers': layer_indices,
        'focus_cats': focus_cats,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li}/{n_layers-1} ---")
        print(f"{'='*70}")

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
                print(f"  Activation collection: {i+1}/{N}")

        handle.remove()
        print(f"  Activations collected ({time.time()-t0_layer:.0f}s)")

        # Compute delta_h
        dh = h_clean - h_corrupt
        obj_labels = [p[0] for p in extended_pairs]
        cat_labels = [p[3] for p in extended_pairs]

        # ANOVA decomposition
        mu = dh.mean(axis=0)

        # Object effect
        unique_objects = sorted(set(obj_labels))
        obj_to_idx = {o: i for i, o in enumerate(unique_objects)}
        obj_centroids = np.zeros((len(unique_objects), d_model))
        obj_counts = np.zeros(len(unique_objects))
        for i, o in enumerate(obj_labels):
            obj_centroids[obj_to_idx[o]] += dh[i]
            obj_counts[obj_to_idx[o]] += 1
        for j in range(len(unique_objects)):
            if obj_counts[j] > 0:
                obj_centroids[j] /= obj_counts[j]
        I_comp = np.zeros_like(dh)
        for i, o in enumerate(obj_labels):
            I_comp[i] = obj_centroids[obj_to_idx[o]] - mu

        # Category effect (residual after object)
        dh_resid_I = dh - mu - I_comp
        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        cat_centroids = np.zeros((len(unique_cats), d_model))
        cat_counts = np.zeros(len(unique_cats))
        for i, c in enumerate(cat_labels):
            cat_centroids[cat_to_idx[c]] += dh_resid_I[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                cat_centroids[j] /= cat_counts[j]
        A_comp = np.zeros_like(dh)
        for i, c in enumerate(cat_labels):
            A_comp[i] = cat_centroids[cat_to_idx[c]]

        # Object-category effect (residual after object+category)
        dh_resid_IC = dh - mu - I_comp - A_comp
        obj_cat_keys = [(obj_labels[i], cat_labels[i]) for i in range(N)]
        unique_obj_cats = sorted(set(obj_cat_keys))
        oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
        oc_centroids = np.zeros((len(unique_obj_cats), d_model))
        oc_counts = np.zeros(len(unique_obj_cats))
        for i, oc in enumerate(obj_cat_keys):
            oc_centroids[oc_to_idx[oc]] += dh_resid_IC[i]
            oc_counts[oc_to_idx[oc]] += 1
        for j in range(len(unique_obj_cats)):
            if oc_counts[j] > 0:
                oc_centroids[j] /= oc_counts[j]
        A_obj_cat = np.zeros_like(dh)
        for i, oc in enumerate(obj_cat_keys):
            A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

        # Group info
        oc_groups = defaultdict(list)
        for i, oc in enumerate(obj_cat_keys):
            oc_groups[oc].append(i)

        multi_sample = sum(1 for c in oc_counts if c > 1)
        three_plus = sum(1 for c in oc_counts if c >= 3)
        print(f"  (obj,cat) groups: {len(unique_obj_cats)} total, "
              f"{multi_sample} with >1 sample, {three_plus} with >=3 samples")

        # Cross-fitted L2 (real LOPO now possible!)
        l2_crossfit = np.zeros((N, d_model), dtype=np.float32)
        for i in range(N):
            oc = obj_cat_keys[i]
            group_indices = oc_groups[oc]
            if len(group_indices) <= 1:
                l2_crossfit[i] = mu + I_comp[i] + A_comp[i]  # fallback to L1
            else:
                other_indices = [j for j in group_indices if j != i]
                l2_resid_avg = np.mean([dh[j] - mu - I_comp[j] - A_comp[j] for j in other_indices], axis=0)
                l2_crossfit[i] = mu + I_comp[i] + A_comp[i] + l2_resid_avg

        # Baseline logits
        print(f"  Computing baseline logits...")
        baseline_target = np.zeros(N)
        baseline_competitor = np.zeros(N)
        for i, (obj, target, comp, cat, t_idx) in enumerate(extended_pairs):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            inputs = tokenizer(prompts_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            if tid is not None:
                baseline_target[i] = float(out.logits[0, -1, tid])
            if cid is not None:
                baseline_competitor[i] = float(out.logits[0, -1, cid])
        print(f"  Baseline done ({time.time()-t0_layer:.0f}s)")

        # Test 3 versions
        test_versions = {
            "L1_category": lambda i: A_comp[i],
            "L2_original": lambda i: A_comp[i] + A_obj_cat[i],
            "L2_crossfit": lambda i: l2_crossfit[i],
        }

        version_results = {}
        for ver_name, direction_fn in test_versions.items():
            print(f"\n  Testing {ver_name}...")
            t0_v = time.time()

            patched_target = np.zeros(N)
            patched_competitor = np.zeros(N)

            for i in range(N):
                obj, target, comp, cat, t_idx = extended_pairs[i]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                if tid is None or cid is None:
                    continue

                delta_np = direction_fn(i)
                pt, pc = test_direction_at_layer(
                    model, tokenizer, layers_list, device, li,
                    delta_np, prompts_corrupt[i], tid, cid
                )
                patched_target[i] = pt
                patched_competitor[i] = pc

                if (i+1) % 30 == 0:
                    print(f"    {ver_name}: {i+1}/{N} ({time.time()-t0_v:.0f}s)")

            td = patched_target - baseline_target
            cd = patched_competitor - baseline_competitor
            add = td - cd

            cat_breakdown = {}
            for cat in unique_cats:
                cat_mask = np.array(cat_labels) == cat
                if cat_mask.sum() == 0:
                    continue
                cat_td = td[cat_mask]
                cat_cd = cd[cat_mask]
                cat_add = add[cat_mask]
                mech = classify_mechanism(float(np.mean(cat_td)), float(np.mean(cat_cd)))
                cat_breakdown[cat] = {
                    'add_mean': float(np.mean(cat_add)),
                    'target_delta_mean': float(np.mean(cat_td)),
                    'competitor_delta_mean': float(np.mean(cat_cd)),
                    'mechanism': mech,
                    'n': int(cat_mask.sum()),
                }

            overall_mech = classify_mechanism(float(np.mean(td)), float(np.mean(cd)))
            ideal_count = sum(1 for c in cat_breakdown.values() if c['mechanism'] == 'IDEAL')

            version_results[ver_name] = {
                'add_mean': float(np.mean(add)),
                'target_delta_mean': float(np.mean(td)),
                'competitor_delta_mean': float(np.mean(cd)),
                'mechanism': overall_mech,
                'ideal_count': ideal_count,
                'total_cats': len(cat_breakdown),
                'category_breakdown': cat_breakdown,
            }

            print(f"    {ver_name}: add={version_results[ver_name]['add_mean']:+.4f}, "
                  f"T={version_results[ver_name]['target_delta_mean']:+.4f}, "
                  f"C={version_results[ver_name]['competitor_delta_mean']:+.4f}, "
                  f"IDEAL={ideal_count}/{len(cat_breakdown)}, "
                  f"mech={overall_mech} ({time.time()-t0_v:.0f}s)")

        # Category comparison
        print(f"\n  === Category: L2_original vs L2_crossfit ===")
        for cat in focus_cats:
            orig = version_results['L2_original']['category_breakdown'].get(cat, {})
            cf = version_results['L2_crossfit']['category_breakdown'].get(cat, {})
            l1 = version_results['L1_category']['category_breakdown'].get(cat, {})
            orig_mech = orig.get('mechanism', '?')
            cf_mech = cf.get('mechanism', '?')
            l1_mech = l1.get('mechanism', '?')
            orig_td = orig.get('target_delta_mean', 0)
            orig_cd = orig.get('competitor_delta_mean', 0)
            cf_td = cf.get('target_delta_mean', 0)
            cf_cd = cf.get('competitor_delta_mean', 0)
            changed = " ← LEAK" if orig_mech == "IDEAL" and cf_mech != "IDEAL" else ""
            if orig_mech != cf_mech:
                changed += " CHANGED" if not changed else ""
            print(f"    {cat:12s}: L2_orig={orig_mech}(T{orig_td:+.3f}C{orig_cd:+.3f}) "
                  f"L2_cf={cf_mech}(T{cf_td:+.3f}C{cf_cd:+.3f}) "
                  f"L1={l1_mech}{changed}")

        # Per-(obj,cat) groups with >=3 samples: detailed analysis
        print(f"\n  === Per-(obj,cat) groups (>=3 samples) ===")
        for oc in sorted(oc_groups.keys()):
            indices = oc_groups[oc]
            if len(indices) < 3:
                continue
            obj, cat = oc
            print(f"    {obj}-{cat}: {len(indices)} samples")

        results['per_layer'][str(li)] = version_results
        elapsed = time.time() - t0_layer
        print(f"\n  L{li} done in {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"--- Phase 394b Summary ---")
    print(f"{'='*70}")
    for li in layer_indices:
        vr = results['per_layer'][str(li)]
        print(f"\n  Layer {li}:")
        for ver in ["L1_category", "L2_original", "L2_crossfit"]:
            r = vr[ver]
            print(f"    {ver:12s}: add={r['add_mean']:+.4f}, T={r['target_delta_mean']:+.4f}, "
                  f"C={r['competitor_delta_mean']:+.4f}, IDEAL={r['ideal_count']}/{r['total_cats']}, "
                  f"mech={r['mechanism']}")

    # Save
    out_dir = ROOT / "results" / "phase394b_extended_crossfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase394b.json"
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
    run_phase394b(model_name)
