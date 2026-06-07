"""
Phase 393: Conditional Centroid Hierarchy + T/C Decomposition
=============================================================
Core prediction: More conditional centroid → more IDEAL (T↑C↓) → more selective.

Test 4 levels of conditioning:
  Level 0 (global):   single centroid for ALL categories
  Level 1 (category): per-category centroid (7 directions)
  Level 2 (obj-cat):  per object-category centroid (~140 directions)
  Level 3 (obj-cat-val): per object-category-value direction (individual pair direction)

At each level, measure:
  - add_effect (gap_delta = target_delta - competitor_delta)
  - target_delta (how much compatible value changes)
  - competitor_delta (how much incompatible value changes)
  - IDEAL ratio: % of categories showing T↑C↓

Key method:
  - Baseline on CORRUPT prompt (consistent with Phase 389-391)
  - Patched forward pass (not W_U projection)
  - Per-category centroid computed via ANOVA (residual after removing object effect)
  - Per-object-category centroid: ANOVA residual grouped by (object, category)
  - Per-object-category-value: raw delta_h for each individual pair

First round: key layers per model (2 layers each)
Second round: confirm key findings if needed
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
# Data definitions (same as Phase 391)
# ============================================================
COLOR_PAIRS = [
    ("apple","red","blue"),("cherry","red","blue"),("tomato","red","blue"),
    ("rose","red","blue"),("ruby","red","blue"),("flame","red","blue"),
    ("blood","red","blue"),("strawberry","red","blue"),("fire","red","blue"),
    ("sky","blue","red"),("ocean","blue","red"),("sapphire","blue","red"),
    ("water","blue","red"),("lake","blue","red"),("river","blue","red"),
    ("banana","yellow","blue"),("sun","yellow","blue"),("gold","yellow","blue"),
    ("lemon","yellow","blue"),("dandelion","yellow","blue"),
    ("grass","green","blue"),("emerald","green","blue"),("forest","green","blue"),
    ("leaf","green","blue"),("lime","green","blue"),
    ("snow","white","black"),("cloud","white","black"),("milk","white","black"),
    ("cotton","white","black"),("bone","white","black"),("salt","white","black"),
    ("coal","black","white"),("shadow","black","white"),("night","black","white"),
    ("smoke","gray","black"),("ash","gray","black"),("pencil","gray","black"),
    ("silver","gray","black"),("steel","gray","black"),
    ("orange","orange","blue"),("carrot","orange","blue"),("pumpkin","orange","blue"),
    ("sunset","orange","blue"),("copper","orange","blue"),
    ("pink","pink","blue"),("flamingo","pink","blue"),("cotton_candy","pink","blue"),
    ("rose_petal","pink","blue"),("salmon","pink","blue"),
]

TEMP_PAIRS = [
    ("fire","hot","cold"),("lava","hot","cold"),("sun","hot","cold"),
    ("oven","hot","cold"),("desert","hot","cold"),("volcano","hot","cold"),
    ("ice","cold","hot"),("snow","cold","hot"),("frost","cold","hot"),
    ("glacier","cold","hot"),("freezer","cold","hot"),("arctic","cold","hot"),
    ("soup","hot","cold"),("summer","hot","cold"),("winter","cold","hot"),
    ("spring","warm","cold"),("autumn","cool","hot"),
    ("tropical","hot","cold"),("polar","cold","hot"),
    ("lava","hot","cold"),("magma","hot","cold"),
    ("cooler","cold","hot"),("refrigerator","cold","hot"),
    ("tea","hot","cold"),("coffee","hot","cold"),
]

MOISTURE_PAIRS = [
    ("ocean","wet","dry"),("rain","wet","dry"),("river","wet","dry"),
    ("lake","wet","dry"),("swamp","wet","dry"),("sponge","wet","dry"),
    ("desert","dry","wet"),("sand","dry","wet"),("dust","dry","wet"),
    ("cracker","dry","wet"),("bone_dry","dry","wet"),
    ("fog","wet","dry"),("dew","wet","dry"),("mist","wet","dry"),
    ("sahara","dry","wet"),("cactus","dry","wet"),
    ("sweat","wet","dry"),("tear","wet","dry"),
    ("cloth","dry","wet"),("paper","dry","wet"),
    ("jungle","wet","dry"),("rainforest","wet","dry"),
    ("arid","dry","wet"),("parched","dry","wet"),
]

SIZE_PAIRS = [
    ("elephant","big","small"),("mountain","big","small"),("planet","big","small"),
    ("whale","big","small"),("building","big","small"),("continent","big","small"),
    ("ant","small","big"),("grain","small","big"),("pin","small","big"),
    ("dot","small","big"),("speck","small","big"),("atom","small","big"),("pixel","small","big"),
    ("ocean","big","small"),("moon","big","small"),("sun","big","small"),
    ("bacteria","small","big"),
]

WEIGHT_PAIRS = [
    ("elephant","heavy","light"),("boulder","heavy","light"),("anchor","heavy","light"),
    ("tank","heavy","light"),("lead","heavy","light"),("iron","heavy","light"),
    ("feather","light","heavy"),("balloon","light","heavy"),("bubble","light","heavy"),
    ("cloud","light","heavy"),("smoke","light","heavy"),("air","light","heavy"),
    ("mountain","heavy","light"),("ship","heavy","light"),
]

SPEED_PAIRS = [
    ("cheetah","fast","slow"),("rocket","fast","slow"),("lightning","fast","slow"),
    ("falcon","fast","slow"),("bullet","fast","slow"),("jet","fast","slow"),
    ("turtle","slow","fast"),("snail","slow","fast"),("sloth","slow","fast"),
    ("walker","slow","fast"),("glacier","slow","fast"),("caterpillar","slow","fast"),
    ("horse","fast","slow"),
]

BRIGHT_PAIRS = [
    ("sun","bright","dark"),("star","bright","dark"),("lamp","bright","dark"),
    ("flashlight","bright","dark"),("candle","bright","dark"),("diamond","bright","dark"),
    ("cave","dark","bright"),("shadow","dark","bright"),("night","dark","bright"),
    ("hole","dark","bright"),("eclipse","dark","bright"),
    ("moon","bright","dark"),("lighthouse","bright","dark"),
    ("dungeon","dark","bright"),
]

CATEGORY_MAP = {
    "color": COLOR_PAIRS, "temperature": TEMP_PAIRS,
    "moisture": MOISTURE_PAIRS, "size": SIZE_PAIRS,
    "weight": WEIGHT_PAIRS, "speed": SPEED_PAIRS,
    "brightness": BRIGHT_PAIRS,
}

ALL_PAIRS = []
PAIR_CATEGORIES = []
_seen = set()
for _cat_name, _pairs in CATEGORY_MAP.items():
    for _obj, _target, _comp in _pairs:
        _key = (_obj, _target)
        if _key not in _seen:
            _seen.add(_key)
            ALL_PAIRS.append((_obj, _target, _comp))
            PAIR_CATEGORIES.append(_cat_name)

# Incorrect pairs: swap target/competitor
INCORRECT_PAIRS = [(obj, comp, target) for obj, target, comp in ALL_PAIRS]
INCORRECT_CATEGORIES = PAIR_CATEGORIES[:]

TEMPLATE = "The {obj} is {attr}."
CORRUPT_BASELINE = "The item"


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
            print(f"  Loading {model_name} with attn_impl={impl}...")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            print(f"  Success with {impl}")
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
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            print(f"    Layer {lid}: {layer_devices[lid]}")

    return model, tokenizer


def anova_decomposition_full(dh_raw, object_labels, category_labels):
    """Full ANOVA decomposition returning all 4 levels of centroid."""
    N, d = dh_raw.shape
    mu = dh_raw.mean(axis=0)

    # Level 0: global centroid
    global_centroid = mu.copy()

    # Object centroids (remove mean)
    unique_objects = sorted(set(object_labels))
    obj_to_idx = {o: i for i, o in enumerate(unique_objects)}
    obj_centroids = np.zeros((len(unique_objects), d))
    obj_counts = np.zeros(len(unique_objects))
    for i, o in enumerate(object_labels):
        obj_centroids[obj_to_idx[o]] += dh_raw[i]
        obj_counts[obj_to_idx[o]] += 1
    for j in range(len(unique_objects)):
        if obj_counts[j] > 0:
            obj_centroids[j] /= obj_counts[j]
    I_comp = np.zeros_like(dh_raw)
    for i, o in enumerate(object_labels):
        I_comp[i] = obj_centroids[obj_to_idx[o]] - mu

    # Level 1: category centroids (residual after removing object effect)
    dh_resid_I = dh_raw - mu - I_comp
    unique_cats = sorted(set(category_labels))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_centroids = np.zeros((len(unique_cats), d))
    cat_counts = np.zeros(len(unique_cats))
    for i, c in enumerate(category_labels):
        cat_centroids[cat_to_idx[c]] += dh_resid_I[i]
        cat_counts[cat_to_idx[c]] += 1
    for j in range(len(unique_cats)):
        if cat_counts[j] > 0:
            cat_centroids[j] /= cat_counts[j]
    A_comp = np.zeros_like(dh_raw)
    for i, c in enumerate(category_labels):
        A_comp[i] = cat_centroids[cat_to_idx[c]]

    # Level 2: object-category centroids (residual after removing object+category effects)
    dh_resid_IC = dh_raw - mu - I_comp - A_comp
    obj_cat_keys = [(object_labels[i], category_labels[i]) for i in range(N)]
    unique_obj_cats = sorted(set(obj_cat_keys))
    oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
    oc_centroids = np.zeros((len(unique_obj_cats), d))
    oc_counts = np.zeros(len(unique_obj_cats))
    for i, oc in enumerate(obj_cat_keys):
        oc_centroids[oc_to_idx[oc]] += dh_resid_IC[i]
        oc_counts[oc_to_idx[oc]] += 1
    for j in range(len(unique_obj_cats)):
        if oc_counts[j] > 0:
            oc_centroids[j] /= oc_counts[j]
    # A_obj_cat = per-(object, category) component
    A_obj_cat = np.zeros_like(dh_raw)
    for i, oc in enumerate(obj_cat_keys):
        A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

    # Level 3: per-pair direction (raw delta_h for each individual pair)
    # This is just dh_raw itself

    return {
        'global_centroid': global_centroid,
        'cat_centroids': cat_centroids,
        'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats,
        'A_comp': A_comp,           # Level 1: per-category component
        'A_obj_cat': A_obj_cat,     # Level 2: per-(object, category) component
        'oc_centroids': oc_centroids,
        'oc_to_idx': oc_to_idx,
        'unique_obj_cats': unique_obj_cats,
        'dh_raw': dh_raw,           # Level 3: per-pair direction
        'mu': mu,
    }


def classify_mechanism(td, cd):
    """Classify the mechanism type based on target_delta and competitor_delta."""
    if td > 0 and cd < 0:
        return "IDEAL"       # T↑C↓ - best selectivity
    elif td > 0 and cd > 0:
        if td > cd:
            return "DOM_BOOST"  # T↑C↑ but T>C
        else:
            return "BOOST_C"    # T↑C↑ but C>T
    elif td < 0 and cd > 0:
        return "REVERSED"    # T↓C↑ - worst
    elif td < 0 and cd < 0:
        if abs(td) > abs(cd):
            return "SUPP_T"     # T↓↓C↓, T suppressed more
        else:
            return "SUPP_C"     # T↓C↓↓, C suppressed more
    else:
        return "MIXED"


def test_direction_at_layer(model, tokenizer, layers_list, device, li, delta_np, prompt, tid, cid):
    """Test a single direction at a layer, return (target_logit, competitor_logit)."""
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


def run_phase393(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 393: Centroid Hierarchy + T/C Decomposition ({model_name}) [{timestamp}] ===")

    # Layer configs - 2 key layers per model (shallow + deep)
    LAYER_CONFIGS = {
        "qwen3": [4, 20],
        "deepseek7b": [4, 20],
        "glm4": [4, 20],
    }
    layer_indices = LAYER_CONFIGS[model_name]

    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  n_layers={n_layers}, d_model={d_model}")

    N = len(ALL_PAIRS)
    obj_labels = [p[0] for p in ALL_PAIRS]
    cat_labels = PAIR_CATEGORIES[:]

    # Build prompts
    correct_prompts = [TEMPLATE.format(obj=obj, attr=target) for obj, target, _ in ALL_PAIRS]
    correct_corrupt = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target) for _, target, _ in ALL_PAIRS]
    incorrect_prompts = [TEMPLATE.format(obj=p[0], attr=p[1]) for p in INCORRECT_PAIRS]
    incorrect_corrupt = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=p[1]) for p in INCORRECT_PAIRS]

    # Get token IDs
    token_ids = {}
    for obj, target, comp in ALL_PAIRS + INCORRECT_PAIRS:
        for tok in [target, comp]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'n_pairs': N, 'layers': layer_indices,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li}/{n_layers-1} ---")
        print(f"{'='*70}")

        # ---- Step 1: Collect activations for correct pairs ----
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
            inputs = tokenizer(correct_prompts[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_clean[i] = captured['h'][0, -1].numpy()

            captured.clear()
            inputs = tokenizer(correct_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_corrupt[i] = captured['h'][0, -1].numpy()

            if (i+1) % 50 == 0:
                print(f"  Activation collection: {i+1}/{N}")

        handle.remove()
        print(f"  Activations collected ({time.time()-t0_layer:.0f}s)")

        # ---- Step 2: Full ANOVA decomposition ----
        dh = h_clean - h_corrupt
        anova = anova_decomposition_full(dh, obj_labels, cat_labels)

        # ---- Step 3: Baseline logits ----
        print(f"  Computing baseline logits...")
        baseline_target_cor = np.zeros(N)
        baseline_competitor_cor = np.zeros(N)
        baseline_target_inc = np.zeros(N)
        baseline_competitor_inc = np.zeros(N)

        for i, (obj, target, comp) in enumerate(ALL_PAIRS):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            # Correct baseline
            inputs = tokenizer(correct_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            if tid is not None:
                baseline_target_cor[i] = float(out.logits[0, -1, tid])
            if cid is not None:
                baseline_competitor_cor[i] = float(out.logits[0, -1, cid])

        # Incorrect baseline
        for i, (obj, target_inc, comp_inc) in enumerate(INCORRECT_PAIRS):
            # In incorrect pair: target_inc = wrong value, comp_inc = correct value
            tid = token_ids.get(comp_inc)   # compatible (correct) token
            cid = token_ids.get(target_inc)  # incompatible (wrong) token
            inputs = tokenizer(incorrect_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            if tid is not None:
                baseline_target_inc[i] = float(out.logits[0, -1, tid])
            if cid is not None:
                baseline_competitor_inc[i] = float(out.logits[0, -1, cid])

            if (i+1) % 50 == 0:
                print(f"    Baseline {i+1}/{N}")

        print(f"  Baseline logits done ({time.time()-t0_layer:.0f}s)")

        # ---- Step 4: Test all 4 hierarchy levels on CORRECT pairs ----
        # Use a SUBSET for Level 2 and 3 (too slow for all pairs)
        # Level 0-1: test all pairs
        # Level 2: test every 2nd pair
        # Level 3: test every 3rd pair

        hierarchy_results = {}

        for level_name, direction_fn, subset_desc in [
            ("L0_global", lambda i: anova['global_centroid'], "all"),
            ("L1_category", lambda i: anova['A_comp'][i], "all"),
            ("L2_obj_cat", lambda i: anova['A_comp'][i] + anova['A_obj_cat'][i], "every2nd"),
            ("L3_pair", lambda i: anova['dh_raw'][i], "every3rd"),
        ]:
            print(f"\n  Testing {level_name} on correct pairs ({subset_desc})...")
            t0_h = time.time()

            if subset_desc == "all":
                test_idx = list(range(N))
            elif subset_desc == "every2nd":
                test_idx = list(range(0, N, 2))
            else:
                test_idx = list(range(0, N, 3))

            patched_target = np.zeros(N)
            patched_competitor = np.zeros(N)

            for i in test_idx:
                obj, target, comp = ALL_PAIRS[i]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                if tid is None or cid is None:
                    continue

                delta_np = direction_fn(i)
                pt, pc = test_direction_at_layer(
                    model, tokenizer, layers_list, device, li,
                    delta_np, correct_corrupt[i], tid, cid
                )
                patched_target[i] = pt
                patched_competitor[i] = pc

                if (i+1) % 30 == 0:
                    elapsed = time.time() - t0_h
                    print(f"    {level_name}: {i+1}/{len(test_idx)} tested ({elapsed:.0f}s)")

            # Compute deltas
            td = patched_target - baseline_target_cor
            cd = patched_competitor - baseline_competitor_cor
            add = td - cd

            # Only use tested indices for aggregation
            td_test = td[test_idx]
            cd_test = cd[test_idx]
            add_test = add[test_idx]

            # Per-category breakdown
            cat_breakdown = {}
            for cat in anova['unique_cats']:
                cat_mask = np.array([cat_labels[j] for j in test_idx]) == cat
                if cat_mask.sum() == 0:
                    continue
                cat_td = td_test[cat_mask]
                cat_cd = cd_test[cat_mask]
                cat_add = add_test[cat_mask]
                # Determine dominant mechanism
                mech = classify_mechanism(float(np.mean(cat_td)), float(np.mean(cat_cd)))
                cat_breakdown[cat] = {
                    'add_mean': float(np.mean(cat_add)),
                    'target_delta_mean': float(np.mean(cat_td)),
                    'competitor_delta_mean': float(np.mean(cat_cd)),
                    'mechanism': mech,
                    'n': int(cat_mask.sum()),
                }

            overall_mech = classify_mechanism(float(np.mean(td_test)), float(np.mean(cd_test)))
            hierarchy_results[level_name] = {
                'add_mean': float(np.mean(add_test)),
                'target_delta_mean': float(np.mean(td_test)),
                'competitor_delta_mean': float(np.mean(cd_test)),
                'add_pos_pct': float(np.mean(add_test > 0) * 100),
                'mechanism': overall_mech,
                'n_tested': len(test_idx),
                'category_breakdown': cat_breakdown,
            }

            elapsed = time.time() - t0_h
            print(f"    {level_name}: add={hierarchy_results[level_name]['add_mean']:+.4f}, "
                  f"T={hierarchy_results[level_name]['target_delta_mean']:+.4f}, "
                  f"C={hierarchy_results[level_name]['competitor_delta_mean']:+.4f}, "
                  f"mech={overall_mech} ({elapsed:.0f}s)")

        # ---- Step 5: Test hierarchy on INCORRECT pairs (Level 0, 1 only for speed) ----
        print(f"\n  Testing L0_global and L1_category on incorrect pairs...")
        inc_hierarchy = {}

        for level_name, direction_fn in [
            ("L0_global", lambda i: anova['global_centroid']),
            ("L1_category", lambda i: anova['A_comp'][i]),
        ]:
            patched_target_inc = np.zeros(N)
            patched_competitor_inc = np.zeros(N)

            for i in range(N):
                obj, target_inc, comp_inc = INCORRECT_PAIRS[i]
                # compatible = comp_inc (correct value), incompatible = target_inc (wrong value)
                tid = token_ids.get(comp_inc)
                cid = token_ids.get(target_inc)
                if tid is None or cid is None:
                    continue

                delta_np = direction_fn(i)
                pt, pc = test_direction_at_layer(
                    model, tokenizer, layers_list, device, li,
                    delta_np, incorrect_corrupt[i], tid, cid
                )
                patched_target_inc[i] = pt
                patched_competitor_inc[i] = pc

                if (i+1) % 30 == 0:
                    print(f"    {level_name} incorrect: {i+1}/{N}")

            td_inc = patched_target_inc - baseline_target_inc
            cd_inc = patched_competitor_inc - baseline_competitor_inc
            add_inc = td_inc - cd_inc

            # Per-category
            cat_breakdown_inc = {}
            for cat in anova['unique_cats']:
                cat_mask = np.array(INCORRECT_CATEGORIES) == cat
                if cat_mask.sum() == 0:
                    continue
                cat_td = td_inc[cat_mask]
                cat_cd = cd_inc[cat_mask]
                cat_add = add_inc[cat_mask]
                mech = classify_mechanism(float(np.mean(cat_td)), float(np.mean(cat_cd)))
                cat_breakdown_inc[cat] = {
                    'add_mean': float(np.mean(cat_add)),
                    'target_delta_mean': float(np.mean(cat_td)),
                    'competitor_delta_mean': float(np.mean(cat_cd)),
                    'mechanism': mech,
                    'n': int(cat_mask.sum()),
                }

            overall_mech = classify_mechanism(float(np.mean(td_inc)), float(np.mean(cd_inc)))
            inc_hierarchy[level_name] = {
                'add_mean': float(np.mean(add_inc)),
                'target_delta_mean': float(np.mean(td_inc)),
                'competitor_delta_mean': float(np.mean(cd_inc)),
                'mechanism': overall_mech,
                'category_breakdown': cat_breakdown_inc,
            }

            print(f"    {level_name} INC: add={inc_hierarchy[level_name]['add_mean']:+.4f}, "
                  f"T={inc_hierarchy[level_name]['target_delta_mean']:+.4f}, "
                  f"C={inc_hierarchy[level_name]['competitor_delta_mean']:+.4f}, "
                  f"mech={overall_mech}")

        # ---- Step 6: SYMMETRIC check for L1 ----
        print(f"\n  --- SYMMETRIC check (L1 category) ---")
        symmetric_count = 0
        total_count = 0
        for cat in anova['unique_cats']:
            ce_cor = hierarchy_results['L1_category']['category_breakdown'].get(cat, {})
            ce_inc = inc_hierarchy['L1_category']['category_breakdown'].get(cat, {})
            add_cor = ce_cor.get('add_mean', 0)
            add_inc = ce_inc.get('add_mean', 0)
            td_cor = ce_cor.get('target_delta_mean', 0)
            cd_cor = ce_cor.get('competitor_delta_mean', 0)
            td_inc = ce_inc.get('target_delta_mean', 0)
            cd_inc = ce_inc.get('competitor_delta_mean', 0)
            mech_cor = ce_cor.get('mechanism', '?')
            mech_inc = ce_inc.get('mechanism', '?')

            is_sym = np.sign(add_cor) != np.sign(add_inc) and add_cor != 0 and add_inc != 0
            if is_sym:
                symmetric_count += 1
            total_count += 1

            sym_str = "SYM" if is_sym else "ASYM"
            print(f"    {cat:12s}: cor={add_cor:+.4f}({mech_cor}) inc={add_inc:+.4f}({mech_inc}) {sym_str}")

        print(f"  SYMMETRIC: {symmetric_count}/{total_count} = {symmetric_count/total_count*100:.0f}%")

        # ---- Step 7: Hierarchy selectivity comparison ----
        print(f"\n  --- Hierarchy Selectivity Comparison ---")
        for level_name in ["L0_global", "L1_category", "L2_obj_cat", "L3_pair"]:
            hr = hierarchy_results[level_name]
            td_mean = hr['target_delta_mean']
            cd_mean = hr['competitor_delta_mean']

            # Selectivity = target_delta - competitor_delta (higher = more selective)
            selectivity = abs(td_mean - cd_mean)

            # IDEAL ratio: how many categories show T↑C↓?
            cat_data = hr.get('category_breakdown', {})
            ideal_count = sum(1 for c in cat_data.values() if c['mechanism'] == 'IDEAL')
            total_cats = len(cat_data)
            ideal_pct = ideal_count / total_cats * 100 if total_cats > 0 else 0

            print(f"    {level_name}: selectivity={selectivity:.4f}, "
                  f"IDEAL={ideal_count}/{total_cats}({ideal_pct:.0f}%), "
                  f"T={td_mean:+.4f}, C={cd_mean:+.4f}, mech={hr['mechanism']}")

        # Save layer results
        results['per_layer'][str(li)] = {
            'hierarchy_correct': hierarchy_results,
            'hierarchy_incorrect': inc_hierarchy,
            'symmetric_count': symmetric_count,
            'symmetric_total': total_count,
        }

        elapsed = time.time() - t0_layer
        print(f"\n  L{li} done in {elapsed:.0f}s")

    # ---- Cross-layer summary ----
    print(f"\n{'='*70}")
    print(f"--- Cross-Layer Hierarchy Summary ---")
    print(f"{'='*70}")

    for level_name in ["L0_global", "L1_category", "L2_obj_cat", "L3_pair"]:
        print(f"\n  {level_name}:")
        for li in layer_indices:
            hr = results['per_layer'][str(li)]['hierarchy_correct'][level_name]
            cat_data = hr.get('category_breakdown', {})
            ideal_count = sum(1 for c in cat_data.values() if c['mechanism'] == 'IDEAL')
            total_cats = len(cat_data)
            print(f"    L{li}: add={hr['add_mean']:+.4f}, T={hr['target_delta_mean']:+.4f}, "
                  f"C={hr['competitor_delta_mean']:+.4f}, IDEAL={ideal_count}/{total_cats}, "
                  f"mech={hr['mechanism']}")

    # ---- Category trajectory across hierarchy levels ----
    print(f"\n--- Category Mechanism Across Hierarchy Levels ---")
    for li in layer_indices:
        print(f"\n  Layer {li}:")
        for cat in sorted(CATEGORY_MAP.keys()):
            mechs = []
            for level_name in ["L0_global", "L1_category", "L2_obj_cat", "L3_pair"]:
                hr = results['per_layer'][str(li)]['hierarchy_correct'][level_name]
                cat_data = hr.get('category_breakdown', {})
                ce = cat_data.get(cat, {})
                mech = ce.get('mechanism', '?')
                add = ce.get('add_mean', 0)
                td = ce.get('target_delta_mean', 0)
                cd = ce.get('competitor_delta_mean', 0)
                mechs.append(f"{mech}(T{td:+.3f}C{cd:+.3f})")
            print(f"    {cat:12s}: {' -> '.join(mechs)}")

    # ---- Save results ----
    out_dir = ROOT / "results" / "phase393_centroid_hierarchy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase393.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model released. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase393(model_name)
