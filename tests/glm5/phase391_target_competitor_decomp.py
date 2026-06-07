"""
Phase 391: Target/Competitor Decomposition + Conditional Centroid Hierarchy
===========================================================================
Core goals:
1. Decompose centroid add_effect into target_delta and competitor_delta
   - target_delta: how much does centroid boost/suppress the compatible value?
   - competitor_delta: how much does centroid boost/suppress the incompatible value?
   This explains: is negative effect from suppressing target or boosting competitor?

2. Compare conditional centroid hierarchy:
   - A_global: single global centroid (all categories averaged)
   - A_category: per-category centroid (7 directions)
   Predict: A_category >> A_global (already proven, but confirm with decomp)

3. Deeper layer coverage for DS7B (L26) and GLM4 (L38)

Key methodology (consistent with Phase 389/390):
- Baseline on CORRUPT prompt
- add_effect = patched(corrupt+delta) - baseline(corrupt)
- target_delta = patched_target_logit - baseline_target_logit
- competitor_delta = patched_competitor_logit - baseline_competitor_logit
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
# Data definitions (same as Phase 389/390)
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

TEMPLATE = "The {obj} is {attr}."
CORRUPT_BASELINE = "The item"


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, then sdpa, then eager
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

    # Show layer allocation for device_map="auto" models
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
        # Show last 3 layers
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            print(f"    Layer {lid}: {layer_devices[lid]}")

    return model, tokenizer


def anova_decomposition(dh_raw, object_labels, category_labels):
    N, d = dh_raw.shape
    mu = dh_raw.mean(axis=0)
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
    # Also compute per-object-category centroids
    obj_cat_keys = [(object_labels[i], category_labels[i]) for i in range(N)]
    unique_obj_cats = sorted(set(obj_cat_keys))
    oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
    oc_centroids = np.zeros((len(unique_obj_cats), d))
    oc_counts = np.zeros(len(unique_obj_cats))
    for i, oc in enumerate(obj_cat_keys):
        oc_centroids[oc_to_idx[oc]] += dh_resid_I[i]
        oc_counts[oc_to_idx[oc]] += 1
    for j in range(len(unique_obj_cats)):
        if oc_counts[j] > 0:
            oc_centroids[j] /= oc_counts[j]
    A_obj_cat = np.zeros_like(dh_raw)
    for i, oc in enumerate(obj_cat_keys):
        A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

    return {
        'mu': mu, 'I_comp': I_comp, 'A_comp': A_comp, 'A_obj_cat': A_obj_cat,
        'cat_centroids': cat_centroids, 'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats,
    }


def run_phase391(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 391: Target/Competitor Decomposition ({model_name}) [{timestamp}] ===")

    # Layer configs - add deeper layers for DS7B and GLM4
    LAYER_CONFIGS = {
        "qwen3": [4, 12, 20, 28],
        "deepseek7b": [4, 12, 20, 26],  # Added L26 for deep layer
        "glm4": [4, 20, 30, 38],         # Added L38 for deep layer
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
    clean_prompts = [TEMPLATE.format(obj=obj, attr=target) for obj, target, _ in ALL_PAIRS]
    corrupt_prompts = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target) for _, target, _ in ALL_PAIRS]

    # Get token IDs
    token_ids = {}
    for obj, target, comp in ALL_PAIRS:
        for tok in [target, comp]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name,
        'timestamp': timestamp,
        'n_pairs': N,
        'layers': layer_indices,
        'per_layer': {},
        'category_trajectory': {},
        'hierarchy_comparison': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*60}")
        print(f"--- Layer {li}/{n_layers-1} ---")
        print(f"{'='*60}")

        # ---- Step 1: Collect activations ----
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
            inputs = tokenizer(clean_prompts[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_clean[i] = captured['h'][0, -1].numpy()

            captured.clear()
            inputs = tokenizer(corrupt_prompts[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_corrupt[i] = captured['h'][0, -1].numpy()

            if (i+1) % 50 == 0:
                print(f"  Activation collection: {i+1}/{N}")

        handle.remove()
        print(f"  Activations collected ({time.time()-t0_layer:.0f}s)")

        # ---- Step 2: ANOVA decomposition ----
        dh = h_clean - h_corrupt
        anova = anova_decomposition(dh, obj_labels, cat_labels)

        # ---- Step 3: Baseline logits on CORRUPT prompt ----
        print(f"  Computing baseline logits (corrupt prompt)...")
        baseline_target = np.zeros(N)
        baseline_competitor = np.zeros(N)
        baseline_lds = np.zeros(N)

        for i, (obj, target, comp) in enumerate(ALL_PAIRS):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            if tid is not None and cid is not None:
                inputs = tokenizer(corrupt_prompts[i], return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out = model(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device))
                baseline_target[i] = float(out.logits[0, -1, tid])
                baseline_competitor[i] = float(out.logits[0, -1, cid])
                baseline_lds[i] = baseline_target[i] - baseline_competitor[i]

        # ---- Step 4: Patched forward with per-category centroid ----
        print(f"  Computing patched add effects with target/competitor decomposition...")
        patched_target = np.zeros(N)
        patched_competitor = np.zeros(N)
        add_effects = np.zeros(N)
        target_delta = np.zeros(N)
        competitor_delta = np.zeros(N)

        for i, (obj, target, comp) in enumerate(ALL_PAIRS):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            if tid is None or cid is None:
                continue

            delta_np = anova['A_comp'][i]  # per-category centroid
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
                inputs_c = tokenizer(corrupt_prompts[i], return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out_patch = model(input_ids=inputs_c["input_ids"].to(device),
                                      attention_mask=inputs_c["attention_mask"].to(device))
                if tid is not None:
                    patched_target[i] = float(out_patch.logits[0, -1, tid])
                if cid is not None:
                    patched_competitor[i] = float(out_patch.logits[0, -1, cid])
            finally:
                handle.remove()

            target_delta[i] = patched_target[i] - baseline_target[i]
            competitor_delta[i] = patched_competitor[i] - baseline_competitor[i]
            add_effects[i] = target_delta[i] - competitor_delta[i]

            if (i+1) % 30 == 0:
                elapsed = time.time() - t0_layer
                rate = (i+1) / elapsed
                eta = (N - i - 1) / rate
                print(f"    {i+1}/{N} pairs ({elapsed:.0f}s, ETA {eta:.0f}s)")

        # ---- Step 5: Hierarchy comparison (global vs category centroid) ----
        print(f"  Computing hierarchy comparison (global vs per-category centroid)...")

        # Global centroid = mean of all A_comp
        global_centroid = anova['A_comp'].mean(axis=0)

        # Test global centroid on a SUBSET (every 3rd pair) for speed
        subset_idx = list(range(0, N, 3))
        global_effects = np.zeros(len(subset_idx))
        global_target_delta = np.zeros(len(subset_idx))
        global_competitor_delta = np.zeros(len(subset_idx))

        delta_global = torch.tensor(global_centroid, dtype=torch.bfloat16, device=device)

        for si, i in enumerate(subset_idx):
            obj, target, comp = ALL_PAIRS[i]
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            if tid is None or cid is None:
                continue

            def make_add_hook_g(delta_vec):
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

            handle = layers_list[li].register_forward_hook(make_add_hook_g(delta_global))
            try:
                inputs_c = tokenizer(corrupt_prompts[i], return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out_patch = model(input_ids=inputs_c["input_ids"].to(device),
                                      attention_mask=inputs_c["attention_mask"].to(device))
                pt = float(out_patch.logits[0, -1, tid]) if tid is not None else 0
                pc = float(out_patch.logits[0, -1, cid]) if cid is not None else 0
                global_target_delta[si] = pt - baseline_target[i]
                global_competitor_delta[si] = pc - baseline_competitor[i]
                global_effects[si] = (pt - pc) - baseline_lds[i]
            finally:
                handle.remove()

        # ---- Step 6: Aggregate results ----
        layer_result = {
            'add_mean': float(np.mean(add_effects)),
            'add_t': float(np.mean(add_effects) / (np.std(add_effects) / np.sqrt(N) + 1e-10)),
            'add_pos_pct': float(np.mean(add_effects > 0) * 100),
            'target_delta_mean': float(np.mean(target_delta)),
            'competitor_delta_mean': float(np.mean(competitor_delta)),
            'target_delta_pos_pct': float(np.mean(target_delta > 0) * 100),
            'competitor_delta_pos_pct': float(np.mean(competitor_delta > 0) * 100),
            'category_effects': {},
            'per_pair_add': add_effects.tolist(),
            'per_pair_target_delta': target_delta.tolist(),
            'per_pair_competitor_delta': competitor_delta.tolist(),
            'hierarchy': {
                'global_add_mean': float(np.mean(global_effects)),
                'global_target_delta_mean': float(np.mean(global_target_delta)),
                'global_competitor_delta_mean': float(np.mean(global_competitor_delta)),
                'category_add_mean': float(np.mean(add_effects[subset_idx])),
                'category_target_delta_mean': float(np.mean(target_delta[subset_idx])),
                'category_competitor_delta_mean': float(np.mean(competitor_delta[subset_idx])),
            },
        }

        # Per-category breakdown
        for cat in anova['unique_cats']:
            mask = np.array([c == cat for c in cat_labels])
            layer_result['category_effects'][cat] = {
                'n': int(mask.sum()),
                'add_mean': float(np.mean(add_effects[mask])),
                'add_pos_pct': float(np.mean(add_effects[mask] > 0) * 100),
                'target_delta_mean': float(np.mean(target_delta[mask])),
                'competitor_delta_mean': float(np.mean(competitor_delta[mask])),
                'target_delta_pos_pct': float(np.mean(target_delta[mask] > 0) * 100),
                'competitor_delta_pos_pct': float(np.mean(competitor_delta[mask] > 0) * 100),
            }

        elapsed = time.time() - t0_layer
        print(f"\n  L{li} done in {elapsed:.0f}s:")
        print(f"    Overall: add={layer_result['add_mean']:+.4f}, t={layer_result['add_t']:+.1f}, pos={layer_result['add_pos_pct']:.0f}%")
        print(f"    Target delta:     {layer_result['target_delta_mean']:+.4f} ({layer_result['target_delta_pos_pct']:.0f}%pos)")
        print(f"    Competitor delta: {layer_result['competitor_delta_mean']:+.4f} ({layer_result['competitor_delta_pos_pct']:.0f}%pos)")
        print(f"    Hierarchy: global={layer_result['hierarchy']['global_add_mean']:+.4f} vs category={layer_result['hierarchy']['category_add_mean']:+.4f}")
        for cat in sorted(layer_result['category_effects'].keys()):
            ce = layer_result['category_effects'][cat]
            print(f"    {cat:12s}: add={ce['add_mean']:+.4f}({ce['add_pos_pct']:.0f}%) "
                  f"T={ce['target_delta_mean']:+.4f}({ce['target_delta_pos_pct']:.0f}%) "
                  f"C={ce['competitor_delta_mean']:+.4f}({ce['competitor_delta_pos_pct']:.0f}%)")

        results['per_layer'][str(li)] = layer_result

    # ---- Cross-layer trajectory ----
    print(f"\n{'='*60}")
    print(f"--- Category Trajectory with Target/Competitor Decomposition ---")
    print(f"{'='*60}")
    for cat in sorted(CATEGORY_MAP.keys()):
        traj = []
        for li in layer_indices:
            ce = results['per_layer'][str(li)]['category_effects'].get(cat, {})
            val = ce.get('add_mean', 0)
            td = ce.get('target_delta_mean', 0)
            cd = ce.get('competitor_delta_mean', 0)
            pos = ce.get('add_pos_pct', 0)
            traj.append((li, val, td, cd, pos))
        vals_str = " -> ".join([f"L{li}={v:+.3f}(T{t:+.3f},C{c:+.3f})"
                                for li, v, t, c, p in traj])
        print(f"  {cat:12s}: {vals_str}")
        results['category_trajectory'][cat] = {
            f"L{li}": {'add_mean': v, 'target_delta': t, 'competitor_delta': c, 'add_pos_pct': p}
            for li, v, t, c, p in traj
        }

    # ---- Direction reversals ----
    print(f"\n--- Direction Reversals ---")
    for cat in sorted(CATEGORY_MAP.keys()):
        traj = results['category_trajectory'].get(cat, {})
        if len(traj) < 2:
            continue
        layers_sorted = sorted([int(k[1:]) for k in traj.keys()])
        signs = [np.sign(traj[f"L{li}"]['add_mean']) for li in layers_sorted]
        reversals = []
        for idx in range(len(signs) - 1):
            if signs[idx] != signs[idx+1] and signs[idx] != 0 and signs[idx+1] != 0:
                reversals.append(f"L{layers_sorted[idx]}->L{layers_sorted[idx+1]}")
        if reversals:
            print(f"  {cat:12s}: REVERSAL at {' '.join(reversals)}")
        else:
            direction = "+" if signs[0] > 0 else ("-" if signs[0] < 0 else "0")
            print(f"  {cat:12s}: stable {direction}")

    # ---- Mechanism summary: is positive effect from boosting target or suppressing competitor? ----
    print(f"\n--- Mechanism Summary ---")
    for li in layer_indices:
        lr = results['per_layer'][str(li)]
        td = lr['target_delta_mean']
        cd = lr['competitor_delta_mean']
        add = lr['add_mean']
        if add > 0:
            if td > 0 and cd <= 0:
                mech = "BOOST_TARGET + SUPPRESS_COMPETITOR"
            elif td > 0 and cd > 0:
                if td > cd:
                    mech = "BOOST_TARGET (dominant, competitor also up)"
                else:
                    mech = "BOOST_COMPETITOR (dominant, target also up but less)"
            elif td <= 0 and cd < 0:
                mech = "SUPPRESS_COMPETITOR (dominant, target also down but less)"
            else:
                mech = "MIXED"
        else:  # add < 0
            if td < 0 and cd >= 0:
                mech = "SUPPRESS_TARGET + BOOST_COMPETITOR"
            elif td < 0 and cd < 0:
                if abs(td) > abs(cd):
                    mech = "SUPPRESS_TARGET (dominant)"
                else:
                    mech = "SUPPRESS_BOTH (competitor more)"
            elif td >= 0 and cd > 0:
                mech = "BOOST_COMPETITOR (dominant)"
            else:
                mech = "MIXED"
        print(f"  L{li}: add={add:+.4f}, T_delta={td:+.4f}, C_delta={cd:+.4f} => {mech}")

    # ---- Save results ----
    out_dir = ROOT / "results" / "phase391_target_competitor_decomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase391.json"
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
    run_phase391(model_name)
