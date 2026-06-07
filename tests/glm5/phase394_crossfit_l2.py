"""
Phase 394: Cross-Fitted L2_obj_cat Validation
==============================================
Core question: Is L2_obj_cat's IDEAL result from real gear surface
or from data leakage (using same sample to estimate and test)?

Method:
  - Leave-one-pair-out (LOPO) cross-fitting for L2_obj_cat
  - For each test pair i in object-category (obj, cat):
    - Estimate L2 direction using ALL OTHER pairs in same (obj, cat)
    - Test on pair i
  - If sample count in (obj, cat) <= 1: skip that pair

Compare:
  - L2_original:  original L2_obj_cat (Phase 393, with leakage)
  - L2_crossfit:  cross-fitted L2_obj_cat (no leakage)
  - L1_category:  for reference (no leakage concern)

If L2_crossfit still shows IDEAL → real gear surface
If L2_crossfit drops to DOM_BOOST → leakage was inflating results

Models: qwen3, deepseek7b, glm4
Layers: key layers per model (focus on GLM4 L20 where L2 was strongest)
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
# Data definitions (same as Phase 391/393)
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
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            print(f"    Layer {lid}: {layer_devices[lid]}")

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


def anova_decomposition_full(dh_raw, object_labels, category_labels):
    """Full ANOVA decomposition returning all levels."""
    N, d = dh_raw.shape
    mu = dh_raw.mean(axis=0)
    global_centroid = mu.copy()

    # Object centroids
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

    # Category centroids (residual after removing object effect)
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

    # Object-category centroids (residual after removing object+category effects)
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
    A_obj_cat = np.zeros_like(dh_raw)
    for i, oc in enumerate(obj_cat_keys):
        A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

    return {
        'global_centroid': global_centroid,
        'cat_centroids': cat_centroids,
        'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats,
        'A_comp': A_comp,
        'A_obj_cat': A_obj_cat,
        'oc_centroids': oc_centroids,
        'oc_to_idx': oc_to_idx,
        'unique_obj_cats': unique_obj_cats,
        'obj_cat_keys': obj_cat_keys,
        'dh_raw': dh_raw,
        'mu': mu,
        'I_comp': I_comp,
    }


def run_phase394(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 394: Cross-Fitted L2_obj_cat ({model_name}) [{timestamp}] ===")

    # Layer configs - focus on key layers where L2 was interesting
    LAYER_CONFIGS = {
        "qwen3": [4, 20],
        "deepseek7b": [4, 20],
        "glm4": [4, 20, 38],  # GLM4 L20 and L38 where L2 was strongest
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

    # Get token IDs
    token_ids = {}
    for obj, target, comp in ALL_PAIRS:
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

        # ---- Step 2: ANOVA decomposition ----
        dh = h_clean - h_corrupt
        anova = anova_decomposition_full(dh, obj_labels, cat_labels)

        # ---- Step 3: Compute cross-fitted L2 directions ----
        # For each pair i, estimate L2_obj_cat direction using ALL OTHER pairs
        # in the same (object, category) group
        print(f"\n  Computing cross-fitted L2 directions...")

        obj_cat_keys = anova['obj_cat_keys']

        # Group pairs by (object, category)
        oc_groups = defaultdict(list)
        for i, oc in enumerate(obj_cat_keys):
            oc_groups[oc].append(i)

        # Count samples per group
        oc_sample_counts = {oc: len(indices) for oc, indices in oc_groups.items()}
        single_sample_groups = sum(1 for c in oc_sample_counts.values() if c <= 1)
        multi_sample_groups = sum(1 for c in oc_sample_counts.values() if c > 1)
        print(f"  (obj,cat) groups: {len(oc_groups)} total, "
              f"{multi_sample_groups} with >1 sample, {single_sample_groups} with <=1 sample")

        # Cross-fitted L2 direction for each pair
        # L2_obj_cat_original = mu + I_comp + A_comp + A_obj_cat
        # L2_obj_cat_crossfit = same but A_obj_cat estimated without pair i
        l2_crossfit = np.zeros((N, d_model), dtype=np.float32)

        skipped_pairs = []
        for i in range(N):
            oc = obj_cat_keys[i]
            group_indices = oc_groups[oc]
            if len(group_indices) <= 1:
                # Only 1 sample in this group — can't do LOPO
                # Fall back to L1 category direction (no leakage)
                l2_crossfit[i] = anova['A_comp'][i]  # Just category component
                skipped_pairs.append(i)
            else:
                # LOPO: compute A_obj_cat without pair i
                other_indices = [j for j in group_indices if j != i]
                # Average the residual (dh - mu - I_comp - A_comp) over other pairs
                l2_residual_avg = np.mean([dh[j] - anova['mu'] - anova['I_comp'][j] - anova['A_comp'][j]
                                          for j in other_indices], axis=0)
                l2_crossfit[i] = anova['mu'] + anova['I_comp'][i] + anova['A_comp'][i] + l2_residual_avg

        print(f"  Cross-fitted L2 computed. Skipped {len(skipped_pairs)} single-sample pairs.")

        # ---- Step 4: Baseline logits ----
        print(f"  Computing baseline logits...")
        baseline_target = np.zeros(N)
        baseline_competitor = np.zeros(N)

        for i, (obj, target, comp) in enumerate(ALL_PAIRS):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            inputs = tokenizer(correct_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            if tid is not None:
                baseline_target[i] = float(out.logits[0, -1, tid])
            if cid is not None:
                baseline_competitor[i] = float(out.logits[0, -1, cid])

        print(f"  Baseline logits done ({time.time()-t0_layer:.0f}s)")

        # ---- Step 5: Test all 3 versions ----
        # L1_category: no leakage (reference)
        # L2_original: with leakage (Phase 393 result)
        # L2_crossfit: without leakage (this phase's key test)

        test_versions = {
            "L1_category": lambda i: anova['A_comp'][i],
            "L2_original": lambda i: anova['A_comp'][i] + anova['A_obj_cat'][i],
            "L2_crossfit": lambda i: l2_crossfit[i],
        }

        version_results = {}

        for ver_name, direction_fn in test_versions.items():
            print(f"\n  Testing {ver_name}...")
            t0_v = time.time()

            patched_target = np.zeros(N)
            patched_competitor = np.zeros(N)

            for i in range(N):
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

                if (i+1) % 50 == 0:
                    elapsed = time.time() - t0_v
                    print(f"    {ver_name}: {i+1}/{N} ({elapsed:.0f}s)")

            # Compute deltas
            td = patched_target - baseline_target
            cd = patched_competitor - baseline_competitor
            add = td - cd

            # Per-category breakdown
            cat_breakdown = {}
            for cat in anova['unique_cats']:
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

            elapsed = time.time() - t0_v
            print(f"    {ver_name}: add={version_results[ver_name]['add_mean']:+.4f}, "
                  f"T={version_results[ver_name]['target_delta_mean']:+.4f}, "
                  f"C={version_results[ver_name]['competitor_delta_mean']:+.4f}, "
                  f"IDEAL={ideal_count}/{len(cat_breakdown)}, "
                  f"mech={overall_mech} ({elapsed:.0f}s)")

        # ---- Step 6: Category-level comparison ----
        print(f"\n  === Category-Level: L2_original vs L2_crossfit ===")
        for cat in sorted(CATEGORY_MAP.keys()):
            orig = version_results['L2_original']['category_breakdown'].get(cat, {})
            cf = version_results['L2_crossfit']['category_breakdown'].get(cat, {})
            l1 = version_results['L1_category']['category_breakdown'].get(cat, {})

            orig_td = orig.get('target_delta_mean', 0)
            orig_cd = orig.get('competitor_delta_mean', 0)
            orig_mech = orig.get('mechanism', '?')
            cf_td = cf.get('target_delta_mean', 0)
            cf_cd = cf.get('competitor_delta_mean', 0)
            cf_mech = cf.get('mechanism', '?')
            l1_td = l1.get('target_delta_mean', 0)
            l1_cd = l1.get('competitor_delta_mean', 0)
            l1_mech = l1.get('mechanism', '?')

            # Check if mechanism changed
            changed = "CHANGED" if orig_mech != cf_mech else "same"
            print(f"    {cat:12s}: L2_orig={orig_mech}(T{orig_td:+.3f}C{orig_cd:+.3f}) "
                  f"L2_cf={cf_mech}(T{cf_td:+.3f}C{cf_cd:+.3f}) "
                  f"L1={l1_mech}(T{l1_td:+.3f}C{l1_cd:+.3f}) [{changed}]")

        # ---- Step 7: Per-(obj,cat) groups with >=3 samples (most reliable crossfit) ----
        print(f"\n  === Per-(obj,cat) groups with >=3 samples ===")
        reliable_groups = {oc: indices for oc, indices in oc_groups.items() if len(indices) >= 3}
        print(f"  Groups with >=3 samples: {len(reliable_groups)}/{len(oc_groups)}")

        # For these groups, compare original vs crossfit at pair level
        if reliable_groups:
            for ver_name in ["L2_original", "L2_crossfit"]:
                td_all = np.zeros(N)
                cd_all = np.zeros(N)
                # We already have these in version_results but need pair-level
                # Reconstruct from patched/baseline
                pass  # Will use aggregate results above

        # ---- Step 8: Leakage analysis ----
        print(f"\n  === Leakage Analysis ===")
        # Compare effect sizes
        for cat in sorted(CATEGORY_MAP.keys()):
            orig = version_results['L2_original']['category_breakdown'].get(cat, {})
            cf = version_results['L2_crossfit']['category_breakdown'].get(cat, {})
            orig_add = abs(orig.get('add_mean', 0))
            cf_add = abs(cf.get('add_mean', 0))
            if orig_add > 0.001:
                ratio = cf_add / orig_add
            else:
                ratio = 0
            leakage_pct = (1 - ratio) * 100 if orig_add > 0.001 else 0
            print(f"    {cat:12s}: |add|_orig={orig_add:.4f}, |add|_cf={cf_add:.4f}, "
                  f"ratio={ratio:.2f}, leakage={leakage_pct:.0f}%")

        # Save layer results
        results['per_layer'][str(li)] = {
            'version_results': version_results,
            'n_skipped_pairs': len(skipped_pairs),
            'n_groups_multi': multi_sample_groups,
            'n_groups_single': single_sample_groups,
        }

        elapsed = time.time() - t0_layer
        print(f"\n  L{li} done in {elapsed:.0f}s")

    # ---- Cross-layer summary ----
    print(f"\n{'='*70}")
    print(f"--- Phase 394 Cross-Layer Summary: L2_original vs L2_crossfit ---")
    print(f"{'='*70}")

    for li in layer_indices:
        vr = results['per_layer'][str(li)]['version_results']
        print(f"\n  Layer {li}:")
        for ver in ["L1_category", "L2_original", "L2_crossfit"]:
            r = vr[ver]
            print(f"    {ver:12s}: add={r['add_mean']:+.4f}, T={r['target_delta_mean']:+.4f}, "
                  f"C={r['competitor_delta_mean']:+.4f}, IDEAL={r['ideal_count']}/{r['total_cats']}, "
                  f"mech={r['mechanism']}")

    # Category comparison across layers
    print(f"\n--- Category Mechanism: L2_original → L2_crossfit ---")
    for li in layer_indices:
        vr = results['per_layer'][str(li)]['version_results']
        print(f"\n  Layer {li}:")
        for cat in sorted(CATEGORY_MAP.keys()):
            orig = vr['L2_original']['category_breakdown'].get(cat, {})
            cf = vr['L2_crossfit']['category_breakdown'].get(cat, {})
            orig_mech = orig.get('mechanism', '?')
            cf_mech = cf.get('mechanism', '?')
            orig_td = orig.get('target_delta_mean', 0)
            orig_cd = orig.get('competitor_delta_mean', 0)
            cf_td = cf.get('target_delta_mean', 0)
            cf_cd = cf.get('competitor_delta_mean', 0)
            changed = " ← LEAK" if orig_mech == "IDEAL" and cf_mech != "IDEAL" else ""
            print(f"    {cat:12s}: {orig_mech}→{cf_mech} "
                  f"(T:{orig_td:+.3f}→{cf_td:+.3f}, C:{orig_cd:+.3f}→{cf_cd:+.3f}){changed}")

    # ---- Save results ----
    out_dir = ROOT / "results" / "phase394_crossfit_l2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase394.json"
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
    run_phase394(model_name)
