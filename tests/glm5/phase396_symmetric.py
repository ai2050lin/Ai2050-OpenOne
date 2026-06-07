"""
Phase 396: SYMMETRIC Verification — Correct vs Incorrect Mirror Test
=====================================================================

Core question: Does the same direction that gives T↑C↓ for correct
statements give T↓C↑ for incorrect statements?

If YES → direction encodes compatibility gradient (not just value preference)
If NO  → direction is a value preference, not a relational mechanism

Design:
- Correct condition: "The elephant is big." (compatible)
  delta_h_correct = h(clean_correct) - h(corrupt_correct)
- Incorrect condition: "The elephant is small." (incompatible)
  delta_h_incorrect = h(clean_incorrect) - h(corrupt_incorrect)

Tests:
1. Apply correct-directions (L1, L2_crossfit) to incorrect-corrupt prompts
   → Expected: compatible_token ↑, incompatible_token ↓ (SYMMETRIC)
2. Cosine similarity between delta_h_correct and delta_h_incorrect
   → Negative = mirror structure
3. Replication: correct-directions on correct-corrupt prompts

Layers (IDEAL + negative control):
- Qwen3:  L4 (color=IDEAL), L20 (no IDEAL)
- DS7B:   L12 (size=IDEAL, L2>L1), L4 (no IDEAL)
- GLM4:   L10 (moisture=IDEAL), L30 (no IDEAL)
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


def build_pairs():
    """Build sample pairs with both correct and incorrect conditions."""
    pairs = []
    for cat, cat_data in RICH_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            prompt_obj = DISPLAY_TO_PROMPT.get(obj_name, obj_name)
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name,
                        'prompt_obj': prompt_obj,
                        'target': target,       # compatible value
                        'comp': comp,            # incompatible value
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


def test_direction(model, tokenizer, layers_list, device, li,
                   delta_np, prompt, tid, cid):
    """Inject direction at layer li and return logit stats."""
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


def run_phase396(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 396: SYMMETRIC Verification ({model_name}) [{timestamp}] ===")

    LAYER_CONFIGS = {
        "qwen3": [4, 20],      # L4=IDEAL, L20=no IDEAL
        "deepseek7b": [4, 12], # L12=IDEAL, L4=no IDEAL
        "glm4": [10, 30],      # L10=IDEAL, L30=no IDEAL
    }
    layer_indices = LAYER_CONFIGS.get(model_name, [4, 20])

    pairs = build_pairs()
    N = len(pairs)

    oc_groups = defaultdict(list)
    for i, p in enumerate(pairs):
        oc_groups[(p['obj'], p['cat'])].append(i)

    print(f"  Total: {N} pairs, {len(oc_groups)} groups")

    # Build prompts for all 4 conditions
    prompts_correct_clean = []     # "The elephant is big."
    prompts_correct_corrupt = []   # "The item is big."
    prompts_incorrect_clean = []   # "The elephant is small."
    prompts_incorrect_corrupt = [] # "The item is small."

    for p in pairs:
        tpl = FRAMES[p['frame_idx']]
        ctpl = CORRUPT_FRAMES[p['frame_idx']]
        prompts_correct_clean.append(tpl.format(obj=p['prompt_obj'], attr=p['target']))
        prompts_correct_corrupt.append(ctpl.format(attr=p['target']))
        prompts_incorrect_clean.append(tpl.format(obj=p['prompt_obj'], attr=p['comp']))
        prompts_incorrect_corrupt.append(ctpl.format(attr=p['comp']))

    # Token IDs for target and comp
    token_ids = {}
    for cat_data in RICH_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer_temp = None  # will set after loading
                        token_ids[tok] = None  # placeholder

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Resolve token IDs
    for cat_data in RICH_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if token_ids.get(tok) is None:
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

        # === Step 1: Collect activations for all 4 conditions ===
        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_hook('h'))

        h_cc = np.zeros((N, d_model), dtype=np.float32)  # correct clean
        h_ccor = np.zeros((N, d_model), dtype=np.float32) # correct corrupt
        h_ic = np.zeros((N, d_model), dtype=np.float32)   # incorrect clean
        h_icor = np.zeros((N, d_model), dtype=np.float32) # incorrect corrupt

        # Also collect baseline logits for corrupt prompts
        baseline_correct = []
        baseline_incorrect = []

        for i in range(N):
            # Correct clean
            captured.clear()
            inputs = tokenizer(prompts_correct_clean[i], return_tensors="pt",
                             truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_cc[i] = captured['h'][0, -1].numpy()

            # Correct corrupt + baseline logits
            captured.clear()
            inputs = tokenizer(prompts_correct_corrupt[i], return_tensors="pt",
                             truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_ccor[i] = captured['h'][0, -1].numpy()
            tid = token_ids.get(pairs[i]['target'])
            cid = token_ids.get(pairs[i]['comp'])
            baseline_correct.append(get_logit_stats(out.logits[0, -1], tid, cid))

            # Incorrect clean
            captured.clear()
            inputs = tokenizer(prompts_incorrect_clean[i], return_tensors="pt",
                             truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_ic[i] = captured['h'][0, -1].numpy()

            # Incorrect corrupt + baseline logits
            captured.clear()
            inputs = tokenizer(prompts_incorrect_corrupt[i], return_tensors="pt",
                             truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_icor[i] = captured['h'][0, -1].numpy()
            # For incorrect corrupt, track same tokens (target=compatible, comp=incompatible)
            baseline_incorrect.append(get_logit_stats(out.logits[0, -1], tid, cid))

            if (i+1) % 40 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()

        # === Step 2: Compute delta_h ===
        dh_correct = h_cc - h_ccor    # correct condition
        dh_incorrect = h_ic - h_icor  # incorrect condition

        # === Step 3: ANOVA decomposition for correct condition ===
        obj_labels = [p['obj'] for p in pairs]
        cat_labels = [p['cat'] for p in pairs]
        mu_correct = dh_correct.mean(axis=0)

        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        cat_centroids_correct = np.zeros((len(unique_cats), d_model))
        cat_counts = np.zeros(len(unique_cats))
        for i, c in enumerate(cat_labels):
            cat_centroids_correct[cat_to_idx[c]] += dh_correct[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                cat_centroids_correct[j] /= cat_counts[j]
        A_cat_correct = np.zeros_like(dh_correct)
        for i, c in enumerate(cat_labels):
            A_cat_correct[i] = cat_centroids_correct[cat_to_idx[c]] - mu_correct

        dh_resid_cat_correct = dh_correct - mu_correct - A_cat_correct
        obj_cat_keys = [(obj_labels[i], cat_labels[i]) for i in range(N)]
        unique_obj_cats = sorted(set(obj_cat_keys))
        oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
        oc_centroids_correct = np.zeros((len(unique_obj_cats), d_model))
        oc_counts = np.zeros(len(unique_obj_cats))
        for i, oc in enumerate(obj_cat_keys):
            oc_centroids_correct[oc_to_idx[oc]] += dh_resid_cat_correct[i]
            oc_counts[oc_to_idx[oc]] += 1
        for j in range(len(unique_obj_cats)):
            if oc_counts[j] > 0:
                oc_centroids_correct[j] /= oc_counts[j]
        A_obj_cat_correct = np.zeros_like(dh_correct)
        for i, oc in enumerate(obj_cat_keys):
            A_obj_cat_correct[i] = oc_centroids_correct[oc_to_idx[oc]]

        # LOPO cross-fit for correct condition
        A_obj_cat_cf_correct = np.zeros_like(dh_correct)
        for oc_key in unique_obj_cats:
            group_indices = oc_groups[oc_key]
            for test_i in group_indices:
                train_indices = [j for j in group_indices if j != test_i]
                if len(train_indices) == 0:
                    A_obj_cat_cf_correct[test_i] = np.zeros(d_model)
                else:
                    A_obj_cat_cf_correct[test_i] = np.mean(
                        [dh_resid_cat_correct[j] for j in train_indices], axis=0)

        # === Step 4: Cosine similarity between correct and incorrect delta_h ===
        cos_per_sample = []
        for i in range(N):
            v1 = dh_correct[i]
            v2 = dh_incorrect[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_per_sample.append(float(np.dot(v1, v2) / (n1 * n2)))
            else:
                cos_per_sample.append(0.0)

        # Category-level cosine similarity
        cat_cos = {}
        for cat in unique_cats:
            cat_mask = np.array(cat_labels) == cat
            dh_c = dh_correct[cat_mask].mean(axis=0)
            dh_i = dh_incorrect[cat_mask].mean(axis=0)
            n1, n2 = np.linalg.norm(dh_c), np.linalg.norm(dh_i)
            if n1 > 1e-10 and n2 > 1e-10:
                cat_cos[cat] = float(np.dot(dh_c, dh_i) / (n1 * n2))
            else:
                cat_cos[cat] = 0.0

        print(f"\n  Cosine similarity (correct vs incorrect delta_h):")
        for cat in unique_cats:
            c = cat_cos[cat]
            sym_label = "MIRROR" if c < -0.3 else ("ALIGNED" if c > 0.3 else "NEUTRAL")
            print(f"    {cat}: cos={c:+.4f} [{sym_label}]")
        mean_cos = np.mean(cos_per_sample)
        print(f"    Mean per-sample: {mean_cos:+.4f}")

        # === Step 5: Define directions from correct condition ===
        directions = {
            "L1_correct": lambda i: mu_correct + A_cat_correct[i],
            "L2_cf_correct": lambda i: mu_correct + A_cat_correct[i] + A_obj_cat_cf_correct[i],
        }

        # === Step 6: Test directions on both conditions ===
        version_results = {}

        for ver_name, direction_fn in directions.items():
            t0_v = time.time()
            # Results for correct condition
            td_correct_arr = np.zeros(N)
            cd_correct_arr = np.zeros(N)
            other_correct_arr = np.zeros(N)

            # Results for incorrect condition (SYMMETRIC test)
            td_incorrect_arr = np.zeros(N)
            cd_incorrect_arr = np.zeros(N)
            other_incorrect_arr = np.zeros(N)

            for i in range(N):
                p = pairs[i]
                tid = token_ids.get(p['target'])  # compatible token
                cid = token_ids.get(p['comp'])     # incompatible token
                if tid is None or cid is None:
                    continue

                delta_np = direction_fn(i)

                # --- Correct condition: inject into correct-corrupt prompt ---
                stats = test_direction(model, tokenizer, layers_list, device, li,
                                      delta_np, prompts_correct_corrupt[i], tid, cid)
                bs = baseline_correct[i]
                td_correct_arr[i] = stats['t_logit'] - bs['t_logit']
                cd_correct_arr[i] = stats['c_logit'] - bs['c_logit']
                other_correct_arr[i] = stats['other_mean'] - bs['other_mean']

                # --- Incorrect condition: inject into incorrect-corrupt prompt ---
                # Same direction, different prompt
                # Track same tokens: tid=compatible, cid=incompatible
                # SYMMETRIC predicts: compatible↑, incompatible↓
                # Which in terms of prompt: prompted(incompatible)↓, alternative(compatible)↑
                stats_inc = test_direction(model, tokenizer, layers_list, device, li,
                                          delta_np, prompts_incorrect_corrupt[i], tid, cid)
                bs_inc = baseline_incorrect[i]
                td_incorrect_arr[i] = stats_inc['t_logit'] - bs_inc['t_logit']
                cd_incorrect_arr[i] = stats_inc['c_logit'] - bs_inc['c_logit']
                other_incorrect_arr[i] = stats_inc['other_mean'] - bs_inc['other_mean']

                if (i+1) % 40 == 0:
                    print(f"    {ver_name}: {i+1}/{N} ({time.time()-t0_v:.0f}s)")

            # Per-category breakdown
            cat_breakdown = {}
            for cat in unique_cats:
                cat_mask = np.array(cat_labels) == cat
                if cat_mask.sum() == 0: continue

                # Correct condition
                c_td = td_correct_arr[cat_mask]
                c_cd = cd_correct_arr[cat_mask]
                c_other = other_correct_arr[cat_mask]
                c_mech = classify_mechanism(float(np.mean(c_td)), float(np.mean(c_cd)))
                c_dmg = 0.0
                if abs(np.mean(c_td)) > 1e-6:
                    c_dmg = abs(np.mean(c_other)) / abs(np.mean(c_td))

                # Incorrect condition (SYMMETRIC)
                i_td = td_incorrect_arr[cat_mask]
                i_cd = cd_incorrect_arr[cat_mask]
                i_other = other_incorrect_arr[cat_mask]
                # For incorrect: td = compatible_token delta, cd = incompatible_token delta
                # SYMMETRIC: compatible↑ (td>0), incompatible↓ (cd<0) → same as IDEAL
                # Or in prompt terms: prompted(incomp)↓, alternative(comp)↑
                i_mech = classify_mechanism(float(np.mean(i_td)), float(np.mean(i_cd)))
                i_dmg = 0.0
                if abs(np.mean(i_td)) > 1e-6:
                    i_dmg = abs(np.mean(i_other)) / abs(np.mean(i_td))

                # SYMMETRIC score: does the mirror pattern hold?
                # Correct: T(comp)↑ C(incomp)↓  → IDEAL
                # SYMMETRIC: same direction on incorrect prompt also gives T(comp)↑ C(incomp)↓
                # i.e., the direction always pushes toward compatible, away from incompatible
                correct_is_ideal = (np.mean(c_td) > 0 and np.mean(c_cd) < 0)
                incorrect_is_ideal = (np.mean(i_td) > 0 and np.mean(i_cd) < 0)
                symmetric_score = "FULL_SYMMETRIC" if (correct_is_ideal and incorrect_is_ideal) else \
                                  "HALF_SYMMETRIC" if correct_is_ideal else \
                                  "NO_SYMMETRIC"

                cat_breakdown[cat] = {
                    # Correct condition
                    'correct_target_delta': float(np.mean(c_td)),    # compatible token
                    'correct_comp_delta': float(np.mean(c_cd)),     # incompatible token
                    'correct_mechanism': c_mech,
                    'correct_damage': float(c_dmg),
                    # Incorrect condition
                    'incorrect_target_delta': float(np.mean(i_td)), # compatible token (same!)
                    'incorrect_comp_delta': float(np.mean(i_cd)),   # incompatible token (same!)
                    'incorrect_mechanism': i_mech,
                    'incorrect_damage': float(i_dmg),
                    # SYMMETRIC
                    'symmetric_score': symmetric_score,
                }

            version_results[ver_name] = {
                'category_breakdown': cat_breakdown,
                'mean_cos_correct_incorrect': float(mean_cos),
                'cat_cos': cat_cos,
            }

            # Print summary
            print(f"\n  {ver_name} results:")
            for cat in unique_cats:
                cb = cat_breakdown[cat]
                print(f"    {cat}: CORRECT T(comp)={cb['correct_target_delta']:+.4f} "
                      f"C(incomp)={cb['correct_comp_delta']:+.4f} [{cb['correct_mechanism']}] | "
                      f"INCORRECT T(comp)={cb['incorrect_target_delta']:+.4f} "
                      f"C(incomp)={cb['incorrect_comp_delta']:+.4f} [{cb['incorrect_mechanism']}] | "
                      f"SYMM={cb['symmetric_score']}")

        results['per_layer'][str(li)] = version_results
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")

    # === Final Summary ===
    print(f"\n{'='*70}")
    print(f"--- Phase 396 SYMMETRIC Summary ({model_name}) ---")
    for li in layer_indices:
        vr = results['per_layer'][str(li)]
        print(f"\n  Layer {li}:")
        print(f"    Mean cos(correct,incorrect): {vr.get('L1_correct', vr.get('L1_category',{})).get('mean_cos_correct_incorrect',0):+.4f}")
        for ver_name in ["L1_correct", "L2_cf_correct"]:
            if ver_name in vr:
                cb = vr[ver_name]['category_breakdown']
                for cat in sorted(cb.keys()):
                    c = cb[cat]
                    print(f"    {ver_name:16s} {cat:10s}: "
                          f"CORR[{c['correct_mechanism']:10s} T={c['correct_target_delta']:+.4f} C={c['correct_comp_delta']:+.4f}] "
                          f"INCORR[{c['incorrect_mechanism']:10s} T={c['incorrect_target_delta']:+.4f} C={c['incorrect_comp_delta']:+.4f}] "
                          f"→ {c['symmetric_score']}")

    # Save
    out_dir = ROOT / "results" / "phase396_symmetric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase396.json"
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
    run_phase396(model_name)
