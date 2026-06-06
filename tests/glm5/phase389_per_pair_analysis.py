"""
Phase 389: Per-Pair Centroid Effect Analysis
=============================================
Goal: Understand why only 30-70% of pairs show positive centroid effects.

Key questions:
1. Which pairs have positive effects? Which have negative?
2. Is the effect distributed by category? By object? By target/competitor?
3. Does the correct-value condition vs incorrect-value condition explain the variance?
4. What is the correlation between baseline logit_diff and add effect?

Design:
- Use ALL_PAIRS (151 correct pairs) + incorrect-value pairs
- Compute ANOVA on all correct pairs
- Test per-pair A centroid add effect
- Analyze: effect vs category, effect vs baseline logit_diff, correct vs incorrect
- Record per-pair results for detailed analysis

This is the most informative next step because:
- Phase 388 showed centroid effects are non-uniform across pairs
- Understanding which pairs benefit vs hurt from centroid addition
  will reveal the mechanism of category encoding
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model

# ============================================================
# Data: ALL_PAIRS (151 correct pairs) + INCORRECT pairs
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
    ("ink","black","white"),("obsidian","black","white"),
    ("carrot","orange","blue"),("sunset","orange","blue"),("pumpkin","orange","blue"),
    ("tiger","orange","blue"),
    ("lavender","purple","blue"),("amethyst","purple","blue"),("plum","purple","blue"),
    ("violet","purple","blue"),
    ("chocolate","brown","blue"),("wood","brown","blue"),("coffee","brown","blue"),
]

TEMP_PAIRS = [
    ("fire","hot","cold"),("sun","hot","cold"),("desert","hot","cold"),
    ("lava","hot","cold"),("oven","hot","cold"),("volcano","hot","cold"),
    ("summer","hot","cold"),("furnace","hot","cold"),("stove","hot","cold"),
    ("ice","cold","hot"),("snow","cold","hot"),("winter","cold","hot"),
    ("arctic","cold","hot"),("freezer","cold","hot"),("glacier","cold","hot"),
    ("fridge","cold","hot"),("frost","cold","hot"),
    ("spring","warm","cold"),("autumn","cool","hot"),
    ("tropics","hot","cold"),("equator","hot","cold"),
    ("polar","cold","hot"),("tundra","cold","hot"),("refrigerator","cold","hot"),
    ("heater","hot","cold"),
]

MOISTURE_PAIRS = [
    ("ocean","wet","dry"),("rain","wet","dry"),("river","wet","dry"),
    ("lake","wet","dry"),("sponge","wet","dry"),("swamp","wet","dry"),
    ("dew","wet","dry"),("fog","wet","dry"),("mist","wet","dry"),
    ("desert","dry","wet"),("sand","dry","wet"),("dust","dry","wet"),
    ("bone","dry","wet"),("cracker","dry","wet"),("tinder","dry","wet"),
    ("sahara","dry","wet"),("cactus","dry","wet"),
    ("jungle","wet","dry"),("forest","wet","dry"),("marsh","wet","dry"),
    ("arid","dry","wet"),("parched","dry","wet"),("thirsty","dry","wet"),
    ("flood","wet","dry"),
]

SIZE_PAIRS = [
    ("elephant","big","small"),("mountain","big","small"),("building","big","small"),
    ("whale","big","small"),("planet","big","small"),("continent","big","small"),
    ("ant","small","big"),("grain","small","big"),("needle","small","big"),
    ("speck","small","big"),("atom","small","big"),("pixel","small","big"),
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

# Build correct pairs
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

    # Build incorrect pairs: same objects, swap target and competitor
    # Only use a SUBSET of incorrect pairs to reduce runtime
    INCORRECT_PAIRS = []
    INCORRECT_CATEGORIES = []
    for i, (obj, target, comp) in enumerate(ALL_PAIRS):
        if i % 2 == 0:  # Only every other pair to reduce test time
            INCORRECT_PAIRS.append((obj, comp, target))
            INCORRECT_CATEGORIES.append(PAIR_CATEGORIES[i])

TEMPLATE = "The {obj} is {attr}."
CORRUPT_BASELINE = "The item"

# ============================================================
# Model loading
# ============================================================
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
    return model, tokenizer

# ============================================================
# ANOVA decomposition
# ============================================================
def anova_decomposition(dh_raw, object_labels, category_labels):
    N, d = dh_raw.shape
    mu = dh_raw.mean(axis=0)
    unique_objects = sorted(set(object_labels))
    obj_to_idx = {o: i for i, o in enumerate(unique_objects)}
    obj_counts = np.zeros(len(unique_objects))
    obj_centroids = np.zeros((len(unique_objects), d))
    for i, o in enumerate(object_labels):
        idx = obj_to_idx[o]
        obj_centroids[idx] += dh_raw[i]
        obj_counts[idx] += 1
    for j in range(len(unique_objects)):
        if obj_counts[j] > 0:
            obj_centroids[j] /= obj_counts[j]
    I_comp = np.zeros_like(dh_raw)
    for i, o in enumerate(object_labels):
        I_comp[i] = obj_centroids[obj_to_idx[o]] - mu
    dh_resid_I = dh_raw - mu - I_comp
    unique_cats = sorted(set(category_labels))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_counts = np.zeros(len(unique_cats))
    cat_centroids = np.zeros((len(unique_cats), d))
    for i, c in enumerate(category_labels):
        idx = cat_to_idx[c]
        cat_centroids[idx] += dh_resid_I[i]
        cat_counts[idx] += 1
    for j in range(len(unique_cats)):
        if cat_counts[j] > 0:
            cat_centroids[j] /= cat_counts[j]
    A_comp = np.zeros_like(dh_raw)
    for i, c in enumerate(category_labels):
        A_comp[i] = cat_centroids[cat_to_idx[c]]
    eps_comp = dh_raw - mu - I_comp - A_comp
    ss_total = np.sum((dh_raw - mu) ** 2)
    r2_I = np.sum(I_comp ** 2) / ss_total if ss_total > 0 else 0
    r2_A = np.sum(A_comp ** 2) / ss_total if ss_total > 0 else 0
    return {
        'mu': mu, 'I_comp': I_comp, 'A_comp': A_comp, 'eps_comp': eps_comp,
        'r2_I': float(r2_I), 'r2_A': float(r2_A),
        'cat_centroids': cat_centroids, 'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats,
    }

# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase389_per_pair_analysis.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    assert model_name in MODEL_CONFIGS

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 389: Per-Pair Centroid Analysis ({model_name}) [{timestamp}] ===")

    LAYER_CONFIGS = {
        "qwen3": [20],
        "glm4": [20],
        "deepseek7b": [4],
    }
    layer_indices = LAYER_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    print(f"  Model loaded: {n_layers} layers")

    # ---- Step 1: Collect activations for ALL pairs (correct + incorrect) ----
    all_pairs = ALL_PAIRS + INCORRECT_PAIRS
    all_cats = PAIR_CATEGORIES + INCORRECT_CATEGORIES
    n_correct = len(ALL_PAIRS)
    n_total = len(all_pairs)
    print(f"\n--- Step 1: Collecting activations for {n_total} pairs ({n_correct} correct + {len(INCORRECT_PAIRS)} incorrect) ---")

    all_dh = {li: [] for li in layer_indices}
    all_baseline_ld = []

    for idx, (obj, target, comp) in enumerate(all_pairs):
        is_correct = idx < n_correct
        cond = "correct" if is_correct else "incorrect"

        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target)

        clean_ids = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(device)
        corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(device)

        target_tok = tokenizer(target, add_special_tokens=False).input_ids[0]
        comp_tok = tokenizer(comp, add_special_tokens=False).input_ids[0]

        clean_acts = {}
        corrupt_acts = {}

        def make_collect_hook(storage, layer_key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    hs = output[0, -1, :].detach().cpu().float().numpy()
                storage[layer_key] = hs
            return hook_fn

        hooks = []
        for li in layer_indices:
            h = layers_list[li].register_forward_hook(make_collect_hook(clean_acts, li))
            hooks.append(h)
        with torch.no_grad():
            model(clean_ids)
        for h in hooks:
            h.remove()

        hooks = []
        for li in layer_indices:
            h = layers_list[li].register_forward_hook(make_collect_hook(corrupt_acts, li))
            hooks.append(h)
        with torch.no_grad():
            corrupt_out = model(corrupt_ids)
        for h in hooks:
            h.remove()

        for li in layer_indices:
            all_dh[li].append(clean_acts[li] - corrupt_acts[li])

        corrupt_logits = corrupt_out.logits[0, -1]
        baseline_ld = (corrupt_logits[target_tok] - corrupt_logits[comp_tok]).float().item()
        all_baseline_ld.append(baseline_ld)

        if idx % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Collected {idx+1}/{n_total} ({cond})")

    for li in layer_indices:
        all_dh[li] = np.stack(all_dh[li], axis=0)
    all_baseline_ld = np.array(all_baseline_ld)
    print(f"  Done: {[(li, all_dh[li].shape) for li in layer_indices]}")

    # ---- Step 2: Compute ANOVA on correct pairs only ----
    print(f"\n--- Step 2: ANOVA on {n_correct} correct pairs ---")
    correct_obj_labels = [ALL_PAIRS[i][0] for i in range(n_correct)]

    results = {
        "model": model_name, "timestamp": timestamp,
        "n_correct": n_correct, "n_incorrect": len(INCORRECT_PAIRS),
        "layers": layer_indices,
        "per_pair": {},  # per-pair results
        "category_summary": {},  # per-category summary
        "condition_comparison": {},  # correct vs incorrect
    }

    for li in layer_indices:
        dh_correct = all_dh[li][:n_correct]
        anova = anova_decomposition(dh_correct, correct_obj_labels, PAIR_CATEGORIES)
        print(f"  L{li}: R2_I={anova['r2_I']:.4f}, R2_A={anova['r2_A']:.6f}")

        # ---- Step 3: Per-pair causal test on ALL pairs ----
        print(f"\n--- Step 3: Per-pair causal test at L{li} ---")
        A_comp_correct = anova['A_comp']  # (n_correct, d)

        per_pair_results = []

        for idx in range(n_total):
            obj, target, comp = all_pairs[idx]
            is_correct = idx < n_correct
            cat = all_cats[idx]

            corrupt_prompt = TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target)
            corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(device)
            target_tok = tokenizer(target, add_special_tokens=False).input_ids[0]
            comp_tok = tokenizer(comp, add_special_tokens=False).input_ids[0]

            # Get A component for this pair
            if is_correct:
                # Use the A component from correct-pair ANOVA
                delta = torch.tensor(A_comp_correct[idx], dtype=torch.bfloat16, device=device)
            else:
                # For incorrect pairs: find the matching correct pair and use its category centroid
                incorrect_idx = idx - n_correct
                # The incorrect pair shares the same category as the correct pair
                # Use the category centroid from the correct-pair ANOVA
                cat_idx = anova['cat_to_idx'].get(cat, None)
                if cat_idx is not None:
                    # Use category centroid from dh_resid_I of correct ANOVA
                    delta_np = anova['cat_centroids'][cat_idx]
                else:
                    delta_np = np.zeros(dh_correct.shape[1])
                delta = torch.tensor(delta_np, dtype=torch.bfloat16, device=device)

            # Baseline logit_diff (pre-computed)
            baseline_ld = all_baseline_ld[idx]

            # Patched forward pass
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
                with torch.no_grad():
                    patched_out = model(corrupt_ids)
                    patched_logits = patched_out.logits[0, -1]
                    # Use COMPATIBLE value - INCOMPATIBLE value regardless of correct/incorrect
                    # For correct pairs: target=compatible, comp=incompatible
                    # For incorrect pairs: target=incompatible, comp=compatible
                    # So we always compute: logit(compatible) - logit(incompatible)
                    if is_correct:
                        patched_ld = (patched_logits[target_tok] - patched_logits[comp_tok]).float().item()
                    else:
                        # incorrect: target is incompatible, comp is compatible
                        patched_ld = (patched_logits[comp_tok] - patched_logits[target_tok]).float().item()
            finally:
                handle.remove()

            # For baseline, also compute compatible - incompatible
            if is_correct:
                baseline_compat_ld = baseline_ld  # already target - comp = compatible - incompatible
            else:
                baseline_compat_ld = -baseline_ld  # target is incompatible, so flip sign

            add_effect = patched_ld - baseline_compat_ld

            per_pair_results.append({
                "idx": idx,
                "object": obj,
                "target": target,
                "competitor": comp,
                "category": cat,
                "condition": "correct" if is_correct else "incorrect",
                "baseline_ld": float(baseline_ld),
                "baseline_compat_ld": float(baseline_compat_ld),
                "add_effect": float(add_effect),
                "A_norm": float(np.linalg.norm(delta.cpu().float().numpy())),
            })

            if idx % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.0f}s] Tested {idx+1}/{n_total}")

        # ---- Step 4: Analyze per-pair results ----
        print(f"\n--- Step 4: Analysis at L{li} ---")

        # By condition
        correct_effects = [r['add_effect'] for r in per_pair_results if r['condition'] == 'correct']
        incorrect_effects = [r['add_effect'] for r in per_pair_results if r['condition'] == 'incorrect']

        print(f"  Correct: mean={np.mean(correct_effects):+.4f}, t={np.mean(correct_effects)/(np.std(correct_effects)/np.sqrt(len(correct_effects))):+.2f}, pos%={100*np.mean([e>0 for e in correct_effects]):.0f}%")
        print(f"  Incorrect: mean={np.mean(incorrect_effects):+.4f}, t={np.mean(incorrect_effects)/(np.std(incorrect_effects)/np.sqrt(len(incorrect_effects))):+.2f}, pos%={100*np.mean([e>0 for e in incorrect_effects]):.0f}%")

        # By category
        cat_effects = defaultdict(lambda: {'correct': [], 'incorrect': []})
        for r in per_pair_results:
            cat_effects[r['category']][r['condition']].append(r['add_effect'])

        print(f"\n  By category:")
        for cat in sorted(cat_effects.keys()):
            ce = cat_effects[cat]
            c_mean = np.mean(ce['correct']) if ce['correct'] else 0
            i_mean = np.mean(ce['incorrect']) if ce['incorrect'] else 0
            c_pos = 100*np.mean([e>0 for e in ce['correct']]) if ce['correct'] else 0
            i_pos = 100*np.mean([e>0 for e in ce['incorrect']]) if ce['incorrect'] else 0
            print(f"    {cat}: correct={c_mean:+.4f}({c_pos:.0f}%pos) incorrect={i_mean:+.4f}({i_pos:.0f}%pos)")

        # Correlation with baseline
        baseline_compat = [r['baseline_compat_ld'] for r in per_pair_results]
        add_effects = [r['add_effect'] for r in per_pair_results]
        if np.std(baseline_compat) > 0 and np.std(add_effects) > 0:
            corr = np.corrcoef(baseline_compat, add_effects)[0, 1]
        else:
            corr = 0.0
        print(f"\n  Correlation(baseline_compat_ld, add_effect): {corr:.3f}")

        # Save results
        results["per_pair"][str(li)] = per_pair_results
        results["category_summary"][str(li)] = {
            cat: {
                "correct_mean": float(np.mean(ce['correct'])) if ce['correct'] else 0,
                "correct_pos_pct": float(100*np.mean([e>0 for e in ce['correct']])) if ce['correct'] else 0,
                "incorrect_mean": float(np.mean(ce['incorrect'])) if ce['incorrect'] else 0,
                "incorrect_pos_pct": float(100*np.mean([e>0 for e in ce['incorrect']])) if ce['incorrect'] else 0,
                "n_correct": len(ce['correct']),
                "n_incorrect": len(ce['incorrect']),
            }
            for cat, ce in cat_effects.items()
        }
        results["condition_comparison"][str(li)] = {
            "correct_mean": float(np.mean(correct_effects)),
            "correct_t": float(np.mean(correct_effects) / (np.std(correct_effects)/np.sqrt(len(correct_effects)))) if np.std(correct_effects) > 0 else 0,
            "correct_pos_pct": float(100*np.mean([e>0 for e in correct_effects])),
            "incorrect_mean": float(np.mean(incorrect_effects)),
            "incorrect_t": float(np.mean(incorrect_effects) / (np.std(incorrect_effects)/np.sqrt(len(incorrect_effects)))) if np.std(incorrect_effects) > 0 else 0,
            "incorrect_pos_pct": float(100*np.mean([e>0 for e in incorrect_effects])),
            "baseline_add_corr": float(corr),
        }

    # Save
    out_dir = ROOT / "results" / "phase389_per_pair_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase389.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Cleanup
    release_model(model)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released, GPU memory cleared.")

if __name__ == "__main__":
    main()
