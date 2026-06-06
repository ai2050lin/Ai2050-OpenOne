"""
Phase 388: Centroid Bootstrap Stability Test
=============================================
Goal: Determine whether category centroid causal effect is stable
across different sample sizes, using only correct-value pairs (ALL_PAIRS).

Key question from Phase 387 failure:
- Phase 386 used 151 correct pairs → A_add positive (t=3-7)
- Phase 387b mixed correct/incorrect (48 pairs) → A_add negative (t=-3)
- Is the reversal due to: (a) sample size, or (b) correct/incorrect mixing?

Design:
- Use ALL_PAIRS (151 correct pairs, 7 categories)
- Pre-collect all activations and baseline logit_diffs
- For each sample size N in [48, 96, 151]:
  - Randomly sample N pairs (stratified by category)
  - Compute ANOVA decomposition on these N pairs (numpy only)
  - Test A centroid causal effect on these N pairs
  - Repeat with 5 random seeds
- Compare with full-data (151 pairs) ANOVA as reference
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
# Data: ALL_PAIRS (151 correct pairs, 7 categories)
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
# ANOVA decomposition (I + A + eps)
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
    r2_eps = np.sum(eps_comp ** 2) / ss_total if ss_total > 0 else 0

    return {
        'mu': mu, 'I_comp': I_comp, 'A_comp': A_comp, 'eps_comp': eps_comp,
        'r2_I': float(r2_I), 'r2_A': float(r2_A), 'r2_eps': float(r2_eps),
        'cat_centroids': cat_centroids, 'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats, 'unique_objects': unique_objects,
        'obj_centroids': obj_centroids, 'obj_to_idx': obj_to_idx,
    }

# ============================================================
# Stratified subsample
# ============================================================
def stratified_subsample(n, rng):
    """Sample n pair indices, maintaining category proportions."""
    cat_indices = defaultdict(list)
    for i, c in enumerate(PAIR_CATEGORIES):
        cat_indices[c].append(i)

    total = len(ALL_PAIRS)
    selected = []
    for cat, indices in cat_indices.items():
        n_cat = max(1, round(n * len(indices) / total))
        n_cat = min(n_cat, len(indices))
        chosen = rng.choice(indices, size=n_cat, replace=False).tolist()
        selected.extend(chosen)

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in range(len(ALL_PAIRS)) if i not in selected]
        if remaining:
            extra = rng.choice(remaining, size=min(n - len(selected), len(remaining)), replace=False).tolist()
            selected.extend(extra)

    return sorted(selected)

# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase388_centroid_bootstrap_stability.py <model_name>")
        print("  model_name: qwen3 | glm4 | deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 388: Centroid Bootstrap Stability ({model_name}) [{timestamp}] ===")

    SAMPLE_SIZES = [48, 96, 151]
    SEEDS = [42, 123, 456, 789, 1024]
    LAYER_CONFIGS = {
        "qwen3": [4, 12, 20, 28],
        "glm4": [4, 12, 20, 30],
        "deepseek7b": [4, 8, 12, 20, 24],
    }
    layer_indices = LAYER_CONFIGS[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    print(f"  Model loaded: {n_layers} layers")

    # ---- Step 1: Collect all activations + baseline logit_diffs ----
    print(f"\n--- Step 1: Collecting activations + baselines for {len(ALL_PAIRS)} pairs ---")
    all_dh = {li: [] for li in layer_indices}  # dh_raw per layer
    all_baseline_ld = []  # baseline logit_diff per pair

    for idx, (obj, target, comp) in enumerate(ALL_PAIRS):
        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target)

        clean_ids = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(device)
        corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(device)

        target_tok = tokenizer(target, add_special_tokens=False).input_ids[0]
        comp_tok = tokenizer(comp, add_special_tokens=False).input_ids[0]

        # Collect activations at all target layers for both clean and corrupt
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

        # Compute dh_raw per layer
        for li in layer_indices:
            dh = clean_acts[li] - corrupt_acts[li]
            all_dh[li].append(dh)

        # Baseline logit_diff (from corrupt)
        corrupt_logits = corrupt_out.logits[0, -1]
        baseline_ld = (corrupt_logits[target_tok] - corrupt_logits[comp_tok]).float().item()
        all_baseline_ld.append(baseline_ld)

        if idx % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Collected {idx+1}/{len(ALL_PAIRS)} pairs")

    # Stack
    for li in layer_indices:
        all_dh[li] = np.stack(all_dh[li], axis=0)
    all_baseline_ld = np.array(all_baseline_ld)

    print(f"  Activations collected: {[(li, all_dh[li].shape) for li in layer_indices]}")

    # ---- Step 2: Full ANOVA (151 pairs) as reference ----
    print(f"\n--- Step 2: Full ANOVA (151 pairs) reference ---")
    full_anova = {}
    for li in layer_indices:
        obj_labels = [ALL_PAIRS[i][0] for i in range(len(ALL_PAIRS))]
        anova = anova_decomposition(all_dh[li], obj_labels, PAIR_CATEGORIES)
        full_anova[li] = anova
        print(f"  L{li}: R2_I={anova['r2_I']:.4f}, R2_A={anova['r2_A']:.6f}, R2_eps={anova['r2_eps']:.4f}")

    # ---- Step 3: Bootstrap subsample tests ----
    print(f"\n--- Step 3: Bootstrap subsample causal tests ---")
    results = {
        "model": model_name, "timestamp": timestamp, "n_total": len(ALL_PAIRS),
        "sample_sizes": SAMPLE_SIZES, "seeds": SEEDS, "layers": layer_indices,
        "full_anova": {}, "bootstrap": {}
    }

    for li in layer_indices:
        anova = full_anova[li]
        results["full_anova"][str(li)] = {
            "r2_I": anova['r2_I'], "r2_A": anova['r2_A'], "r2_eps": anova['r2_eps'],
        }

    total_tests = len(SAMPLE_SIZES) * len(SEEDS) * len(layer_indices)
    test_count = 0

    for n_sample in SAMPLE_SIZES:
        results["bootstrap"][str(n_sample)] = {}
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            sub_indices = stratified_subsample(n_sample, rng)
            sub_pairs = [ALL_PAIRS[i] for i in sub_indices]
            sub_cats = [PAIR_CATEGORIES[i] for i in sub_indices]
            sub_obj_labels = [ALL_PAIRS[i][0] for i in sub_indices]

            seed_results = {}

            for li in layer_indices:
                test_count += 1
                elapsed = time.time() - start_time
                print(f"\n  [{elapsed:.0f}s] Test {test_count}/{total_tests}: "
                      f"n={n_sample}, seed={seed}, L{li}")

                # Subsample activations
                dh_sub = all_dh[li][sub_indices]

                # ANOVA on subsample
                anova_sub = anova_decomposition(dh_sub, sub_obj_labels, sub_cats)
                A_comp = anova_sub['A_comp']  # (n_sample, d)

                # Causal test: add A_comp[i] to corrupt at layer li
                add_effects = []
                for i in range(n_sample):
                    obj, target, comp_word = sub_pairs[i]
                    corrupt_prompt = TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target)
                    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(device)
                    target_tok = tokenizer(target, add_special_tokens=False).input_ids[0]
                    comp_tok = tokenizer(comp_word, add_special_tokens=False).input_ids[0]

                    delta = torch.tensor(A_comp[i], dtype=torch.bfloat16, device=device)

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
                            patched_ld = (patched_logits[target_tok] - patched_logits[comp_tok]).float().item()
                    finally:
                        handle.remove()

                    # Use pre-computed baseline
                    original_idx = sub_indices[i]
                    baseline_ld = all_baseline_ld[original_idx]
                    add_effects.append(patched_ld - baseline_ld)

                add_effects = np.array(add_effects)
                mean_eff = float(add_effects.mean())
                std_eff = float(add_effects.std())
                t_val = mean_eff / (std_eff / np.sqrt(len(add_effects))) if std_eff > 0 else 0
                n_positive = int((add_effects > 0).sum())
                n_negative = int((add_effects <= 0).sum())

                # Per-category centroid cosine similarity with full-data
                cat_cosine_sims = {}
                full_cc = full_anova[li]['cat_centroids']
                full_cti = full_anova[li]['cat_to_idx']
                sub_cc = anova_sub['cat_centroids']
                sub_cti = anova_sub['cat_to_idx']
                for cat_name in anova_sub['unique_cats']:
                    if cat_name in full_cti:
                        fc = full_cc[full_cti[cat_name]]
                        sc = sub_cc[sub_cti[cat_name]]
                        fn, sn = np.linalg.norm(fc), np.linalg.norm(sc)
                        if fn > 1e-10 and sn > 1e-10:
                            cat_cosine_sims[cat_name] = float(np.dot(fc, sc) / (fn * sn))

                # Average cosine similarity across categories
                avg_cosine = float(np.mean(list(cat_cosine_sims.values()))) if cat_cosine_sims else 0.0

                seed_results[str(li)] = {
                    "n_sample": n_sample, "seed": seed,
                    "r2_I": anova_sub['r2_I'], "r2_A": anova_sub['r2_A'], "r2_eps": anova_sub['r2_eps'],
                    "A_add_mean": mean_eff, "A_add_std": std_eff, "A_add_t": t_val,
                    "A_n_positive": n_positive, "A_n_negative": n_negative,
                    "A_direction": "positive" if mean_eff > 0 else "negative",
                    "avg_cosine_with_full": avg_cosine,
                    "cat_cosine_sims": cat_cosine_sims,
                }

                print(f"    A_add={mean_eff:+.4f}(t={t_val:+.2f}), "
                      f"R2_A={anova_sub['r2_A']:.6f}, "
                      f"pos/neg={n_positive}/{n_negative}, "
                      f"cos_full={avg_cosine:.3f}")

            results["bootstrap"][str(n_sample)][str(seed)] = seed_results

    # ---- Step 4: Summary ----
    print(f"\n--- Step 4: Summary ---")
    summary = {}
    for li in layer_indices:
        summary[str(li)] = {}
        for n_sample in SAMPLE_SIZES:
            a_adds = []
            a_ts = []
            n_pos = 0
            n_neg = 0
            cosines = []
            for seed in SEEDS:
                r = results["bootstrap"][str(n_sample)][str(seed)][str(li)]
                a_adds.append(r["A_add_mean"])
                a_ts.append(r["A_add_t"])
                if r["A_direction"] == "positive":
                    n_pos += 1
                else:
                    n_neg += 1
                cosines.append(r["avg_cosine_with_full"])

            summary[str(li)][str(n_sample)] = {
                "A_add_mean_of_means": float(np.mean(a_adds)),
                "A_add_std_of_means": float(np.std(a_adds)),
                "A_t_mean": float(np.mean(a_ts)),
                "direction_consistency": f"{n_pos}pos/{n_neg}neg",
                "all_positive": n_pos == len(SEEDS),
                "all_negative": n_neg == len(SEEDS),
                "avg_cosine_with_full": float(np.mean(cosines)),
            }

            s = summary[str(li)][str(n_sample)]
            marker = "+" if s["all_positive"] else ("-" if s["all_negative"] else "~")
            print(f"  L{li} n={n_sample}: A_add={s['A_add_mean_of_means']:+.4f}(sd={s['A_add_std_of_means']:.4f}), "
                  f"t={s['A_t_mean']:+.2f}, dir={s['direction_consistency']}, "
                  f"cos_full={s['avg_cosine_with_full']:.3f} {marker}")

    results["summary"] = summary

    # Save
    out_dir = ROOT / "results" / "phase388_centroid_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase388.json"
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
