"""
Phase 391b: Incorrect-Condition Target/Competitor Decomposition (Confirmation Round)
===================================================================================
Goal: Verify whether incorrect-value pairs show REVERSED mechanism (T↓C↑)
compared to correct-value pairs showing IDEAL (T↑C↓).

This is the critical test for the SYMMETRIC mechanism hypothesis:
- Correct: centroid boosts target + suppresses competitor
- Incorrect: centroid suppresses target + boosts competitor

Only test the MOST INFORMATIVE layer per model (fast confirmation):
- Qwen3: L4 (where color shows IDEAL) + L20 (where brightness reverses)
- DS7B: L4 (where moisture/temperature show IDEAL)
- GLM4: L20 (where brightness shows IDEAL)
"""
import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model

# Same data as Phase 391
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

# Build incorrect pairs: swap target and competitor
INCORRECT_PAIRS = []
INCORRECT_CATEGORIES = []
for i, (obj, target, comp) in enumerate(ALL_PAIRS):
    INCORRECT_PAIRS.append((obj, comp, target))  # target becomes competitor
    INCORRECT_CATEGORIES.append(PAIR_CATEGORIES[i])

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
            print(f"  Failed {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
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
    return {
        'mu': mu, 'A_comp': A_comp,
        'cat_centroids': cat_centroids, 'cat_to_idx': cat_to_idx,
        'unique_cats': unique_cats,
    }


def run_phase391b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 391b: Incorrect T/C Decomposition ({model_name}) [{timestamp}] ===")

    # Only test key layers for speed
    LAYER_CONFIGS = {
        "qwen3": [4, 20],
        "deepseek7b": [4],
        "glm4": [20],
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

    N_correct = len(ALL_PAIRS)
    N_incorrect = len(INCORRECT_PAIRS)
    obj_labels = [p[0] for p in ALL_PAIRS]
    cat_labels = PAIR_CATEGORIES[:]

    # Build prompts for BOTH correct and incorrect
    # Correct: "The apple is red" → target=red, competitor=blue
    # Incorrect: "The apple is blue" → target=blue (wrong), competitor=red (right)
    correct_prompts = [TEMPLATE.format(obj=obj, attr=target) for obj, target, _ in ALL_PAIRS]
    # INCORRECT_PAIRS[i] = (obj, wrong_value, correct_value) e.g. (apple, blue, red)
    # The incorrect prompt should use the WRONG value: "The apple is blue"
    incorrect_prompts = [TEMPLATE.format(obj=p[0], attr=p[1]) for p in INCORRECT_PAIRS]
    # Corrupt prompts
    correct_corrupt = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target) for _, target, _ in ALL_PAIRS]
    incorrect_corrupt = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=p[1]) for p in INCORRECT_PAIRS]

    # Get token IDs for all values
    token_ids = {}
    for obj, target, comp in ALL_PAIRS + INCORRECT_PAIRS:
        for tok in [target, comp]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'n_correct': N_correct, 'n_incorrect': N_incorrect,
        'layers': layer_indices,
        'per_layer': {},
    }

    for li in layer_indices:
        t0 = time.time()
        print(f"\n--- Layer {li} ---")

        # 1. Collect activations for CORRECT pairs (to compute centroid)
        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_hook('h'))
        h_clean = np.zeros((N_correct, d_model), dtype=np.float32)
        h_corrupt = np.zeros((N_correct, d_model), dtype=np.float32)

        for i in range(N_correct):
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
        handle.remove()
        print(f"  Correct activations collected ({time.time()-t0:.0f}s)")

        # 2. ANOVA on correct pairs → get per-category centroid
        dh = h_clean - h_corrupt
        anova = anova_decomposition(dh, obj_labels, cat_labels)

        # 3. Test on INCORRECT pairs: what does the correct-pair centroid do to incorrect pairs?
        # For incorrect pairs:
        # - "compatible" value = the one that SHOULD be correct (was target in correct pair)
        # - "incompatible" value = the one currently in the prompt (was competitor in correct pair)
        # - logit_diff = logit(compatible) - logit(incompatible)
        # We use the SAME category centroid from correct pairs

        print(f"  Testing incorrect pairs...")
        baseline_target_incorrect = np.zeros(N_incorrect)   # logit of "compatible" (correct) value
        baseline_competitor_incorrect = np.zeros(N_incorrect)  # logit of "incompatible" (wrong) value
        patched_target_incorrect = np.zeros(N_incorrect)
        patched_competitor_incorrect = np.zeros(N_incorrect)

        for i, (obj, target_inc, comp_inc) in enumerate(INCORRECT_PAIRS):
            # In incorrect pair: target_inc = wrong value (e.g., "blue" for apple)
            #                    comp_inc = correct value (e.g., "red" for apple)
            # We measure: logit(comp_inc) - logit(target_inc) = logit(correct) - logit(wrong)
            tid = token_ids.get(comp_inc)   # compatible (correct) token
            cid = token_ids.get(target_inc)  # incompatible (wrong) token

            if tid is None or cid is None:
                continue

            # Baseline on corrupt prompt
            inputs = tokenizer(incorrect_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            baseline_target_incorrect[i] = float(out.logits[0, -1, tid])   # correct value logit
            baseline_competitor_incorrect[i] = float(out.logits[0, -1, cid])  # wrong value logit

            # Patched: add correct-pair centroid to corrupt prompt
            # Use the centroid for this pair's category
            cat = INCORRECT_CATEGORIES[i]
            cat_idx = anova['cat_to_idx'][cat]
            delta_np = anova['cat_centroids'][cat_idx]
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
                with torch.no_grad():
                    out_patch = model(input_ids=inputs["input_ids"].to(device),
                                      attention_mask=inputs["attention_mask"].to(device))
                patched_target_incorrect[i] = float(out_patch.logits[0, -1, tid])
                patched_competitor_incorrect[i] = float(out_patch.logits[0, -1, cid])
            finally:
                handle.remove()

            if (i+1) % 30 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{N_incorrect} incorrect pairs ({elapsed:.0f}s)")

        # 4. Compute effects for incorrect pairs
        # add_effect = (patched_target - patched_competitor) - (baseline_target - baseline_competitor)
        # = (patched_correct - patched_wrong) - (baseline_correct - baseline_wrong)
        target_delta_inc = patched_target_incorrect - baseline_target_incorrect
        competitor_delta_inc = patched_competitor_incorrect - baseline_competitor_incorrect
        add_effects_inc = target_delta_inc - competitor_delta_inc

        # 5. Also compute correct-pair effects (reuse Phase 391 data or compute fresh for comparison)
        # For a fair comparison, compute correct-pair effects at the same layer
        print(f"  Computing correct-pair effects for comparison...")
        baseline_target_cor = np.zeros(N_correct)
        baseline_competitor_cor = np.zeros(N_correct)
        patched_target_cor = np.zeros(N_correct)
        patched_competitor_cor = np.zeros(N_correct)

        for i, (obj, target, comp) in enumerate(ALL_PAIRS):
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            if tid is None or cid is None:
                continue

            inputs = tokenizer(correct_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
            baseline_target_cor[i] = float(out.logits[0, -1, tid])
            baseline_competitor_cor[i] = float(out.logits[0, -1, cid])

            delta_np = anova['A_comp'][i]
            delta = torch.tensor(delta_np, dtype=torch.bfloat16, device=device)

            def make_add_hook2(delta_vec):
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

            handle = layers_list[li].register_forward_hook(make_add_hook2(delta))
            try:
                with torch.no_grad():
                    out_patch = model(input_ids=inputs["input_ids"].to(device),
                                      attention_mask=inputs["attention_mask"].to(device))
                patched_target_cor[i] = float(out_patch.logits[0, -1, tid])
                patched_competitor_cor[i] = float(out_patch.logits[0, -1, cid])
            finally:
                handle.remove()

        target_delta_cor = patched_target_cor - baseline_target_cor
        competitor_delta_cor = patched_competitor_cor - baseline_competitor_cor
        add_effects_cor = target_delta_cor - competitor_delta_cor

        # 6. Compare correct vs incorrect
        layer_result = {
            'correct': {
                'add_mean': float(np.mean(add_effects_cor)),
                'target_delta_mean': float(np.mean(target_delta_cor)),
                'competitor_delta_mean': float(np.mean(competitor_delta_cor)),
                'category_effects': {},
            },
            'incorrect': {
                'add_mean': float(np.mean(add_effects_inc)),
                'target_delta_mean': float(np.mean(target_delta_inc)),
                'competitor_delta_mean': float(np.mean(competitor_delta_inc)),
                'category_effects': {},
            },
        }

        for cat in anova['unique_cats']:
            mask_cor = np.array([c == cat for c in cat_labels])
            mask_inc = np.array([c == cat for c in INCORRECT_CATEGORIES])

            ce_cor = {
                'n': int(mask_cor.sum()),
                'add_mean': float(np.mean(add_effects_cor[mask_cor])),
                'target_delta_mean': float(np.mean(target_delta_cor[mask_cor])),
                'competitor_delta_mean': float(np.mean(competitor_delta_cor[mask_cor])),
            }
            ce_inc = {
                'n': int(mask_inc.sum()),
                'add_mean': float(np.mean(add_effects_inc[mask_inc])),
                'target_delta_mean': float(np.mean(target_delta_inc[mask_inc])),
                'competitor_delta_mean': float(np.mean(competitor_delta_inc[mask_inc])),
            }
            layer_result['correct']['category_effects'][cat] = ce_cor
            layer_result['incorrect']['category_effects'][cat] = ce_inc

        # Print results
        elapsed = time.time() - t0
        print(f"\n  L{li} done in {elapsed:.0f}s:")
        print(f"  CORRECT:   add={layer_result['correct']['add_mean']:+.4f}, "
              f"T={layer_result['correct']['target_delta_mean']:+.4f}, "
              f"C={layer_result['correct']['competitor_delta_mean']:+.4f}")
        print(f"  INCORRECT: add={layer_result['incorrect']['add_mean']:+.4f}, "
              f"T={layer_result['incorrect']['target_delta_mean']:+.4f}, "
              f"C={layer_result['incorrect']['competitor_delta_mean']:+.4f}")

        print(f"\n  Per-category SYMMETRIC check:")
        for cat in sorted(anova['unique_cats']):
            ce_c = layer_result['correct']['category_effects'].get(cat, {})
            ce_i = layer_result['incorrect']['category_effects'].get(cat, {})
            add_c = ce_c.get('add_mean', 0)
            add_i = ce_i.get('add_mean', 0)
            td_c = ce_c.get('target_delta_mean', 0)
            cd_c = ce_c.get('competitor_delta_mean', 0)
            td_i = ce_i.get('target_delta_mean', 0)
            cd_i = ce_i.get('competitor_delta_mean', 0)

            # Check if signs are opposite
            sign_match = "SYMMETRIC" if np.sign(add_c) != np.sign(add_i) and add_c != 0 and add_i != 0 else "ASYMMETRIC"
            # Check mechanism
            mech_c = f"T{'↑' if td_c > 0 else '↓'}C{'↑' if cd_c > 0 else '↓'}"
            mech_i = f"T{'↑' if td_i > 0 else '↓'}C{'↑' if cd_i > 0 else '↓'}"
            print(f"    {cat:12s}: cor={add_c:+.4f}({mech_c}) inc={add_i:+.4f}({mech_i}) {sign_match}")

        results['per_layer'][str(li)] = layer_result

    # Save
    out_dir = ROOT / "results" / "phase391_target_competitor_decomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase391b.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model released.")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase391b(model_name)
