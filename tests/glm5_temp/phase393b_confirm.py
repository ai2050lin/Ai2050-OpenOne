"""
Phase 393b: Confirmation Round — Focus on L2_obj_cat
====================================================
Round 1发现GLM4 L2_obj_cat在L4/L20都有43% IDEAL。
本测试在更多层验证L2_obj_cat是否持续优于L1/L0。

关键问题：
1. GLM4 L2_obj_cat是否在深层(L30/L38)也保持高IDEAL比例？
2. DS7B L2_obj_cat在L12/L26是否改善？
3. Qwen3 L2_obj_cat在L12/L28是否改善？

只测L0/L1/L2（L3太不稳定），省时间。
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

# Same data as Phase 393
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
            print(f"  Failed {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer


def classify_mechanism(td, cd):
    if td > 0 and cd < 0: return "IDEAL"
    elif td > 0 and cd > 0: return "DOM_BOOST" if td > cd else "BOOST_C"
    elif td < 0 and cd > 0: return "REVERSED"
    elif td < 0 and cd < 0: return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    return "MIXED"


def run_phase393b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 393b: L2_obj_cat Deep Verification ({model_name}) [{timestamp}] ===")

    # Test more layers per model
    LAYER_CONFIGS = {
        "qwen3": [12, 28],       # Layers not tested in Phase 393
        "deepseek7b": [12, 26],  # Layers not tested in Phase 393
        "glm4": [30, 38],        # Layers not tested in Phase 393
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

    correct_prompts = [TEMPLATE.format(obj=obj, attr=target) for obj, target, _ in ALL_PAIRS]
    correct_corrupt = [TEMPLATE.format(obj=CORRUPT_BASELINE, attr=target) for _, target, _ in ALL_PAIRS]

    token_ids = {}
    for obj, target, comp in ALL_PAIRS:
        for tok in [target, comp]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None

    results = {'model': model_name, 'timestamp': timestamp, 'layers': layer_indices, 'per_layer': {}}

    for li in layer_indices:
        t0 = time.time()
        print(f"\n--- Layer {li} ---")

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
                print(f"  Activation: {i+1}/{N}")
        handle.remove()
        print(f"  Activations done ({time.time()-t0:.0f}s)")

        # ANOVA
        dh = h_clean - h_corrupt
        mu = dh.mean(axis=0)
        unique_objects = sorted(set(obj_labels))
        obj_to_idx = {o: i2 for i2, o in enumerate(unique_objects)}
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

        dh_resid_I = dh - mu - I_comp
        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i2 for i2, c in enumerate(unique_cats)}
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

        dh_resid_IC = dh - mu - I_comp - A_comp
        obj_cat_keys = [(obj_labels[i], cat_labels[i]) for i in range(N)]
        unique_obj_cats = sorted(set(obj_cat_keys))
        oc_to_idx = {oc: i2 for i2, oc in enumerate(unique_obj_cats)}
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

        # Baseline
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

        # Test L0, L1, L2 on ALL pairs
        hierarchy_results = {}
        for level_name, direction_fn in [
            ("L0_global", lambda i: mu),
            ("L1_category", lambda i: A_comp[i]),
            ("L2_obj_cat", lambda i: A_comp[i] + A_obj_cat[i]),
        ]:
            patched_target = np.zeros(N)
            patched_competitor = np.zeros(N)

            for i in range(N):
                obj, target, comp = ALL_PAIRS[i]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                if tid is None or cid is None:
                    continue

                delta_np = direction_fn(i)
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
                    inputs = tokenizer(correct_corrupt[i], return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out = model(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device))
                    if tid is not None:
                        patched_target[i] = float(out.logits[0, -1, tid])
                    if cid is not None:
                        patched_competitor[i] = float(out.logits[0, -1, cid])
                finally:
                    handle.remove()

                if (i+1) % 30 == 0:
                    elapsed = time.time() - t0
                    print(f"    {level_name}: {i+1}/{N} ({elapsed:.0f}s)")

            td = patched_target - baseline_target
            cd = patched_competitor - baseline_competitor
            add = td - cd

            cat_breakdown = {}
            for cat in unique_cats:
                cat_mask = np.array(cat_labels) == cat
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
            hierarchy_results[level_name] = {
                'add_mean': float(np.mean(add)),
                'target_delta_mean': float(np.mean(td)),
                'competitor_delta_mean': float(np.mean(cd)),
                'mechanism': overall_mech,
                'ideal_count': ideal_count,
                'ideal_pct': ideal_count / len(cat_breakdown) * 100 if cat_breakdown else 0,
                'category_breakdown': cat_breakdown,
            }
            print(f"  {level_name}: add={hierarchy_results[level_name]['add_mean']:+.4f}, "
                  f"T={hierarchy_results[level_name]['target_delta_mean']:+.4f}, "
                  f"C={hierarchy_results[level_name]['competitor_delta_mean']:+.4f}, "
                  f"IDEAL={ideal_count}/{len(cat_breakdown)}, mech={overall_mech}")

        results['per_layer'][str(li)] = hierarchy_results
        elapsed = time.time() - t0
        print(f"  L{li} done in {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"--- Phase 393b Summary: L2_obj_cat across layers ---")
    print(f"{'='*70}")
    for level_name in ["L0_global", "L1_category", "L2_obj_cat"]:
        print(f"\n  {level_name}:")
        for li in layer_indices:
            hr = results['per_layer'][str(li)][level_name]
            print(f"    L{li}: add={hr['add_mean']:+.4f}, T={hr['target_delta_mean']:+.4f}, "
                  f"C={hr['competitor_delta_mean']:+.4f}, IDEAL={hr['ideal_count']}/7({hr['ideal_pct']:.0f}%)")

    # Category detail for L2_obj_cat
    print(f"\n  L2_obj_cat per-category detail:")
    for li in layer_indices:
        hr = results['per_layer'][str(li)]['L2_obj_cat']
        print(f"    L{li}:")
        for cat in sorted(hr['category_breakdown'].keys()):
            ce = hr['category_breakdown'][cat]
            print(f"      {cat:12s}: add={ce['add_mean']:+.4f}, T={ce['target_delta_mean']:+.4f}, "
                  f"C={ce['competitor_delta_mean']:+.4f}, mech={ce['mechanism']}")

    out_dir = ROOT / "results" / "phase393_centroid_hierarchy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase393b.json"
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
    run_phase393b(model_name)
