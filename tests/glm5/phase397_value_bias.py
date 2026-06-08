"""
Phase 397: Value Bias vs Compatibility Separation
==================================================

Core question: Is FULL_SYMMETRIC just value preference alignment,
or is there a genuine compatibility gradient mechanism?

Three tests:
1. Direction Flip Test:
   - If -L1 makes REVERSED objects IDEAL → L1 is pure value preference
   - If -L1 does NOT make REVERSED objects IDEAL → more complex mechanism

2. Per-Object Direction Cosine Similarity:
   - cos(ant_dir, elephant_dir) within size category
   - If negative → objects use opposite directions → no single compatibility gradient
   - If positive → shared direction → possible compatibility gradient

3. Cross-Object Direction Test (subset):
   - Test ant's direction on elephant's prompt, and vice versa
   - If ant's direction helps ant but hurts elephant → separate mechanisms
   - If ant's direction helps both → shared mechanism

Layer config: 1 key layer per model (from Phase 396b results)
- Qwen3: L4
- DS7B: L12
- GLM4: L10
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
}

DISPLAY_TO_PROMPT = {"ocean_c": "ocean"}

# Value bias alignment per object: which value does this object prefer?
VALUE_ALIGNMENT = {
    # size: big-compatible vs small-compatible
    "elephant": "big", "mountain": "big", "whale": "big",
    "ant": "small", "grain": "small", "pin": "small",
    # moisture: wet-compatible vs dry-compatible
    "ocean": "wet", "rain": "wet", "river": "wet",
    "desert": "dry", "sand": "dry", "dust": "dry",
    # color: red-compatible vs blue-compatible vs white-compatible vs green-compatible
    "apple": "red", "cherry": "red",
    "sky": "blue", "ocean_c": "blue",
    "snow": "white",
    "grass": "green",
}


def build_pairs():
    pairs = []
    for cat, cat_data in RICH_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            prompt_obj = DISPLAY_TO_PROMPT.get(obj_name, obj_name)
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name, 'prompt_obj': prompt_obj,
                        'target': target, 'comp': comp,
                        'cat': cat, 'frame_idx': f_idx, 'value_idx': v_idx,
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
        'other_abs_mean': float(np.mean(np.abs(other_logits))),
    }


def test_direction(model, tokenizer, layers_list, device, li,
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


def classify_mechanism(td, cd):
    if td > 0 and cd < 0: return "IDEAL"
    elif td > 0 and cd > 0: return "DOM_BOOST" if td > cd else "BOOST_C"
    elif td < 0 and cd > 0: return "REVERSED"
    elif td < 0 and cd < 0: return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    else: return "MIXED"


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_phase397(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 397: Value Bias Separation ({model_name}) [{timestamp}] ===")

    LAYER_CONFIGS = {
        "qwen3": [4],
        "deepseek7b": [12],
        "glm4": [10],
    }
    layer_indices = LAYER_CONFIGS.get(model_name, [4])

    pairs = build_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs (3 categories × 6 objects × 2 values × 4 frames)")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device

    # Resolve token IDs
    token_ids = {}
    for cat_data in RICH_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")

        # === Collect activations ===
        captured = {}
        def make_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook_fn

        handle = layers_list[li].register_forward_hook(make_hook('h'))

        h_correct = np.zeros((N, d_model), dtype=np.float32)
        h_correct_corrupt = np.zeros((N, d_model), dtype=np.float32)
        h_incorrect = np.zeros((N, d_model), dtype=np.float32)
        h_incorrect_corrupt = np.zeros((N, d_model), dtype=np.float32)

        baseline_correct = []
        baseline_incorrect = []

        for i in range(N):
            p = pairs[i]
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])

            # Correct clean + corrupt
            tpl = FRAMES[p['frame_idx']]
            ctpl = CORRUPT_FRAMES[p['frame_idx']]
            correct_clean = tpl.format(obj=p['prompt_obj'], attr=p['target'])
            correct_corrupt = ctpl.format(attr=p['target'])

            captured.clear()
            inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_correct[i] = captured['h'][0, -1].numpy()

            captured.clear()
            inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_correct_corrupt[i] = captured['h'][0, -1].numpy()
            baseline_correct.append(get_logit_stats(out.logits[0, -1], tid, cid))

            # Incorrect clean + corrupt
            incorrect_clean = tpl.format(obj=p['prompt_obj'], attr=p['comp'])
            incorrect_corrupt = ctpl.format(attr=p['comp'])

            captured.clear()
            inputs = tokenizer(incorrect_clean, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_incorrect[i] = captured['h'][0, -1].numpy()

            captured.clear()
            inputs = tokenizer(incorrect_corrupt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_incorrect_corrupt[i] = captured['h'][0, -1].numpy()
            baseline_incorrect.append(get_logit_stats(out.logits[0, -1], tid, cid))

            if (i+1) % 48 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()

        # Compute delta_h
        dh_correct = h_correct - h_correct_corrupt
        dh_incorrect = h_incorrect - h_incorrect_corrupt

        # ====== ANALYSIS 1: Per-Object Direction Cosine Similarity ======
        print(f"\n  === Analysis 1: Per-Object Direction Cosine Similarity ===")

        cat_labels = [p['cat'] for p in pairs]
        obj_labels = [p['obj'] for p in pairs]

        # Compute per-object direction: mean(delta_h_correct) for each object
        obj_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            obj_groups[p['obj']].append(i)

        per_obj_dirs = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs[obj] = np.mean(dh_correct[indices], axis=0)

        # Also compute per-object direction from INCORRECT condition
        per_obj_dirs_inc = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs_inc[obj] = np.mean(dh_incorrect[indices], axis=0)

        # Cosine similarity within each category
        unique_cats = sorted(set(cat_labels))
        cos_results = {}

        for cat in unique_cats:
            cat_objs = sorted([obj for obj in per_obj_dirs.keys()
                              if cat_labels[obj_groups[obj][0]] == cat])
            print(f"\n  Category: {cat}")
            print(f"  Objects: {cat_objs}")

            # Pairwise cosine similarity (correct condition)
            cos_matrix = np.zeros((len(cat_objs), len(cat_objs)))
            for i, o1 in enumerate(cat_objs):
                for j, o2 in enumerate(cat_objs):
                    cos_matrix[i, j] = cos_sim(per_obj_dirs[o1], per_obj_dirs[o2])

            # Print condensed: mean within value-aligned group, mean across groups
            # Identify value-aligned groups
            aligned_groups = defaultdict(list)
            for obj in cat_objs:
                va = VALUE_ALIGNMENT.get(obj, "unknown")
                aligned_groups[va].append(obj)

            print(f"  Value groups: {dict((k, v) for k, v in aligned_groups.items())}")

            # Within-group similarity
            within_sims = {}
            for va, objs in aligned_groups.items():
                if len(objs) > 1:
                    sims = []
                    for i in range(len(objs)):
                        for j in range(i+1, len(objs)):
                            sims.append(cos_sim(per_obj_dirs[objs[i]], per_obj_dirs[objs[j]]))
                    within_sims[va] = float(np.mean(sims))
                else:
                    within_sims[va] = 1.0  # single object

            # Cross-group similarity
            cross_sims = {}
            va_keys = list(aligned_groups.keys())
            for i in range(len(va_keys)):
                for j in range(i+1, len(va_keys)):
                    sims = []
                    for o1 in aligned_groups[va_keys[i]]:
                        for o2 in aligned_groups[va_keys[j]]:
                            sims.append(cos_sim(per_obj_dirs[o1], per_obj_dirs[o2]))
                    cross_sims[f"{va_keys[i]}_vs_{va_keys[j]}"] = float(np.mean(sims))

            # Correct vs Incorrect direction similarity per object
            corr_inc_sims = {}
            for obj in cat_objs:
                corr_inc_sims[obj] = cos_sim(per_obj_dirs[obj], per_obj_dirs_inc[obj])

            print(f"  Within-group cos (correct): {within_sims}")
            print(f"  Cross-group cos (correct): {cross_sims}")
            print(f"  Correct-vs-Incorrect cos: {dict((k, f'{v:.3f}') for k, v in corr_inc_sims.items())}")

            cos_results[cat] = {
                'within_group': within_sims,
                'cross_group': cross_sims,
                'corr_inc_sim': corr_inc_sims,
                'full_matrix': {o1: {o2: float(cos_matrix[i, j])
                            for j, o2 in enumerate(cat_objs)}
                           for i, o1 in enumerate(cat_objs)},
            }

        # ====== ANALYSIS 2: Direction Flip Test ======
        print(f"\n  === Analysis 2: Direction Flip Test ===")

        # Compute L1_category directions
        mu = dh_correct.mean(axis=0)
        unique_cats_list = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats_list)}
        cat_centroids = np.zeros((len(unique_cats_list), d_model))
        cat_counts = np.zeros(len(unique_cats_list))
        for i, c in enumerate(cat_labels):
            cat_centroids[cat_to_idx[c]] += dh_correct[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats_list)):
            if cat_counts[j] > 0:
                cat_centroids[j] /= cat_counts[j]

        # Directions to test
        # For each object, we test:
        # 1. L1_category (+ and -)
        # 2. Per-object pure direction (+ and -)
        flip_results = {}

        unique_objs = sorted(set(obj_labels))
        for obj in unique_objs:
            obj_mask = np.array(obj_labels) == obj
            obj_indices = np.where(obj_mask)[0]
            if len(obj_indices) == 0:
                continue

            cat = cat_labels[obj_indices[0]]
            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])
            val_align = VALUE_ALIGNMENT.get(obj, "?")

            # Direction 1: L1_category
            dir_l1 = mu + (cat_centroids[cat_to_idx[cat]] - mu)
            # Direction 2: Per-object pure
            dir_pobj = per_obj_dirs[obj]

            for dir_name, direction in [("L1", dir_l1), ("POBJ", dir_pobj)]:
                for flip_label, sign in [("+", 1.0), ("-", -1.0)]:
                    test_dir = sign * direction

                    # Test on correct-corrupt prompts
                    td_corr_list = []
                    cd_corr_list = []
                    for idx in obj_indices:
                        p = pairs[idx]
                        ctpl = CORRUPT_FRAMES[p['frame_idx']]
                        prompt = ctpl.format(attr=p['target'])
                        stats = test_direction(model, tokenizer, layers_list, device, li,
                                              test_dir, prompt, tid, cid)
                        bs = baseline_correct[idx]
                        td_corr_list.append(stats['t_logit'] - bs['t_logit'])
                        cd_corr_list.append(stats['c_logit'] - bs['c_logit'])

                    # Test on incorrect-corrupt prompts
                    td_inc_list = []
                    cd_inc_list = []
                    for idx in obj_indices:
                        p = pairs[idx]
                        ctpl = CORRUPT_FRAMES[p['frame_idx']]
                        prompt = ctpl.format(attr=p['comp'])
                        stats = test_direction(model, tokenizer, layers_list, device, li,
                                              test_dir, prompt, tid, cid)
                        bs = baseline_incorrect[idx]
                        td_inc_list.append(stats['t_logit'] - bs['t_logit'])
                        cd_inc_list.append(stats['c_logit'] - bs['c_logit'])

                    corr_mech = classify_mechanism(np.mean(td_corr_list), np.mean(cd_corr_list))
                    inc_mech = classify_mechanism(np.mean(td_inc_list), np.mean(cd_inc_list))

                    corr_ideal = (np.mean(td_corr_list) > 0 and np.mean(cd_corr_list) < 0)
                    inc_ideal = (np.mean(td_inc_list) > 0 and np.mean(cd_inc_list) < 0)
                    symm = "FULL" if (corr_ideal and inc_ideal) else \
                           "HALF" if corr_ideal else "NO"

                    key = f"{obj}_{dir_name}_{flip_label}"
                    flip_results[key] = {
                        'cat': cat,
                        'value_align': val_align,
                        'direction': dir_name,
                        'flip': flip_label,
                        'correct_td': float(np.mean(td_corr_list)),
                        'correct_cd': float(np.mean(cd_corr_list)),
                        'correct_mech': corr_mech,
                        'incorrect_td': float(np.mean(td_inc_list)),
                        'incorrect_cd': float(np.mean(cd_inc_list)),
                        'incorrect_mech': inc_mech,
                        'symmetric': symm,
                    }

                    print(f"    {obj:10s} {dir_name}{flip_label}: "
                          f"CORR[{corr_mech:10s} T={np.mean(td_corr_list):+.4f} C={np.mean(cd_corr_list):+.4f}] "
                          f"INCORR[{inc_mech:10s} T={np.mean(td_inc_list):+.4f} C={np.mean(cd_inc_list):+.4f}] "
                          f"SYMM={symm} align={val_align}")

            # Progress log
            obj_idx_in_list = unique_objs.index(obj)
            if (obj_idx_in_list + 1) % 6 == 0:
                elapsed = time.time() - t0_layer
                print(f"  --- Progress: {obj_idx_in_list+1}/{len(unique_objs)} objects ({elapsed:.0f}s) ---")

        # ====== ANALYSIS 3: Cross-Object Direction Test ======
        print(f"\n  === Analysis 3: Cross-Object Direction Test ===")

        # Select key pairs: opposite value-aligned objects within same category
        cross_pairs = [
            ("ant", "elephant", "size"),      # small vs big
            ("ocean", "desert", "moisture"),   # wet vs dry
            ("apple", "sky", "color"),         # red vs blue
        ]

        cross_results = {}
        for src_obj, tgt_obj, cat in cross_pairs:
            if src_obj not in per_obj_dirs or tgt_obj not in per_obj_dirs:
                continue

            src_dir = per_obj_dirs[src_obj]
            tgt_indices = [i for i, p in enumerate(pairs) if p['obj'] == tgt_obj]
            if not tgt_indices:
                continue

            p0 = pairs[tgt_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])

            # Test src direction on tgt's correct prompt
            td_corr_list = []
            cd_corr_list = []
            for idx in tgt_indices:
                p = pairs[idx]
                ctpl = CORRUPT_FRAMES[p['frame_idx']]
                prompt = ctpl.format(attr=p['target'])
                stats = test_direction(model, tokenizer, layers_list, device, li,
                                      src_dir, prompt, tid, cid)
                bs = baseline_correct[idx]
                td_corr_list.append(stats['t_logit'] - bs['t_logit'])
                cd_corr_list.append(stats['c_logit'] - bs['c_logit'])

            # Test src direction on tgt's incorrect prompt
            td_inc_list = []
            cd_inc_list = []
            for idx in tgt_indices:
                p = pairs[idx]
                ctpl = CORRUPT_FRAMES[p['frame_idx']]
                prompt = ctpl.format(attr=p['comp'])
                stats = test_direction(model, tokenizer, layers_list, device, li,
                                      src_dir, prompt, tid, cid)
                bs = baseline_incorrect[idx]
                td_inc_list.append(stats['t_logit'] - bs['t_logit'])
                cd_inc_list.append(stats['c_logit'] - bs['c_logit'])

            corr_mech = classify_mechanism(np.mean(td_corr_list), np.mean(cd_corr_list))
            inc_mech = classify_mechanism(np.mean(td_inc_list), np.mean(cd_inc_list))

            cross_key = f"{src_obj}_to_{tgt_obj}"
            cross_results[cross_key] = {
                'cat': cat,
                'src_value_align': VALUE_ALIGNMENT.get(src_obj, "?"),
                'tgt_value_align': VALUE_ALIGNMENT.get(tgt_obj, "?"),
                'correct_td': float(np.mean(td_corr_list)),
                'correct_cd': float(np.mean(cd_corr_list)),
                'correct_mech': corr_mech,
                'incorrect_td': float(np.mean(td_inc_list)),
                'incorrect_cd': float(np.mean(cd_inc_list)),
                'incorrect_mech': inc_mech,
            }

            # Also test tgt direction on src prompt
            tgt_dir = per_obj_dirs[tgt_obj]
            src_indices = [i for i, p in enumerate(pairs) if p['obj'] == src_obj]
            p0s = pairs[src_indices[0]]
            tid_s = token_ids.get(p0s['target'])
            cid_s = token_ids.get(p0s['comp'])

            td_corr_s = []
            cd_corr_s = []
            for idx in src_indices:
                p = pairs[idx]
                ctpl = CORRUPT_FRAMES[p['frame_idx']]
                prompt = ctpl.format(attr=p['target'])
                stats = test_direction(model, tokenizer, layers_list, device, li,
                                      tgt_dir, prompt, tid_s, cid_s)
                bs = baseline_correct[idx]
                td_corr_s.append(stats['t_logit'] - bs['t_logit'])
                cd_corr_s.append(stats['c_logit'] - bs['c_logit'])

            td_inc_s = []
            cd_inc_s = []
            for idx in src_indices:
                p = pairs[idx]
                ctpl = CORRUPT_FRAMES[p['frame_idx']]
                prompt = ctpl.format(attr=p['comp'])
                stats = test_direction(model, tokenizer, layers_list, device, li,
                                      tgt_dir, prompt, tid_s, cid_s)
                bs = baseline_incorrect[idx]
                td_inc_s.append(stats['t_logit'] - bs['t_logit'])
                cd_inc_s.append(stats['c_logit'] - bs['c_logit'])

            corr_mech_s = classify_mechanism(np.mean(td_corr_s), np.mean(cd_corr_s))
            inc_mech_s = classify_mechanism(np.mean(td_inc_s), np.mean(cd_inc_s))

            reverse_key = f"{tgt_obj}_to_{src_obj}"
            cross_results[reverse_key] = {
                'cat': cat,
                'src_value_align': VALUE_ALIGNMENT.get(tgt_obj, "?"),
                'tgt_value_align': VALUE_ALIGNMENT.get(src_obj, "?"),
                'correct_td': float(np.mean(td_corr_s)),
                'correct_cd': float(np.mean(cd_corr_s)),
                'correct_mech': corr_mech_s,
                'incorrect_td': float(np.mean(td_inc_s)),
                'incorrect_cd': float(np.mean(cd_inc_s)),
                'incorrect_mech': inc_mech_s,
            }

            print(f"    {src_obj}→{tgt_obj}: CORR[{corr_mech:10s} T={np.mean(td_corr_list):+.4f} C={np.mean(cd_corr_list):+.4f}] "
                  f"INCORR[{inc_mech:10s}]")
            print(f"    {tgt_obj}→{src_obj}: CORR[{corr_mech_s:10s} T={np.mean(td_corr_s):+.4f} C={np.mean(cd_corr_s):+.4f}] "
                  f"INCORR[{inc_mech_s:10s}]")

        # ====== Summary ======
        print(f"\n  === Summary for Layer {li} ===")

        # Count FULL_SYMMETRIC for each direction type and flip
        for dir_name in ["L1", "POBJ"]:
            for flip_label in ["+", "-"]:
                full_count = 0
                half_count = 0
                for key, val in flip_results.items():
                    if val['direction'] == dir_name and val['flip'] == flip_label:
                        if val['symmetric'] == "FULL":
                            full_count += 1
                        elif val['symmetric'] == "HALF":
                            half_count += 1
                print(f"    {dir_name}{flip_label}: FULL_SYMMETRIC={full_count}, HALF={half_count}")

        # Value bias analysis: does -L1 make REVERSED objects IDEAL?
        print(f"\n  === Value Bias Analysis ===")
        for obj in unique_objs:
            l1_plus = flip_results.get(f"{obj}_L1_+")
            l1_minus = flip_results.get(f"{obj}_L1_-")
            pobj_plus = flip_results.get(f"{obj}_POBJ_+")
            pobj_minus = flip_results.get(f"{obj}_POBJ_-")

            if l1_plus and l1_minus:
                flip_effect = "L1_FLIP→IDEAL" if l1_minus['correct_mech'] == 'IDEAL' else f"L1_FLIP→{l1_minus['correct_mech']}"
                print(f"    {obj:10s} (align={VALUE_ALIGNMENT.get(obj,'?'):5s}): "
                      f"L1+={l1_plus['correct_mech']:10s} L1-={l1_minus['correct_mech']:10s} | "
                      f"POBJ+={pobj_plus['correct_mech']:10s} POBJ-={pobj_minus['correct_mech']:10s} | {flip_effect}")

        results['per_layer'][str(li)] = {
            'cosine_similarity': cos_results,
            'flip_test': flip_results,
            'cross_object': cross_results,
        }
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")

    # Save
    out_dir = ROOT / "results" / "phase397_value_bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase397.json"
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
    run_phase397(model_name)
