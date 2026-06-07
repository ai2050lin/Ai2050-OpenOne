"""
Phase 396b: Per-Object SYMMETRIC + Neutral Prompt Test
======================================================

Resolve ambiguity from Phase 396:
1. Is the T/C swapping for size/speed an averaging artifact?
   → Per-object analysis: test each object separately
2. Is the direction a value preference or compatibility interaction?
   → Test on neutral prompt: "The elephant is ___." (no value)
   → If direction boosts compatible value on neutral prompt → value preference
   → If direction has no effect on neutral prompt → pure interaction

Focus on the key IDEAL cases:
- DS7B L12: size (per-object)
- GLM4 L10: moisture (per-object + neutral prompt)
- Qwen3 L4: color (per-object)
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

# Neutral prompt templates (no value specified)
NEUTRAL_FRAMES = [
    "The {obj} is",
    "An {obj} is",
    "This {obj} is",
    "That {obj} is",
]

NEUTRAL_CORRUPT_FRAMES = [
    "The item is",
    "An item is",
    "This item is",
    "That item is",
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


def run_phase396b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 396b: Per-Object SYMMETRIC + Neutral ({model_name}) [{timestamp}] ===")

    LAYER_CONFIGS = {
        "qwen3": [4],       # color=IDEAL at L4
        "deepseek7b": [12], # size=IDEAL at L12
        "glm4": [10],       # moisture=IDEAL at L10
    }
    layer_indices = LAYER_CONFIGS.get(model_name, [4])

    pairs = build_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs (3 categories)")

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

        # Per-sample storage
        h_correct = np.zeros((N, d_model), dtype=np.float32)
        h_correct_corrupt = np.zeros((N, d_model), dtype=np.float32)
        h_incorrect = np.zeros((N, d_model), dtype=np.float32)
        h_incorrect_corrupt = np.zeros((N, d_model), dtype=np.float32)
        h_neutral = np.zeros((N, d_model), dtype=np.float32)
        h_neutral_corrupt = np.zeros((N, d_model), dtype=np.float32)

        baseline_correct = []
        baseline_incorrect = []
        baseline_neutral = []

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

            # Neutral clean + corrupt
            ntpl = NEUTRAL_FRAMES[p['frame_idx']]
            nctpl = NEUTRAL_CORRUPT_FRAMES[p['frame_idx']]
            neutral_clean = ntpl.format(obj=p['prompt_obj'])
            neutral_corrupt = nctpl.format()

            captured.clear()
            inputs = tokenizer(neutral_clean, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(device),
                           attention_mask=inputs["attention_mask"].to(device))
            h_neutral[i] = captured['h'][0, -1].numpy()
            baseline_neutral.append(get_logit_stats(out.logits[0, -1], tid, cid))

            captured.clear()
            inputs = tokenizer(neutral_corrupt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            h_neutral_corrupt[i] = captured['h'][0, -1].numpy()

            if (i+1) % 40 == 0:
                print(f"  Activation: {i+1}/{N} ({time.time()-t0_layer:.0f}s)")

        handle.remove()

        # Compute delta_h for all conditions
        dh_correct = h_correct - h_correct_corrupt
        dh_incorrect = h_incorrect - h_incorrect_corrupt
        dh_neutral = h_neutral - h_neutral_corrupt

        # ANOVA for correct condition
        cat_labels = [p['cat'] for p in pairs]
        obj_labels = [p['obj'] for p in pairs]
        mu = dh_correct.mean(axis=0)
        unique_cats = sorted(set(cat_labels))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        cat_centroids = np.zeros((len(unique_cats), d_model))
        cat_counts = np.zeros(len(unique_cats))
        for i, c in enumerate(cat_labels):
            cat_centroids[cat_to_idx[c]] += dh_correct[i]
            cat_counts[cat_to_idx[c]] += 1
        for j in range(len(unique_cats)):
            if cat_counts[j] > 0:
                cat_centroids[j] /= cat_counts[j]
        A_cat = np.zeros_like(dh_correct)
        for i, c in enumerate(cat_labels):
            A_cat[i] = cat_centroids[cat_to_idx[c]] - mu

        # Per-object L2_crossfit directions (LOPO)
        dh_resid_cat = dh_correct - mu - A_cat
        oc_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            oc_groups[(p['obj'], p['cat'])].append(i)
        obj_cat_keys = [(obj_labels[i], cat_labels[i]) for i in range(N)]
        unique_obj_cats = sorted(set(obj_cat_keys))
        oc_to_idx = {oc: i for i, oc in enumerate(unique_obj_cats)}
        oc_centroids = np.zeros((len(unique_obj_cats), d_model))
        oc_counts = np.zeros(len(unique_obj_cats))
        for i, oc in enumerate(obj_cat_keys):
            oc_centroids[oc_to_idx[oc]] += dh_resid_cat[i]
            oc_counts[oc_to_idx[oc]] += 1
        for j in range(len(unique_obj_cats)):
            if oc_counts[j] > 0:
                oc_centroids[j] /= oc_counts[j]
        A_obj_cat = np.zeros_like(dh_correct)
        for i, oc in enumerate(obj_cat_keys):
            A_obj_cat[i] = oc_centroids[oc_to_idx[oc]]

        # LOPO cross-fit
        A_obj_cat_cf = np.zeros_like(dh_correct)
        for oc_key in unique_obj_cats:
            group_indices = oc_groups[oc_key]
            for test_i in group_indices:
                train_indices = [j for j in group_indices if j != test_i]
                if len(train_indices) == 0:
                    A_obj_cat_cf[test_i] = np.zeros(d_model)
                else:
                    A_obj_cat_cf[test_i] = np.mean(
                        [dh_resid_cat[j] for j in train_indices], axis=0)

        # === Per-object SYMMETRIC test ===
        print(f"\n  === Per-Object SYMMETRIC Test ===")

        obj_results = {}
        unique_objs = sorted(set(obj_labels))
        for obj in unique_objs:
            obj_mask = np.array(obj_labels) == obj
            obj_indices = np.where(obj_mask)[0]
            if len(obj_indices) == 0:
                continue

            cat = cat_labels[obj_indices[0]]  # all same cat for this obj
            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])

            # L1_category direction for this object
            direction_l1 = mu + A_cat[obj_indices[0]]
            # L2_crossfit direction
            direction_l2cf = mu + A_cat[obj_indices[0]] + A_obj_cat_cf[obj_indices[0]]

            for ver_name, direction in [("L1", direction_l1), ("L2_cf", direction_l2cf)]:
                # Test on correct-corrupt prompts
                td_corr_list = []
                cd_corr_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl = CORRUPT_FRAMES[p['frame_idx']]
                    prompt = ctpl.format(attr=p['target'])
                    stats = test_direction(model, tokenizer, layers_list, device, li,
                                          direction, prompt, tid, cid)
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
                                          direction, prompt, tid, cid)
                    bs = baseline_incorrect[idx]
                    td_inc_list.append(stats['t_logit'] - bs['t_logit'])
                    cd_inc_list.append(stats['c_logit'] - bs['c_logit'])

                # Test on neutral-corrupt prompts
                td_neu_list = []
                cd_neu_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    nctpl = NEUTRAL_CORRUPT_FRAMES[p['frame_idx']]
                    prompt = nctpl.format()
                    stats = test_direction(model, tokenizer, layers_list, device, li,
                                          direction, prompt, tid, cid)
                    bs = baseline_neutral[idx]
                    td_neu_list.append(stats['t_logit'] - bs['t_logit'])
                    cd_neu_list.append(stats['c_logit'] - bs['c_logit'])

                corr_mech = classify_mechanism(np.mean(td_corr_list), np.mean(cd_corr_list))
                inc_mech = classify_mechanism(np.mean(td_inc_list), np.mean(cd_inc_list))
                neu_mech = classify_mechanism(np.mean(td_neu_list), np.mean(cd_neu_list))

                # SYMMETRIC: correct=IDEAL AND incorrect also pushes same direction
                corr_ideal = (np.mean(td_corr_list) > 0 and np.mean(cd_corr_list) < 0)
                inc_ideal = (np.mean(td_inc_list) > 0 and np.mean(cd_inc_list) < 0)
                symm = "FULL" if (corr_ideal and inc_ideal) else \
                       "HALF" if corr_ideal else "NO"

                obj_results[f"{obj}_{ver_name}"] = {
                    'cat': cat,
                    'correct_td': float(np.mean(td_corr_list)),
                    'correct_cd': float(np.mean(cd_corr_list)),
                    'correct_mech': corr_mech,
                    'incorrect_td': float(np.mean(td_inc_list)),
                    'incorrect_cd': float(np.mean(cd_inc_list)),
                    'incorrect_mech': inc_mech,
                    'neutral_td': float(np.mean(td_neu_list)),
                    'neutral_cd': float(np.mean(cd_neu_list)),
                    'neutral_mech': neu_mech,
                    'symmetric': symm,
                }

                print(f"    {obj:10s} {ver_name:5s}: "
                      f"CORR[{corr_mech:10s} T={np.mean(td_corr_list):+.4f} C={np.mean(cd_corr_list):+.4f}] "
                      f"INCORR[{inc_mech:10s} T={np.mean(td_inc_list):+.4f} C={np.mean(cd_inc_list):+.4f}] "
                      f"NEUTRAL[{neu_mech:10s} T={np.mean(td_neu_list):+.4f} C={np.mean(cd_neu_list):+.4f}] "
                      f"SYMM={symm}")

        results['per_layer'][str(li)] = obj_results
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")

    # Save
    out_dir = ROOT / "results" / "phase396b_per_object"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase396b.json"
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
    run_phase396b(model_name)
