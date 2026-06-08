"""
Phase 397b: Layer Trajectory — Confirm Nonlinearity Across Layers
=================================================================

Round 2 confirmation test for Phase 397's key finding:
- DS7B ant: L1+ and L1- both give IDEAL (nonlinear)
- elephant: no direction gives IDEAL (attractor)

Test at MULTIPLE layers to find:
1. Where does the "size→small" attractor emerge?
2. Is there any layer where elephant CAN achieve IDEAL?
3. Does the nonlinearity pattern change across layers?

Focus on DS7B (most interesting patterns) + quick check on Qwen3/GLM4
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

# Focus on SIZE category only (most dramatic pattern)
SIZE_DATA = {
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

VALUE_ALIGNMENT = {
    "elephant": "big", "mountain": "big", "whale": "big",
    "ant": "small", "grain": "small", "pin": "small",
}

# Layers to test - more granular
LAYER_CONFIGS = {
    "qwen3": [2, 4, 8, 12, 16, 20, 24, 28, 32, 35],
    "deepseek7b": [2, 4, 8, 12, 16, 20, 24, 27],
    "glm4": [2, 5, 10, 15, 20, 25, 30, 35, 39],
}


def build_size_pairs():
    pairs = []
    for cat, cat_data in SIZE_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name, 'prompt_obj': obj_name,
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
    return model, tokenizer


def get_logit_stats(logits_tensor, target_id, comp_id):
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return {'t_logit': t_logit, 'c_logit': c_logit}


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


def run_phase397b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 397b: Layer Trajectory ({model_name}) [{timestamp}] ===")

    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    pairs = build_size_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs (size category only)")

    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device

    # Resolve token IDs
    token_ids = {}
    for cat_data in SIZE_DATA.values():
        for obj_name, value_combos in cat_data["objects"].items():
            for target, comp in value_combos:
                for tok in [target, comp]:
                    if tok not in token_ids:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        token_ids[tok] = ids[0] if ids else None

    results = {
        'model': model_name, 'timestamp': timestamp,
        'trajectory': {},
    }

    # For efficiency: collect activations at ALL layers in one forward pass
    # Then test L1+ and L1- on correct prompt only (most informative)
    # Focus on 2 key objects: ant (small-compatible) and elephant (big-compatible)

    key_objects = ["ant", "elephant"]
    key_obj_indices = {obj: [i for i, p in enumerate(pairs) if p['obj'] == obj]
                       for obj in key_objects}

    for li in layer_indices:
        t0 = time.time()
        print(f"\n--- Layer {li} ---")

        # Collect activations for correct condition only
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
        baseline_correct = []

        for i in range(N):
            p = pairs[i]
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])

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

        handle.remove()

        # Compute delta_h and directions
        dh_correct = h_correct - h_correct_corrupt
        mu = dh_correct.mean(axis=0)

        # L1 size direction
        cat_centroid = dh_correct.mean(axis=0)  # all size, same as mu

        # Per-object directions
        obj_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            obj_groups[p['obj']].append(i)

        per_obj_dirs = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs[obj] = np.mean(dh_correct[indices], axis=0)

        # Test L1+ and L1- on correct prompt for ant and elephant
        layer_result = {}

        for obj in key_objects:
            obj_indices = key_obj_indices[obj]
            if not obj_indices:
                continue

            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])

            dir_l1 = cat_centroid  # same as mu for single category

            for dir_label, direction in [("L1+", dir_l1), ("L1-", -dir_l1)]:
                td_list = []
                cd_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl = CORRUPT_FRAMES[p['frame_idx']]
                    prompt = ctpl.format(attr=p['target'])
                    stats = test_direction(model, tokenizer, layers_list, device, li,
                                          direction, prompt, tid, cid)
                    bs = baseline_correct[idx]
                    td_list.append(stats['t_logit'] - bs['t_logit'])
                    cd_list.append(stats['c_logit'] - bs['c_logit'])

                td_mean = float(np.mean(td_list))
                cd_mean = float(np.mean(cd_list))
                mech = classify_mechanism(td_mean, cd_mean)

                key = f"{obj}_{dir_label}"
                layer_result[key] = {
                    'td': td_mean, 'cd': cd_mean, 'mech': mech
                }

                print(f"  L{li} {obj:10s} {dir_label}: {mech:10s} T={td_mean:+.4f} C={cd_mean:+.4f}")

        results['trajectory'][str(li)] = layer_result
        elapsed = time.time() - t0
        print(f"  L{li} done ({elapsed:.0f}s)")

    # Save
    out_dir = ROOT / "results" / "phase397b_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase397b.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Print trajectory summary
    print(f"\n=== Trajectory Summary for {model_name} ===")
    print(f"{'Layer':>6s} {'ant_L1+':>12s} {'ant_L1-':>12s} {'elephant_L1+':>14s} {'elephant_L1-':>14s}")
    for li in layer_indices:
        lr = results['trajectory'].get(str(li), {})
        ant_p = lr.get('ant_L1+', {})
        ant_m = lr.get('ant_L1-', {})
        elp_p = lr.get('elephant_L1+', {})
        elp_m = lr.get('elephant_L1-', {})
        ap = f"{ant_p.get('mech','?'):8s} T={ant_p.get('td',0):+.2f}" if ant_p else "N/A"
        am = f"{ant_m.get('mech','?'):8s} T={ant_m.get('td',0):+.2f}" if ant_m else "N/A"
        ep = f"{elp_p.get('mech','?'):8s} T={elp_p.get('td',0):+.2f}" if elp_p else "N/A"
        em = f"{elp_m.get('mech','?'):8s} T={elp_m.get('td',0):+.2f}" if elp_m else "N/A"
        print(f"{li:>6d} {ap:>12s} {am:>12s} {ep:>14s} {em:>14s}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase397b(model_name)
