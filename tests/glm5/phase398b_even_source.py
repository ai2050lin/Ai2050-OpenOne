"""
Phase 398b: Even Component Source Decomposition
================================================

Round 2 confirmation for Phase 398. The key finding is Even >> Odd,
but we need to know WHERE the Even component comes from.

Three candidate sources of Even:
1. Norm effect: ±d both change the norm, and norm change has sign-independent effect
2. Attractor effect: subsequent layers map ±d to the same output region
3. RMSNorm effect: RMSNorm partially removes sign information

Method:
A. Pure norm injection test:
   - Inject a random direction (orthogonal to d) with same norm as d
   - If random_orthogonal gives similar Even → norm is the main source
   - If random_orthogonal gives much smaller Even → attractor is the main source

B. Norm-matched vs direction-matched:
   - Inject d with alpha that matches the norm change of alpha*d
   - Compare: does pure norm boost give same Even as d?

C. Post-RMSNorm residual analysis:
   - Capture residual BEFORE and AFTER RMSNorm at next layer
   - Measure how much sign information is preserved after RMSNorm

Focus: DS7B (most nonlinear) at L4 and L12
Plus quick check on Qwen3 L4
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

SIZE_DATA = {
    "size": {
        "objects": {
            "ant":     [("small","big"),("tiny","large")],
            "elephant":[("big","small"),("large","tiny")],
            "mountain":[("big","small"),("large","tiny")],
        },
    },
}

VALUE_ALIGNMENT = {
    "ant": "small", "elephant": "big", "mountain": "big",
}

ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
POS_ALPHAS = [0.5, 1.0, 2.0]

LAYER_CONFIGS = {
    "qwen3": [4, 16, 32],
    "deepseek7b": [4, 12, 20],
}

N_RANDOM_DIRS = 5  # number of random orthogonal directions to average


def build_pairs():
    pairs = []
    for cat, cat_data in SIZE_DATA.items():
        for obj_name, value_combos in cat_data["objects"].items():
            for v_idx, (target, comp) in enumerate(value_combos):
                for f_idx in range(len(FRAMES)):
                    pairs.append({
                        'obj': obj_name,
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


def get_logit_diff(logits_tensor, target_id, comp_id):
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return t_logit - c_logit, t_logit, c_logit


def test_injection(model, tokenizer, layers_list, device, li,
                   delta_np, alpha, prompt, tid, cid):
    """Inject alpha * delta at layer li"""
    scaled = alpha * delta_np
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
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
        logit_diff, t_logit, c_logit = get_logit_diff(out.logits[0, -1], tid, cid)
    finally:
        handle.remove()
    return logit_diff, t_logit, c_logit


def make_orthogonal_directions(d, n, rng=None):
    """Generate n random directions orthogonal to d, with same norm as d"""
    if rng is None:
        rng = np.random.RandomState(42)
    d_norm = np.linalg.norm(d)
    results = []
    for _ in range(n):
        r = rng.randn(len(d)).astype(np.float32)
        # Project out d component
        r = r - (np.dot(r, d) / (np.dot(d, d) + 1e-10)) * d
        # Normalize to same norm as d
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-10:
            r = r * (d_norm / r_norm)
        results.append(r)
    return results


def test_rmsnorm_sign_preservation(model, tokenizer, layers_list, device, li,
                                    direction, alpha, prompt):
    """
    Capture residual BEFORE and AFTER next layer's RMSNorm.
    Measure how much sign information is preserved.
    """
    scaled = alpha * direction
    delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)

    pre_norm = {}
    post_norm = {}

    # Hook on current layer (to get injected state)
    def make_pre_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                pre_norm[key] = output[0].detach().float().cpu()
            else:
                pre_norm[key] = output.detach().float().cpu()
        return hook_fn

    # Hook on next layer's input_layernorm
    next_li = min(li + 1, len(layers_list) - 1)
    if next_li == li:
        return None  # can't test if same layer

    next_layer = layers_list[next_li]
    # Get the input_layernorm
    norm_module = None
    if hasattr(next_layer, 'input_layernorm'):
        norm_module = next_layer.input_layernorm
    elif hasattr(next_layer, 'ln_1'):
        norm_module = next_layer.ln_1
    else:
        return None

    def make_post_norm_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                post_norm[key] = output[0].detach().float().cpu()
            else:
                post_norm[key] = output.detach().float().cpu()
        return hook_fn

    # Run with injection
    h_pre = layers_list[li].register_forward_hook(make_pre_hook('pre'))
    h_post = norm_module.register_forward_hook(make_post_norm_hook('post'))

    # Also need the baseline (no injection)
    pre_norm_base = {}
    post_norm_base = {}

    def make_pre_base_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                pre_norm_base[key] = output[0].detach().float().cpu()
            else:
                pre_norm_base[key] = output.detach().float().cpu()
        return hook_fn

    def make_post_norm_base_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                post_norm_base[key] = output[0].detach().float().cpu()
            else:
                post_norm_base[key] = output.detach().float().cpu()
        return hook_fn

    # Run with injection
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            hs[0, -1, :] += dv
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook_fn

    h_inject = layers_list[li].register_forward_hook(make_add_hook(delta))

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h_inject.remove()
        h_pre.remove()
        h_post.remove()

    if 'pre' not in pre_norm or 'post' not in post_norm:
        return None

    # Now run without injection for baseline
    h_pre_base = layers_list[li].register_forward_hook(make_pre_base_hook('pre'))
    h_post_base = norm_module.register_forward_hook(make_post_norm_base_hook('post'))

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
    finally:
        h_pre_base.remove()
        h_post_base.remove()

    if 'pre' not in pre_norm_base or 'post' not in post_norm_base:
        return None

    # Compute residual after injection minus baseline
    delta_pre = (pre_norm['pre'][0, -1] - pre_norm_base['pre'][0, -1]).numpy()
    delta_post = (post_norm['post'][0, -1] - post_norm_base['post'][0, -1]).numpy()

    # Sign preservation: cos(delta_pre, delta_post)
    norm_pre = np.linalg.norm(delta_pre)
    norm_post = np.linalg.norm(delta_post)
    if norm_pre < 1e-10 or norm_post < 1e-10:
        return None

    cos_preserved = float(np.dot(delta_pre, delta_post) / (norm_pre * norm_post))

    # Norm ratio
    norm_ratio = float(norm_post / norm_pre)

    return {
        'cos_preserved': cos_preserved,  # how much sign is preserved after RMSNorm
        'norm_ratio': norm_ratio,         # how much norm changes after RMSNorm
        'delta_pre_norm': float(norm_pre),
        'delta_post_norm': float(norm_post),
    }


def classify_mechanism(td, cd):
    if td > 0 and cd < 0: return "IDEAL"
    elif td > 0 and cd > 0: return "DOM_BOOST" if td > cd else "BOOST_C"
    elif td < 0 and cd > 0: return "REVERSED"
    elif td < 0 and cd < 0: return "SUPP_T" if abs(td) > abs(cd) else "SUPP_C"
    else: return "MIXED"


def run_phase398b(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 398b: Even Source Decomposition ({model_name}) [{timestamp}] ===")

    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    pairs = build_pairs()
    N = len(pairs)
    print(f"  Total: {N} pairs (size category: ant, elephant, mountain)")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Layers: {layer_indices}")
    print(f"  Random orthogonal dirs: {N_RANDOM_DIRS}")

    # Load model
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
        'alphas': ALPHAS,
        'per_layer': {},
    }

    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")

        # === Step 1: Collect activations ===
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
        baseline_diffs = np.zeros(N, dtype=np.float32)
        baseline_t = np.zeros(N, dtype=np.float32)
        baseline_c = np.zeros(N, dtype=np.float32)

        for i in range(N):
            p = pairs[i]
            tid = token_ids.get(p['target'])
            cid = token_ids.get(p['comp'])

            tpl = FRAMES[p['frame_idx']]
            ctpl = CORRUPT_FRAMES[p['frame_idx']]
            correct_clean = tpl.format(obj=p['obj'], attr=p['target'])
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
            diff, tl, cl = get_logit_diff(out.logits[0, -1], tid, cid)
            baseline_diffs[i] = diff
            baseline_t[i] = tl
            baseline_c[i] = cl

        handle.remove()

        # Compute delta_h
        dh_correct = h_correct - h_correct_corrupt

        # Per-object directions
        obj_labels = [p['obj'] for p in pairs]
        obj_groups = defaultdict(list)
        for i, p in enumerate(pairs):
            obj_groups[p['obj']].append(i)

        per_obj_dirs = {}
        for obj, indices in obj_groups.items():
            per_obj_dirs[obj] = np.mean(dh_correct[indices], axis=0)

        # === Test A: Random orthogonal directions (norm-matched) ===
        print(f"\n  === Test A: Random Orthogonal Directions (norm-matched) ===")

        layer_result = {}

        for obj in ["ant", "elephant", "mountain"]:
            obj_indices = obj_groups.get(obj, [])
            if len(obj_indices) == 0:
                continue

            p0 = pairs[obj_indices[0]]
            tid = token_ids.get(p0['target'])
            cid = token_ids.get(p0['comp'])
            val_align = VALUE_ALIGNMENT.get(obj, "?")
            dir_l1 = per_obj_dirs[obj]  # use per-object direction

            # Generate random orthogonal directions
            ortho_dirs = make_orthogonal_directions(dir_l1, N_RANDOM_DIRS)

            # Test L1 direction at all alphas
            l1_effects = {}
            for alpha in ALPHAS:
                td_list = []
                cd_list = []
                diff_list = []
                for idx in obj_indices:
                    p = pairs[idx]
                    ctpl = CORRUPT_FRAMES[p['frame_idx']]
                    prompt = ctpl.format(attr=p['target'])
                    diff, tl, cl = test_injection(
                        model, tokenizer, layers_list, device, li,
                        dir_l1, alpha, prompt, tid, cid)
                    diff_list.append(diff - baseline_diffs[idx])
                    td_list.append(tl - baseline_t[idx])
                    cd_list.append(cl - baseline_c[idx])
                l1_effects[alpha] = {
                    'delta_diff': float(np.mean(diff_list)),
                    'delta_t': float(np.mean(td_list)),
                    'delta_c': float(np.mean(cd_list)),
                }

            # Compute Odd/Even for L1
            l1_odd = {}
            l1_even = {}
            for alpha in POS_ALPHAS:
                ep = l1_effects[alpha]
                en = l1_effects[-alpha]
                l1_odd[alpha] = {
                    'diff': (ep['delta_diff'] - en['delta_diff']) / 2,
                    't': (ep['delta_t'] - en['delta_t']) / 2,
                    'c': (ep['delta_c'] - en['delta_c']) / 2,
                }
                l1_even[alpha] = {
                    'diff': (ep['delta_diff'] + en['delta_diff']) / 2,
                    't': (ep['delta_t'] + en['delta_t']) / 2,
                    'c': (ep['delta_c'] + en['delta_c']) / 2,
                }

            # Test random orthogonal directions at alpha=1.0 and alpha=-1.0
            ortho_effects = {}
            for r_idx, ortho_dir in enumerate(ortho_dirs):
                r_effects = {}
                for alpha in [-1.0, 1.0]:
                    diff_list = []
                    td_list = []
                    cd_list = []
                    for idx in obj_indices:
                        p = pairs[idx]
                        ctpl = CORRUPT_FRAMES[p['frame_idx']]
                        prompt = ctpl.format(attr=p['target'])
                        diff, tl, cl = test_injection(
                            model, tokenizer, layers_list, device, li,
                            ortho_dir, alpha, prompt, tid, cid)
                        diff_list.append(diff - baseline_diffs[idx])
                        td_list.append(tl - baseline_t[idx])
                        cd_list.append(cl - baseline_c[idx])
                    r_effects[alpha] = {
                        'delta_diff': float(np.mean(diff_list)),
                        'delta_t': float(np.mean(td_list)),
                        'delta_c': float(np.mean(cd_list)),
                    }

                # Odd/Even for this random direction
                ep_r = r_effects[1.0]
                en_r = r_effects[-1.0]
                r_odd_diff = (ep_r['delta_diff'] - en_r['delta_diff']) / 2
                r_even_diff = (ep_r['delta_diff'] + en_r['delta_diff']) / 2

                ortho_effects[r_idx] = {
                    'alpha_pos': ep_r,
                    'alpha_neg': en_r,
                    'odd_diff': r_odd_diff,
                    'even_diff': r_even_diff,
                }

            # Average random orthogonal effects
            avg_ortho_even_diff = np.mean([ortho_effects[r]['even_diff'] for r in range(N_RANDOM_DIRS)])
            avg_ortho_odd_diff = np.mean([ortho_effects[r]['odd_diff'] for r in range(N_RANDOM_DIRS)])

            # Compare L1 Even vs random Even
            l1_even_diff_a1 = l1_even[1.0]['diff']
            l1_odd_diff_a1 = l1_odd[1.0]['diff']

            # If random_orthogonal Even ≈ L1 Even → norm effect dominates
            # If random_orthogonal Even << L1 Even → attractor in d's subspace
            even_ratio = avg_ortho_even_diff / (l1_even_diff_a1 + 1e-10)
            odd_ratio = avg_ortho_odd_diff / (l1_odd_diff_a1 + 1e-10)

            print(f"\n  {obj} (align={val_align}):")
            print(f"    L1 Even(alpha=1) = {l1_even_diff_a1:+.4f}, L1 Odd = {l1_odd_diff_a1:+.4f}")
            print(f"    Ortho Even(avg)  = {avg_ortho_even_diff:+.4f}, Ortho Odd(avg) = {avg_ortho_odd_diff:+.4f}")
            print(f"    Even ratio (ortho/L1) = {even_ratio:.3f}")
            print(f"    Odd ratio (ortho/L1)  = {odd_ratio:.3f}")

            if abs(even_ratio) < 0.3:
                source = "ATTRACTOR_DOM"  # norm alone can't explain Even
            elif abs(even_ratio) > 0.7:
                source = "NORM_DOM"  # norm explains most of Even
            else:
                source = "MIXED"

            print(f"    → Even source: {source}")

            # === Test B: RMSNorm sign preservation ===
            print(f"\n  === Test B: RMSNorm Sign Preservation ({obj}) ===")

            rmsnorm_results = {}
            for alpha in [1.0, -1.0]:
                # Use first prompt as representative
                idx = obj_indices[0]
                p = pairs[idx]
                tpl = FRAMES[p['frame_idx']]
                prompt = tpl.format(obj=p['obj'], attr=p['target'])

                rn_result = test_rmsnorm_sign_preservation(
                    model, tokenizer, layers_list, device, li,
                    dir_l1, alpha, prompt)

                if rn_result:
                    rmsnorm_results[alpha] = rn_result
                    print(f"    alpha={alpha:+.1f}: cos_preserved={rn_result['cos_preserved']:.4f}, "
                          f"norm_ratio={rn_result['norm_ratio']:.4f}")
                else:
                    print(f"    alpha={alpha:+.1f}: RMSNorm test failed (module not found)")

            # Compare sign preservation for +d vs -d
            if 1.0 in rmsnorm_results and -1.0 in rmsnorm_results:
                cos_pos = rmsnorm_results[1.0]['cos_preserved']
                cos_neg = rmsnorm_results[-1.0]['cos_preserved']
                avg_cos = (cos_pos + cos_neg) / 2

                if avg_cos < 0.3:
                    norm_role = "SIGN_DESTROY"  # RMSNorm removes most sign info
                elif avg_cos > 0.7:
                    norm_role = "SIGN_PRESERVE"  # RMSNorm preserves sign
                else:
                    norm_role = "PARTIAL"

                print(f"    Avg cos_preserved = {avg_cos:.4f} → {norm_role}")
            else:
                norm_role = "UNKNOWN"
                avg_cos = None

            layer_result[obj] = {
                'value_align': val_align,
                'l1_effects': l1_effects,
                'l1_odd': l1_odd,
                'l1_even': l1_even,
                'ortho_effects': ortho_effects,
                'avg_ortho_even': float(avg_ortho_even_diff),
                'avg_ortho_odd': float(avg_ortho_odd_diff),
                'even_ratio': float(even_ratio),
                'even_source': source,
                'rmsnorm_results': rmsnorm_results,
                'norm_role': norm_role,
                'avg_cos_preserved': float(avg_cos) if avg_cos is not None else None,
            }

        # === Layer Summary ===
        print(f"\n  === Layer {li} Summary ===")
        print(f"  {'Object':10s} {'L1 Even':>10s} {'Ortho Even':>10s} {'Ratio':>7s} {'Source':15s} {'RMSNorm':>12s}")
        for obj in ["ant", "elephant", "mountain"]:
            r = layer_result.get(obj, {})
            if not r:
                continue
            print(f"  {obj:10s} {r.get('l1_even',{}).get(1.0,{}).get('diff',0):+10.4f} "
                  f"{r.get('avg_ortho_even',0):+10.4f} "
                  f"{r.get('even_ratio',0):7.3f} "
                  f"{r.get('even_source','?'):15s} "
                  f"{r.get('norm_role','?'):>12s}")

        results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s")

    # Save
    out_dir = ROOT / "results" / "phase398b_even_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase398b.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Even Source Summary ({model_name}) ===")
    for li in layer_indices:
        lr = results['per_layer'].get(str(li), {})
        print(f"\n  Layer {li}:")
        for obj in ["ant", "elephant", "mountain"]:
            r = lr.get(obj, {})
            if not r:
                continue
            print(f"    {obj:10s}: Even_src={r.get('even_source','?'):15s} "
                  f"ortho/L1={r.get('even_ratio',0):.3f} "
                  f"RMSNorm={r.get('norm_role','?'):12s} "
                  f"cos_pres={r.get('avg_cos_preserved','N/A')}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase398b(model_name)
