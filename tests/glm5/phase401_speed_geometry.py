"""
Phase 401: Speed Semantic Geometry Analysis
=============================================

Phase 400b showed speed is the ONLY category where direction > norm (ATTRACTOR_DOM),
but with only 3 objects, cross-generalization was inconsistent (cheetah→rocket reverses).

This phase addresses the core bottleneck:
1. Expand to 12+ speed objects with natural/man-made labels
2. Full cross-generalization matrix (12×12 = 144 pairs)
3. Geometric structure: pairwise cosine, clustering by type/speed
4. Layer-wise evolution of speed directions

Key questions:
A. Do natural objects share a speed subspace distinct from man-made?
B. Is speed level (fast/slow) correlated with direction similarity?
C. Does the geometric structure explain the cheetah→rocket reversal?
D. How do speed directions evolve across layers?

Usage:
  python tests/glm5/phase401_speed_geometry.py qwen3
  python tests/glm5/phase401_speed_geometry.py deepseek7b
  python tests/glm5/phase401_speed_geometry.py glm4
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

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

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

# 12 speed objects: natural animals, man-made vehicles, natural phenomena
# speed_level: 1=extremely slow, 5=extremely fast
SPEED_OBJECTS = {
    # Natural Animals
    "snail":      {"type": "animal",  "speed_level": 1, "target": "slow",    "comp": "fast"},
    "turtle":     {"type": "animal",  "speed_level": 2, "target": "slow",    "comp": "fast"},
    "horse":      {"type": "animal",  "speed_level": 4, "target": "fast",    "comp": "slow"},
    "cheetah":    {"type": "animal",  "speed_level": 5, "target": "fast",    "comp": "slow"},
    "falcon":     {"type": "animal",  "speed_level": 5, "target": "fast",    "comp": "slow"},
    # Man-made Vehicles
    "bicycle":    {"type": "vehicle", "speed_level": 2, "target": "slow",    "comp": "fast"},
    "ship":       {"type": "vehicle", "speed_level": 2, "target": "slow",    "comp": "fast"},
    "train":      {"type": "vehicle", "speed_level": 4, "target": "fast",    "comp": "slow"},
    "rocket":     {"type": "vehicle", "speed_level": 5, "target": "fast",    "comp": "slow"},
    # Natural Phenomena
    "glacier":    {"type": "phenomenon", "speed_level": 1, "target": "slow", "comp": "fast"},
    "wind":       {"type": "phenomenon", "speed_level": 4, "target": "fast", "comp": "slow"},
    "lightning":  {"type": "phenomenon", "speed_level": 5, "target": "fast", "comp": "slow"},
}

SPEED_CANDIDATES = [
    "fast", "slow", "rapid", "sluggish", "quick", "swift",
    "leisurely", "speedy", "quickly", "slowly", "hastily", "gradually",
]

LAYER_CONFIGS = {
    "qwen3": [4, 16, 28],
    "deepseek7b": [4, 12, 20],
    "glm4": [5, 15, 25, 35],
}

N_RANDOM = 20  # For odd/even decomposition


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


def get_logit_diff(logits_tensor, target_id, comp_id):
    logits = logits_tensor.float().cpu().numpy()
    t_logit = float(logits[target_id]) if target_id is not None else 0.0
    c_logit = float(logits[comp_id]) if comp_id is not None else 0.0
    return t_logit - c_logit, t_logit, c_logit


def make_orthogonal_directions(d, n, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    d_norm = np.linalg.norm(d)
    results = []
    for i in range(n):
        r = rng.randn(len(d)).astype(np.float32)
        r = r - (np.dot(r, d) / (np.dot(d, d) + 1e-10)) * d
        r_norm = np.linalg.norm(r)
        if r_norm > 1e-10:
            r = r * (d_norm / r_norm)
        results.append(r)
    return results


def compute_direction(model, tokenizer, layers_list, device, li, obj_name, obj_data,
                      token_ids, captured, hook_handle_list):
    """Compute the speed direction for a single object at a given layer."""
    target = obj_data["target"]
    comp = obj_data["comp"]
    tid = token_ids.get(target)
    cid = token_ids.get(comp)

    h_correct_list = []
    h_corrupt_list = []
    baseline_diffs = []

    # We use a shared hook - register once, compute per object
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn

    handle = layers_list[li].register_forward_hook(make_hook('h'))
    hook_handle_list.append(handle)

    for f_idx in range(len(FRAMES)):
        correct_clean = FRAMES[f_idx].format(obj=obj_name, attr=target)
        correct_corrupt = CORRUPT_FRAMES[f_idx].format(attr=target)

        captured.clear()
        inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
        h_correct_list.append(captured['h'][0, -1].numpy())

        captured.clear()
        inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device))
        h_corrupt_list.append(captured['h'][0, -1].numpy())
        diff, _, _ = get_logit_diff(out.logits[0, -1], tid, cid)
        baseline_diffs.append(diff)

    # Remove hook
    handle.remove()
    hook_handle_list.remove(handle)

    dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
    baseline = float(np.mean(baseline_diffs))

    return dh, baseline


def injection_test(model, tokenizer, layers_list, device, li, direction, prompt,
                   target_id, comp_id, baseline_diff):
    """Inject direction and measure effect."""
    results = {}
    for alpha in [-1.0, 1.0]:
        scaled = alpha * direction
        delta = torch.tensor(scaled, dtype=torch.bfloat16, device=device)
        diff_list = []
        for f_idx in range(len(CORRUPT_FRAMES)):
            p = CORRUPT_FRAMES[f_idx].format(
                attr=tokenizer.decode([target_id]) if target_id else "fast")
            def make_add_hook(dv):
                def hook_fn(module, input, output):
                    hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                    hs[0, -1, :] += dv
                    return (hs,) + output[1:] if isinstance(output, tuple) else hs
                return hook_fn
            h = layers_list[li].register_forward_hook(make_add_hook(delta))
            try:
                inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out2 = model(input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device))
                d2, _, _ = get_logit_diff(out2.logits[0, -1], target_id, comp_id)
            finally:
                h.remove()
            diff_list.append(d2 - baseline_diff)
        results[alpha] = float(np.mean(diff_list))

    even = (results[1.0] + results[-1.0]) / 2
    odd = (results[1.0] - results[-1.0]) / 2
    return even, odd


def run_phase401(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== Phase 401: Speed Semantic Geometry ({model_name}) [{timestamp}] ===")

    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    obj_names = sorted(SPEED_OBJECTS.keys())
    n_obj = len(obj_names)
    print(f"  Layers: {layer_indices}")
    print(f"  Objects: {n_obj} ({', '.join(obj_names)})")
    print(f"  Cross-generalization pairs: {n_obj * (n_obj - 1)}")

    # Load model
    print(f"\n--- Loading {model_name} ---")
    model, tokenizer = load_model_bf16(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    device = next(model.parameters()).device

    # Get unembedding matrix
    W_U = get_W_U(model, model_name)
    W_U_np = W_U.astype(np.float32)

    # Resolve token IDs
    token_ids = {}
    for obj_name, obj_data in SPEED_OBJECTS.items():
        for tok in [obj_data["target"], obj_data["comp"]]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    for tok in SPEED_CANDIDATES:
        if tok not in token_ids:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            token_ids[tok] = ids[0] if ids else None

    print(f"  Token IDs: fast={token_ids.get('fast')}, slow={token_ids.get('slow')}")

    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'objects': {k: v for k, v in SPEED_OBJECTS.items()},
        'per_layer': {},
    }

    for li in layer_indices:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")

        # ============================================================
        # Step 1: Compute speed directions for ALL 12 objects
        # ============================================================
        print(f"\n  Step 1: Computing speed directions for {n_obj} objects...")

        obj_dirs = {}
        obj_baselines = {}
        captured = {}
        hook_list = []

        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            dh, baseline = compute_direction(
                model, tokenizer, layers_list, device, li,
                obj_name, obj_data, token_ids, captured, hook_list)
            obj_dirs[obj_name] = dh
            obj_baselines[obj_name] = baseline
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.4f} baseline_gap={baseline:.4f}")

        # ============================================================
        # Step 2: Odd/Even decomposition for each object
        # ============================================================
        print(f"\n  Step 2: Odd/Even decomposition...")

        obj_decomp = {}
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            tid = token_ids.get(obj_data["target"])
            cid = token_ids.get(obj_data["comp"])
            if tid is None or cid is None:
                continue

            dir_l1 = obj_dirs[obj_name]
            baseline = obj_baselines[obj_name]

            # Self-injection test
            even, odd = injection_test(
                model, tokenizer, layers_list, device, li,
                dir_l1, None, tid, cid, baseline)

            # Random direction baseline
            ortho_dirs = make_orthogonal_directions(dir_l1, N_RANDOM)
            ortho_even_list = []
            for ortho_dir in ortho_dirs:
                o_even, o_odd = injection_test(
                    model, tokenizer, layers_list, device, li,
                    ortho_dir, None, tid, cid, baseline)
                ortho_even_list.append(o_even)

            avg_ortho_even = float(np.mean(ortho_even_list))
            std_ortho_even = float(np.std(ortho_even_list))
            even_ratio = avg_ortho_even / (even + 1e-10)
            odd_pct = abs(odd) / (abs(even) + abs(odd) + 1e-10)

            source = "NORM_DOM" if abs(even_ratio) > 0.7 else (
                "ATTRACTOR_DOM" if abs(even_ratio) < 0.3 else "MIXED")

            obj_decomp[obj_name] = {
                'even': even, 'odd': odd,
                'avg_ortho_even': avg_ortho_even, 'std_ortho_even': std_ortho_even,
                'even_ratio': even_ratio, 'odd_pct': float(odd_pct),
                'source': source,
            }
            print(f"    {obj_name}: even={even:+.4f} odd={odd:+.4f} "
                  f"odd%={odd_pct:.1%} source={source}")

        # ============================================================
        # Step 3: Pairwise cosine similarity matrix
        # ============================================================
        print(f"\n  Step 3: Pairwise cosine similarity...")

        cos_matrix = np.zeros((n_obj, n_obj))
        for i, name_i in enumerate(obj_names):
            di = obj_dirs[name_i]
            di_norm = np.linalg.norm(di)
            if di_norm < 1e-10:
                continue
            for j, name_j in enumerate(obj_names):
                dj = obj_dirs[name_j]
                dj_norm = np.linalg.norm(dj)
                if dj_norm < 1e-10:
                    continue
                cos_matrix[i, j] = float(np.dot(di, dj) / (di_norm * dj_norm))

        # Print matrix
        print(f"\n    Cosine Similarity Matrix:")
        header = f"    {'':>12s}"
        for name in obj_names:
            header += f" {name[:6]:>6s}"
        print(header)
        for i, name_i in enumerate(obj_names):
            row = f"    {name_i:>12s}"
            for j in range(n_obj):
                row += f" {cos_matrix[i, j]:>+6.3f}"
            print(row)

        # ============================================================
        # Step 4: Cross-generalization matrix (12×12)
        # ============================================================
        print(f"\n  Step 4: Cross-generalization matrix ({n_obj}×{n_obj})...")

        cross_even_matrix = np.zeros((n_obj, n_obj))
        cross_odd_matrix = np.zeros((n_obj, n_obj))

        for i, src_name in enumerate(obj_names):
            src_data = SPEED_OBJECTS[src_name]
            src_tid = token_ids.get(src_data["target"])
            src_cid = token_ids.get(src_data["comp"])
            src_dir = obj_dirs[src_name]

            if src_tid is None or src_cid is None:
                continue

            src_baseline = obj_baselines[src_name]

            for j, tgt_name in enumerate(obj_names):
                if i == j:
                    continue

                tgt_data = SPEED_OBJECTS[tgt_name]
                tgt_tid = token_ids.get(tgt_data["target"])
                tgt_cid = token_ids.get(tgt_data["comp"])

                if tgt_tid is None or tgt_cid is None:
                    continue

                # Inject src direction on tgt's context
                tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt_data["target"])
                tgt_baseline = obj_baselines[tgt_name]

                # +direction
                delta = torch.tensor(src_dir, dtype=torch.bfloat16, device=device)
                def make_add_hook(dv):
                    def hook_fn(module, input, output):
                        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
                        hs[0, -1, :] += dv
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook_fn

                h = layers_list[li].register_forward_hook(make_add_hook(delta))
                try:
                    inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out_plus = model(input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device))
                    diff_plus, _, _ = get_logit_diff(out_plus.logits[0, -1], tgt_tid, tgt_cid)
                finally:
                    h.remove()

                # -direction
                neg_delta = torch.tensor(-src_dir, dtype=torch.bfloat16, device=device)
                h2 = layers_list[li].register_forward_hook(make_add_hook(neg_delta))
                try:
                    inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        out_minus = model(input_ids=inputs["input_ids"].to(device),
                                         attention_mask=inputs["attention_mask"].to(device))
                    diff_minus, _, _ = get_logit_diff(out_minus.logits[0, -1], tgt_tid, tgt_cid)
                finally:
                    h2.remove()

                # Baseline for tgt
                inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out_base = model(input_ids=inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device))
                base_diff, _, _ = get_logit_diff(out_base.logits[0, -1], tgt_tid, tgt_cid)

                plus_effect = diff_plus - base_diff
                minus_effect = diff_minus - base_diff
                cross_even = (plus_effect + minus_effect) / 2
                cross_odd = (plus_effect - minus_effect) / 2

                cross_even_matrix[i, j] = cross_even
                cross_odd_matrix[i, j] = cross_odd

            print(f"    {src_name} -> others: done")

        # Print cross-odd matrix (most informative for direction semantics)
        print(f"\n    Cross-Odd Matrix (direction transfer):")
        header = f"    {'src\\tgt':>12s}"
        for name in obj_names:
            header += f" {name[:6]:>6s}"
        print(header)
        for i, name_i in enumerate(obj_names):
            row = f"    {name_i:>12s}"
            for j in range(n_obj):
                if i == j:
                    row += f" {'---':>6s}"
                else:
                    row += f" {cross_odd_matrix[i, j]:>+6.3f}"
            print(row)

        # ============================================================
        # Step 5: Type-based aggregation
        # ============================================================
        print(f"\n  Step 5: Type-based aggregation...")

        types = defaultdict(list)
        for name in obj_names:
            types[SPEED_OBJECTS[name]["type"]].append(name)

        # Within-type cross-odd
        type_stats = {}
        for t1 in types:
            for t2 in types:
                pairs = []
                for n1 in types[t1]:
                    for n2 in types[t2]:
                        if n1 != n2:
                            i = obj_names.index(n1)
                            j = obj_names.index(n2)
                            pairs.append(cross_odd_matrix[i, j])
                if pairs:
                    key = f"{t1}->{t2}"
                    type_stats[key] = {
                        'mean_odd': float(np.mean(pairs)),
                        'std_odd': float(np.std(pairs)),
                        'n_pairs': len(pairs),
                        'positive_rate': float(sum(1 for p in pairs if p > 0) / len(pairs)),
                    }
                    print(f"    {key}: mean_odd={np.mean(pairs):+.4f} "
                          f"pos_rate={sum(1 for p in pairs if p > 0) / len(pairs):.1%} "
                          f"n={len(pairs)}")

        # Speed-level correlation
        print(f"\n  Speed-level analysis:")
        same_speed_pairs = []  # Both fast or both slow
        opposite_speed_pairs = []  # One fast, one slow
        for i, name_i in enumerate(obj_names):
            for j, name_j in enumerate(obj_names):
                if i >= j:
                    continue
                si = SPEED_OBJECTS[name_i]["speed_level"]
                sj = SPEED_OBJECTS[name_j]["speed_level"]
                odd_val = cross_odd_matrix[i, j]
                if si >= 4 and sj >= 4:  # Both fast
                    same_speed_pairs.append(('fast-fast', odd_val))
                elif si <= 2 and sj <= 2:  # Both slow
                    same_speed_pairs.append(('slow-slow', odd_val))
                elif (si >= 4 and sj <= 2) or (si <= 2 and sj >= 4):  # Opposite
                    opposite_speed_pairs.append(odd_val)

        for label, pairs_list in [('fast-fast', [v for l, v in same_speed_pairs if l == 'fast-fast']),
                                   ('slow-slow', [v for l, v in same_speed_pairs if l == 'slow-slow']),
                                   ('opposite', opposite_speed_pairs)]:
            if pairs_list:
                print(f"    {label}: mean_odd={np.mean(pairs_list):+.4f} "
                      f"pos_rate={sum(1 for v in pairs_list if v > 0) / len(pairs_list):.1%} "
                      f"n={len(pairs_list)}")

        # ============================================================
        # Step 6: W_U projection for all objects
        # ============================================================
        print(f"\n  Step 6: W_U projection analysis...")

        wu_results = {}
        for obj_name in obj_names:
            dir_l1 = obj_dirs[obj_name]
            dir_norm = np.linalg.norm(dir_l1)
            if dir_norm < 1e-10:
                continue
            dir_normalized = dir_l1 / dir_norm

            proj_fast = 0.0
            proj_slow = 0.0
            top_projs = []

            for tok in SPEED_CANDIDATES:
                c_id = token_ids.get(tok)
                if c_id is not None and c_id < W_U_np.shape[0]:
                    w = W_U_np[c_id]
                    w_norm = np.linalg.norm(w)
                    if w_norm > 1e-10:
                        cos = float(np.dot(dir_normalized, w / w_norm))
                        top_projs.append((tok, cos))
                        if tok in ['fast', 'rapid', 'quick', 'swift', 'speedy', 'quickly', 'hastily']:
                            proj_fast = max(proj_fast, cos) if proj_fast == 0 else (proj_fast + cos) / 2
                        if tok in ['slow', 'sluggish', 'leisurely', 'slowly', 'gradually']:
                            proj_slow = max(proj_slow, cos) if proj_slow == 0 else (proj_slow + cos) / 2

            top_projs.sort(key=lambda x: -x[1])
            wu_results[obj_name] = {
                'proj_fast_avg': float(proj_fast),
                'proj_slow_avg': float(proj_slow),
                'fast_minus_slow': float(proj_fast - proj_slow),
                'top3': top_projs[:3],
            }
            print(f"    {obj_name}: fast_proj={proj_fast:.4f} slow_proj={proj_slow:.4f} "
                  f"gap={proj_fast-proj_slow:+.4f} top={top_projs[:2]}")

        # ============================================================
        # Step 7: Direction clustering analysis
        # ============================================================
        print(f"\n  Step 7: Direction clustering...")

        # Build direction matrix
        dir_matrix = np.array([obj_dirs[n] for n in obj_names])
        # Normalize
        dir_norms = np.linalg.norm(dir_matrix, axis=1, keepdims=True)
        dir_norms = np.maximum(dir_norms, 1e-10)
        dir_normalized_matrix = dir_matrix / dir_norms

        # Compute within-type and across-type cosine similarities
        within_type_cos = []
        across_type_cos = []

        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                cos_val = cos_matrix[i, j]
                ti = SPEED_OBJECTS[obj_names[i]]["type"]
                tj = SPEED_OBJECTS[obj_names[j]]["type"]
                if ti == tj:
                    within_type_cos.append(cos_val)
                else:
                    across_type_cos.append(cos_val)

        print(f"    Within-type cosine: mean={np.mean(within_type_cos):+.4f} "
              f"std={np.std(within_type_cos):.4f} n={len(within_type_cos)}")
        print(f"    Across-type cosine: mean={np.mean(across_type_cos):+.4f} "
              f"std={np.std(across_type_cos):.4f} n={len(across_type_cos)}")

        # Speed-level cosine correlation
        speed_sims = []
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                si = SPEED_OBJECTS[obj_names[i]]["speed_level"]
                sj = SPEED_OBJECTS[obj_names[j]]["speed_level"]
                speed_diff = abs(si - sj)
                cos_val = cos_matrix[i, j]
                speed_sims.append((speed_diff, cos_val))

        speed_diffs = [s[0] for s in speed_sims]
        cos_vals = [s[1] for s in speed_sims]
        # Simple correlation
        if len(speed_diffs) > 2:
            mean_sd = np.mean(speed_diffs)
            mean_cv = np.mean(cos_vals)
            cov = np.mean([(sd - mean_sd) * (cv - mean_cv) for sd, cv in speed_sims])
            std_sd = np.std(speed_diffs)
            std_cv = np.std(cos_vals)
            corr = cov / (std_sd * std_cv + 1e-10) if std_sd > 0 and std_cv > 0 else 0.0
            print(f"    Speed-level vs cosine correlation: r={corr:+.4f}")

        # ============================================================
        # Store layer results
        # ============================================================
        layer_result = {
            'obj_dirs_norm': {n: float(np.linalg.norm(obj_dirs[n])) for n in obj_names},
            'obj_baselines': obj_baselines,
            'decomposition': obj_decomp,
            'cosine_matrix': cos_matrix.tolist(),
            'cross_even_matrix': cross_even_matrix.tolist(),
            'cross_odd_matrix': cross_odd_matrix.tolist(),
            'type_stats': type_stats,
            'wu_results': {k: {kk: vv for kk, vv in v.items() if kk != 'top3'} 
                          for k, v in wu_results.items()},
            'wu_top3': {k: v['top3'] for k, v in wu_results.items()},
            'within_type_cos_mean': float(np.mean(within_type_cos)) if within_type_cos else 0,
            'across_type_cos_mean': float(np.mean(across_type_cos)) if across_type_cos else 0,
            'speed_cosine_corr': float(corr) if len(speed_diffs) > 2 else 0,
        }

        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0:.0f}s")

    # ============================================================
    # Cross-layer summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")

    print(f"\nOdd% by object and layer:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        decomp = lr.get('decomposition', {})
        row = f"  L{li}: "
        for name in obj_names:
            if name in decomp:
                row += f" {name}={decomp[name]['odd_pct']:.0%}"
        print(row)

    print(f"\nWithin-type vs Across-type cosine similarity:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        within = lr.get('within_type_cos_mean', 0)
        across = lr.get('across_type_cos_mean', 0)
        print(f"  L{li}: within={within:+.4f} across={across:+.4f} diff={within-across:+.4f}")

    print(f"\nSpeed-level vs cosine correlation:")
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        corr = lr.get('speed_cosine_corr', 0)
        print(f"  L{li}: r={corr:+.4f}")

    # Type-based cross-odd summary
    print(f"\nType-based cross-odd (deepest layer):")
    deepest_li = str(layer_indices[-1])
    lr = all_results['per_layer'].get(deepest_li, {})
    for key, val in lr.get('type_stats', {}).items():
        print(f"  {key}: mean_odd={val['mean_odd']:+.4f} pos_rate={val['positive_rate']:.1%}")

    # Save
    out_dir = ROOT / "results" / "phase401_speed_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase401.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase401(model_name)
