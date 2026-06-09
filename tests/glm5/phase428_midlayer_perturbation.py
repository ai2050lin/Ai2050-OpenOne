"""
Phase 428: Mid-Layer Residual Perturbation + Manifold Detection
================================================================

Based on Phase 426 findings:
1. Qwen3: α_c ≈ 0.91, category-property decoupled
2. GLM4: α_c ≈ 0.30, category-property coupled
3. DS7B: no stable category transition

Key Questions:
1. WHERE do category trajectories form? (embedding vs mid-layers)
   - If embedding perturbation is most effective → category determined at input
   - If mid-layer perturbation is more effective → category forms at that layer
   - If last-layer perturbation is effective → category direction directly influences readout

2. Is GLM4's low threshold real coupling or manifold-out artifact?
   - If full_entropy spikes before top-1 switches → manifold-out (model confused)
   - If top-1 switches cleanly with low full_entropy → real semantic coupling

3. Does DS7B become sensitive at mid-layers?
   - If mid-layer perturbation works where embedding doesn't → category in deeper layers

Design:
- Same category direction (embedding-space) applied at different depths
- Track manifold indicators: full_entropy, confidence, residual_norm
- Compare embedding vs mid-layer effectiveness

Usage:
  python tests/glm5/phase428_midlayer_perturbation.py qwen3 1
  python tests/glm5/phase428_midlayer_perturbation.py glm4 1
  python tests/glm5/phase428_midlayer_perturbation.py deepseek7b 1
"""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_model_info, get_W_U, get_layers

# ===== Alpha grids =====
ALPHA_R1 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
ALPHA_R2 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5]

# ===== Layer fractions (0.0 = embedding) =====
LAYER_FRACS_R1 = [0.0, 0.25, 0.50, 0.75]
LAYER_FRACS_R2 = [0.0, 0.20, 0.40, 0.60, 0.80]

# ===== Objects and categories =====
OBJECT_CATEGORIES = OrderedDict([
    ("fruit", {
        "objects": ["apple", "banana", "orange", "grape", "lemon", "mango", "peach"],
        "opposing": "animal",
    }),
    ("animal", {
        "objects": ["dog", "cat", "horse", "lion", "fish", "bird", "bear"],
        "opposing": "fruit",
    }),
    ("tool", {
        "objects": ["knife", "hammer", "spoon", "ruler", "nail", "chisel"],
        "opposing": "vehicle",
    }),
    ("vehicle", {
        "objects": ["car", "train", "bus", "truck", "boat", "ship"],
        "opposing": "tool",
    }),
    ("place", {
        "objects": ["desert", "forest", "ocean", "city", "island", "valley"],
        "opposing": "fruit",
    }),
])

R1_OBJECTS = ["apple", "dog", "knife"]
R2_OBJECTS = ["apple", "dog", "knife", "car", "cat"]

# ===== Tasks =====
KNOWLEDGE_TASKS = OrderedDict([
    ("category", {
        "template": "A {obj} is a kind of",
        "candidates": OrderedDict([
            ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
        ]),
    }),
    ("property", {
        "template": "The most notable property of a {obj} is that it is",
        "candidates": OrderedDict([
            ("edible", 1), ("alive", 2), ("sharp", 3), ("fast", 4), ("vast", 5),
        ]),
    }),
])

PERTURBATION_TYPES = ["remove_category", "add_random"]


def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention"""
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            print(f"  Loaded with attn_implementation={impl}")
            break
        except Exception as e:
            print(f"  {impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def verify_single_token(tokenizer, obj_word):
    tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    return len(tok_ids) == 1, tok_ids


def get_object_category(obj_word):
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        if obj_word in cat_info["objects"]:
            return cat_name
    return None


def get_category_directions(model, tokenizer, device):
    """Compute category directions from embedding space"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()

    category_centers = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        vecs = []
        for obj_word in cat_info["objects"]:
            tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
            if tok_ids:
                vecs.append(W_E[tok_ids[0]])
        if vecs:
            category_centers[cat_name] = np.mean(vecs, axis=0)

    category_directions = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        opposing = cat_info["opposing"]
        if cat_name in category_centers and opposing in category_centers:
            d = category_centers[cat_name] - category_centers[opposing]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            category_directions[cat_name] = d

    global_center = np.mean(list(category_centers.values()), axis=0)
    return category_directions, category_centers, global_center, W_E


def get_candidate_ids(tokenizer, candidates):
    cand_ids = {}
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    return cand_ids


def compute_entropy(probs_dict):
    probs = np.array(list(probs_dict.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def make_orthogonal_random(direction, d_model, rng, norm=1.0):
    d = rng.randn(d_model)
    proj = np.dot(d, direction) / max(np.dot(direction, direction), 1e-10) * direction
    d = d - proj
    d_norm = np.linalg.norm(d)
    if d_norm > 0:
        d = d / d_norm * norm
    return d


def get_target_layer_indices(n_layers, fracs):
    """Convert layer fractions to actual layer indices.
    0.0 means embedding (no layer), others mean after that fraction of layers."""
    result = []
    for f in fracs:
        if f == 0.0:
            result.append(-1)  # embedding
        else:
            idx = min(int(f * n_layers), n_layers - 1)
            result.append(idx)
    return result


def find_obj_positions(input_ids, obj_tok_ids):
    """Find positions of object tokens in input_ids"""
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            return list(range(i, i + len(obj_tok_ids)))
    return [1]  # fallback


def run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates):
    """Run baseline (no perturbation), return full metrics"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    next_logits = outputs.logits[0, -1, :]
    full_probs = torch.softmax(next_logits.float().cpu(), dim=-1)

    # Candidate probs
    result = {}
    for cand, tid in cand_ids.items():
        if tid < full_probs.shape[-1]:
            result[cand] = float(full_probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total

    level = sum(candidates[c] * result.get(c, 0.0) for c in candidates)
    cand_entropy = compute_entropy(result)
    top_cand = max(result, key=result.get) if result else "N/A"

    # Full distribution metrics (manifold detection)
    full_entropy = float(-torch.sum(full_probs * torch.log2(full_probs + 1e-10)).item())
    confidence = float(full_probs.max().item())

    # Top-2 gap
    sorted_probs, sorted_ids = torch.sort(full_probs, descending=True)
    top2_gap = float((sorted_probs[0] - sorted_probs[1]).item())

    return {
        "level": round(level, 4),
        "entropy": round(cand_entropy, 4),
        "top": top_cand,
        "probs": {k: round(v, 4) for k, v in result.items()},
        "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4),
        "top2_gap": round(top2_gap, 4),
    }


def run_with_embedding_perturbation(model, tokenizer, device, template, obj_word,
                                     perturbation, alpha, cand_ids, candidates):
    """Perturb at embedding layer (same as Phase 426)"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    obj_positions = find_obj_positions(input_ids, obj_tok_ids)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)

    perturbation_tensor = torch.tensor(
        perturbation * alpha, dtype=inputs_embeds.dtype, device=device
    )
    for pos in obj_positions:
        inputs_embeds[0, pos, :] += perturbation_tensor

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        try:
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        except Exception:
            empty = {c: 1.0/len(candidates) for c in candidates}
            level = sum(candidates[c]*empty[c] for c in candidates)
            return {"level": round(level, 4), "delta": 0.0, "entropy": 0, "top": "N/A",
                    "full_entropy": 0, "confidence": 0, "top2_gap": 0}

    next_logits = outputs.logits[0, -1, :]
    full_probs = torch.softmax(next_logits.float().cpu(), dim=-1)

    result = {}
    for cand, tid in cand_ids.items():
        if tid < full_probs.shape[-1]:
            result[cand] = float(full_probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total

    level = sum(candidates[c] * result.get(c, 0.0) for c in candidates)
    cand_entropy = compute_entropy(result)
    top_cand = max(result, key=result.get) if result else "N/A"
    full_entropy = float(-torch.sum(full_probs * torch.log2(full_probs + 1e-10)).item())
    confidence = float(full_probs.max().item())
    sorted_probs, _ = torch.sort(full_probs, descending=True)
    top2_gap = float((sorted_probs[0] - sorted_probs[1]).item())

    return {
        "level": round(level, 4),
        "entropy": round(cand_entropy, 4),
        "top": top_cand,
        "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4),
        "top2_gap": round(top2_gap, 4),
    }


def run_with_midlayer_perturbation(model, tokenizer, device, template, obj_word,
                                    perturbation, alpha, cand_ids, candidates,
                                    target_layer_idx, obj_tok_ids):
    """Perturb at a specific mid-layer using forward hook.
    
    target_layer_idx: the layer AFTER which perturbation is added.
    The hook intercepts the output of this layer and adds perturbation.
    """
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    obj_positions = find_obj_positions(input_ids, obj_tok_ids)
    attention_mask = torch.ones_like(input_ids)

    # Residual norm tracking
    residual_metrics = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Record pre-perturbation norm at object position
        obj_pos = obj_positions[0] if obj_positions else 0
        pre_norm = float(hidden_states[0, obj_pos, :].detach().float().norm().item())
        residual_metrics["pre_obj_norm"] = round(pre_norm, 4)

        # Add perturbation
        pert_tensor = torch.tensor(
            perturbation * alpha,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        modified = hidden_states.clone()
        for pos in obj_positions:
            modified[0, pos, :] += pert_tensor

        post_norm = float(modified[0, obj_pos, :].detach().float().norm().item())
        residual_metrics["post_obj_norm"] = round(post_norm, 4)
        residual_metrics["norm_change_pct"] = round((post_norm - pre_norm) / max(pre_norm, 1e-6) * 100, 2)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    layers = get_layers(model)
    handle = layers[target_layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception:
            handle.remove()
            empty = {c: 1.0/len(candidates) for c in candidates}
            level = sum(candidates[c]*empty[c] for c in candidates)
            return {"level": round(level, 4), "delta": 0.0, "entropy": 0, "top": "N/A",
                    "full_entropy": 0, "confidence": 0, "top2_gap": 0,
                    "residual_metrics": residual_metrics}

    handle.remove()

    next_logits = outputs.logits[0, -1, :]
    full_probs = torch.softmax(next_logits.float().cpu(), dim=-1)

    result = {}
    for cand, tid in cand_ids.items():
        if tid < full_probs.shape[-1]:
            result[cand] = float(full_probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total

    level = sum(candidates[c] * result.get(c, 0.0) for c in candidates)
    cand_entropy = compute_entropy(result)
    top_cand = max(result, key=result.get) if result else "N/A"
    full_entropy = float(-torch.sum(full_probs * torch.log2(full_probs + 1e-10)).item())
    confidence = float(full_probs.max().item())
    sorted_probs, _ = torch.sort(full_probs, descending=True)
    top2_gap = float((sorted_probs[0] - sorted_probs[1]).item())

    return {
        "level": round(level, 4),
        "entropy": round(cand_entropy, 4),
        "top": top_cand,
        "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4),
        "top2_gap": round(top2_gap, 4),
        "residual_metrics": residual_metrics,
    }


def run_phase428(model_name, round_num=1):
    """Run Phase 428 experiment"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 428: Mid-Layer Perturbation + Manifold Detection ({model_name}) R{round_num} [{timestamp}] ===")
    print(f"{'='*80}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # Category directions
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing category directions...")
    category_directions, category_centers, global_center, W_E = get_category_directions(model, tokenizer, device)

    # Target layers
    if round_num == 1:
        layer_fracs = LAYER_FRACS_R1
        alpha_grid = ALPHA_R1
        test_objects = R1_OBJECTS
    else:
        layer_fracs = LAYER_FRACS_R2
        alpha_grid = ALPHA_R2
        test_objects = R2_OBJECTS

    target_layers = get_target_layer_indices(info.n_layers, layer_fracs)
    layer_names = {}
    for frac, lidx in zip(layer_fracs, target_layers):
        if lidx == -1:
            layer_names[frac] = "embed"
        else:
            layer_names[frac] = f"L{lidx}"
    
    print(f"\n  Target layers: {layer_names}")
    print(f"  Alpha grid: {alpha_grid}")
    print(f"  Test objects: {test_objects}")

    # Filter single-token objects
    single_token_objects = []
    for obj_word in test_objects:
        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if is_single:
            single_token_objects.append(obj_word)
        else:
            print(f"  WARNING: '{obj_word}' is multi-token ({tok_ids}), skipping")
    test_objects = single_token_objects

    rng = np.random.RandomState(42)
    d_model = W_E.shape[1]

    # Results structure
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": timestamp,
        "phase": "428",
        "round": round_num,
        "target_layers": {str(f): layer_names[f] for f in layer_fracs},
        "alpha_grid": alpha_grid,
        "test_objects": test_objects,
        "per_object": {},
    }

    total_tests = 0
    t_start = time.time()

    for obj_idx, obj_word in enumerate(test_objects):
        obj_cat = get_object_category(obj_word)
        if obj_cat is None:
            continue
        opposing_cat = OBJECT_CATEGORIES[obj_cat]["opposing"]

        # Identity residual
        obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
        obj_embedding = W_E[obj_tok_ids[0]].copy()
        identity_residual = obj_embedding - category_centers.get(obj_cat, np.zeros(d_model))
        id_norm = np.linalg.norm(identity_residual)
        if id_norm > 0:
            identity_direction = identity_residual / id_norm
        else:
            identity_direction = np.zeros(d_model)

        print(f"\n[{time.strftime('%H:%M:%S')}] === Object {obj_idx+1}/{len(test_objects)}: "
              f"{obj_word} (cat={obj_cat}) ===")

        obj_results = {
            "category": obj_cat,
            "opposing": opposing_cat,
            "baselines": {},
            "perturbations": {},
        }

        for task_name, task_info in KNOWLEDGE_TASKS.items():
            template = task_info["template"]
            candidates = task_info["candidates"]
            cand_ids = get_candidate_ids(tokenizer, candidates)

            print(f"\n  Task: {task_name}")

            # ===== Baseline =====
            baseline = run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates)
            base_level = baseline["level"]
            obj_results["baselines"][task_name] = baseline
            print(f"    Baseline: level={base_level:.3f}, top={baseline['top']}, "
                  f"full_H={baseline['full_entropy']:.2f}, conf={baseline['confidence']:.3f}")

            # ===== Perturbation at each layer =====
            for perturb_type in PERTURBATION_TYPES:
                if perturb_type == "remove_category":
                    if obj_cat not in category_directions:
                        continue
                    direction = -category_directions[obj_cat]
                elif perturb_type == "add_random":
                    if obj_cat not in category_directions:
                        continue
                    direction = make_orthogonal_random(category_directions[obj_cat], d_model, rng)
                else:
                    continue

                perturb_data = {}

                for frac, lidx in zip(layer_fracs, target_layers):
                    layer_key = layer_names[frac]
                    curve_data = {}

                    for alpha in alpha_grid:
                        if alpha == 0.0:
                            curve_data["0.0"] = {
                                "level": round(base_level, 4),
                                "delta": 0.0,
                                "top": baseline["top"],
                                "full_entropy": baseline["full_entropy"],
                                "confidence": baseline["confidence"],
                                "top2_gap": baseline["top2_gap"],
                            }
                            continue

                        # Run perturbation
                        if lidx == -1:
                            # Embedding perturbation
                            res = run_with_embedding_perturbation(
                                model, tokenizer, device, template, obj_word,
                                direction, alpha, cand_ids, candidates
                            )
                        else:
                            # Mid-layer perturbation
                            res = run_with_midlayer_perturbation(
                                model, tokenizer, device, template, obj_word,
                                direction, alpha, cand_ids, candidates,
                                lidx, obj_tok_ids
                            )

                        delta = res["level"] - base_level
                        entry = {
                            "level": res["level"],
                            "delta": round(delta, 4),
                            "top": res["top"],
                            "full_entropy": res["full_entropy"],
                            "confidence": res["confidence"],
                            "top2_gap": res["top2_gap"],
                        }
                        if "residual_metrics" in res:
                            entry["residual_metrics"] = res["residual_metrics"]

                        curve_data[str(alpha)] = entry
                        total_tests += 1

                    perturb_data[layer_key] = curve_data

                    # Print key alpha points
                    key_alphas = ["0.0", "0.3", "0.5", "1.0", "1.5"]
                    pts = []
                    for ka in key_alphas:
                        if ka in curve_data:
                            d = curve_data[ka]
                            pts.append(f"a={ka}→Δ{d['delta']:+.2f},{d['top'][:3]},H{d['full_entropy']:.1f}")
                    print(f"    {perturb_type}@{layer_key}: {' | '.join(pts)}")

                obj_results["perturbations"][f"{task_name}_{perturb_type}"] = perturb_data

            # GPU log
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                if gpu > 10:
                    print(f"    [GPU: {gpu:.2f}GB]")

        results["per_object"][obj_word] = obj_results

        # Progress
        elapsed = time.time() - t_start
        est_total = elapsed / (obj_idx + 1) * len(test_objects)
        print(f"  [{time.strftime('%H:%M:%S')}] Progress: {obj_idx+1}/{len(test_objects)}, "
              f"elapsed={elapsed/60:.1f}min, est={est_total/60:.1f}min")

    # ===== Summary =====
    print(f"\n{'='*80}")
    print(f"=== Phase 428 Summary ({model_name}) R{round_num} ===")
    print(f"{'='*80}")

    # 1. Layer effectiveness comparison
    print("\n--- Layer Effectiveness (|Δ| at α=1.0, remove_category, category task) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        base_level = obj_data["baselines"]["category"]["level"]
        key = "category_remove_category"
        if key not in obj_data["perturbations"]:
            continue
        perturb_data = obj_data["perturbations"][key]

        print(f"  {obj_word} (base_level={base_level:.2f}):")
        for layer_key in [layer_names[f] for f in layer_fracs]:
            if layer_key in perturb_data and "1.0" in perturb_data[layer_key]:
                d = perturb_data[layer_key]["1.0"]
                print(f"    {layer_key}: Δ={d['delta']:+.3f}, top={d['top']}, "
                      f"full_H={d['full_entropy']:.2f}, conf={d['confidence']:.3f}")

    # 2. Manifold detection: entropy change at critical alpha
    print("\n--- Manifold Detection: Full Entropy at Critical Alpha ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        base_entropy = obj_data["baselines"]["category"]["full_entropy"]
        base_conf = obj_data["baselines"]["category"]["confidence"]
        key = "category_remove_category"
        if key not in obj_data["perturbations"]:
            continue
        perturb_data = obj_data["perturbations"][key]

        print(f"  {obj_word} (base_H={base_entropy:.2f}, base_conf={base_conf:.3f}):")
        for layer_key in [layer_names[f] for f in layer_fracs]:
            if layer_key not in perturb_data:
                continue
            curve = perturb_data[layer_key]
            # Find first alpha where top changes
            base_top = obj_data["baselines"]["category"]["top"]
            for alpha_str in sorted([float(a) for a in curve.keys()]):
                if alpha_str == 0:
                    continue
                d = curve[str(alpha_str)]
                if d["top"] != base_top:
                    delta_entropy = d["full_entropy"] - base_entropy
                    print(f"    {layer_key}: switch at α={alpha_str}, "
                          f"{base_top}→{d['top']}, "
                          f"ΔH={delta_entropy:+.2f}, conf={d['confidence']:.3f}, "
                          f"{'CLEAN' if delta_entropy < 0.5 else 'CONFUSED'}")
                    break

    # 3. Category-Property coupling at different layers
    print("\n--- Category-Property Coupling (|Δ| at α=1.0, remove_category) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        cat_key = "category_remove_category"
        prop_key = "property_remove_category"
        if cat_key not in obj_data["perturbations"] or prop_key not in obj_data["perturbations"]:
            continue
        cat_data = obj_data["perturbations"][cat_key]
        prop_data = obj_data["perturbations"][prop_key]

        print(f"  {obj_word}:")
        for layer_key in [layer_names[f] for f in layer_fracs]:
            cat_delta = abs(cat_data.get(layer_key, {}).get("1.0", {}).get("delta", 0))
            prop_delta = abs(prop_data.get(layer_key, {}).get("1.0", {}).get("delta", 0))
            ratio = cat_delta / prop_delta if prop_delta > 0.01 else float('inf')
            coupling = "DECOUPLED" if ratio > 5 else ("COUPLED" if ratio < 2 else "PARTIAL")
            print(f"    {layer_key}: cat|Δ|={cat_delta:.3f}, prop|Δ|={prop_delta:.3f}, "
                  f"ratio={ratio:.1f} ({coupling})")

    # Save results
    results_dir = ROOT / "results" / "phase428_midlayer_perturbation"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase428_r{round_num}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    results = convert(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model released. GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase428(model_name, round_num)


if __name__ == "__main__":
    main()
