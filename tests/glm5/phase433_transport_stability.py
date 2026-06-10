"""
Phase 433: Transport Operator Stability — Cross-Object Consistency
===================================================================

KEY QUESTION: Do same-category objects share the same transport direction?

If same-category objects (apple, orange, lemon) produce SIMILAR transported
directions at each layer, then T_{0→l} is category-specific (shared pathway).
If they produce DIFFERENT directions, then transport is object-specific.

This is the most critical validation experiment for the "natural transport" theory.

METHOD:
1. For each object in a category, compute δ_l = h_l(perturbed) - h_l(clean)
   using the SAME embedding perturbation direction (category direction)
2. Compare δ_l across objects in the same category using cosine similarity
3. Also compare across categories to check category-specificity

MEASUREMENTS:
1. Within-category cosine: cos(δ_l(obj_A), δ_l(obj_B)) for same-category pairs
2. Cross-category cosine: cos(δ_l(obj_A), δ_l(obj_C)) for different-category pairs
3. Transport direction norm consistency across objects
4. Category separability: within-category vs cross-category cosine gap

R1: 3 objects/category × 4 categories = 12 objects, alpha=2.0
R2: 5-7 objects/category × 4 categories, multi-alpha

Usage:
  python tests/glm5/phase433_transport_stability.py qwen3 1
  python tests/glm5/phase433_transport_stability.py glm4 1
  python tests/glm5/phase433_transport_stability.py deepseek7b 1
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

from model_utils import MODEL_CONFIGS, get_model_info, get_layers, get_W_U

# ===== Category definitions =====
CATEGORIES = OrderedDict([
    ("fruit", {
        "r1_objects": ["apple", "orange", "lemon"],
        "r2_objects": ["apple", "orange", "lemon", "banana", "grape"],
        "opposing": "animal",
    }),
    ("animal", {
        "r1_objects": ["dog", "cat", "horse"],
        "r2_objects": ["dog", "cat", "horse", "lion", "fish"],
        "opposing": "fruit",
    }),
    ("tool", {
        "r1_objects": ["knife", "hammer", "spoon"],
        "r2_objects": ["knife", "hammer", "spoon", "ruler", "nail"],
        "opposing": "vehicle",
    }),
    ("vehicle", {
        "r1_objects": ["car", "train", "bus"],
        "r2_objects": ["car", "train", "bus", "truck", "boat"],
        "opposing": "tool",
    }),
])

TEMPLATE = "A {obj} is a kind of"
CANDIDATES = OrderedDict([
    ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
])

EMBED_ALPHA_R1 = [2.0]
EMBED_ALPHA_R2 = [1.0, 2.0, 4.0]

# Layer sampling: every 10% + first + last
LAYER_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_model_bf16(model_name):
    """Load model with bf16 + device_map=auto, try flash_attention first"""
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


def get_category_directions(W_E, tokenizer):
    """Compute category directions at embedding level"""
    category_centers = {}
    for cat_name, cat_info in CATEGORIES.items():
        vecs = []
        all_objs = cat_info["r2_objects"]
        for obj_word in all_objs:
            tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
            if tok_ids:
                vecs.append(W_E[tok_ids[0]])
        if vecs:
            category_centers[cat_name] = np.mean(vecs, axis=0)

    category_directions = {}
    for cat_name, cat_info in CATEGORIES.items():
        opposing = cat_info["opposing"]
        if cat_name in category_centers and opposing in category_centers:
            d = category_centers[cat_name] - category_centers[opposing]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            category_directions[(cat_name, opposing)] = d

    return category_directions


def compute_cosine(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_transport_directions(model, tokenizer, device, n_layers, W_E, category_dir, 
                                 obj_word, alpha, target_layers):
    """
    Collect transported direction δ_l at each target layer for a given object.
    Returns: {layer_idx: {'obj': delta_vec, 'last': delta_vec, 'obj_norm': float, 'last_norm': float}}
    """
    prompt = TEMPLATE.format(obj=obj_word)
    tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    # Find obj position
    prompt_ids = input_ids[0].cpu().tolist()
    obj_pos = 1  # default
    for i in range(len(prompt_ids) - len(tok_ids) + 1):
        if prompt_ids[i:i+len(tok_ids)] == tok_ids:
            obj_pos = i
            break
    last_pos = seq_len - 1
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Clean run
    with torch.no_grad():
        clean_embeds = embed_layer(input_ids).detach().clone()
    
    # Perturbed run: add category direction at obj position
    perturbed_embeds = clean_embeds.clone()
    direction_tensor = torch.tensor(category_dir, dtype=clean_embeds.dtype, device=device)
    perturbed_embeds[0, obj_pos, :] += (alpha * direction_tensor).to(clean_embeds.dtype)
    
    attention_mask = torch.ones_like(input_ids)
    
    # Collect activations via hooks
    captured_clean = {}
    captured_perturbed = {}
    layers = get_layers(model)
    
    def make_hook(captured_dict, li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].detach().float().cpu()
            else:
                hs = output.detach().float().cpu()
            captured_dict[li] = {
                "obj": hs[0, obj_pos, :].numpy().copy(),
                "last": hs[0, last_pos, :].numpy().copy(),
            }
        return hook_fn
    
    # Clean forward
    hooks = [layers[li].register_forward_hook(make_hook(captured_clean, li)) for li in target_layers]
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=clean_embeds, attention_mask=attention_mask)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    
    # Perturbed forward
    hooks = [layers[li].register_forward_hook(make_hook(captured_perturbed, li)) for li in target_layers]
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    
    # Compute deltas
    results = {}
    for li in target_layers:
        if li in captured_clean and li in captured_perturbed:
            for pos in ['obj', 'last']:
                delta = captured_perturbed[li][pos] - captured_clean[li][pos]
                delta_norm = float(np.linalg.norm(delta))
                results[f"L{li}/{pos}"] = {
                    "delta": delta,
                    "norm": delta_norm,
                }
                # Store clean norm for reference
                clean_norm = float(np.linalg.norm(captured_clean[li][pos]))
                results[f"L{li}/{pos}"]["clean_norm"] = clean_norm
    
    return results, obj_pos, last_pos


def run_experiment(model_name, round_num):
    """Main experiment"""
    print(f"\n{'='*70}")
    print(f"Phase 433: Transport Operator Stability")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    # Target layers
    target_layers = sorted(set(
        int(f * (n_layers - 1)) for f in LAYER_FRACS
    ))
    print(f"  Target layers: {target_layers}")
    
    # Get W_E and category directions
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()
    category_directions = get_category_directions(W_E, tokenizer)
    
    # Get W_U for candidate IDs
    W_U = get_W_U(model, model_name)
    
    # Get candidate token IDs
    cand_ids = {}
    for cand in CANDIDATES:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    
    # Select objects for this round
    alphas = EMBED_ALPHA_R1 if round_num == 1 else EMBED_ALPHA_R2
    
    # Verify single-token objects
    valid_objects = {}
    for cat_name, cat_info in CATEGORIES.items():
        obj_list = cat_info[f"r{round_num}_objects"]
        valid = []
        for obj in obj_list:
            is_single, tok_ids = verify_single_token(tokenizer, obj)
            if is_single:
                valid.append((obj, tok_ids[0]))
            else:
                print(f"  WARNING: '{obj}' is multi-token, skipping")
        valid_objects[cat_name] = valid
    
    # Results storage
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": target_layers,
        "alphas": alphas,
        "per_category": {},
    }
    
    # Main loop: for each category, collect transport directions for all objects
    total = sum(len(v) for v in valid_objects.values()) * len(alphas)
    count = 0
    t_start = time.time()
    
    for cat_name, cat_info in CATEGORIES.items():
        opposing = cat_info["opposing"]
        cat_dir_key = (cat_name, opposing)
        if cat_dir_key not in category_directions:
            print(f"  Skipping {cat_name}: no direction")
            continue
        
        cat_dir = category_directions[cat_dir_key]
        print(f"\n--- Category: {cat_name} vs {opposing} ---")
        
        cat_results = {
            "category": cat_name,
            "opposing": opposing,
            "objects": {},
        }
        
        for obj_word, obj_tok_id in valid_objects[cat_name]:
            for alpha in alphas:
                count += 1
                elapsed = time.time() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate / 60 if rate > 0 else 0
                print(f"  [{count}/{total}] {obj_word} alpha={alpha} "
                      f"({elapsed/60:.1f}min, ETA={eta:.1f}min)")
                
                try:
                    transport_data, obj_pos, last_pos = collect_transport_directions(
                        model, tokenizer, device, n_layers, W_E, cat_dir,
                        obj_word, alpha, target_layers
                    )
                    
                    # Store raw deltas and norms
                    obj_data = {
                        "obj_tok_id": obj_tok_id,
                        "obj_pos": obj_pos,
                        "last_pos": last_pos,
                        "alpha": alpha,
                        "transport": {},
                    }
                    
                    for layer_key_pos, data in transport_data.items():
                        obj_data["transport"][layer_key_pos] = {
                            "norm": data["norm"],
                            "clean_norm": data["clean_norm"],
                            # Don't store the full delta vector (too large), 
                            # we'll compute cosines on the fly
                        }
                    
                    # Store full delta vectors for cross-object comparison (only for last alpha)
                    if alpha == alphas[-1]:
                        obj_data["delta_vectors"] = {}
                        for layer_key_pos, data in transport_data.items():
                            obj_data["delta_vectors"][layer_key_pos] = data["delta"].tolist()
                    
                    # Compute baseline output
                    prompt = TEMPLATE.format(obj=obj_word)
                    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
                    input_ids = input_ids.to(device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, -1, :].float().cpu().numpy()
                    probs = np.exp(logits - logits.max())
                    probs = probs / probs.sum()
                    
                    cand_probs = {c: float(probs[cid]) for c, cid in cand_ids.items()}
                    obj_data["baseline_cand_probs"] = cand_probs
                    
                    cat_results["objects"][f"{obj_word}_a{alpha}"] = obj_data
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
        
        all_results["per_category"][cat_name] = cat_results
    
    # ===== Cross-object cosine similarity analysis =====
    print(f"\n{'='*70}")
    print("CROSS-OBJECT COSINE ANALYSIS")
    print(f"{'='*70}")
    
    cosine_analysis = {}
    
    for cat_name, cat_info in CATEGORIES.items():
        opposing = cat_info["opposing"]
        cat_dir_key = (cat_name, opposing)
        if cat_dir_key not in category_directions:
            continue
        
        if cat_name not in all_results["per_category"]:
            continue
        
        cat_data = all_results["per_category"][cat_name]
        
        # Get objects with delta_vectors (last alpha only)
        objects_with_vectors = {}
        for key, obj_data in cat_data["objects"].items():
            if "delta_vectors" in obj_data:
                base_name = key.split("_a")[0]
                objects_with_vectors[base_name] = obj_data["delta_vectors"]
        
        if len(objects_with_vectors) < 2:
            continue
        
        obj_names = sorted(objects_with_vectors.keys())
        
        # Get all layer/position keys
        all_layer_keys = set()
        for name in obj_names:
            all_layer_keys.update(objects_with_vectors[name].keys())
        all_layer_keys = sorted(all_layer_keys)
        
        cat_cosine = {
            "within_category": {},  # {layer_key: [cos_values]}
            "within_mean": {},
            "within_std": {},
        }
        
        for lk in all_layer_keys:
            # Collect delta vectors for this layer/position
            deltas = {}
            for name in obj_names:
                if lk in objects_with_vectors[name]:
                    deltas[name] = np.array(objects_with_vectors[name][lk])
            
            if len(deltas) < 2:
                continue
            
            # Within-category cosine
            within_cos = []
            names = sorted(deltas.keys())
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    cos = compute_cosine(deltas[names[i]], deltas[names[j]])
                    within_cos.append(cos)
            
            cat_cosine["within_category"][lk] = within_cos
            cat_cosine["within_mean"][lk] = float(np.mean(within_cos)) if within_cos else 0
            cat_cosine["within_std"][lk] = float(np.std(within_cos)) if within_cos else 0
        
        cosine_analysis[cat_name] = cat_cosine
        
        # Print summary
        print(f"\n--- {cat_name} (within-category) ---")
        for lk in all_layer_keys:
            if lk in cat_cosine["within_mean"]:
                mean = cat_cosine["within_mean"][lk]
                std = cat_cosine["within_std"][lk]
                n_pairs = len(cat_cosine["within_category"].get(lk, []))
                print(f"  {lk}: cos={mean:.3f}±{std:.3f} (n={n_pairs})")
    
    # ===== Cross-category cosine comparison =====
    print(f"\n{'='*70}")
    print("CROSS-CATEGORY COSINE COMPARISON")
    print(f"{'='*70}")
    
    # For each layer/position, compare within-category vs cross-category cosine
    all_layer_keys = set()
    for cat_name in cosine_analysis:
        all_layer_keys.update(cosine_analysis[cat_name]["within_mean"].keys())
    all_layer_keys = sorted(all_layer_keys)
    
    # Collect all delta vectors across categories for cross-category comparison
    all_deltas_by_category = {}
    for cat_name, cat_info in CATEGORIES.items():
        if cat_name not in all_results["per_category"]:
            continue
        cat_data = all_results["per_category"][cat_name]
        objects_with_vectors = {}
        for key, obj_data in cat_data["objects"].items():
            if "delta_vectors" in obj_data:
                base_name = key.split("_a")[0]
                objects_with_vectors[base_name] = obj_data["delta_vectors"]
        if objects_with_vectors:
            all_deltas_by_category[cat_name] = objects_with_vectors
    
    cross_cat_results = {}
    for lk in all_layer_keys:
        # Collect deltas per category
        cat_deltas = {}
        for cat_name, obj_vecs in all_deltas_by_category.items():
            deltas = []
            for name, vecs in obj_vecs.items():
                if lk in vecs:
                    deltas.append(np.array(vecs[lk]))
            if deltas:
                cat_deltas[cat_name] = deltas
        
        # Cross-category cosine
        cat_names = sorted(cat_deltas.keys())
        cross_cos = []
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                for d1 in cat_deltas[cat_names[i]]:
                    for d2 in cat_deltas[cat_names[j]]:
                        cross_cos.append(compute_cosine(d1, d2))
        
        # Within-category cosine (average across all categories)
        within_cos_all = []
        for cat_name in cat_names:
            deltas = cat_deltas[cat_name]
            for i in range(len(deltas)):
                for j in range(i+1, len(deltas)):
                    within_cos_all.append(compute_cosine(deltas[i], deltas[j]))
        
        cross_mean = float(np.mean(cross_cos)) if cross_cos else 0
        within_mean = float(np.mean(within_cos_all)) if within_cos_all else 0
        gap = within_mean - cross_mean
        
        cross_cat_results[lk] = {
            "within_mean": within_mean,
            "cross_mean": cross_mean,
            "gap": gap,
            "n_within": len(within_cos_all),
            "n_cross": len(cross_cos),
        }
        
        if abs(gap) > 0.01:
            print(f"  {lk}: within={within_mean:.3f}, cross={cross_mean:.3f}, gap={gap:+.3f}")
    
    # Save results
    all_results["cosine_analysis"] = cosine_analysis
    all_results["cross_category"] = cross_cat_results
    
    # Save to file
    out_dir = ROOT / "results" / "phase433_transport_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase433_r{round_num}.json"
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {out_file}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("PHASE 433 SUMMARY")
    print(f"{'='*70}")
    
    for cat_name in cosine_analysis:
        ca = cosine_analysis[cat_name]
        # Find best within-category layer/position
        best_lk = None
        best_mean = -1
        for lk, mean in ca["within_mean"].items():
            if mean > best_mean:
                best_mean = mean
                best_lk = lk
        if best_lk:
            print(f"  {cat_name}: best within-category cos={best_mean:.3f} at {best_lk}")
    
    print(f"\nCross-category gap (within - cross):")
    for lk in sorted(cross_cat_results.keys()):
        gap = cross_cat_results[lk]["gap"]
        if abs(gap) > 0.01:
            within = cross_cat_results[lk]["within_mean"]
            cross = cross_cat_results[lk]["cross_mean"]
            marker = "***" if gap > 0.1 else ("**" if gap > 0.05 else "*")
            print(f"  {lk}: gap={gap:+.3f} (W={within:.3f}, C={cross:.3f}) {marker}")
    
    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU after release: {gpu_after:.2f} GB")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phase433_transport_stability.py <model> <round>")
        print("  model: qwen3, glm4, deepseek7b")
        print("  round: 1 or 2")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_experiment(model_name, round_num)
