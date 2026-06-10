"""
Phase 429B: Norm-Scaled Mid-Layer Perturbation
================================================

Phase 429 R1 showed that even layer-specific probe directions have ZERO effect
at mid-layers! But sanity check (alpha=50 random) CAN change output.

Critical insight: mid-layer residual norms are ~10-100x larger than embedding norms.
So alpha=4 in embedding space is a large perturbation, but alpha=4 in mid-layer
residual space is tiny.

This experiment uses NORM-SCALED perturbation:
  perturbation = alpha * (direction / ||direction||) * ||h_l||

So alpha now represents a FRACTION of the residual norm, not absolute magnitude.
alpha=0.1 means 10% of residual norm added in the category direction.

Key tests:
1. Norm-scaled perturbation at each layer (alpha as fraction of ||h||)
2. Compare: embedding dir vs layer-probe dir vs random dir
3. Compare: obj position vs last token position
4. Sanity: measure residual norms at each layer

Usage:
  python tests/glm5/phase429b_norm_scaled_perturbation.py qwen3 1
  python tests/glm5/phase429b_norm_scaled_perturbation.py glm4 1
  python tests/glm5/phase429b_norm_scaled_perturbation.py deepseek7b 1
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

R1_OBJECTS = ["apple", "knife", "car"]
R2_OBJECTS = ["apple", "dog", "knife", "car", "cat", "orange", "hammer"]

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

# Alpha as fraction of residual norm
# Include NEGATIVE alpha to test pushing AWAY from current category
ALPHA_FRAC_R1 = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
ALPHA_FRAC_R2 = [-2.0, -1.5, -1.0, -0.8, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# Layer sampling
LAYER_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8]


def load_model_bf16(model_name):
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


def find_obj_position(input_ids, obj_tok_ids):
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            return i
    return 1


def compute_embedding_directions(model, tokenizer, device):
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
            category_directions[(cat_name, opposing)] = d
    
    return category_directions, category_centers, W_E


def collect_category_activations(model, tokenizer, device, n_layers):
    """Collect residual stream activations for all category members"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Collecting category activations...")
    
    layers = get_layers(model)
    result = {}
    
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        cat_activations = {}
        obj_words = cat_info["objects"]
        
        for obj_word in obj_words:
            is_single, tok_ids = verify_single_token(tokenizer, obj_word)
            if not is_single:
                continue
            
            prompt = KNOWLEDGE_TASKS["category"]["template"].format(obj=obj_word)
            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            
            obj_pos = find_obj_position(input_ids, tok_ids)
            last_pos = input_ids.shape[1] - 1
            
            captured = {}
            def make_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].detach().float().cpu()
                    else:
                        hs = output.detach().float().cpu()
                    captured[li] = hs
                return hook_fn
            
            hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception:
                    pass
            
            for h in hooks:
                h.remove()
            
            for li, hs in captured.items():
                if li not in cat_activations:
                    cat_activations[li] = {"obj": [], "last": [], "norms": []}
                cat_activations[li]["obj"].append(hs[0, obj_pos, :].numpy())
                cat_activations[li]["last"].append(hs[0, last_pos, :].numpy())
                cat_activations[li]["norms"].append(float(hs[0, obj_pos, :].norm().item()))
        
        result[cat_name] = cat_activations
        n_collected = cat_activations.get(0, {}).get("n_obj", len(cat_activations.get(0, {}).get("obj", [])))
        print(f"  {cat_name}: collected {len(cat_activations.get(0, {}).get('obj', []))} objects")
    
    return result


def compute_layer_probe_directions(category_activations, d_model):
    """Compute layer-specific category probe directions"""
    result = {}
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        opposing = cat_info["opposing"]
        key = (cat_name, opposing)
        
        if cat_name not in category_activations or opposing not in category_activations:
            continue
        
        cat_data = category_activations[cat_name]
        opp_data = category_activations[opposing]
        
        layer_dirs = {}
        common_layers = set(cat_data.keys()) & set(opp_data.keys())
        
        for li in common_layers:
            dirs = {}
            for pos in ["obj", "last"]:
                cat_vecs = cat_data[li].get(pos, [])
                opp_vecs = opp_data[li].get(pos, [])
                
                if not cat_vecs or not opp_vecs:
                    continue
                
                cat_mean = np.mean(cat_vecs, axis=0)
                opp_mean = np.mean(opp_vecs, axis=0)
                d = cat_mean - opp_mean
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    d = d / norm
                dirs[pos] = d
            
            if dirs:
                layer_dirs[li] = dirs
        
        result[key] = layer_dirs
    
    return result


def measure_residual_norms(model, tokenizer, device, n_layers, test_objects):
    """Measure residual stream norms at each layer for test objects"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Measuring residual norms...")
    
    layers = get_layers(model)
    norms = {}
    
    for obj_word in test_objects[:3]:  # Just a few objects
        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if not is_single:
            continue
        
        prompt = KNOWLEDGE_TASKS["category"]["template"].format(obj=obj_word)
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        
        obj_pos = find_obj_position(input_ids, tok_ids)
        last_pos = input_ids.shape[1] - 1
        
        captured = {}
        def make_hook(li):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0].detach().float().cpu()
                else:
                    hs = output.detach().float().cpu()
                captured[li] = hs
            return hook_fn
        
        hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks:
            h.remove()
        
        obj_norms = {}
        last_norms = {}
        for li, hs in captured.items():
            obj_norms[li] = float(hs[0, obj_pos, :].norm().item())
            last_norms[li] = float(hs[0, last_pos, :].norm().item())
        
        norms[obj_word] = {"obj": obj_norms, "last": last_norms}
        
        # Print key layers
        print(f"  {obj_word} residual norms:")
        for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            on = obj_norms.get(li, 0)
            ln = last_norms.get(li, 0)
            print(f"    L{li}: obj_norm={on:.2f}, last_norm={ln:.2f}")
    
    return norms


def run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates):
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
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
        "level": round(level, 4), "entropy": round(cand_entropy, 4),
        "top": top_cand, "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4), "top2_gap": round(top2_gap, 4),
    }


def run_norm_scaled_midlayer_perturbation(model, tokenizer, device, template, obj_word,
                                           direction, alpha_frac, cand_ids, candidates,
                                           target_layer_idx, perturb_position,
                                           obj_tok_ids, residual_norm_at_layer):
    """
    Perturb at mid-layer with NORM-SCALED perturbation.
    
    actual_perturbation_magnitude = alpha_frac * residual_norm_at_layer
    direction is unit-normalized, so:
    perturbation_vector = alpha_frac * residual_norm * direction
    """
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1
    
    # Scale perturbation by residual norm
    actual_magnitude = alpha_frac * residual_norm_at_layer
    perturbation = direction * actual_magnitude  # direction is already unit-normalized
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        
        pert_tensor = torch.tensor(
            perturbation, dtype=hs.dtype, device=hs.device
        )
        modified = hs.clone()
        
        if perturb_position == "obj":
            modified[0, obj_pos, :] += pert_tensor
        elif perturb_position == "last":
            modified[0, last_pos, :] += pert_tensor
        elif perturb_position == "all":
            modified[0, :, :] += pert_tensor.unsqueeze(0)
        
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
            return {"level": round(level, 4), "delta": 0.0, "top": "ERROR",
                    "full_entropy": 0, "confidence": 0, "top2_gap": 0}
    
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
        "level": round(level, 4), "entropy": round(cand_entropy, 4),
        "top": top_cand, "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4), "top2_gap": round(top2_gap, 4),
    }


def run_norm_scaled_embedding_perturbation(model, tokenizer, device, template, obj_word,
                                            direction, alpha_frac, cand_ids, candidates,
                                            perturb_position, embedding_norm):
    """Norm-scaled embedding perturbation"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    # Scale by embedding norm
    actual_magnitude = alpha_frac * embedding_norm
    pert_tensor = torch.tensor(
        direction * actual_magnitude, dtype=inputs_embeds.dtype, device=device
    )
    
    if perturb_position == "obj":
        inputs_embeds[0, obj_pos, :] += pert_tensor
    elif perturb_position == "last":
        inputs_embeds[0, last_pos, :] += pert_tensor
    elif perturb_position == "all":
        inputs_embeds[0, :, :] += pert_tensor.unsqueeze(0)
    
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        try:
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        except Exception:
            empty = {c: 1.0/len(candidates) for c in candidates}
            level = sum(candidates[c]*empty[c] for c in candidates)
            return {"level": round(level, 4), "delta": 0.0, "top": "ERROR",
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
        "level": round(level, 4), "entropy": round(cand_entropy, 4),
        "top": top_cand, "full_entropy": round(full_entropy, 4),
        "confidence": round(confidence, 4), "top2_gap": round(top2_gap, 4),
    }


def run_phase429b(model_name, round_num=1):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 429B: Norm-Scaled Mid-Layer Perturbation ({model_name}) R{round_num} [{timestamp}] ===")
    print(f"{'='*80}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # Test objects
    test_objects = R1_OBJECTS if round_num == 1 else R2_OBJECTS
    alpha_fracs = ALPHA_FRAC_R1 if round_num == 1 else ALPHA_FRAC_R2
    
    # Filter single-token
    single_token_objects = []
    for obj_word in test_objects:
        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if is_single:
            single_token_objects.append(obj_word)
        else:
            print(f"  WARNING: '{obj_word}' is multi-token, skipping")
    test_objects = single_token_objects
    
    # Step 1: Measure residual norms
    residual_norms = measure_residual_norms(model, tokenizer, device, info.n_layers, test_objects)
    
    # Step 2: Collect category activations and compute probe directions
    category_activations = collect_category_activations(model, tokenizer, device, info.n_layers)
    layer_probe_dirs = compute_layer_probe_directions(category_activations, info.d_model)
    
    # Step 3: Compute embedding directions
    embed_directions, category_centers, W_E = compute_embedding_directions(model, tokenizer, device)
    
    # Step 4: Measure embedding norms
    embed_norms = {}
    for obj_word in test_objects:
        tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
        embed_norms[obj_word] = float(np.linalg.norm(W_E[tok_ids[0]]))
    print(f"\n  Embedding norms: {embed_norms}")
    
    # Target layers
    target_layers = {}
    for f in LAYER_FRACS:
        if f == 0.0:
            target_layers[f] = -1
        else:
            target_layers[f] = min(int(f * info.n_layers), info.n_layers - 1)
    
    results = {
        "model": model_name, "model_class": info.model_class,
        "n_layers": info.n_layers, "d_model": info.d_model,
        "timestamp": timestamp, "phase": "429b", "round": round_num,
        "test_objects": test_objects, "alpha_fracs": alpha_fracs,
        "target_layers": {str(f): f"L{target_layers[f]}" if target_layers[f] >= 0 else "embed" 
                         for f in LAYER_FRACS},
        "residual_norms_summary": {},
        "per_object": {},
    }
    
    # Summarize residual norms
    for obj_word in test_objects:
        if obj_word in residual_norms:
            norms = residual_norms[obj_word]
            summary = {}
            for li in [0, info.n_layers//4, info.n_layers//2, 3*info.n_layers//4, info.n_layers-1]:
                summary[f"L{li}"] = {
                    "obj": round(norms["obj"].get(li, 0), 2),
                    "last": round(norms["last"].get(li, 0), 2),
                }
            results["residual_norms_summary"][obj_word] = summary
    
    total_tests = 0
    t_start = time.time()
    
    for obj_idx, obj_word in enumerate(test_objects):
        obj_cat = get_object_category(obj_word)
        if obj_cat is None:
            continue
        opposing_cat = OBJECT_CATEGORIES[obj_cat]["opposing"]
        direction_key = (obj_cat, opposing_cat)
        
        obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
        
        print(f"\n[{time.strftime('%H:%M:%S')}] === Object {obj_idx+1}/{len(test_objects)}: "
              f"{obj_word} (cat={obj_cat}) ===")
        
        obj_results = {
            "category": obj_cat, "opposing": opposing_cat,
            "baselines": {}, "perturbations": {},
        }
        
        for task_name, task_info in KNOWLEDGE_TASKS.items():
            template = task_info["template"]
            candidates = task_info["candidates"]
            cand_ids = get_candidate_ids(tokenizer, candidates)
            
            if round_num == 1 and task_name != "category":
                continue  # Only test category task for R1
            
            print(f"\n  Task: {task_name}")
            
            # Baseline
            baseline = run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates)
            base_level = baseline["level"]
            base_entropy = baseline["full_entropy"]
            obj_results["baselines"][task_name] = baseline
            print(f"    Baseline: level={base_level:.3f}, top={baseline['top']}, "
                  f"full_H={base_entropy:.2f}, conf={baseline['confidence']:.3f}")
            
            # Test direction types
            for dir_type in ["embed_dir", "layer_probe"]:
                dir_results = {}
                
                for pos_type in ["obj", "last"]:
                    pos_results = {}
                    
                    for frac in LAYER_FRACS:
                        lidx = target_layers[frac]
                        layer_key = f"L{lidx}" if lidx >= 0 else "embed"
                        
                        # Get direction
                        if dir_type == "embed_dir":
                            if direction_key not in embed_directions:
                                continue
                            direction = embed_directions[direction_key]
                        elif dir_type == "layer_probe":
                            if direction_key in layer_probe_dirs and lidx in layer_probe_dirs[direction_key]:
                                layer_dirs = layer_probe_dirs[direction_key][lidx]
                                if pos_type in layer_dirs:
                                    direction = layer_dirs[pos_type]
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                        
                        # Get residual norm for scaling
                        if lidx == -1:
                            res_norm = embed_norms.get(obj_word, 10.0)
                        else:
                            res_norm = residual_norms.get(obj_word, {}).get("obj", {}).get(lidx, 50.0)
                            if pos_type == "last":
                                res_norm = residual_norms.get(obj_word, {}).get("last", {}).get(lidx, 50.0)
                        
                        curve_data = {}
                        
                        for alpha_frac in alpha_fracs:
                            if alpha_frac == 0.0:
                                curve_data["0.0"] = {
                                    "level": round(base_level, 4), "delta": 0.0,
                                    "top": baseline["top"],
                                    "full_entropy": base_entropy,
                                    "confidence": baseline["confidence"],
                                    "actual_magnitude": 0.0,
                                }
                                continue
                            
                            actual_mag = alpha_frac * res_norm
                            
                            if lidx == -1:
                                res = run_norm_scaled_embedding_perturbation(
                                    model, tokenizer, device, template, obj_word,
                                    direction, alpha_frac, cand_ids, candidates,
                                    perturb_position=pos_type,
                                    embedding_norm=res_norm
                                )
                            else:
                                res = run_norm_scaled_midlayer_perturbation(
                                    model, tokenizer, device, template, obj_word,
                                    direction, alpha_frac, cand_ids, candidates,
                                    lidx, pos_type, obj_tok_ids,
                                    residual_norm_at_layer=res_norm
                                )
                            
                            delta = res["level"] - base_level
                            curve_data[str(alpha_frac)] = {
                                "level": res["level"],
                                "delta": round(delta, 4),
                                "top": res["top"],
                                "full_entropy": res["full_entropy"],
                                "confidence": res["confidence"],
                                "actual_magnitude": round(actual_mag, 2),
                            }
                            total_tests += 1
                        
                        pos_results[layer_key] = curve_data
                    
                    dir_results[pos_type] = pos_results
                
                obj_results["perturbations"][f"{task_name}_{dir_type}"] = dir_results
                
                # Print summary
                print(f"\n    --- {dir_type} | {task_name} (norm-scaled) ---")
                for pos_type in ["obj", "last"]:
                    for frac in LAYER_FRACS:
                        lidx = target_layers[frac]
                        layer_key = f"L{lidx}" if lidx >= 0 else "embed"
                        if layer_key in dir_results.get(pos_type, {}):
                            curve = dir_results[pos_type][layer_key]
                            for af in ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0"]:
                                if af in curve:
                                    c = curve[af]
                                    print(f"      {dir_type}@{layer_key}/{pos_type} a_frac={af}: "
                                          f"D={c['delta']:+.3f}, top={c['top'][:4]}, "
                                          f"H={c['full_entropy']:.1f}, c={c['confidence']:.3f}, "
                                          f"mag={c['actual_magnitude']:.1f}")
            
            # GPU log
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                if gpu > 10:
                    print(f"    [GPU: {gpu:.2f}GB]")
        
        results["per_object"][obj_word] = obj_results
        
        elapsed = time.time() - t_start
        est_total = elapsed / (obj_idx + 1) * len(test_objects)
        print(f"  [{time.strftime('%H:%M:%S')}] Progress: {obj_idx+1}/{len(test_objects)}, "
              f"elapsed={elapsed/60:.1f}min, est={est_total/60:.1f}min")
    
    # ===== Summary =====
    print(f"\n{'='*80}")
    print(f"=== Phase 429B Summary ({model_name}) R{round_num} ===")
    print(f"{'='*80}")
    
    print("\n--- Residual Norms by Layer ---")
    for obj_word in test_objects:
        if obj_word in results["residual_norms_summary"]:
            print(f"  {obj_word}:")
            for layer_key, norms in results["residual_norms_summary"][obj_word].items():
                print(f"    {layer_key}: obj={norms['obj']:.1f}, last={norms['last']:.1f}")
    
    print("\n--- Direction Comparison (norm-scaled, alpha_frac=+/-1.0, category, last pos) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        od = results["per_object"][obj_word]
        base = od["baselines"]["category"]["level"]
        base_top = od["baselines"]["category"]["top"]
        print(f"  {obj_word} (base: level={base:.2f}, top={base_top}):")
        
        for dir_type in ["embed_dir", "layer_probe"]:
            key = f"category_{dir_type}"
            if key not in od["perturbations"]:
                continue
            pd = od["perturbations"][key]
            for pos in ["obj", "last"]:
                for frac in LAYER_FRACS:
                    lidx = target_layers[frac]
                    layer_key = f"L{lidx}" if lidx >= 0 else "embed"
                    if layer_key in pd.get(pos, {}):
                        curve = pd[pos][layer_key]
                        for af in ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0"]:
                            if af in curve:
                                c = curve[af]
                                print(f"    {dir_type}@{layer_key}/{pos} a_frac={af}: "
                                      f"D={c['delta']:+.3f} top={c['top'][:5]} H={c['full_entropy']:.1f} "
                                      f"c={c['confidence']:.3f} mag={c['actual_magnitude']:.1f}")
    
    # Save
    results_dir = ROOT / "results" / "phase429b_norm_scaled"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase429b_r{round_num}.json"
    
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
        if isinstance(obj, tuple):
            return str(obj)
        return obj
    
    results = convert(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
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
    
    run_phase429b(model_name, round_num)


if __name__ == "__main__":
    main()
