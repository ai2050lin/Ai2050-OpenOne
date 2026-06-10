"""
Phase 429: Layer-Specific Probe Directions + Position Routing
=============================================================

Phase 428 showed embedding-space category direction has ZERO effect at mid-layers.
This is the most critical follow-up: find each layer's OWN category direction.

Key Questions:
1. Does each layer have its own category direction in residual stream?
   - Compute: d_{l,p}^{cat} = mean(h_{l,p}(cat_A)) - mean(h_{l,p}(cat_B))
   - If this direction causes category switch when perturbed → YES
   - If not → category info is not in residual stream at that layer

2. Position routing: which token position carries category info at mid-layers?
   - Object token position (same as Phase 428)
   - Last token position (where readout happens)
   - All token positions

3. Sanity check: verify hooks work by injecting large random vectors

Design:
- For each model, collect residual stream activations for category members
- Compute layer-specific category directions at each position
- Perturb with these directions and measure effect
- Compare: embedding direction vs layer-specific direction vs random direction
- Compare: object position vs last token position

Usage:
  python tests/glm5/phase429_layer_probe_direction.py qwen3 1
  python tests/glm5/phase429_layer_probe_direction.py glm4 1
  python tests/glm5/phase429_layer_probe_direction.py deepseek7b 1
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

# R1: core objects, R2: expanded
R1_OBJECTS = ["apple", "knife", "car"]
R2_OBJECTS = ["apple", "dog", "knife", "car", "cat", "orange", "hammer"]

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

# Alpha grid
ALPHA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]

# Layer sampling
LAYER_SAMPLE_R1 = [0.0, 0.2, 0.4, 0.6, 0.8]  # fraction of total layers
LAYER_SAMPLE_R2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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
    """Find start position of object token in input_ids"""
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            return i
    return 1


# ========================================================================
# STEP 1: Collect residual stream activations for all category members
# ========================================================================

def collect_category_activations(model, tokenizer, device, n_layers, category_template):
    """
    Run each category member through the model, collect residual stream at each layer.
    Returns: {cat_name: {layer_idx: {position: mean_activation_vector}}}
    
    position can be:
      - "obj": object token position
      - "last": last token position
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Collecting category activations...")
    
    layers = get_layers(model)
    result = {}
    
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        cat_activations = {}  # {layer_idx: {"obj": [vectors], "last": [vectors]}}
        obj_words = cat_info["objects"]
        
        for obj_word in obj_words:
            is_single, tok_ids = verify_single_token(tokenizer, obj_word)
            if not is_single:
                continue
            
            prompt = category_template.format(obj=obj_word)
            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            
            obj_pos = find_obj_position(input_ids, tok_ids)
            last_pos = input_ids.shape[1] - 1
            
            # Collect hidden states using hooks
            captured = {}
            
            def make_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].detach().float().cpu()
                    else:
                        hs = output.detach().float().cpu()
                    captured[li] = hs
                return hook_fn
            
            hooks = []
            sample_layers = list(range(n_layers))  # all layers
            for li in sample_layers:
                hooks.append(layers[li].register_forward_hook(make_hook(li)))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"  Forward failed for {obj_word}: {e}")
                    for h in hooks:
                        h.remove()
                    continue
            
            for h in hooks:
                h.remove()
            
            # Extract activations at obj and last positions
            for li, hs in captured.items():
                if li not in cat_activations:
                    cat_activations[li] = {"obj": [], "last": []}
                
                # Object position
                obj_vec = hs[0, obj_pos, :].numpy()
                cat_activations[li]["obj"].append(obj_vec)
                
                # Last position
                last_vec = hs[0, last_pos, :].numpy()
                cat_activations[li]["last"].append(last_vec)
        
        # Compute means
        result[cat_name] = {}
        for li in cat_activations:
            obj_vecs = cat_activations[li]["obj"]
            last_vecs = cat_activations[li]["last"]
            result[cat_name][li] = {
                "obj_mean": np.mean(obj_vecs, axis=0) if obj_vecs else None,
                "last_mean": np.mean(last_vecs, axis=0) if last_vecs else None,
                "n_obj": len(obj_vecs),
            }
        
        print(f"  {cat_name}: collected {result[cat_name].get(0, {}).get('n_obj', 0)} objects")
    
    return result


def compute_layer_probe_directions(category_activations, d_model):
    """
    Compute layer-specific category probe directions.
    
    For each (layer, position), compute:
    d_{l,p}^{cat} = mean(h_{l,p}(cat)) - mean(h_{l,p}(opposing))
    
    Returns: {(cat_name, opposing): {layer_idx: {"obj": direction, "last": direction}}}
    """
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
                cat_mean = cat_data[li].get(f"{pos}_mean")
                opp_mean = opp_data[li].get(f"{pos}_mean")
                
                if cat_mean is None or opp_mean is None:
                    continue
                
                d = cat_mean - opp_mean
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    d = d / norm
                dirs[pos] = d
            
            if dirs:
                layer_dirs[li] = dirs
        
        result[key] = layer_dirs
    
    return result


def compute_embedding_directions(model, tokenizer, device):
    """Compute embedding-space category directions (same as Phase 426/428)"""
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


# ========================================================================
# STEP 2: Run perturbation experiments with different directions
# ========================================================================

def run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates):
    """Run baseline (no perturbation)"""
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


def run_midlayer_perturbation(model, tokenizer, device, template, obj_word,
                               direction, alpha, cand_ids, candidates,
                               target_layer_idx, perturb_position,
                               obj_tok_ids):
    """
    Perturb at a specific mid-layer and position.
    
    perturb_position: "obj" = object token, "last" = last token, "all" = all tokens
    """
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1
    seq_len = input_ids.shape[1]
    
    residual_metrics = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        pert_tensor = torch.tensor(
            direction * alpha,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        modified = hidden_states.clone()
        
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
        except Exception as e:
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


def run_embedding_perturbation(model, tokenizer, device, template, obj_word,
                               direction, alpha, cand_ids, candidates,
                               perturb_position="obj"):
    """Perturb at embedding layer"""
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    obj_tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    pert_tensor = torch.tensor(
        direction * alpha, dtype=inputs_embeds.dtype, device=device
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


# ========================================================================
# STEP 3: Sanity check - verify hooks work with large random perturbation
# ========================================================================

def sanity_check_hooks(model, tokenizer, device, n_layers, test_obj="apple"):
    """Verify hooks work by injecting a large random vector"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Sanity check: large random perturbation...")
    
    layers = get_layers(model)
    template = KNOWLEDGE_TASKS["category"]["template"]
    candidates = KNOWLEDGE_TASKS["category"]["candidates"]
    cand_ids = get_candidate_ids(tokenizer, candidates)
    
    prompt = template.format(obj=test_obj)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Baseline
    with torch.no_grad():
        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
    base_logits = out_base.logits[0, -1, :].float().cpu()
    base_top5 = torch.argsort(base_logits, descending=True)[:5]
    base_words = [tokenizer.decode([int(t)]) for t in base_top5]
    print(f"  Baseline top5: {base_words}")
    
    # Test at mid-layer (50%) with huge random vector
    mid_layer = n_layers // 2
    rng = np.random.RandomState(42)
    
    for alpha in [10.0, 50.0]:
        random_dir = rng.randn(base_logits.shape[0])  # d_model from logit shape; wrong
        # Get d_model from model
        d_model = model.config.hidden_size
        random_dir = rng.randn(d_model)
        norm = np.linalg.norm(random_dir)
        if norm > 0:
            random_dir = random_dir / norm
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            pert = torch.tensor(
                random_dir * alpha, dtype=hs.dtype, device=hs.device
            )
            modified = hs.clone()
            modified[0, -1, :] += pert
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        handle = layers[mid_layer].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        handle.remove()
        
        pert_logits = out.logits[0, -1, :].float().cpu()
        # Compare
        logit_diff = float((pert_logits - base_logits).norm().item())
        pert_top5 = torch.argsort(pert_logits, descending=True)[:5]
        pert_words = [tokenizer.decode([int(t)]) for t in pert_top5]
        top_changed = base_words != pert_words
        
        print(f"  L{mid_layer} random α={alpha}: logit_diff={logit_diff:.2f}, "
              f"top5 changed={top_changed}, top5={pert_words[:3]}")
    
    print("  Sanity check complete.")


# ========================================================================
# MAIN EXPERIMENT
# ========================================================================

def run_phase429(model_name, round_num=1):
    """Run Phase 429 experiment"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 429: Layer-Specific Probe Directions + Position Routing ({model_name}) R{round_num} [{timestamp}] ===")
    print(f"{'='*80}")
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # Sanity check first!
    sanity_check_hooks(model, tokenizer, device, info.n_layers)
    
    # Step 1: Collect category activations
    category_activations = collect_category_activations(
        model, tokenizer, device, info.n_layers,
        KNOWLEDGE_TASKS["category"]["template"]
    )
    
    # Step 2: Compute layer-specific probe directions
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing layer-specific probe directions...")
    layer_probe_dirs = compute_layer_probe_directions(category_activations, info.d_model)
    
    # Print direction similarity between layers
    for key, layer_dirs in layer_probe_dirs.items():
        cat_name, opposing = key
        # Check similarity between embedding direction and each layer's direction
        print(f"\n  Direction ({cat_name} vs {opposing}):")
        sorted_layers = sorted(layer_dirs.keys())
        for li in sorted_layers:
            dirs = layer_dirs[li]
            for pos in ["obj", "last"]:
                if pos in dirs:
                    d = dirs[pos]
                    print(f"    L{li}@{pos}: norm_check={np.linalg.norm(d):.4f}")
    
    # Step 3: Compute embedding-space directions (Phase 428 baseline)
    embed_directions, category_centers, W_E = compute_embedding_directions(model, tokenizer, device)
    
    # Step 4: Test objects
    if round_num == 1:
        test_objects = R1_OBJECTS
        layer_fracs = LAYER_SAMPLE_R1
        alpha_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    else:
        test_objects = R2_OBJECTS
        layer_fracs = LAYER_SAMPLE_R2
        alpha_grid = [0.0, 0.3, 0.5, 1.0, 2.0, 4.0]
    
    # Map fractions to layer indices
    target_layers = {}
    for f in layer_fracs:
        if f == 0.0:
            target_layers[f] = -1  # embedding
        else:
            target_layers[f] = min(int(f * info.n_layers), info.n_layers - 1)
    
    # Filter single-token objects
    single_token_objects = []
    for obj_word in test_objects:
        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if is_single:
            single_token_objects.append(obj_word)
        else:
            print(f"  WARNING: '{obj_word}' is multi-token, skipping")
    test_objects = single_token_objects
    
    # Results structure
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": timestamp,
        "phase": "429",
        "round": round_num,
        "test_objects": test_objects,
        "alpha_grid": alpha_grid,
        "target_layers": {str(f): f"L{target_layers[f]}" if target_layers[f] >= 0 else "embed" 
                         for f in layer_fracs},
        "per_object": {},
    }
    
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
            
            # Baseline
            baseline = run_baseline(model, tokenizer, device, template, obj_word, cand_ids, candidates)
            base_level = baseline["level"]
            obj_results["baselines"][task_name] = baseline
            print(f"    Baseline: level={base_level:.3f}, top={baseline['top']}, "
                  f"full_H={baseline['full_entropy']:.2f}, conf={baseline['confidence']:.3f}")
            
            # ===== Test 3 direction types × 2 positions × multiple layers =====
            # Direction types:
            #   1. "embed_dir": embedding-space category direction (Phase 428 style)
            #   2. "layer_probe": layer-specific probe direction at target layer
            #   3. "random": random orthogonal direction (control)
            
            rng = np.random.RandomState(42)
            
            for dir_type in ["embed_dir", "layer_probe", "random"]:
                # Only test category task for R1 (save time)
                if round_num == 1 and task_name != "category" and dir_type != "embed_dir":
                    continue
                
                dir_results = {}
                
                for pos_type in ["obj", "last"]:
                    pos_results = {}
                    
                    for frac in layer_fracs:
                        lidx = target_layers[frac]
                        layer_key = f"L{lidx}" if lidx >= 0 else "embed"
                        
                        # Get direction for this (dir_type, layer, position)
                        if dir_type == "embed_dir":
                            if direction_key in embed_directions:
                                direction = embed_directions[direction_key]
                            else:
                                continue
                        elif dir_type == "layer_probe":
                            if direction_key in layer_probe_dirs and lidx in layer_probe_dirs[direction_key]:
                                layer_dirs = layer_probe_dirs[direction_key][lidx]
                                if pos_type in layer_dirs:
                                    direction = layer_dirs[pos_type]
                                else:
                                    continue
                            else:
                                continue
                        elif dir_type == "random":
                            if direction_key in embed_directions:
                                ref_dir = embed_directions[direction_key]
                            else:
                                ref_dir = rng.randn(info.d_model)
                                ref_dir = ref_dir / np.linalg.norm(ref_dir)
                            # Make orthogonal random direction
                            d = rng.randn(info.d_model)
                            proj = np.dot(d, ref_dir) / max(np.dot(ref_dir, ref_dir), 1e-10) * ref_dir
                            d = d - proj
                            d_norm = np.linalg.norm(d)
                            if d_norm > 0:
                                direction = d / d_norm
                            else:
                                continue
                        else:
                            continue
                        
                        curve_data = {}
                        
                        for alpha in alpha_grid:
                            if alpha == 0.0:
                                curve_data["0.0"] = {
                                    "level": round(base_level, 4), "delta": 0.0,
                                    "top": baseline["top"],
                                    "full_entropy": baseline["full_entropy"],
                                    "confidence": baseline["confidence"],
                                }
                                continue
                            
                            if lidx == -1:
                                # Embedding perturbation
                                res = run_embedding_perturbation(
                                    model, tokenizer, device, template, obj_word,
                                    direction, alpha, cand_ids, candidates,
                                    perturb_position=pos_type
                                )
                            else:
                                # Mid-layer perturbation
                                res = run_midlayer_perturbation(
                                    model, tokenizer, device, template, obj_word,
                                    direction, alpha, cand_ids, candidates,
                                    lidx, pos_type, obj_tok_ids
                                )
                            
                            delta = res["level"] - base_level
                            curve_data[str(alpha)] = {
                                "level": res["level"],
                                "delta": round(delta, 4),
                                "top": res["top"],
                                "full_entropy": res["full_entropy"],
                                "confidence": res["confidence"],
                            }
                            total_tests += 1
                        
                        pos_results[layer_key] = curve_data
                    
                    dir_results[pos_type] = pos_results
                
                obj_results["perturbations"][f"{task_name}_{dir_type}"] = dir_results
                
                # Print summary for this direction type
                if task_name == "category":
                    print(f"\n    --- {dir_type} | {task_name} ---")
                    for pos_type in ["obj", "last"]:
                        for frac in layer_fracs:
                            lidx = target_layers[frac]
                            layer_key = f"L{lidx}" if lidx >= 0 else "embed"
                            if layer_key in dir_results.get(pos_type, {}):
                                curve = dir_results[pos_type][layer_key]
                                # Print alpha=1.0 and alpha=4.0
                                for a in ["1.0", "4.0"]:
                                    if a in curve:
                                        d = curve[a]
                                        print(f"      {dir_type}@{layer_key}/{pos_type} α={a}: "
                                              f"Δ={d['delta']:+.3f}, top={d['top'][:4]}, "
                                              f"H={d['full_entropy']:.1f}, conf={d['confidence']:.3f}")
            
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
    print(f"=== Phase 429 Summary ({model_name}) R{round_num} ===")
    print(f"{'='*80}")
    
    # 1. Embedding vs Layer-probe direction effectiveness
    print("\n--- Direction Comparison: embed_dir vs layer_probe vs random (α=4.0, category task, last pos) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        base_level = obj_data["baselines"]["category"]["level"]
        base_top = obj_data["baselines"]["category"]["top"]
        
        print(f"  {obj_word} (base: level={base_level:.2f}, top={base_top}):")
        
        for dir_type in ["embed_dir", "layer_probe", "random"]:
            key = f"category_{dir_type}"
            if key not in obj_data["perturbations"]:
                continue
            perturb_data = obj_data["perturbations"][key]
            
            # Get best delta across all layers and positions at α=4.0
            best_delta = 0
            best_info = ""
            for pos_type in ["obj", "last"]:
                if pos_type not in perturb_data:
                    continue
                for layer_key, curve in perturb_data[pos_type].items():
                    if "4.0" in curve:
                        d = curve["4.0"]
                        if abs(d["delta"]) > abs(best_delta):
                            best_delta = d["delta"]
                            best_info = f"{layer_key}/{pos_type}(Δ={d['delta']:+.3f},top={d['top'][:4]},H={d['full_entropy']:.1f},c={d['confidence']:.3f})"
            
            print(f"    {dir_type}: best={best_info}")
    
    # 2. Position routing: obj vs last
    print("\n--- Position Routing: obj vs last (layer_probe, α=4.0, category) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        key = "category_layer_probe"
        if key not in obj_data["perturbations"]:
            continue
        perturb_data = obj_data["perturbations"][key]
        
        print(f"  {obj_word}:")
        for pos_type in ["obj", "last"]:
            if pos_type not in perturb_data:
                continue
            max_delta = 0
            best_layer = ""
            for layer_key, curve in perturb_data[pos_type].items():
                if "4.0" in curve:
                    d = curve["4.0"]
                    if abs(d["delta"]) > abs(max_delta):
                        max_delta = d["delta"]
                        best_layer = layer_key
            print(f"    {pos_type}: max|Δ|={abs(max_delta):.3f} at {best_layer}")
    
    # 3. Layer-by-layer effectiveness
    print("\n--- Layer-by-Layer Effectiveness (layer_probe, last pos, α=4.0, category) ---")
    for obj_word in test_objects:
        if obj_word not in results["per_object"]:
            continue
        obj_data = results["per_object"][obj_word]
        key = "category_layer_probe"
        if key not in obj_data["perturbations"]:
            continue
        perturb_data = obj_data["perturbations"].get(key, {})
        last_data = perturb_data.get("last", {})
        
        print(f"  {obj_word}:")
        for layer_key in sorted(last_data.keys()):
            curve = last_data[layer_key]
            if "4.0" in curve:
                d = curve["4.0"]
                print(f"    {layer_key}: Δ={d['delta']:+.3f}, top={d['top'][:4]}, "
                      f"H={d['full_entropy']:.1f}, conf={d['confidence']:.3f}")
    
    # 4. Direction similarity analysis
    print("\n--- Direction Similarity: Embed vs Layer-Probe (cosine) ---")
    for key_tuple, layer_dirs in layer_probe_dirs.items():
        cat_name, opposing = key_tuple
        if key_tuple not in embed_directions:
            continue
        embed_dir = embed_directions[key_tuple]
        
        print(f"  ({cat_name} vs {opposing}):")
        for li in sorted(layer_dirs.keys()):
            dirs = layer_dirs[li]
            for pos in ["obj", "last"]:
                if pos in dirs:
                    cos_sim = float(np.dot(embed_dir, dirs[pos]) / 
                                   (np.linalg.norm(embed_dir) * np.linalg.norm(dirs[pos]) + 1e-10))
                    if li % max(1, info.n_layers // 5) == 0 or li < 3 or li > info.n_layers - 4:
                        print(f"    L{li}@{pos}: cos(embed,probe)={cos_sim:.4f}")
    
    # Save results
    results_dir = ROOT / "results" / "phase429_layer_probe_direction"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase429_r{round_num}.json"
    
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
    
    # Save direction similarity separately (large matrix)
    direction_similarity = {}
    for key_tuple, layer_dirs in layer_probe_dirs.items():
        cat_name, opposing = key_tuple
        sim_key = f"{cat_name}_vs_{opposing}"
        if key_tuple not in embed_directions:
            continue
        embed_dir = embed_directions[key_tuple]
        
        sim_data = {}
        for li in sorted(layer_dirs.keys()):
            dirs = layer_dirs[li]
            for pos in ["obj", "last"]:
                if pos in dirs:
                    cos_sim = float(np.dot(embed_dir, dirs[pos]) / 
                                   (np.linalg.norm(embed_dir) * np.linalg.norm(dirs[pos]) + 1e-10))
                    sim_data[f"L{li}_{pos}"] = round(cos_sim, 4)
        direction_similarity[sim_key] = sim_data
    
    results["direction_similarity"] = direction_similarity
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
    
    run_phase429(model_name, round_num)


if __name__ == "__main__":
    main()
