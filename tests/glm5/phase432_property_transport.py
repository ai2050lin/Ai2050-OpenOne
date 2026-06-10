"""
Phase 432: Property Natural Transport
======================================

KEY QUESTION: Does the natural transport mechanism work for PROPERTIES
(not just category)?

If the model transports category info (fruit/animal/tool/vehicle),
does it also transport property info (red/sharp/fast)?

If yes → the transport mechanism is GENERAL, not category-specific.
If no → the transport mechanism only handles category information.

METHOD:
1. Define properties for each object: color, attribute, material
2. Inject property direction at embedding (obj position)
3. Track how property perturbation propagates through layers
4. Measure whether property perturbation changes the target property output
5. Compare with category transport effectiveness

PROPERTY DEFINITIONS:
- apple → red (color), sweet (taste), round (shape)
- orange → orange (color), sour (taste), round (shape)  
- dog → brown (color), furry (texture), loud (attribute)
- cat → gray (color), soft (texture), quiet (attribute)
- knife → silver (color), sharp (attribute), metal (material)
- hammer → gray (color), heavy (attribute), metal (material)
- car → black (color), fast (attribute), engine (component)
- train → gray (color), long (attribute), track (component)

PROPERTY TASK: "The {obj} is" → check if target property appears in top candidates

MEASUREMENTS:
1. Baseline: model's natural property prediction
2. Property perturbation effect: inject property direction, check change
3. Natural transport: track δ_l through layers (like Phase 430)
4. Compare property transport vs category transport effectiveness
5. Cross-property interference: does injecting "red" affect "sharp"?

R1: 4 objects × 2 properties each = 8 combinations
R2: 8 objects × 2-3 properties each

Usage:
  python tests/glm5/phase432_property_transport.py qwen3 1
  python tests/glm5/phase432_property_transport.py glm4 1
  python tests/glm5/phase432_property_transport.py deepseek7b 1
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

# ===== Object-Property definitions =====
# Each object has: category, opposing, properties (property_word → [related_words for evaluation])
OBJECT_PROPERTIES = OrderedDict([
    ("apple", {
        "category": "fruit",
        "opposing": "animal",
        "properties": OrderedDict([
            ("red", {"related": ["red", "green", "yellow"], "type": "color"}),
            ("sweet", {"related": ["sweet", "sour", "bitter"], "type": "taste"}),
        ]),
    }),
    ("orange", {
        "category": "fruit",
        "opposing": "animal",
        "properties": OrderedDict([
            ("orange", {"related": ["orange", "yellow", "red"], "type": "color"}),
            ("sour", {"related": ["sour", "sweet", "bitter"], "type": "taste"}),
        ]),
    }),
    ("dog", {
        "category": "animal",
        "opposing": "fruit",
        "properties": OrderedDict([
            ("brown", {"related": ["brown", "black", "white"], "type": "color"}),
            ("furry", {"related": ["furry", "hairy", "soft"], "type": "texture"}),
        ]),
    }),
    ("cat", {
        "category": "animal",
        "opposing": "fruit",
        "properties": OrderedDict([
            ("gray", {"related": ["gray", "black", "white"], "type": "color"}),
            ("soft", {"related": ["soft", "furry", "smooth"], "type": "texture"}),
        ]),
    }),
    ("knife", {
        "category": "tool",
        "opposing": "vehicle",
        "properties": OrderedDict([
            ("sharp", {"related": ["sharp", "dull", "pointed"], "type": "attribute"}),
            ("metal", {"related": ["metal", "steel", "iron"], "type": "material"}),
        ]),
    }),
    ("hammer", {
        "category": "tool",
        "opposing": "vehicle",
        "properties": OrderedDict([
            ("heavy", {"related": ["heavy", "solid", "strong"], "type": "attribute"}),
            ("metal", {"related": ["metal", "steel", "iron"], "type": "material"}),
        ]),
    }),
    ("car", {
        "category": "vehicle",
        "opposing": "tool",
        "properties": OrderedDict([
            ("fast", {"related": ["fast", "quick", "rapid"], "type": "attribute"}),
            ("engine", {"related": ["engine", "motor", "wheel"], "type": "component"}),
        ]),
    }),
    ("train", {
        "category": "vehicle",
        "opposing": "tool",
        "properties": OrderedDict([
            ("long", {"related": ["long", "big", "large"], "type": "attribute"}),
            ("track", {"related": ["track", "rail", "station"], "type": "component"}),
        ]),
    }),
])

# Templates for different tasks
CATEGORY_TEMPLATE = "A {obj} is a kind of"
PROPERTY_TEMPLATE = "The {obj} is"

CANDIDATES_CATEGORY = OrderedDict([
    ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
])

EMBED_ALPHA_R1 = [1.0, 2.0, 4.0]
EMBED_ALPHA_R2 = [0.5, 1.0, 2.0, 3.0, 4.0, 8.0]

R1_OBJECTS = ["apple", "dog", "knife", "car"]
R2_OBJECTS = list(OBJECT_PROPERTIES.keys())

# Layer sampling
LAYER_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_model_bf16(model_name):
    """Load model with bf16 + device_map=auto"""
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


def verify_single_token(tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    return len(tok_ids) == 1, tok_ids


def compute_cosine(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_logits_and_probs(model, input_ids, attention_mask):
    """Get logits and probability distribution"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return logits, probs


def compute_entropy(probs):
    probs = probs[probs > 1e-10]
    return float(-np.sum(probs * np.log2(probs)))


def find_obj_position(input_ids, obj_tok_ids):
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            return i
    return 1


def collect_transport_at_layers(model, tokenizer, device, n_layers, W_E, W_U,
                                 obj_word, direction, alpha, target_layers,
                                 template, obj_tok_ids):
    """
    Collect transported direction at each target layer.
    Returns: {layer_idx: {'obj_delta_norm': float, 'last_delta_norm': float, 
                           'cos_obj': float, 'cos_last': float}}
    """
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = seq_len - 1
    
    embed_layer = model.get_input_embeddings()
    
    # Clean embeddings
    with torch.no_grad():
        clean_embeds = embed_layer(input_ids).detach().clone()
    
    # Perturbed: add direction at obj position
    perturbed_embeds = clean_embeds.clone()
    direction_tensor = torch.tensor(direction, dtype=clean_embeds.dtype, device=device)
    perturbed_embeds[0, obj_pos, :] += (alpha * direction_tensor).to(clean_embeds.dtype)
    
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
    
    # Compute deltas and cosine with original direction
    results = {}
    for li in target_layers:
        if li in captured_clean and li in captured_perturbed:
            for pos in ['obj', 'last']:
                delta = captured_perturbed[li][pos] - captured_clean[li][pos]
                delta_norm = float(np.linalg.norm(delta))
                cos_with_dir = compute_cosine(delta, direction)
                clean_norm = float(np.linalg.norm(captured_clean[li][pos]))
                results[f"L{li}/{pos}"] = {
                    "delta_norm": delta_norm,
                    "cos_with_inject": cos_with_dir,
                    "clean_norm": clean_norm,
                }
    
    return results, obj_pos, last_pos


def measure_property_effect(model, tokenizer, device, obj_word, direction, alpha,
                            template, obj_tok_ids, target_tokens):
    """
    Measure how property injection affects the output distribution.
    
    Returns: {
        'clean': {token: prob},
        'perturbed': {token: prob},
        'delta': {token: float},
        'top_shift': (token, delta_prob),
    }
    """
    prompt = template.format(obj=obj_word)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    
    embed_layer = model.get_input_embeddings()
    
    # Clean
    _, clean_probs = get_logits_and_probs(model, input_ids, attention_mask)
    
    # Perturbed
    with torch.no_grad():
        clean_embeds = embed_layer(input_ids).detach().clone()
    perturbed_embeds = clean_embeds.clone()
    direction_tensor = torch.tensor(direction, dtype=clean_embeds.dtype, device=device)
    perturbed_embeds[0, obj_pos, :] += (alpha * direction_tensor).to(clean_embeds.dtype)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    perturbed_probs = np.exp(logits - logits.max())
    perturbed_probs = perturbed_probs / perturbed_probs.sum()
    
    # Compute effects on target tokens
    clean_target = {}
    perturbed_target = {}
    delta_target = {}
    for token_word in target_tokens:
        tok_ids = tokenizer.encode(" " + token_word, add_special_tokens=False)
        if tok_ids:
            tid = tok_ids[-1]
            clean_target[token_word] = float(clean_probs[tid])
            perturbed_target[token_word] = float(perturbed_probs[tid])
            delta_target[token_word] = float(perturbed_probs[tid] - clean_probs[tid])
    
    # Find top shift
    top_shift = max(delta_target.items(), key=lambda x: abs(x[1])) if delta_target else ("none", 0)
    
    return {
        "clean": clean_target,
        "perturbed": perturbed_target,
        "delta": delta_target,
        "top_shift": top_shift,
        "clean_entropy": compute_entropy(clean_probs),
        "perturbed_entropy": compute_entropy(perturbed_probs),
    }


def run_experiment(model_name, round_num):
    """Main experiment"""
    print(f"\n{'='*70}")
    print(f"Phase 432: Property Natural Transport")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    target_layers = sorted(set(
        int(f * (n_layers - 1)) for f in LAYER_FRACS
    ))
    print(f"  Target layers: {target_layers}")
    
    # Get weight matrices
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    
    # Get category directions (for comparison)
    category_centers = {}
    for cat_name in ["fruit", "animal", "tool", "vehicle"]:
        vecs = []
        for obj_word, obj_info in OBJECT_PROPERTIES.items():
            if obj_info["category"] == cat_name:
                tok_ids = tokenizer.encode(obj_word, add_special_tokens=False)
                if tok_ids:
                    vecs.append(W_E[tok_ids[0]])
        if vecs:
            category_centers[cat_name] = np.mean(vecs, axis=0)
    
    category_directions = {}
    for cat_name in ["fruit", "animal", "tool", "vehicle"]:
        opp_map = {"fruit": "animal", "animal": "fruit", "tool": "vehicle", "vehicle": "tool"}
        opp = opp_map[cat_name]
        if cat_name in category_centers and opp in category_centers:
            d = category_centers[cat_name] - category_centers[opp]
            norm = np.linalg.norm(d)
            if norm > 0:
                category_directions[cat_name] = d / norm
    
    # Property directions from W_U
    def get_property_direction(prop_word):
        tok_ids = tokenizer.encode(prop_word, add_special_tokens=False)
        if tok_ids:
            tid = tok_ids[-1]
            direction = W_U[tid].copy()
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm, tid
        return None, None
    
    # Category candidate IDs
    cand_ids = {}
    for cand in CANDIDATES_CATEGORY:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    
    # Select objects
    objects = R1_OBJECTS if round_num == 1 else R2_OBJECTS
    alphas = EMBED_ALPHA_R1 if round_num == 1 else EMBED_ALPHA_R2
    
    # Verify single-token
    valid_objects = []
    for obj in objects:
        is_single, tok_ids = verify_single_token(tokenizer, obj)
        if is_single:
            valid_objects.append(obj)
        else:
            print(f"  WARNING: '{obj}' is multi-token, skipping")
    
    # Results
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": target_layers,
        "per_object": {},
    }
    
    total = len(valid_objects) * len(alphas) * 3  # per prop, per alpha: baseline + transport + effect
    count = 0
    t_start = time.time()
    
    for obj_word in valid_objects:
        obj_info = OBJECT_PROPERTIES[obj_word]
        cat = obj_info["category"]
        opp = obj_info["opposing"]
        
        print(f"\n--- {obj_word} ({cat}) ---")
        
        obj_result = {
            "category": cat,
            "opposing": opp,
            "properties": {},
        }
        
        # First: measure baseline category and property outputs
        cat_prompt = CATEGORY_TEMPLATE.format(obj=obj_word)
        cat_input_ids = tokenizer.encode(cat_prompt, add_special_tokens=True, return_tensors="pt")
        cat_input_ids = cat_input_ids.to(device)
        cat_attention_mask = torch.ones_like(cat_input_ids)
        _, cat_probs = get_logits_and_probs(model, cat_input_ids, cat_attention_mask)
        
        cat_cand_probs = {c: float(cat_probs[cid]) for c, cid in cand_ids.items()}
        obj_result["baseline_category"] = cat_cand_probs
        
        # Property baseline
        prop_prompt = PROPERTY_TEMPLATE.format(obj=obj_word)
        prop_input_ids = tokenizer.encode(prop_prompt, add_special_tokens=True, return_tensors="pt")
        prop_input_ids = prop_input_ids.to(device)
        prop_attention_mask = torch.ones_like(prop_input_ids)
        _, prop_probs = get_logits_and_probs(model, prop_input_ids, prop_attention_mask)
        
        # Top 20 tokens for property prompt
        top20_idx = np.argsort(prop_probs)[::-1][:20]
        top20 = [(tokenizer.decode([i]).strip(), float(prop_probs[i])) for i in top20_idx]
        obj_result["baseline_property_top20"] = top20
        
        is_single, obj_tok_ids = verify_single_token(tokenizer, obj_word)
        
        for prop_word, prop_info in obj_info["properties"].items():
            prop_type = prop_info["type"]
            related = prop_info["related"]
            
            print(f"  Property: {prop_word} ({prop_type})")
            
            prop_dir, prop_tid = get_property_direction(prop_word)
            if prop_dir is None:
                print(f"    Cannot get direction for '{prop_word}', skipping")
                continue
            
            prop_result = {
                "property_word": prop_word,
                "property_type": prop_type,
                "related_words": related,
                "direction_source": "W_U",
            }
            
            # Also get category direction for comparison
            cat_dir = category_directions.get(cat)
            cos_cat_prop = compute_cosine(prop_dir, cat_dir) if cat_dir is not None else 0
            prop_result["cos_with_category_dir"] = cos_cat_prop
            
            for alpha in alphas:
                count += 1
                elapsed = time.time() - t_start
                print(f"    [{count}] alpha={alpha} ({elapsed/60:.1f}min)")
                
                # 1. Measure property injection effect
                try:
                    effect = measure_property_effect(
                        model, tokenizer, device, obj_word, prop_dir, alpha,
                        PROPERTY_TEMPLATE, obj_tok_ids, related
                    )
                    prop_result[f"effect_a{alpha}"] = effect
                except Exception as e:
                    print(f"      Effect measurement error: {e}")
                
                # 2. Measure category injection effect (for comparison)
                if cat_dir is not None:
                    try:
                        cat_effect = measure_property_effect(
                            model, tokenizer, device, obj_word, cat_dir, alpha,
                            CATEGORY_TEMPLATE, obj_tok_ids, list(cand_ids.keys())
                        )
                        prop_result[f"category_effect_a{alpha}"] = cat_effect
                    except Exception as e:
                        print(f"      Category effect error: {e}")
                
                # 3. Transport direction tracking
                try:
                    transport, obj_pos, last_pos = collect_transport_at_layers(
                        model, tokenizer, device, n_layers, W_E, W_U,
                        obj_word, prop_dir, alpha, target_layers,
                        PROPERTY_TEMPLATE, obj_tok_ids
                    )
                    prop_result[f"transport_a{alpha}"] = transport
                except Exception as e:
                    print(f"      Transport error: {e}")
            
            obj_result["properties"][prop_word] = prop_result
        
        all_results["per_object"][obj_word] = obj_result
    
    # Save results
    out_dir = ROOT / "results" / "phase432_property_transport"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase432_r{round_num}.json"
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {out_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PHASE 432 SUMMARY")
    print(f"{'='*70}")
    
    for obj_word, obj_data in all_results["per_object"].items():
        cat = obj_data["category"]
        print(f"\n  {obj_word} ({cat}):")
        for prop_word, prop_data in obj_data["properties"].items():
            ptype = prop_data["property_type"]
            cos_cp = prop_data.get("cos_with_category_dir", 0)
            print(f"    {prop_word} ({ptype}): cos(cat_dir)={cos_cp:.3f}")
            
            for alpha in alphas:
                key = f"effect_a{alpha}"
                if key in prop_data:
                    eff = prop_data[key]
                    top_shift = eff.get("top_shift", ("?", 0))
                    H_clean = eff.get("clean_entropy", 0)
                    H_pert = eff.get("perturbed_entropy", 0)
                    print(f"      alpha={alpha}: top_shift={top_shift[0]}({top_shift[1]:+.4f}), "
                          f"H: {H_clean:.1f}→{H_pert:.1f}")
    
    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU after release: {gpu_after:.2f} GB")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phase432_property_transport.py <model> <round>")
        print("  model: qwen3, glm4, deepseek7b")
        print("  round: 1 or 2")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_experiment(model_name, round_num)
