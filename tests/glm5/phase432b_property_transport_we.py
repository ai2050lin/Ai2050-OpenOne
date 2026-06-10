"""
Phase 432b: Property Transport with W_E Directions
====================================================

CRITICAL FIX: Phase 432 used W_U (readout) directions which don't work
because they're OUTPUT directions, not INPUT directions.

This experiment uses W_E (embedding) property differences, analogous to
how category directions are computed.

Property directions in W_E space:
- sweet direction = W_E("sweet") - W_E("bitter") 
- red direction = W_E("red") - W_E("green")
- sharp direction = W_E("sharp") - W_E("dull")
- fast direction = W_E("fast") - W_E("slow")
- heavy direction = W_E("heavy") - W_E("light")
- furry direction = W_E("furry") - W_E("smooth")

This is the CORRECT comparison to category directions.

R1: 4 objects x 1 property each, alpha=2.0
R2: 8 objects x 2 properties each, multi-alpha

Usage:
  python tests/glm5/phase432b_property_transport_we.py qwen3 1
  python tests/glm5/phase432b_property_transport_we.py glm4 1
  python tests/glm5/phase432b_property_transport_we.py deepseek7b 1
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

# ===== Property definitions in W_E space =====
# Each property: (positive_word, negative_word, type)
# Direction = W_E(positive) - W_E(negative)
WE_PROPERTIES = OrderedDict([
    # Color properties
    ("red", {"pos": "red", "neg": "green", "type": "color"}),
    ("brown", {"pos": "brown", "neg": "white", "type": "color"}),
    # Taste/texture properties  
    ("sweet", {"pos": "sweet", "neg": "bitter", "type": "taste"}),
    ("furry", {"pos": "furry", "neg": "smooth", "type": "texture"}),
    # Attribute properties
    ("sharp", {"pos": "sharp", "neg": "dull", "type": "attribute"}),
    ("heavy", {"pos": "heavy", "neg": "light", "type": "attribute"}),
    ("fast", {"pos": "fast", "neg": "slow", "type": "attribute"}),
    ("long", {"pos": "long", "neg": "short", "type": "attribute"}),
    # Material properties
    ("metal", {"pos": "metal", "neg": "wood", "type": "material"}),
])

# Object -> relevant properties
OBJECT_PROPERTY_MAP = OrderedDict([
    ("apple", {
        "category": "fruit", "opposing": "animal",
        "properties": ["red", "sweet"],
    }),
    ("dog", {
        "category": "animal", "opposing": "fruit",
        "properties": ["brown", "furry"],
    }),
    ("knife", {
        "category": "tool", "opposing": "vehicle",
        "properties": ["sharp", "metal"],
    }),
    ("car", {
        "category": "vehicle", "opposing": "tool",
        "properties": ["fast", "heavy"],
    }),
])

# Category direction comparison objects
CATEGORY_OBJECTS = {
    "fruit": ["apple", "orange", "banana", "grape", "mango", "peach"],
    "animal": ["dog", "cat", "horse", "lion", "fish", "bird"],
    "tool": ["knife", "hammer", "spoon", "ruler", "nail"],
    "vehicle": ["car", "train", "bus", "truck", "boat"],
}

TEMPLATE = "The {obj} is"
CANDIDATES = OrderedDict([
    ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
])

EMBED_ALPHA_R1 = [1.0, 2.0, 4.0]
LAYER_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def verify_single_token(tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    return len(tok_ids) == 1, tok_ids


def compute_cosine(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_we_direction(W_E, tokenizer, pos_word, neg_word):
    """Get W_E direction = W_E(pos) - W_E(neg), normalized"""
    pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
    
    if not pos_ids or not neg_ids:
        return None, None, None
    
    pos_vec = W_E[pos_ids[0]]
    neg_vec = W_E[neg_ids[0]]
    
    direction = pos_vec - neg_vec
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction, pos_ids[0], neg_ids[0]


def get_category_direction(W_E, tokenizer, cat_name, opp_name):
    """Compute category direction from embedding centers"""
    cat_objs = CATEGORY_OBJECTS.get(cat_name, [])
    opp_objs = CATEGORY_OBJECTS.get(opp_name, [])
    
    cat_vecs = []
    for obj in cat_objs:
        ids = tokenizer.encode(obj, add_special_tokens=False)
        if ids:
            cat_vecs.append(W_E[ids[0]])
    
    opp_vecs = []
    for obj in opp_objs:
        ids = tokenizer.encode(obj, add_special_tokens=False)
        if ids:
            opp_vecs.append(W_E[ids[0]])
    
    if not cat_vecs or not opp_vecs:
        return None
    
    cat_center = np.mean(cat_vecs, axis=0)
    opp_center = np.mean(opp_vecs, axis=0)
    
    d = cat_center - opp_center
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    return d


def find_obj_position(input_ids, obj_tok_ids):
    prompt_ids = input_ids[0].cpu().tolist()
    for i in range(len(prompt_ids) - len(obj_tok_ids) + 1):
        if prompt_ids[i:i+len(obj_tok_ids)] == obj_tok_ids:
            return i
    return 1


def run_experiment(model_name, round_num):
    print(f"\n{'='*70}")
    print(f"Phase 432b: Property Transport with W_E Directions")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    target_layers = sorted(set(int(f * (n_layers - 1)) for f in LAYER_FRACS))
    print(f"  n_layers={n_layers}, d_model={d_model}")
    print(f"  Target layers: {target_layers}")
    
    # Get weight matrices
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    
    # Category candidate IDs
    cand_ids = {}
    for cand in CANDIDATES:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if ids:
            cand_ids[cand] = ids[-1]
    
    # Compute W_E property directions
    we_dirs = {}
    print("\nW_E Property Directions:")
    for prop_name, prop_info in WE_PROPERTIES.items():
        direction, pos_id, neg_id = get_we_direction(
            W_E, tokenizer, prop_info["pos"], prop_info["neg"]
        )
        if direction is not None:
            we_dirs[prop_name] = {
                "direction": direction,
                "pos_id": pos_id,
                "neg_id": neg_id,
                "pos_word": prop_info["pos"],
                "neg_word": prop_info["neg"],
                "type": prop_info["type"],
            }
            # Also get W_U direction for comparison
            wu_dir = W_U[pos_id].copy()
            wu_norm = np.linalg.norm(wu_dir)
            if wu_norm > 0:
                wu_dir = wu_dir / wu_norm
            cos_we_wu = compute_cosine(direction, wu_dir)
            print(f"  {prop_name}: {prop_info['pos']}-{prop_info['neg']}, "
                  f"cos(W_E, W_U)={cos_we_wu:.3f}")
        else:
            print(f"  {prop_name}: FAILED to compute direction")
    
    # Compute category directions
    cat_dirs = {}
    for cat_name in ["fruit", "animal", "tool", "vehicle"]:
        opp_map = {"fruit": "animal", "animal": "fruit", "tool": "vehicle", "vehicle": "tool"}
        d = get_category_direction(W_E, tokenizer, cat_name, opp_map[cat_name])
        if d is not None:
            cat_dirs[cat_name] = d
    
    # Print cosines between property and category directions
    print("\nCosine(property_W_E, category_W_E):")
    for prop_name in we_dirs:
        cos_vals = []
        for cat_name, cat_dir in cat_dirs.items():
            cos = compute_cosine(we_dirs[prop_name]["direction"], cat_dir)
            cos_vals.append(f"{cat_name}={cos:.3f}")
        print(f"  {prop_name}: {', '.join(cos_vals)}")
    
    alphas = EMBED_ALPHA_R1
    
    # Results
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": target_layers,
        "we_directions": {
            k: {"cos_we_wu": compute_cosine(v["direction"], 
                W_U[v["pos_id"]]/max(np.linalg.norm(W_U[v["pos_id"]]),1e-10)),
                "type": v["type"]}
            for k, v in we_dirs.items()
        },
        "per_object": {},
    }
    
    total = len(OBJECT_PROPERTY_MAP)
    count = 0
    t_start = time.time()
    
    for obj_word, obj_info in OBJECT_PROPERTY_MAP.items():
        count += 1
        cat = obj_info["category"]
        opp = obj_info["opposing"]
        props = obj_info["properties"]
        
        elapsed = time.time() - t_start
        print(f"\n--- [{count}/{total}] {obj_word} ({cat}) ---")
        
        is_single, obj_tok_ids = verify_single_token(tokenizer, obj_word)
        if not is_single:
            print(f"  Multi-token, skipping")
            continue
        
        obj_result = {
            "category": cat,
            "opposing": opp,
            "properties": {},
        }
        
        # Baseline output
        prompt = TEMPLATE.format(obj=obj_word)
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :].float().cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        
        top20_idx = np.argsort(probs)[::-1][:20]
        top20 = [(tokenizer.decode([i]).strip(), float(probs[i])) for i in top20_idx]
        obj_result["baseline_top20"] = top20
        
        for prop_name in props:
            if prop_name not in we_dirs:
                continue
            
            prop_data = we_dirs[prop_name]
            prop_dir = prop_data["direction"]
            prop_type = prop_data["type"]
            
            # Category direction for comparison
            cat_dir = cat_dirs.get(cat)
            cos_prop_cat = compute_cosine(prop_dir, cat_dir) if cat_dir is not None else 0
            
            print(f"  Property: {prop_name} ({prop_type}), cos(cat)={cos_prop_cat:.3f}")
            
            prop_result = {
                "property": prop_name,
                "type": prop_type,
                "cos_with_category": cos_prop_cat,
                "cos_we_wu": compute_cosine(prop_dir, 
                    W_U[prop_data["pos_id"]]/max(np.linalg.norm(W_U[prop_data["pos_id"]]),1e-10)),
            }
            
            for alpha in alphas:
                # Inject property direction at obj position in embedding
                perturbed_embeds = embed_layer(input_ids).detach().clone()
                direction_tensor = torch.tensor(prop_dir, dtype=perturbed_embeds.dtype, device=device)
                obj_pos = find_obj_position(input_ids, obj_tok_ids)
                perturbed_embeds[0, obj_pos, :] += (alpha * direction_tensor).to(perturbed_embeds.dtype)
                
                # Get perturbed output
                with torch.no_grad():
                    pert_outputs = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
                pert_logits = pert_outputs.logits[0, -1, :].float().cpu().numpy()
                pert_probs = np.exp(pert_logits - pert_logits.max())
                pert_probs = pert_probs / pert_probs.sum()
                
                # Effect on property-related tokens
                prop_effect = {}
                for word in [prop_data["pos_word"], prop_data["neg_word"]]:
                    ids = tokenizer.encode(" " + word, add_special_tokens=False)
                    if ids:
                        tid = ids[-1]
                        delta_p = float(pert_probs[tid] - probs[tid])
                        prop_effect[word] = {
                            "clean_prob": float(probs[tid]),
                            "perturbed_prob": float(pert_probs[tid]),
                            "delta": delta_p,
                        }
                
                # Effect on category candidates
                cat_effect = {}
                for cand, cid in cand_ids.items():
                    delta_p = float(pert_probs[cid] - probs[cid])
                    cat_effect[cand] = delta_p
                
                # Top shift
                delta_all = pert_probs - probs
                top_shift_idx = np.argmax(np.abs(delta_all))
                top_shift_word = tokenizer.decode([int(top_shift_idx)]).strip()
                top_shift_delta = float(delta_all[top_shift_idx])
                
                # Entropy
                H_clean = float(-np.sum(probs[probs > 1e-10] * np.log2(probs[probs > 1e-10])))
                H_pert = float(-np.sum(pert_probs[pert_probs > 1e-10] * np.log2(pert_probs[pert_probs > 1e-10])))
                
                alpha_result = {
                    "prop_effect": prop_effect,
                    "cat_effect": cat_effect,
                    "top_shift": (top_shift_word, top_shift_delta),
                    "entropy": {"clean": H_clean, "perturbed": H_pert},
                }
                
                prop_result[f"alpha_{alpha}"] = alpha_result
                
                # Transport tracking: collect deltas at key layers
                captured_clean = {}
                captured_pert = {}
                layers = get_layers(model)
                last_pos = input_ids.shape[1] - 1
                
                def make_hook(cap_dict, li):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hs = output[0].detach().float().cpu()
                        else:
                            hs = output.detach().float().cpu()
                        cap_dict[li] = {
                            "obj": hs[0, obj_pos, :].numpy().copy(),
                            "last": hs[0, last_pos, :].numpy().copy(),
                        }
                    return hook_fn
                
                hooks = [layers[li].register_forward_hook(make_hook(captured_clean, li)) for li in target_layers]
                with torch.no_grad():
                    clean_embeds = embed_layer(input_ids).detach().clone()
                    try:
                        _ = model(inputs_embeds=clean_embeds, attention_mask=attention_mask)
                    except Exception:
                        pass
                for h in hooks:
                    h.remove()
                
                hooks = [layers[li].register_forward_hook(make_hook(captured_pert, li)) for li in target_layers]
                with torch.no_grad():
                    try:
                        _ = model(inputs_embeds=perturbed_embeds, attention_mask=attention_mask)
                    except Exception:
                        pass
                for h in hooks:
                    h.remove()
                
                # Compute transport deltas
                transport = {}
                for li in target_layers:
                    if li in captured_clean and li in captured_pert:
                        for pos in ['obj', 'last']:
                            delta = captured_pert[li][pos] - captured_clean[li][pos]
                            delta_norm = float(np.linalg.norm(delta))
                            cos_with_inject = compute_cosine(delta, prop_dir)
                            cos_with_cat = compute_cosine(delta, cat_dir) if cat_dir is not None else 0
                            transport[f"L{li}/{pos}"] = {
                                "delta_norm": delta_norm,
                                "cos_with_prop": cos_with_inject,
                                "cos_with_cat": cos_with_cat,
                            }
                
                alpha_result["transport"] = transport
                
                print(f"    alpha={alpha}: prop_effect={prop_effect}, "
                      f"top_shift={top_shift_word}({top_shift_delta:+.4f}), "
                      f"H: {H_clean:.1f}->{H_pert:.1f}")
            
            obj_result["properties"][prop_name] = prop_result
        
        all_results["per_object"][obj_word] = obj_result
    
    # Save
    out_dir = ROOT / "results" / "phase432b_property_transport_we"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase432b_r{round_num}.json"
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {out_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PHASE 432b SUMMARY: W_E Property Direction Transport")
    print(f"{'='*70}")
    
    for obj_word, obj_data in all_results["per_object"].items():
        cat = obj_data["category"]
        print(f"\n  {obj_word} ({cat}):")
        for prop_name, prop_data in obj_data.get("properties", {}).items():
            ptype = prop_data["type"]
            cos_cat = prop_data.get("cos_with_category", 0)
            cos_wu = prop_data.get("cos_we_wu", 0)
            print(f"    {prop_name} ({ptype}): cos(cat)={cos_cat:.3f}, cos(W_E,W_U)={cos_wu:.3f}")
            
            for alpha in alphas:
                key = f"alpha_{alpha}"
                if key in prop_data:
                    adata = prop_data[key]
                    pe = adata.get("prop_effect", {})
                    cat_eff = adata.get("cat_effect", {})
                    top = adata.get("top_shift", ("?", 0))
                    H_c = adata.get("entropy", {}).get("clean", 0)
                    H_p = adata.get("entropy", {}).get("perturbed", 0)
                    
                    # Property target delta
                    pos_word = WE_PROPERTIES[prop_name]["pos"]
                    pos_delta = pe.get(pos_word, {}).get("delta", 0)
                    
                    # Category delta
                    cat_delta = cat_eff.get(cat, 0)
                    
                    # Transport cosine at mid-layer last pos
                    mid_li = target_layers[len(target_layers)//2]
                    mid_key = f"L{mid_li}/last"
                    transport = adata.get("transport", {})
                    mid_cos_prop = transport.get(mid_key, {}).get("cos_with_prop", 0)
                    mid_cos_cat = transport.get(mid_key, {}).get("cos_with_cat", 0)
                    
                    print(f"      alpha={alpha}: pos_delta={pos_delta:+.4f}, cat_delta={cat_delta:+.4f}, "
                          f"top={top[0]}({top[1]:+.4f}), "
                          f"H:{H_c:.1f}->{H_p:.1f}, "
                          f"transport_L{mid_li}/last: cos_prop={mid_cos_prop:.3f}, cos_cat={mid_cos_cat:.3f}")
    
    # Release
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU after release: {gpu_after:.2f} GB")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phase432b_property_transport_we.py <model> <round>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_experiment(model_name, round_num)
