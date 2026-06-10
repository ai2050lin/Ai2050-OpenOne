"""
Phase 430: Natural Transport Direction + Causal Tracing
=========================================================

Phase 429B showed layer-probe directions CAN cause mid-layer category switching,
but they are STATISTICAL (mean difference), not CAUSAL.

This experiment tests two questions:

Q1: What does the embedding-space category perturbation become after being
    naturally transported through the network?
    - Run model with embedding perturbation: h_L0' = h_L0 + α*d_embed
    - Record δ_l = h_l(perturbed) - h_l(clean) at each layer
    - This δ_l is the NATURALLY TRANSPORTED direction

Q2: Is the naturally transported direction more effective than the probe direction?
    - Inject δ_l into clean run at layer l, same position
    - Compare: does it reproduce the same category switch?
    - If yes: the transport operator T_{0→l} preserves semantic content
    - If no: the probe direction captures something else

Q3: Causal tracing - which layers/positions are causally critical?
    - Corrupt the input (swap category word)
    - Restore clean activation at each layer/position
    - Measure recovery of correct category
    - Find the "causal trace" of category information

Usage:
  python tests/glm5/phase430_natural_transport_direction.py qwen3 1
  python tests/glm5/phase430_natural_transport_direction.py glm4 1
  python tests/glm5/phase430_natural_transport_direction.py deepseek7b 1
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
R2_OBJECTS = ["apple", "dog", "knife", "car", "orange", "hammer", "train"]

KNOWLEDGE_TASKS = OrderedDict([
    ("category", {
        "template": "A {obj} is a kind of",
        "candidates": OrderedDict([
            ("fruit", 1), ("animal", 2), ("tool", 3), ("vehicle", 4), ("place", 5),
        ]),
    }),
])

# Embedding perturbation alpha (for generating transported directions)
EMBED_ALPHA_R1 = [1.0, 2.0, 4.0, 8.0]
EMBED_ALPHA_R2 = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

# Alpha fraction for injecting transported/probe direction at mid-layers
INJECT_ALPHA_FRAC = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]

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


def get_logits_and_probs(model, input_ids, attention_mask, device):
    """Get logits and probability distribution"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1).numpy()
    return logits.numpy(), probs


def analyze_category_output(probs, cand_ids, candidates):
    """Analyze category output"""
    cand_probs = {}
    for cand, cid in cand_ids.items():
        cand_probs[cand] = float(probs[cid])

    # Find top candidate
    top_idx = np.argmax(probs)
    top_prob = float(probs[top_idx])

    # Category level (-1 to +1)
    cat_name = list(candidates.keys())[0]  # target category
    opp_name = None
    for c, cid in cand_ids.items():
        if cand_probs.get(c, 0) > 0 and c != cat_name:
            opp_name = c
            break

    target_prob = cand_probs.get(list(candidates.keys())[0], 0)
    opp_prob = max(cand_probs.get(c, 0) for c in cand_ids if c != list(candidates.keys())[0]) if len(cand_probs) > 1 else 0
    level = target_prob - opp_prob

    # Confidence
    top5_probs = np.sort(probs)[::-1][:5]
    confidence = float(top5_probs[0] - top5_probs[1]) if len(top5_probs) > 1 else float(top5_probs[0])

    # Full entropy
    full_H = compute_entropy({i: p for i, p in enumerate(probs) if p > 1e-10})

    # Top 5 tokens
    top5_idx = np.argsort(probs)[::-1][:5]

    return {
        "level": float(level),
        "top_prob": float(top_prob),
        "confidence": float(confidence),
        "full_entropy": full_H,
        "cand_probs": cand_probs,
        "top": [int(x) for x in top5_idx[:5]],
    }


def collect_layer_activations(model, tokenizer, device, n_layers, prompt, obj_tok_ids):
    """Collect residual stream at every layer for both obj and last positions"""
    layers = get_layers(model)

    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1

    captured = {}

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].detach().float().cpu()
            else:
                hs = output.detach().float().cpu()
            captured[li] = {
                "obj": hs[0, obj_pos, :].numpy().copy(),
                "last": hs[0, last_pos, :].numpy().copy(),
                "obj_norm": float(hs[0, obj_pos, :].norm().item()),
                "last_norm": float(hs[0, last_pos, :].norm().item()),
            }
        return hook_fn

    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]

    with torch.no_grad():
        try:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return captured, input_ids, attention_mask, obj_pos, last_pos


def run_with_perturbation_at_layer(model, tokenizer, device, n_layers, prompt,
                                     obj_tok_ids, perturbation_dict, cand_ids, candidates):
    """
    Run model with perturbation injected at specific layer/position.

    perturbation_dict: {layer_idx: {"obj": vector, "last": vector}}
    """
    layers = get_layers(model)

    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    obj_pos = find_obj_position(input_ids, obj_tok_ids)
    last_pos = input_ids.shape[1] - 1

    injected = {}

    def make_inject_hook(li, perturb):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output

            if "obj" in perturb and perturb["obj"] is not None:
                p = torch.tensor(perturb["obj"], dtype=hs.dtype, device=hs.device)
                hs[0, obj_pos, :] += p
            if "last" in perturb and perturb["last"] is not None:
                p = torch.tensor(perturb["last"], dtype=hs.dtype, device=hs.device)
                hs[0, last_pos, :] += p

            injected[li] = True

            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        return hook_fn

    hooks = []
    for li, perturb in perturbation_dict.items():
        hooks.append(layers[li].register_forward_hook(make_inject_hook(li, perturb)))

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks:
                h.remove()
            return None

    for h in hooks:
        h.remove()

    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1).numpy()

    return analyze_category_output(probs, cand_ids, candidates)


def main():
    if len(sys.argv) < 3:
        print("Usage: python phase430_natural_transport_direction.py <model> <round>")
        print("  model: qwen3, glm4, deepseek7b")
        print("  round: 1 (R1 basic), 2 (R2 confirm)")
        sys.exit(1)

    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    is_r1 = round_num == 1

    test_objects = R1_OBJECTS if is_r1 else R2_OBJECTS
    embed_alphas = EMBED_ALPHA_R1 if is_r1 else EMBED_ALPHA_R2

    print(f"\n{'='*70}")
    print(f"Phase 430: Natural Transport Direction + Causal Tracing")
    print(f"Model: {model_name}, Round: R{round_num}")
    print(f"Objects: {test_objects}")
    print(f"Embed alphas: {embed_alphas}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Target layers
    target_layers = {}
    for frac in LAYER_FRACS:
        target_layers[frac] = int(frac * (n_layers - 1))

    # Compute embedding directions
    category_directions, category_centers, W_E = compute_embedding_directions(model, tokenizer, device)
    cand_ids = get_candidate_ids(tokenizer, KNOWLEDGE_TASKS["category"]["candidates"])

    # ===== Part 1: Collect natural transported directions =====
    print(f"\n[{time.strftime('%H:%M:%S')}] Part 1: Collecting natural transported directions...")

    results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": {str(k): v for k, v in target_layers.items()},
        "per_object": {},
    }

    for obj_word in test_objects:
        cat_name = get_object_category(obj_word)
        if cat_name is None:
            continue

        is_single, tok_ids = verify_single_token(tokenizer, obj_word)
        if not is_single:
            print(f"  Skipping {obj_word} (multi-token)")
            continue

        opposing = OBJECT_CATEGORIES[cat_name]["opposing"]
        dir_key = (cat_name, opposing)
        if dir_key not in category_directions:
            continue

        d_embed = category_directions[dir_key]
        prompt = KNOWLEDGE_TASKS["category"]["template"].format(obj=obj_word)

        print(f"\n  [{time.strftime('%H:%M:%S')}] === {obj_word} ({cat_name}) ===")

        # Step 1a: Get clean baseline
        clean_acts, input_ids, attention_mask, obj_pos, last_pos = \
            collect_layer_activations(model, tokenizer, device, n_layers, prompt, tok_ids)

        # Get baseline output
        _, baseline_probs = get_logits_and_probs(model, input_ids, attention_mask, device)
        baseline_result = analyze_category_output(baseline_probs, cand_ids,
                                                   KNOWLEDGE_TASKS["category"]["candidates"])

        print(f"  Baseline: level={baseline_result['level']:.3f}, "
              f"top_cand={max(baseline_result['cand_probs'], key=baseline_result['cand_probs'].get)}, "
              f"H={baseline_result['full_entropy']:.1f}, c={baseline_result['confidence']:.3f}")

        obj_result = {
            "baseline": baseline_result,
            "category": cat_name,
            "opposing": opposing,
            "obj_pos": obj_pos,
            "last_pos": last_pos,
        }

        # Step 1b: For each embed alpha, collect perturbed activations and compute δ_l
        transported_directions = {}  # {alpha: {li: {"obj": δ, "last": δ, "cos_obj": ..., "cos_last": ...}}}
        perturbed_outputs = {}  # {alpha: output_result}

        for alpha in embed_alphas:
            print(f"  [{time.strftime('%H:%M:%S')}] Computing δ_l for α={alpha}...")

            # Run with embedding perturbation
            embed_layer = model.get_input_embeddings()
            input_ids_p = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            input_ids_p = input_ids_p.to(device)

            inputs_embeds_base = embed_layer(input_ids_p).detach().clone()
            d_tensor = torch.tensor(d_embed, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_pert = inputs_embeds_base.clone()

            # Add perturbation at obj position in embedding
            inputs_embeds_pert[0, obj_pos, :] += (alpha * d_tensor).to(inputs_embeds_pert.dtype)

            # Collect perturbed activations
            pert_captured = {}
            def make_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].detach().float().cpu()
                    else:
                        hs = output.detach().float().cpu()
                    pert_captured[li] = {
                        "obj": hs[0, obj_pos, :].numpy().copy(),
                        "last": hs[0, last_pos, :].numpy().copy(),
                    }
                return hook_fn

            layers = get_layers(model)
            hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]

            with torch.no_grad():
                try:
                    outputs = model(inputs_embeds=inputs_embeds_pert,
                                   attention_mask=torch.ones(1, input_ids_p.shape[1], device=device))
                except Exception:
                    pass

            for h in hooks:
                h.remove()

            # Compute δ_l (natural transported direction)
            alpha_dirs = {}
            for li in range(n_layers):
                if li in clean_acts and li in pert_captured:
                    delta_obj = pert_captured[li]["obj"] - clean_acts[li]["obj"]
                    delta_last = pert_captured[li]["last"] - clean_acts[li]["last"]

                    # Cosine similarity with d_embed
                    cos_obj = float(np.dot(delta_obj, d_embed) /
                                   (max(np.linalg.norm(delta_obj), 1e-10) * max(np.linalg.norm(d_embed), 1e-10)))
                    cos_last = float(np.dot(delta_last, d_embed) /
                                    (max(np.linalg.norm(delta_last), 1e-10) * max(np.linalg.norm(d_embed), 1e-10)))

                    alpha_dirs[li] = {
                        "obj": delta_obj,
                        "last": delta_last,
                        "obj_norm": float(np.linalg.norm(delta_obj)),
                        "last_norm": float(np.linalg.norm(delta_last)),
                        "cos_obj": cos_obj,
                        "cos_last": cos_last,
                    }

            transported_directions[alpha] = alpha_dirs

            # Get perturbed output
            logits = outputs.logits[0, -1, :].float().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            perturbed_outputs[alpha] = analyze_category_output(probs, cand_ids,
                                                                KNOWLEDGE_TASKS["category"]["candidates"])

            # Print transport summary for key layers
            for frac in LAYER_FRACS:
                li = target_layers[frac]
                if li in alpha_dirs:
                    ad = alpha_dirs[li]
                    layer_key = f"L{li}" if li > 0 else "embed"
                    print(f"    {layer_key}: ||δ_obj||={ad['obj_norm']:.3f} "
                          f"||δ_last||={ad['last_norm']:.3f} "
                          f"cos_obj={ad['cos_obj']:.3f} cos_last={ad['cos_last']:.3f}")

        # ===== Part 2: Inject transported directions into clean run =====
        print(f"\n  [{time.strftime('%H:%M:%S')}] Part 2: Injecting transported directions...")

        inject_results = {}  # {alpha_source: {layer_frac: {a_frac: result}}}

        # Use alpha=2.0 and alpha=4.0 as source of transported directions
        source_alphas = [a for a in embed_alphas if a in [2.0, 4.0, 8.0]]

        for src_alpha in source_alphas:
            if src_alpha not in transported_directions:
                continue

            print(f"\n  Source α={src_alpha}:")
            src_dirs = transported_directions[src_alpha]
            inject_results[str(src_alpha)] = {}

            for frac in LAYER_FRACS:
                li = target_layers[frac]
                if li not in src_dirs:
                    continue

                layer_key = f"L{li}" if li > 0 else "embed"
                inject_results[str(src_alpha)][layer_key] = {}

                for pos_type in ["obj", "last"]:
                    delta_vec = src_dirs[li][pos_type]
                    delta_norm = np.linalg.norm(delta_vec)
                    if delta_norm < 1e-10:
                        continue

                    # Normalized direction
                    delta_unit = delta_vec / delta_norm

                    for a_frac in INJECT_ALPHA_FRAC:
                        # Inject: h_l += a_frac * ||h_l|| * delta_unit
                        clean_norm = clean_acts[li][f"{pos_type}_norm"]
                        perturbation = a_frac * clean_norm * delta_unit

                        perturb_dict = {li: {pos_type: perturbation}}

                        result = run_with_perturbation_at_layer(
                            model, tokenizer, device, n_layers, prompt,
                            tok_ids, perturb_dict, cand_ids,
                            KNOWLEDGE_TASKS["category"]["candidates"]
                        )

                        if result is not None:
                            delta_level = result["level"] - baseline_result["level"]
                            inject_results[str(src_alpha)][layer_key][f"{pos_type}_a{a_frac}"] = {
                                "delta": delta_level,
                                "level": result["level"],
                                "top_cand": max(result["cand_probs"], key=result["cand_probs"].get),
                                "full_entropy": result["full_entropy"],
                                "confidence": result["confidence"],
                                "actual_magnitude": float(abs(a_frac) * clean_norm),
                            }

                            if abs(a_frac) >= 0.5 and abs(delta_level) > 0.1:
                                print(f"    {layer_key}/{pos_type} a_frac={a_frac}: "
                                      f"Δ={delta_level:+.3f} top={max(result['cand_probs'], key=result['cand_probs'].get)} "
                                      f"H={result['full_entropy']:.1f} c={result['confidence']:.3f}")

        # ===== Part 3: Compare probe direction vs transported direction =====
        print(f"\n  [{time.strftime('%H:%M:%S')}] Part 3: Probe vs Transported comparison...")

        # Collect probe directions (same as Phase 429B)
        all_cat_acts = {}
        for cn, ci in OBJECT_CATEGORIES.items():
            cat_acts = {}
            for ow in ci["objects"][:7]:  # Use up to 7 objects
                is_s, tk = verify_single_token(tokenizer, ow)
                if not is_s:
                    continue
                p = KNOWLEDGE_TASKS["category"]["template"].format(obj=ow)
                acts, _, _, _, _ = collect_layer_activations(model, tokenizer, device, n_layers, p, tk)
                for li, a in acts.items():
                    if li not in cat_acts:
                        cat_acts[li] = {"obj": [], "last": []}
                    cat_acts[li]["obj"].append(a["obj"])
                    cat_acts[li]["last"].append(a["last"])
            all_cat_acts[cn] = cat_acts

        # Compute probe directions
        probe_directions = {}
        for cat_name, cat_info in OBJECT_CATEGORIES.items():
            opposing = cat_info["opposing"]
            if cat_name not in all_cat_acts or opposing not in all_cat_acts:
                continue
            cat_data = all_cat_acts[cat_name]
            opp_data = all_cat_acts[opposing]
            layer_dirs = {}
            for li in set(cat_data.keys()) & set(opp_data.keys()):
                dirs = {}
                for pos in ["obj", "last"]:
                    cv = cat_data[li].get(pos, [])
                    ov = opp_data[li].get(pos, [])
                    if not cv or not ov:
                        continue
                    cat_mean = np.mean(cv, axis=0)
                    opp_mean = np.mean(ov, axis=0)
                    d = cat_mean - opp_mean
                    norm = np.linalg.norm(d)
                    if norm > 0:
                        d = d / norm
                    dirs[pos] = d
                if dirs:
                    layer_dirs[li] = dirs
            if layer_dirs:
                probe_directions[(cat_name, opposing)] = layer_dirs

        # Compare transported vs probe at key layers using cosine similarity
        comparison = {}
        for src_alpha in source_alphas:
            if src_alpha not in transported_directions:
                continue
            src_dirs = transported_directions[src_alpha]
            comp = {}
            for frac in LAYER_FRACS:
                li = target_layers[frac]
                if li not in src_dirs:
                    continue
                layer_key = f"L{li}" if li > 0 else "embed"
                comp[layer_key] = {}

                for pos in ["obj", "last"]:
                    delta_vec = src_dirs[li][pos]
                    delta_norm = np.linalg.norm(delta_vec)
                    if delta_norm < 1e-10:
                        continue

                    # Probe direction at this layer
                    if dir_key in probe_directions and li in probe_directions[dir_key]:
                        probe_d = probe_directions[dir_key][li].get(pos)
                        if probe_d is not None:
                            cos = float(np.dot(delta_vec, probe_d) /
                                       (max(delta_norm, 1e-10) * max(np.linalg.norm(probe_d), 1e-10)))
                            comp[layer_key][pos] = {
                                "transported_norm": delta_norm,
                                "cosine_with_probe": cos,
                            }

            comparison[str(src_alpha)] = comp

        # Print comparison
        print(f"\n  Transported vs Probe cosine similarity:")
        for src_alpha, comp in comparison.items():
            for layer_key, pos_data in comp.items():
                for pos, data in pos_data.items():
                    print(f"    α={src_alpha} {layer_key}/{pos}: "
                          f"||δ||={data['transported_norm']:.3f} "
                          f"cos(probe)={data['cosine_with_probe']:.4f}")

        # ===== Part 4: Inject probe direction (for comparison) =====
        print(f"\n  [{time.strftime('%H:%M:%S')}] Part 4: Injecting probe directions (comparison)...")

        probe_inject_results = {}

        for frac in LAYER_FRACS:
            li = target_layers[frac]
            if dir_key not in probe_directions or li not in probe_directions[dir_key]:
                continue
            layer_key = f"L{li}" if li > 0 else "embed"
            probe_inject_results[layer_key] = {}

            for pos in ["obj", "last"]:
                if pos not in probe_directions[dir_key][li]:
                    continue
                probe_d = probe_directions[dir_key][li][pos]

                for a_frac in INJECT_ALPHA_FRAC:
                    clean_norm = clean_acts[li][f"{pos}_norm"]
                    perturbation = a_frac * clean_norm * probe_d

                    perturb_dict = {li: {pos: perturbation}}

                    result = run_with_perturbation_at_layer(
                        model, tokenizer, device, n_layers, prompt,
                        tok_ids, perturb_dict, cand_ids,
                        KNOWLEDGE_TASKS["category"]["candidates"]
                    )

                    if result is not None:
                        delta_level = result["level"] - baseline_result["level"]
                        probe_inject_results[layer_key][f"{pos}_a{a_frac}"] = {
                            "delta": delta_level,
                            "level": result["level"],
                            "top_cand": max(result["cand_probs"], key=result["cand_probs"].get),
                            "full_entropy": result["full_entropy"],
                            "confidence": result["confidence"],
                        }

        # ===== Part 5: Causal tracing =====
        print(f"\n  [{time.strftime('%H:%M:%S')}] Part 5: Causal tracing...")

        # Use "corrupt-then-restore" method
        # Corrupt: replace object word with opposing category word
        # Then restore clean activation at each layer/position

        # Find a corrupting word from opposing category
        opp_cat = opposing
        opp_words = OBJECT_CATEGORIES[opp_cat]["objects"]
        corrupt_word = None
        for ow in opp_words:
            is_s, _ = verify_single_token(tokenizer, ow)
            if is_s:
                corrupt_word = ow
                break

        if corrupt_word:
            corrupt_prompt = KNOWLEDGE_TASKS["category"]["template"].format(obj=corrupt_word)
            corrupt_acts, _, _, _, _ = collect_layer_activations(
                model, tokenizer, device, n_layers, corrupt_prompt,
                tokenizer.encode(corrupt_word, add_special_tokens=False)
            )

            # Run corrupt baseline
            corrupt_input_ids = tokenizer.encode(corrupt_prompt, add_special_tokens=True, return_tensors="pt")
            corrupt_input_ids = corrupt_input_ids.to(device)
            _, corrupt_probs = get_logits_and_probs(model, corrupt_input_ids,
                                                     torch.ones_like(corrupt_input_ids), device)
            corrupt_result = analyze_category_output(corrupt_probs, cand_ids,
                                                      KNOWLEDGE_TASKS["category"]["candidates"])

            print(f"  Corrupt baseline ({corrupt_word}): level={corrupt_result['level']:.3f}")

            # Restore clean activation at each layer
            causal_trace = {}

            for frac in LAYER_FRACS:
                li = target_layers[frac]
                if li not in clean_acts or li not in corrupt_acts:
                    continue
                layer_key = f"L{li}" if li > 0 else "embed"

                for pos in ["obj", "last"]:
                    # Compute the clean - corrupt difference at this position
                    clean_vec = clean_acts[li][pos]
                    corrupt_vec = corrupt_acts[li][pos]
                    diff = clean_vec - corrupt_vec

                    # Inject this difference into the corrupt run
                    perturb_dict = {li: {pos: diff}}

                    result = run_with_perturbation_at_layer(
                        model, tokenizer, device, n_layers, corrupt_prompt,
                        tokenizer.encode(corrupt_word, add_special_tokens=False),
                        perturb_dict, cand_ids,
                        KNOWLEDGE_TASKS["category"]["candidates"]
                    )

                    if result is not None:
                        # Recovery: how much does restoring this layer recover the clean output?
                        clean_level = baseline_result["level"]
                        corrupt_level = corrupt_result["level"]
                        restored_level = result["level"]

                        if abs(clean_level - corrupt_level) > 1e-6:
                            recovery = (restored_level - corrupt_level) / (clean_level - corrupt_level)
                        else:
                            recovery = 0.0

                        causal_trace[f"{layer_key}/{pos}"] = {
                            "recovery": float(recovery),
                            "restored_level": result["level"],
                            "clean_level": clean_level,
                            "corrupt_level": corrupt_level,
                        }

                        if abs(recovery) > 0.05:
                            print(f"    {layer_key}/{pos}: recovery={recovery:.3f} "
                                  f"(clean={clean_level:.2f}, corrupt={corrupt_level:.2f}, "
                                  f"restored={result['level']:.2f})")
        else:
            causal_trace = {"error": "No valid corrupt word found"}

        # ===== Save results =====
        obj_result["transported_directions_norms"] = {}
        for alpha, dirs in transported_directions.items():
            obj_result["transported_directions_norms"][str(alpha)] = {}
            for li, d in dirs.items():
                layer_key = f"L{li}"
                obj_result["transported_directions_norms"][str(alpha)][layer_key] = {
                    "obj_norm": d["obj_norm"],
                    "last_norm": d["last_norm"],
                    "cos_obj": d["cos_obj"],
                    "cos_last": d["cos_last"],
                }

        obj_result["perturbed_outputs"] = {}
        for alpha, r in perturbed_outputs.items():
            obj_result["perturbed_outputs"][str(alpha)] = r

        obj_result["inject_results"] = inject_results
        obj_result["probe_inject_results"] = probe_inject_results
        obj_result["comparison"] = comparison
        obj_result["causal_trace"] = causal_trace

        # Store residual norms
        obj_result["residual_norms"] = {}
        for li, a in clean_acts.items():
            obj_result["residual_norms"][f"L{li}"] = {
                "obj_norm": a["obj_norm"],
                "last_norm": a["last_norm"],
            }

        results["per_object"][obj_word] = obj_result

        # Clean up
        del clean_acts, transported_directions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save to file
    out_dir = ROOT / "results" / "phase430_natural_transport"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_phase430_r{round_num}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    results = convert(results)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to {out_file}")

    # ===== Print summary =====
    print(f"\n{'='*70}")
    print(f"PHASE 430 SUMMARY - {model_name} R{round_num}")
    print(f"{'='*70}")

    for obj_word, obj_data in results["per_object"].items():
        cat = obj_data["category"]
        opp = obj_data["opposing"]
        base = obj_data["baseline"]

        print(f"\n  {obj_word} ({cat} vs {opp}):")
        print(f"    Baseline: level={base['level']:.3f}, "
              f"top={max(base['cand_probs'], key=base['cand_probs'].get)}, "
              f"H={base['full_entropy']:.1f}")

        # Embedding perturbation effect
        print(f"    Embedding perturbation effect:")
        for alpha_str, r in sorted(obj_data.get("perturbed_outputs", {}).items(),
                                    key=lambda x: float(x[0])):
            print(f"      α={alpha_str}: Δ={r['level']-base['level']:+.3f} "
                  f"top={max(r['cand_probs'], key=r['cand_probs'].get)} "
                  f"H={r['full_entropy']:.1f}")

        # Transported direction norms (how does perturbation propagate)
        print(f"    Transported direction norms (cos with d_embed):")
        for alpha_str in ["2.0", "4.0"]:
            if alpha_str in obj_data.get("transported_directions_norms", {}):
                for layer_key in ["L0", f"L{target_layers[0.2]}", f"L{target_layers[0.4]}",
                                  f"L{target_layers[0.6]}", f"L{target_layers[0.8]}"]:
                    if layer_key in obj_data["transported_directions_norms"][alpha_str]:
                        d = obj_data["transported_directions_norms"][alpha_str][layer_key]
                        print(f"      α={alpha_str} {layer_key}: "
                              f"||δ_obj||={d['obj_norm']:.3f} ||δ_last||={d['last_norm']:.3f} "
                              f"cos_obj={d['cos_obj']:.3f} cos_last={d['cos_last']:.3f}")

        # Injected transported direction effect
        print(f"    Injected transported direction (best effects):")
        for src_alpha_str, layer_data in obj_data.get("inject_results", {}).items():
            for layer_key, pos_data in layer_data.items():
                for key, r in pos_data.items():
                    if abs(r["delta"]) > 0.3:
                        print(f"      src_α={src_alpha_str} {layer_key}/{key}: "
                              f"Δ={r['delta']:+.3f} H={r['full_entropy']:.1f}")

        # Probe direction injection
        print(f"    Probe direction injection (best effects):")
        for layer_key, pos_data in obj_data.get("probe_inject_results", {}).items():
            for key, r in pos_data.items():
                if abs(r["delta"]) > 0.3:
                    print(f"      {layer_key}/{key}: Δ={r['delta']:+.3f} "
                          f"H={r['full_entropy']:.1f}")

        # Causal trace
        print(f"    Causal trace (recovery > 0.1):")
        for key, r in obj_data.get("causal_trace", {}).items():
            if isinstance(r, dict) and abs(r.get("recovery", 0)) > 0.1:
                print(f"      {key}: recovery={r['recovery']:.3f}")

    # Release model
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[{time.strftime('%H:%M:%S')}] Done! Model released.")


if __name__ == "__main__":
    main()
