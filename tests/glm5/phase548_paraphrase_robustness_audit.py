#!/usr/bin/env python3
"""
Phase 548: Paraphrase Candidate Robustness and Human-Readable Sample Audit

Phase 547 discovered label gate / paraphrase gate separation.
This phase verifies GLM4 vehicle_tool residual_perp's clean paraphrase candidate
with random controls, object-echo control, and human-readable sample tables.

Design: 
  - Use single-step logits analysis (like Phase 542) + short greedy generation
  - 7 conditions: baseline, residual_parallel, residual_full, residual_perp, readout, random_same_norm, random_perp
  - 3 forbidden scaffolds + 4 category pairs
  - Classify generated output quality: clean_synonym, label_violation, object_echo, wrong_synonym, generic, other
  - S_clean_para_net = P(clean_synonym) - P(wrong_synonym) - P(label_violation) - P(object_echo)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import (  # noqa: E402
    PEAK_LAYERS,
    decompose,
    hidden_at_layer,
    load_model_bf16_flash,
    mean_dir,
    score_logits,
    token_ids,
)
from phase532_multi_seed_controls import normalize, random_orthogonal, cos  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt, encode_batch  # noqa: E402
from phase539_interface_cluster_mechanism import (  # noqa: E402
    CORE_SOURCES,
    build_candidates,
    build_components,
    layer_windows,
    pair_targets,
    pair_competitors,
    logits_with_patch,
    patch_site,
    interventions_for as interventions_539,
)


OUT_ROOT = Path("results/glm5_phase548_paraphrase_robustness_audit")

# ===== Enlarged object sets =====
EXTENDED_VEHICLES = ["car", "truck", "bus", "train", "bicycle", "motorcycle", "boat", "airplane", "scooter", "van", "taxi", "tram"]
EXTENDED_TOOLS = ["hammer", "saw", "drill", "wrench", "screwdriver", "pliers", "knife", "chisel", "axe", "shovel", "scissors", "clamp"]
EXTENDED_FRUITS = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "plum", "cherry", "lemon", "kiwi", "melon"]
EXTENDED_VEGETABLES = ["carrot", "potato", "onion", "lettuce", "cabbage", "broccoli", "spinach", "pepper", "tomato", "cucumber", "celery", "radish"]
EXTENDED_ANIMALS = ["dog", "cat", "horse", "cow", "sheep", "goat", "lion", "tiger", "bear", "wolf", "fox", "deer"]

PAIR_SPECS = {
    "vehicle_tool": ("vehicle", "tool"),
    "fruit_vegetable": ("fruit", "vegetable"),
    "animal_tool": ("animal", "tool"),
    "fruit_tool": ("fruit", "tool"),
}

EXTENDED_BANK = {
    "vehicle": EXTENDED_VEHICLES,
    "tool": EXTENDED_TOOLS,
    "fruit": EXTENDED_FRUITS,
    "vegetable": EXTENDED_VEGETABLES,
    "animal": EXTENDED_ANIMALS,
}

# ===== Forbidden scaffolds =====
FORBIDDEN_SCAFFOLDS = {
    "forbidden_definition": "Do not use the words vehicle, tool, fruit, vegetable, animal, clothing, or furniture. Describe {x} in other words:",
    "forbidden_sentence_completion": "Without using vehicle, tool, fruit, vegetable, animal, clothing, or furniture, complete: A {x} is a type of",
    "forbidden_natural_qa": "You must not say vehicle, tool, fruit, vegetable, animal, clothing, or furniture. What kind of thing is {x}?",
}

FORBIDDEN_LABELS = ["vehicle", "tool", "fruit", "vegetable", "animal", "clothing", "furniture",
    "Vehicle", "Tool", "Fruit", "Vegetable", "Animal", "Clothing", "Furniture",
    "vehicles", "tools", "fruits", "vegetables", "animals", "clothes", "furnitures",
    " Vehicles", " Tools", " Fruits", " Vegetables", " Animals", " Clothes",
]

CLEAN_SYNONYMS = {
    "vehicle": ["transport", "transportation", "automobile", "motor", "conveyance", "carriage",
        "wheeled", "driving", "riding", "moving", "travel", "journey", "road", "machine", "engine", "motorized"],
    "tool": ["instrument", "device", "equipment", "implement", "apparatus", "utensil", "gadget",
        "mechanism", "contrivance", "hand-held", "manual", "working", "cutting", "fixing", "building", "repairing"],
    "fruit": ["produce", "crop", "harvest", "edible", "sweet", "juicy", "fresh", "tree-grown",
        "orchard", "berry", "tropical", "snack", "food", "dessert", "plant"],
    "vegetable": ["produce", "crop", "harvest", "edible", "greens", "leafy", "plant", "garden-grown",
        "root", "stalk", "cooking", "meal", "dish", "ingredient"],
    "animal": ["creature", "beast", "living", "organism", "species", "pet", "wildlife", "fauna",
        "mammal", "bird", "fish", "breathing", "moving"],
}

GENERIC_WORDS = ["object", "thing", "item", "something", "entity", "concept",
    "Object", "Thing", "Item", "Something", "kind", "type", "sort", "category", "class", "group"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def forbidden_prompt(scaffold: str, obj: str) -> str:
    return FORBIDDEN_SCAFFOLDS[scaffold].format(x=obj, cap=obj.capitalize())


def classify_output(text: str, obj: str, pos_category: str, neg_category: str) -> dict[str, Any]:
    """Classify generated suffix into quality categories."""
    text_lower = text.lower().strip()
    obj_lower = obj.lower()

    # Check label violation
    label_violation = False
    violation_word = ""
    for label in FORBIDDEN_LABELS:
        if label.lower() in text_lower:
            label_violation = True
            violation_word = label
            break

    # Check object echo
    object_echo = False
    echo_word = ""
    # Check if object name appears (but not as the prompt object)
    if obj_lower in text_lower:
        object_echo = True
        echo_word = obj

    # Check clean synonym (positive category, excluding object itself)
    clean_synonym = False
    synonym_word = ""
    for syn in CLEAN_SYNONYMS.get(pos_category, []):
        if syn.lower() in text_lower and syn.lower() != obj_lower:
            clean_synonym = True
            synonym_word = syn
            break

    # Check wrong synonym (negative category)
    wrong_synonym = False
    wrong_word = ""
    for syn in CLEAN_SYNONYMS.get(neg_category, []):
        if syn.lower() in text_lower:
            wrong_synonym = True
            wrong_word = syn
            break

    # Check generic
    generic = False
    generic_word = ""
    for gw in GENERIC_WORDS:
        if gw.lower() in text_lower:
            generic = True
            generic_word = gw
            break

    # Degenerate
    degenerate = len(text_lower.strip()) < 3 or text_lower.strip() in ("the", "a", "an", "is", "it")

    # Priority: violation > wrong > clean > echo > generic > degenerate > other
    if label_violation:
        quality = "label_violation"
    elif wrong_synonym:
        quality = "wrong_synonym"
    elif clean_synonym:
        quality = "clean_synonym"
    elif object_echo:
        quality = "object_echo"
    elif generic:
        quality = "generic"
    elif degenerate:
        quality = "degenerate"
    else:
        quality = "other"

    return {
        "quality": quality,
        "label_violation": label_violation,
        "violation_word": violation_word,
        "object_echo": object_echo,
        "echo_word": echo_word,
        "clean_synonym": clean_synonym,
        "synonym_word": synonym_word,
        "wrong_synonym": wrong_synonym,
        "wrong_word": wrong_word,
        "generated": text,
    }


def generate_greedy(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    max_new_tokens: int,
    max_length: int,
) -> str:
    """Greedy generate with optional intervention on first step only.
    
    Uses model.generate() for baseline (no intervention) and manual 
    single-step + hook for intervention conditions.
    """
    input_device = next(model.parameters()).device
    
    if interventions is None:
        # Baseline: use model.generate directly
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        with torch.inference_mode():
            gen_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.2, use_cache=False,
            )
        
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        suffix = full_text[len(prompt_decoded):].strip() if full_text.startswith(prompt_decoded) else full_text.strip()
        
        del gen_ids, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return suffix
    
    # With intervention: manual greedy generation loop
    prepared = {}
    for layer_id, (direction, alpha) in interventions.items():
        prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    prompt_len = int(attention_mask.sum(dim=1).item())
    prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    
    for step in range(max_new_tokens):
        # Register hooks only on first step
        handles = []
        if step == 0:
            pos = current_mask.sum(dim=1) - 1
            for layer_id, d_tensor in prepared.items():
                layer = layers[layer_id]
                layer_device = next(layer.parameters()).device
                d_local = d_tensor.to(layer_device)
                pos_local = pos.to(layer_device)

                def make_hook(d_vec, pos_vec):
                    def hook_fn(_module, _inp, output):
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            actual_device = hs.device
                            if str(actual_device) == "meta":
                                return output
                            d_on_dev = d_vec.to(actual_device).to(hs.dtype)
                            p_on_dev = pos_vec.to(actual_device)
                            hs[torch.arange(hs.shape[0], device=actual_device), p_on_dev] += d_on_dev
                            return (hs,) + output[1:]
                        hs = output.clone()
                        actual_device = hs.device
                        if str(actual_device) == "meta":
                            return output
                        d_on_dev = d_vec.to(actual_device).to(hs.dtype)
                        p_on_dev = pos_vec.to(actual_device)
                        hs[torch.arange(hs.shape[0], device=actual_device), p_on_dev] += d_on_dev
                        return hs
                    return hook_fn

                handles.append(layer.register_forward_hook(make_hook(d_local, pos_local)))

        with torch.inference_mode():
            out = model(current_ids, attention_mask=current_mask, use_cache=False, return_dict=True)

        for handle in handles:
            handle.remove()

        # Get logits at last position
        last_pos = current_ids.shape[1] - 1
        logits = out.logits[0, last_pos].float().cpu().numpy().astype(np.float32)
        new_tok = int(np.argmax(logits))
        
        # Append token
        new_tok_tensor = torch.tensor([[new_tok]], device=input_device)
        current_ids = torch.cat([current_ids, new_tok_tensor], dim=1)
        current_mask = torch.cat([current_mask, torch.ones_like(new_tok_tensor)], dim=1)

        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if current_ids.shape[1] >= max_length:
            break

    full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    suffix = full_text[len(prompt_decoded):].strip() if full_text.startswith(prompt_decoded) else full_text.strip()
    
    del current_ids, current_mask, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return suffix


def compute_quality_stats(classifications: list[dict], n: int) -> dict[str, float]:
    if n == 0:
        return {}
    counts = {}
    for c in classifications:
        q = c["quality"]
        counts[q] = counts.get(q, 0) + 1
    rates = {k: v / n for k, v in counts.items()}

    p_clean = rates.get("clean_synonym", 0.0)
    p_wrong = rates.get("wrong_synonym", 0.0)
    p_label_viol = rates.get("label_violation", 0.0)
    p_echo = rates.get("object_echo", 0.0)
    s_net = p_clean - p_wrong - p_label_viol - p_echo
    label_dep = (p_clean + p_label_viol) / (p_clean + rates.get("generic", 0.0) + p_label_viol + 1e-8)

    return {
        "n": n,
        "p_clean_synonym": p_clean,
        "p_wrong_synonym": p_wrong,
        "p_label_violation": p_label_viol,
        "p_object_echo": p_echo,
        "p_generic": rates.get("generic", 0.0),
        "p_other": rates.get("other", 0.0),
        "s_clean_para_net": s_net,
        "label_dependency": label_dep,
    }


def add_extended_components(
    comps: dict[str, dict[str, np.ndarray]],
    W_U: np.ndarray,
    tokenizer: Any,
    seeds: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Add readout, random_same_norm, random_perp conditions."""
    for pair in CORE_SOURCES:
        if pair not in comps:
            continue
        pair_comps = comps[pair]
        perp_vec = pair_comps["residual_perp"]
        full_vec = pair_comps["residual_full"]
        readout_vec = pair_comps["_readout"]
        perp_norm = float(np.linalg.norm(perp_vec))
        full_norm = float(np.linalg.norm(full_vec))

        # Readout direction (same norm as full)
        pair_comps["readout"] = (normalize(readout_vec) * full_norm).astype(np.float32)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            v_raw = rng.standard_normal(full_vec.shape[0]).astype(np.float32)
            pair_comps[f"random_same_norm_{seed}"] = (normalize(v_raw) * full_norm).astype(np.float32)
            pair_comps[f"random_perp_{seed}"] = random_orthogonal(
                perp_vec.shape[0], [normalize(readout_vec)], perp_norm, seed=seed + 1000
            ).astype(np.float32)

    return comps


def build_all_conditions(seeds: list[int]) -> list[str]:
    base = ["baseline", "residual_parallel", "residual_full", "residual_perp", "readout"]
    random = []
    for seed in seeds:
        random.extend([f"random_same_norm_{seed}", f"random_perp_{seed}"])
    return base + random


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = float(args.alpha)
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        W_U = get_W_U(model, args.model).astype(np.float32)
        all_conditions = build_all_conditions(seeds)

        log(f"{args.model}: paraphrase robustness, windows={windows}, alpha={alpha}, seeds={seeds}")

        # Build components
        candidates = build_candidates(args.train_n)
        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            comps = build_components(dirs, W_U, tokenizer, seeds)
            comps = add_extended_components(comps, W_U, tokenizer, seeds)
            components_by_layer[str(layer_id)] = comps

        # Use center window
        center_window = windows.get("center", windows.get("extended", [PEAK_LAYERS[args.model]]))
        log(f"Using window: {center_window}")

        # Run experiments per pair+scaffold
        pairs_to_test = ["vehicle_tool"] if not args.full_pairs else list(PAIR_SPECS.keys())
        experiment_results = {}

        for pair in pairs_to_test:
            pos_cat, neg_cat = PAIR_SPECS[pair]
            source_pair = pair if pair in CORE_SOURCES else "vehicle_tool"

            all_objs = EXTENDED_BANK.get(pos_cat, CATEGORY_BANK.get(pos_cat, []))
            test_objs = all_objs[args.train_n:args.train_n + args.test_n]
            if len(test_objs) < args.test_n:
                test_objs = all_objs[-args.test_n:]

            scaffolds = list(FORBIDDEN_SCAFFOLDS.keys()) if not args.smoke else ["forbidden_definition"]

            for scaffold in scaffolds:
                exp_key = f"{pair}_{scaffold}"
                log(f"Experiment: {exp_key} ({len(test_objs)} objects)")

                # Build prompts
                prompts = [forbidden_prompt(scaffold, obj) for obj in test_objs]

                # Run all conditions
                condition_results = {}
                for condition in all_conditions:
                    # Build interventions
                    if condition == "baseline":
                        interventions = None
                    else:
                        vec_key = condition
                        interventions = {}
                        for lid in center_window:
                            vec = components_by_layer[str(lid)][source_pair].get(vec_key)
                            if vec is not None:
                                interventions[lid] = (vec, alpha)

                    # Run greedy generation for each prompt
                    classifications = []
                    for i, prompt_text in enumerate(prompts):
                        suffix = generate_greedy(
                            model, tokenizer, device, layers, prompt_text,
                            interventions, args.max_new_tokens, args.max_length,
                        )
                        cls = classify_output(suffix, test_objs[i], pos_cat, neg_cat)
                        classifications.append(cls)
                        if i % 3 == 0:
                            log(f"    ...{i}/{len(test_objs)} objects done: {suffix[:40]}")

                    stats = compute_quality_stats(classifications, len(classifications))
                    stats["samples"] = [
                        {
                            "object": test_objs[i],
                            "suffix": classifications[i]["generated"][:80],
                            "quality": classifications[i]["quality"],
                            "synonym": classifications[i].get("synonym_word", ""),
                            "violation": classifications[i].get("violation_word", ""),
                            "echo": classifications[i].get("echo_word", ""),
                        }
                        for i in range(min(20, len(classifications)))
                    ]
                    condition_results[condition] = stats

                    p_clean = stats.get("p_clean_synonym", 0.0)
                    p_viol = stats.get("p_label_violation", 0.0)
                    p_echo = stats.get("p_object_echo", 0.0)
                    s_net = stats.get("s_clean_para_net", 0.0)
                    log(f"  {condition}: clean={p_clean:.3f} viol={p_viol:.3f} echo={p_echo:.3f} net={s_net:.3f}")

                # Compute deltas
                base = condition_results.get("baseline", {})
                deltas = {}
                for condition in all_conditions:
                    if condition == "baseline":
                        deltas[condition] = {"delta_clean": 0.0, "delta_viol": 0.0, "delta_echo": 0.0, "delta_net": 0.0}
                        continue
                    cond = condition_results[condition]
                    deltas[condition] = {
                        "delta_clean": cond.get("p_clean_synonym", 0.0) - base.get("p_clean_synonym", 0.0),
                        "delta_viol": cond.get("p_label_violation", 0.0) - base.get("p_label_violation", 0.0),
                        "delta_echo": cond.get("p_object_echo", 0.0) - base.get("p_object_echo", 0.0),
                        "delta_net": cond.get("s_clean_para_net", 0.0) - base.get("s_clean_para_net", 0.0),
                    }

                # Robustness check
                perp_net_delta = deltas.get("residual_perp", {}).get("delta_net", 0.0)
                perp_viol_delta = deltas.get("residual_perp", {}).get("delta_viol", 0.0)
                rand_net_max = max(deltas.get(f"random_same_norm_{s}", {}).get("delta_net", -999.0) for s in seeds)
                rand_perp_net_max = max(deltas.get(f"random_perp_{s}", {}).get("delta_net", -999.0) for s in seeds)
                full_net_delta = deltas.get("residual_full", {}).get("delta_net", 0.0)

                beats_random_perp = perp_net_delta > rand_perp_net_max
                no_label_viol = abs(perp_viol_delta) < 0.05
                passes_robustness = beats_random_perp and no_label_viol and perp_net_delta > 0.05

                log(f"  SUMMARY {exp_key}: perp_net={perp_net_delta:+.3f} rand_perp={rand_perp_net_max:+.3f} "
                    f"beats={beats_random_perp} viol={no_label_viol} passes={passes_robustness}")

                experiment_results[exp_key] = {
                    "pair": pair,
                    "scaffold": scaffold,
                    "source_pair_used": source_pair,
                    "objects": test_objs,
                    "alpha": alpha,
                    "conditions": condition_results,
                    "deltas": deltas,
                    "robustness": {
                        "perp_net_delta": perp_net_delta,
                        "perp_clean_delta": deltas.get("residual_perp", {}).get("delta_clean", 0.0),
                        "perp_viol_delta": perp_viol_delta,
                        "full_net_delta": full_net_delta,
                        "rand_net_max": rand_net_max,
                        "rand_perp_net_max": rand_perp_net_max,
                        "beats_random_perp": beats_random_perp,
                        "beats_random_same_norm": perp_net_delta > rand_net_max,
                        "no_label_violation": no_label_viol,
                        "passes_robustness": passes_robustness,
                    },
                }

        # Final summary
        summary = {}
        for exp_key, exp_data in experiment_results.items():
            summary[exp_key] = exp_data["robustness"]

        return {
            "phase": 548,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair_specs": PAIR_SPECS,
            "forbidden_scaffolds": list(FORBIDDEN_SCAFFOLDS.keys()),
            "conditions": all_conditions,
            "windows": windows,
            "center_window": center_window,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "alpha": alpha,
            "random_seeds": seeds,
            "max_new_tokens": args.max_new_tokens,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "experiments": experiment_results,
            "summary": summary,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=6)
    parser.add_argument("--alpha", default="8")
    parser.add_argument("--random-seeds", default="11,23,37")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-pairs", action="store_true")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase548_{args.model}_paraphrase_robustness{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")

    log("=" * 60)
    log("KEY RESULTS:")
    for exp_key, s in result["summary"].items():
        log(f"  {exp_key}: perp_net={s['perp_net_delta']:+.3f} rand_perp={s['rand_perp_net_max']:+.3f} "
            f"beats={s['beats_random_perp']} viol={s['no_label_violation']} passes={s['passes_robustness']}")
    log("=" * 60)

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
