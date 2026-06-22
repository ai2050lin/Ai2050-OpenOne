#!/usr/bin/env python3
"""
Phase 575: Closed Semantic Micro-World v1 — Object-Category Causal Closure
闭合语义微世界v1：对象—类别因果闭包

Step 0: Tokenizer audit — select low-similarity artificial symbols
Step 1: Truth table construction with counterfactual variants
Step 2: Prompt protocol with knowledge/distractor/query blocks
Step 3: Layerwise state decoding via linear probes
Step 4: Causal swap — inject category state from sample A into sample B
Step 5: Orthogonality test — does category swap preserve object state?
Step 6: Retrieval vs state test — activation transplant to new context

Run:
  python tests/glm5/phase575_closed_micro_world.py qwen3 --smoke
  python tests/glm5/phase575_closed_micro_world.py qwen3
  python tests/glm5/phase575_closed_micro_world.py glm4
  python tests/glm5/phase575_closed_micro_world.py deepseek7b
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402


OUT_ROOT = Path("results/glm5_phase575_closed_micro_world")

# Candidate artificial symbols
CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]
CANDIDATE_VALUES = ["v05", "v91", "v44", "v68"]
CANDIDATE_RELATIONS = ["r31", "r64", "r27", "r88"]

# Distractor symbols
DISTRACTOR_SYMS = ["x72", "m19", "z03", "q08", "p44", "w52", "k37", "n85"]

# Probe layers (relative to n_layers)
PROBE_LAYER_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Step 0: Tokenizer audit
# ============================================================================

def tokenizer_audit(
    tokenizer: Any,
    model: Any,
    objects: list[str],
    categories: list[str],
    values: list[str],
    relations: list[str],
) -> dict[str, Any]:
    """Audit tokenization of artificial symbols."""
    all_syms = objects + categories + values + relations + DISTRACTOR_SYMS
    audit: dict[str, Any] = {"symbols": {}}

    # Get embedding matrix
    embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    W_U = get_W_U(model, "").astype(np.float32) if hasattr(model, 'lm_head') else None
    # Re-get W_U properly
    try:
        W_U = model.lm_head.weight.detach().float().cpu().numpy()
    except Exception:
        W_U = None

    sym_info = {}
    for sym in all_syms:
        ids_with_space = tokenizer.encode(" " + sym, add_special_tokens=False)
        ids_no_space = tokenizer.encode(sym, add_special_tokens=False)

        # Embedding similarity with other symbols
        emb_similarities = {}
        if ids_with_space and ids_with_space[0] < embed.shape[0]:
            emb = embed[ids_with_space[0]]
            for other in all_syms:
                if other == sym:
                    continue
                other_ids = tokenizer.encode(" " + other, add_special_tokens=False)
                if other_ids and other_ids[0] < embed.shape[0]:
                    other_emb = embed[other_ids[0]]
                    cos_sim = float(np.dot(emb, other_emb) / (
                        np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-8))
                    emb_similarities[other] = cos_sim

        sym_info[sym] = {
            "token_ids_space": ids_with_space,
            "token_ids_no_space": ids_no_space,
            "n_tokens": len(ids_with_space),
            "max_emb_sim": max(emb_similarities.values()) if emb_similarities else 0.0,
            "most_similar": max(emb_similarities, key=emb_similarities.get) if emb_similarities else None,
        }

    audit["symbols"] = sym_info

    # Select best symbols (low similarity, single token preferred)
    selected_objects = [s for s in objects if sym_info[s]["n_tokens"] <= 2][:4]
    selected_categories = [s for s in categories if sym_info[s]["n_tokens"] <= 2][:2]
    selected_values = [s for s in values if sym_info[s]["n_tokens"] <= 2][:2]
    selected_relations = [s for s in relations if sym_info[s]["n_tokens"] <= 2][:2]

    audit["selected"] = {
        "objects": selected_objects,
        "categories": selected_categories,
        "values": selected_values,
        "relations": selected_relations,
    }

    log(f"  Tokenizer audit: objects={selected_objects}, categories={selected_categories}")
    for s in selected_objects + selected_categories:
        info = sym_info[s]
        log(f"    {s}: tokens={info['n_tokens']} max_sim={info['max_emb_sim']:.3f}")

    return audit


# ============================================================================
# Step 1+2: Truth table and prompt construction
# ============================================================================

def build_truth_tables(
    objects: list[str],
    categories: list[str],
    n_tables: int = 3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build multiple random truth tables for object->category mapping."""
    rng = random.Random(seed)
    tables = []
    for t in range(n_tables):
        mapping = {}
        for obj in objects:
            mapping[obj] = rng.choice(categories)
        tables.append(mapping)
    return tables


def build_prompt(
    truth_table: dict[str, str],
    query_object: str,
    distractors: list[str] | None = None,
    n_distractors: int = 3,
    seed: int = 42,
) -> str:
    """Build a prompt with knowledge block, distractors, and query."""
    rng = random.Random(seed)

    # Knowledge block
    rules = [f"{obj} belongs to {cat}." for obj, cat in truth_table.items()]
    rng.shuffle(rules)

    # Distractor block
    if distractors is None:
        distractors = DISTRACTOR_SYMS[:n_distractors]
    distractor_lines = []
    for d in distractors[:n_distractors]:
        r = rng.choice(CANDIDATE_RELATIONS[:2])
        v = rng.choice(CANDIDATE_VALUES[:2])
        o = rng.choice(CANDIDATE_OBJECTS[:4])
        distractor_lines.append(f"{r} {o} {v}.")

    prompt = "Rules:\n"
    prompt += "\n".join(rules)
    prompt += "\n\nDistractors:\n"
    prompt += "\n".join(distractor_lines)
    prompt += f"\n\nQuery:\n{query_object} belongs to ?"
    return prompt


# ============================================================================
# Step 3: Layerwise state decoding via linear probes
# ============================================================================

def collect_hidden_states(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    layers: list[Any],
    max_length: int = 256,
) -> dict[int, np.ndarray]:
    """Collect hidden states at each layer for all prompts.
    Returns {layer_idx: [n_prompts, d_model]}
    """
    n_layers = len(layers)
    probe_layers = [int(f * n_layers) for f in PROBE_LAYER_FRACTIONS]
    probe_layers = sorted(set([min(l, n_layers - 1) for l in probe_layers]))

    captured: dict[int, np.ndarray] = {}

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    hooks = []
    for lid in probe_layers:
        def make_hook(layer_id: int):
            def hook(_module, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[layer_id] = hs[:, answer_pos, :].detach().float().cpu().numpy()
            return hook
        hooks.append(layers[lid].register_forward_hook(make_hook(lid)))

    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

    for h in hooks:
        h.remove()

    return captured


def train_probes(
    hidden_states: dict[int, np.ndarray],
    labels: np.ndarray,
    label_name: str,
) -> dict[int, dict[str, float]]:
    """Train linear probes at each layer."""
    results = {}
    n = len(labels)
    if n < 4:
        return results

    # Check we have at least 2 classes with >=2 samples each
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_classes = unique_labels[counts >= 2]
    if len(valid_classes) < 2:
        return results

    # Filter to valid classes only
    mask = np.isin(labels, valid_classes)
    h_filtered = {lid: h[mask] for lid, h in hidden_states.items()}
    labels_filtered = labels[mask]

    n_filtered = len(labels_filtered)
    cv_folds = min(5, n_filtered // 2, min(counts[counts >= 2]))

    for lid, h in h_filtered.items():
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            scores = cross_val_score(clf, h, labels_filtered, cv=max(2, cv_folds), scoring='accuracy')
            results[lid] = {
                "accuracy_mean": float(np.nanmean(scores)),
                "accuracy_std": float(np.nanstd(scores)),
            }
        except Exception as e:
            results[lid] = {"error": str(e), "accuracy_mean": 0.0, "accuracy_std": 0.0}

    return results


# ============================================================================
# Step 4+5: Causal swap and orthogonality test
# ============================================================================

def collect_single_hidden_states(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    layers: list[Any],
    probe_layers: list[int],
    max_length: int = 256,
) -> dict[int, np.ndarray]:
    """Collect hidden states for a single prompt."""
    captured: dict[int, np.ndarray] = {}

    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1

    hooks = []
    for lid in probe_layers:
        def make_hook(layer_id: int):
            def hook(_module, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[layer_id] = hs[0, answer_pos, :].detach().float().cpu().numpy()
            return hook
        hooks.append(layers[lid].register_forward_hook(make_hook(lid)))

    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

    for h in hooks:
        h.remove()

    return captured


def causal_swap_experiment(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    selected_objects: list[str],
    selected_categories: list[str],
    truth_tables: list[dict[str, str]],
    probe_layers: list[int],
    peak_layer: int,
    max_length: int = 256,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Step 4+5: Causal swap and orthogonality test.

    For each pair (sample_A, sample_B):
    - Collect hidden states at peak_layer for both
    - Swap the category-related component from A into B
    - Check if B now outputs A's category
    - Check if B's object state is preserved
    """
    results: dict[str, Any] = {"swaps": [], "summary": {}}

    # We use a simple approach: compute the direction difference between
    # two category states, then add that direction to B's hidden state
    # via a hook on the peak layer.

    n_swaps = 0
    n_category_changed = 0
    n_object_preserved = 0

    # For each truth table, pick two objects with different categories
    for tt_idx, tt in enumerate(truth_tables):
        cats = list(set(tt.values()))
        if len(cats) < 2:
            continue

        # Find objects with different categories
        obj_by_cat = {}
        for obj, cat in tt.items():
            obj_by_cat.setdefault(cat, []).append(obj)

        if len(obj_by_cat) < 2:
            continue

        cat_a, cat_b = list(obj_by_cat.keys())[:2]
        obj_a = obj_by_cat[cat_a][0]
        obj_b = obj_by_cat[cat_b][0]

        # Build prompts
        prompt_a = build_prompt(tt, obj_a, seed=tt_idx * 100)
        prompt_b = build_prompt(tt, obj_b, seed=tt_idx * 100 + 1)

        # Collect hidden states
        hs_a = collect_single_hidden_states(model, tokenizer, device, prompt_a, layers, probe_layers, max_length)
        hs_b = collect_single_hidden_states(model, tokenizer, device, prompt_b, layers, probe_layers, max_length)

        if peak_layer not in hs_a or peak_layer not in hs_b:
            continue

        # Category direction: difference between A and B at peak layer
        # This is approximate — the difference includes both object and category info
        delta = hs_a[peak_layer] - hs_b[peak_layer]

        # Baseline: what does B generate?
        baseline_b = generate_text(model, tokenizer, device, prompt_b, max_new_tokens, max_length)

        # Swapped: inject delta into B at peak layer
        swapped_b = generate_with_hidden_injection(
            model, tokenizer, device, layers, prompt_b,
            peak_layer, delta, max_new_tokens, max_length,
        )

        # Check if swapped output contains cat_a (A's category)
        cat_a_in_output = cat_a in swapped_b.lower()
        cat_b_in_output = cat_b in swapped_b.lower()
        obj_b_in_output = obj_b.lower() in swapped_b.lower()

        n_swaps += 1
        if cat_a_in_output and not cat_b_in_output:
            n_category_changed += 1
        if obj_b_in_output or (not cat_a_in_output and not cat_b_in_output):
            n_object_preserved += 1

        results["swaps"].append({
            "table_idx": tt_idx,
            "obj_a": obj_a, "cat_a": cat_a,
            "obj_b": obj_b, "cat_b": cat_b,
            "baseline_b": baseline_b,
            "swapped_b": swapped_b,
            "cat_a_in_swapped": cat_a_in_output,
            "cat_b_in_swapped": cat_b_in_output,
            "obj_b_in_swapped": obj_b_in_output,
        })

        log(f"  Swap {tt_idx}: {obj_b}(cat={cat_b}) + delta from {obj_a}(cat={cat_a})")
        log(f"    baseline_b: '{baseline_b}'")
        log(f"    swapped_b:  '{swapped_b}'")
        log(f"    cat_a in output: {cat_a_in_output}, cat_b in output: {cat_b_in_output}")

    results["summary"] = {
        "n_swaps": n_swaps,
        "n_category_changed": n_category_changed,
        "n_object_preserved": n_object_preserved,
        "category_change_rate": n_category_changed / max(1, n_swaps),
        "object_preserve_rate": n_object_preserved / max(1, n_swaps),
    }

    return results


def generate_text(
    model: Any, tokenizer: Any, device: torch.device,
    prompt: str, max_new_tokens: int, max_length: int,
) -> str:
    """Generate text from a prompt."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_hidden_injection(
    model: Any, tokenizer: Any, device: torch.device,
    layers: list[Any], prompt: str,
    inject_layer: int, inject_delta: np.ndarray,
    max_new_tokens: int, max_length: int,
) -> str:
    """Generate text with a hidden state delta injected at a specific layer."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Convert delta to tensor
    delta_t = torch.tensor(inject_delta, dtype=torch.float16, device=device)

    # Hook to inject delta at the answer position
    def inject_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Inject at the last position (answer position)
        hs[0, -1, :] += delta_t.to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = layers[inject_layer].register_forward_hook(inject_hook)

    try:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# ============================================================================
# Step 6: Retrieval vs state test
# ============================================================================

def retrieval_vs_state_test(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    selected_objects: list[str],
    selected_categories: list[str],
    truth_tables: list[dict[str, str]],
    probe_layers: list[int],
    peak_layer: int,
    max_length: int = 256,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Test if activation transplant works in a new context."""
    results: dict[str, Any] = {"tests": [], "summary": {}}

    n_tests = 0
    n_transplant_works = 0

    for tt_idx, tt in enumerate(truth_tables[:2]):
        cats = list(set(tt.values()))
        if len(cats) < 2:
            continue

        obj_by_cat = {}
        for obj, cat in tt.items():
            obj_by_cat.setdefault(cat, []).append(obj)

        cat_a, cat_b = list(obj_by_cat.keys())[:2]
        obj_a = obj_by_cat[cat_a][0]
        obj_b = obj_by_cat[cat_b][0]

        # Source prompt: has rules, queries obj_a
        source_prompt = build_prompt(tt, obj_a, seed=tt_idx * 100)

        # Target prompt: NO rules, queries obj_b
        target_prompt = f"Query:\n{obj_b} belongs to ?"

        # Collect source hidden state
        hs_source = collect_single_hidden_states(
            model, tokenizer, device, source_prompt, layers, probe_layers, max_length
        )

        if peak_layer not in hs_source:
            continue

        # Baseline: what does target generate without rules?
        baseline_target = generate_text(model, tokenizer, device, target_prompt, max_new_tokens, max_length)

        # Transplant: inject source state into target
        # Use the full hidden state (not just delta) as injection
        source_h = hs_source[peak_layer]

        transplanted_target = generate_with_hidden_injection(
            model, tokenizer, device, layers, target_prompt,
            peak_layer, source_h * 0.5,  # Scale down to avoid overwhelming
            max_new_tokens, max_length,
        )

        cat_a_in_output = cat_a in transplanted_target.lower()
        cat_b_in_output = cat_b in transplanted_target.lower()

        n_tests += 1
        if cat_a_in_output:
            n_transplant_works += 1

        results["tests"].append({
            "table_idx": tt_idx,
            "source_obj": obj_a, "source_cat": cat_a,
            "target_obj": obj_b, "target_cat": cat_b,
            "baseline_target": baseline_target,
            "transplanted_target": transplanted_target,
            "source_cat_in_output": cat_a_in_output,
            "target_cat_in_output": cat_b_in_output,
        })

        log(f"  Transplant {tt_idx}: source={obj_a}({cat_a}) → target={obj_b}({cat_b})")
        log(f"    baseline:  '{baseline_target}'")
        log(f"    transplant: '{transplanted_target}'")
        log(f"    source_cat in output: {cat_a_in_output}")

    results["summary"] = {
        "n_tests": n_tests,
        "n_transplant_works": n_transplant_works,
        "transplant_rate": n_transplant_works / max(1, n_tests),
    }

    return results


# ============================================================================
# Main
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        n_layers = info.n_layers

        probe_layers = sorted(set([min(int(f * n_layers), n_layers - 1) for f in PROBE_LAYER_FRACTIONS]))

        log(f"{args.model}: phase575 n_layers={n_layers}, probe_layers={probe_layers}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "phase": 575, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "n_layers": n_layers,
            "model_info": {"n_layers": n_layers, "d_model": info.d_model, "class": info.model_class},
        }

        # === Step 0: Tokenizer audit ===
        log("=== Step 0: Tokenizer audit ===")
        audit = tokenizer_audit(tokenizer, model, CANDIDATE_OBJECTS, CANDIDATE_CATEGORIES,
                                CANDIDATE_VALUES, CANDIDATE_RELATIONS)
        result["step0_audit"] = audit
        selected = audit["selected"]
        sel_objects = selected["objects"]
        sel_categories = selected["categories"]

        if len(sel_objects) < 2 or len(sel_categories) < 2:
            log("WARNING: Not enough valid symbols, using fallback")
            sel_objects = CANDIDATE_OBJECTS[:4]
            sel_categories = CANDIDATE_CATEGORIES[:2]

        # === Step 1: Build truth tables ===
        log("=== Step 1: Build truth tables ===")
        n_tables = args.n_tables
        truth_tables = build_truth_tables(sel_objects, sel_categories, n_tables)
        result["step1_truth_tables"] = truth_tables
        for i, tt in enumerate(truth_tables):
            log(f"  Table {i}: {tt}")

        # === Step 2+3: Build prompts and collect hidden states ===
        log("=== Step 2+3: Collect hidden states and train probes ===")
        all_prompts = []
        object_labels = []
        category_labels = []
        table_labels = []

        for tt_idx, tt in enumerate(truth_tables):
            for obj in sel_objects:
                prompt = build_prompt(tt, obj, seed=tt_idx * 100 + hash(obj) % 1000)
                all_prompts.append(prompt)
                object_labels.append(sel_objects.index(obj))
                category_labels.append(sel_categories.index(tt[obj]))
                table_labels.append(tt_idx)

        object_labels = np.array(object_labels)
        category_labels = np.array(category_labels)
        table_labels = np.array(table_labels)

        log(f"  Collected {len(all_prompts)} prompts across {n_tables} tables")

        # Collect hidden states
        hidden_by_layer = collect_hidden_states(
            model, tokenizer, device, all_prompts, layers, args.max_length,
        )

        # Train probes
        log("  Training object probes...")
        object_probes = train_probes(hidden_by_layer, object_labels, "object")
        log("  Training category probes...")
        category_probes = train_probes(hidden_by_layer, category_labels, "category")

        # Find peak layer for category
        peak_layer = max(category_probes.keys(),
                        key=lambda l: category_probes[l].get("accuracy_mean", 0))
        peak_acc = category_probes[peak_layer].get("accuracy_mean", 0)
        log(f"  Category probe peak: L{peak_layer} acc={peak_acc:.3f}")

        # Log probe results
        for lid in sorted(object_probes.keys()):
            op = object_probes[lid].get("accuracy_mean", 0)
            cp = category_probes[lid].get("accuracy_mean", 0)
            log(f"    L{lid}: object_acc={op:.3f} category_acc={cp:.3f}")

        result["step3_probes"] = {
            "object_probes": object_probes,
            "category_probes": category_probes,
            "peak_layer": peak_layer,
            "peak_category_acc": peak_acc,
        }

        # === Step 4+5: Causal swap ===
        log("=== Step 4+5: Causal swap and orthogonality ===")
        swap_results = causal_swap_experiment(
            model, tokenizer, device, layers,
            sel_objects, sel_categories, truth_tables,
            probe_layers, peak_layer, args.max_length, args.max_new_tokens,
        )
        result["step4_causal_swap"] = swap_results
        log(f"  Category change rate: {swap_results['summary']['category_change_rate']:.2f}")
        log(f"  Object preserve rate: {swap_results['summary']['object_preserve_rate']:.2f}")

        # === Step 6: Retrieval vs state ===
        log("=== Step 6: Retrieval vs state test ===")
        transplant_results = retrieval_vs_state_test(
            model, tokenizer, device, layers,
            sel_objects, sel_categories, truth_tables,
            probe_layers, peak_layer, args.max_length, args.max_new_tokens,
        )
        result["step6_transplant"] = transplant_results
        log(f"  Transplant success rate: {transplant_results['summary']['transplant_rate']:.2f}")

        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 4
        log("SMOKE TEST MODE: n_tables=4")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase575_{args.model}_closed_micro_world{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
