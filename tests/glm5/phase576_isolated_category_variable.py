#!/usr/bin/env python3
"""
Phase 576: Closed Micro-World v2 — Isolated Category Variable Intervention
闭合微世界v2：隔离类别变量干预

Key improvements over Phase 575:
1. Expanded micro-world: 10+ truth tables, 8 objects, 4 categories
2. Tokenizer/embedding audit to ensure symbols don't leak category
3. Linear category subspace extraction via class-mean SVD
4. Orthogonality test: P_C @ P_O ≈ 0
5. Subspace-isolated swap: h' = h + alpha * P_C @ (h_A - h_B)
6. Strength scan: alpha in {0.5, 1.0, 2.0}
7. Multi-layer combination intervention (single, two-layer, range)
8. Cross-table generalization test
9. Retrieval ablation: masked rules, no-rules transplant

Run:
  python tests/glm5/phase576_isolated_category_variable.py qwen3 --smoke
  python tests/glm5/phase576_isolated_category_variable.py qwen3
  python tests/glm5/phase576_isolated_category_variable.py glm4
  python tests/glm5/phase576_isolated_category_variable.py deepseek7b
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

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402

OUT_ROOT = Path("results/glm5_phase576_isolated_category")

# Expanded symbol sets — 8 objects, 4 categories
CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]
CANDIDATE_VALUES = ["v05", "v91", "v44", "v68"]
CANDIDATE_RELATIONS = ["r31", "r64", "r27", "r88"]
DISTRACTOR_SYMS = ["x72", "m19", "z03", "q08", "p44", "w52", "k37", "n85"]

# Probe layers — 9 fractions
PROBE_LAYER_FRACTIONS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# Strength scan values
ALPHAS = [0.5, 1.0, 2.0]


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
) -> dict[str, Any]:
    """Audit tokenization of artificial symbols — ensure low embedding similarity."""
    all_syms = objects + categories + CANDIDATE_VALUES + CANDIDATE_RELATIONS + DISTRACTOR_SYMS
    audit: dict[str, Any] = {"symbols": {}}

    embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()

    sym_info: dict[str, Any] = {}
    for sym in all_syms:
        ids = tokenizer.encode(" " + sym, add_special_tokens=False)
        ids_ns = tokenizer.encode(sym, add_special_tokens=False)

        emb_sims: dict[str, float] = {}
        if ids and ids[0] < embed.shape[0]:
            emb = embed[ids[0]]
            for other in all_syms:
                if other == sym:
                    continue
                oids = tokenizer.encode(" " + other, add_special_tokens=False)
                if oids and oids[0] < embed.shape[0]:
                    oemb = embed[oids[0]]
                    cos = float(np.dot(emb, oemb) / (np.linalg.norm(emb) * np.linalg.norm(oemb) + 1e-8))
                    emb_sims[other] = cos

        sym_info[sym] = {
            "token_ids": ids,
            "n_tokens": len(ids),
            "max_emb_sim": max(emb_sims.values()) if emb_sims else 0.0,
            "most_similar": max(emb_sims, key=emb_sims.get) if emb_sims else None,
        }

    audit["symbols"] = sym_info

    # Select objects: single/double token, low similarity
    sel_objects = [s for s in objects if sym_info[s]["n_tokens"] <= 2][:8]
    sel_categories = [s for s in categories if sym_info[s]["n_tokens"] <= 2][:4]

    if len(sel_objects) < 4:
        sel_objects = objects[:8]
    if len(sel_categories) < 2:
        sel_categories = categories[:4]

    audit["selected"] = {"objects": sel_objects, "categories": sel_categories}

    log(f"  Tokenizer audit: objects={sel_objects}, categories={sel_categories}")
    for s in sel_objects + sel_categories:
        info = sym_info[s]
        log(f"    {s}: tokens={info['n_tokens']} max_sim={info['max_emb_sim']:.3f}")

    return audit


# ============================================================================
# Step 1: Truth table construction
# ============================================================================

def build_truth_tables(
    objects: list[str],
    categories: list[str],
    n_tables: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build truth tables ensuring each category has at least 2 objects."""
    rng = random.Random(seed)
    n_cats = len(categories)
    tables: list[dict[str, str]] = []

    for t in range(n_tables):
        mapping: dict[str, str] = {}
        shuffled = list(objects)
        rng.shuffle(shuffled)

        # Assign 2 objects per category first (balanced)
        idx = 0
        for c in categories:
            for _ in range(2):
                if idx < len(shuffled):
                    mapping[shuffled[idx]] = c
                    idx += 1

        # Remaining objects: random
        while idx < len(shuffled):
            mapping[shuffled[idx]] = rng.choice(categories)
            idx += 1

        tables.append(mapping)

    return tables


# ============================================================================
# Step 2: Prompt construction
# ============================================================================

def build_prompt(
    truth_table: dict[str, str],
    query_object: str,
    seed: int = 42,
) -> str:
    """Build prompt with rules and Answer: format."""
    rng = random.Random(seed)
    rules = [f"{obj} belongs to {cat}." for obj, cat in truth_table.items()]
    rng.shuffle(rules)

    prompt = "Rules:\n" + "\n".join(rules)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"
    return prompt


def build_masked_prompt(
    truth_table: dict[str, str],
    query_object: str,
    seed: int = 42,
) -> str:
    """Build prompt with categories masked — prevents direct retrieval."""
    rng = random.Random(seed)
    rules = [f"{obj} belongs to ???." for obj in truth_table]
    rng.shuffle(rules)

    prompt = "Rules:\n" + "\n".join(rules)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"
    return prompt


def build_no_rules_prompt(query_object: str) -> str:
    """Build prompt with no rules — pure query."""
    return f"Question: {query_object} belongs to ?\nAnswer:"


# ============================================================================
# Step 3: Hidden state collection
# ============================================================================

def collect_hidden_states(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    layers: list[Any],
    probe_layers: list[int],
    max_length: int = 384,
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Collect hidden states at each probe layer for all prompts (batched)."""
    n_layers = len(layers)
    all_captured: dict[int, list[np.ndarray]] = {lid: [] for lid in probe_layers}

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        answer_pos = input_ids.shape[1] - 1

        captured: dict[int, np.ndarray] = {}
        hooks: list[Any] = []

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

        for lid in probe_layers:
            if lid in captured:
                all_captured[lid].append(captured[lid])

        if (start // batch_size) % 4 == 0:
            log(f"    Collected batch {start // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

    tokenizer.padding_side = old_padding

    return {lid: np.concatenate(vs, axis=0) for lid, vs in all_captured.items() if vs}


# ============================================================================
# Step 4: Probe training (verify decodability)
# ============================================================================

def train_probes(
    hidden_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Train cross-validated linear probes at each layer."""
    results: dict[int, dict[str, float]] = {}

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_classes = unique_labels[counts >= 2]
    if len(valid_classes) < 2:
        return results

    mask = np.isin(labels, valid_classes)
    labels_f = labels[mask]

    for lid, h in hidden_by_layer.items():
        h_f = h[mask]
        n = len(labels_f)
        cv_folds = min(5, n // 2, min(counts[counts >= 2]))
        cv_folds = max(2, cv_folds)

        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            scores = cross_val_score(clf, h_f, labels_f, cv=cv_folds, scoring="accuracy")
            results[lid] = {
                "accuracy_mean": float(np.nanmean(scores)),
                "accuracy_std": float(np.nanstd(scores)),
            }
        except Exception as e:
            results[lid] = {"accuracy_mean": 0.0, "accuracy_std": 0.0, "error": str(e)}

    return results


# ============================================================================
# Step 5: Subspace extraction (class-mean SVD)
# ============================================================================

def extract_subspace(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract class-discriminative subspace via class-mean SVD.

    Returns (U [d, k-1], singular_values [k-1]) or (None, None) if failed.
    """
    global_mean = hidden_states.mean(axis=0)
    class_diffs: list[np.ndarray] = []

    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_diffs.append(hidden_states[mask].mean(axis=0) - global_mean)

    if len(class_diffs) < 2:
        return None, None

    M = np.stack(class_diffs, axis=0)  # [k, d]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    n_comp = min(n_classes - 1, len(S))
    return Vt[:n_comp].T.copy(), S[:n_comp].copy()  # [d, k-1]


def extract_all_subspaces(
    hidden_by_layer: dict[int, np.ndarray],
    cat_labels: np.ndarray,
    obj_labels: np.ndarray,
    n_cats: int,
    n_objs: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Extract category and object subspaces at each layer.

    Returns (cat_subspaces, obj_subspaces, singular_values).
    """
    cat_subs: dict[int, np.ndarray] = {}
    obj_subs: dict[int, np.ndarray] = {}
    sing_vals: dict[int, np.ndarray] = {}

    for lid, h in hidden_by_layer.items():
        U_C, S_C = extract_subspace(h, cat_labels, n_cats)
        U_O, S_O = extract_subspace(h, obj_labels, n_objs)
        if U_C is not None:
            cat_subs[lid] = U_C
        if U_O is not None:
            obj_subs[lid] = U_O
        if S_C is not None:
            sing_vals[lid] = S_C

    return cat_subs, obj_subs, sing_vals


# ============================================================================
# Step 6: Orthogonality test
# ============================================================================

def orthogonality_test(
    cat_subspaces: dict[int, np.ndarray],
    obj_subspaces: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Test orthogonality between category and object subspaces.

    overlap = ||U_C^T @ U_O||_F^2 / min(k_c, k_o)
    """
    results: dict[int, dict[str, float]] = {}

    for lid in cat_subspaces:
        if lid not in obj_subspaces:
            continue
        U_C = cat_subspaces[lid]  # [d, k_c]
        U_O = obj_subspaces[lid]  # [d, k_o]

        cross = U_C.T @ U_O  # [k_c, k_o]
        overlap_frob = float(np.linalg.norm(cross, "fro"))
        overlap_sq = float(np.sum(cross ** 2))
        min_dim = min(cross.shape)
        normalized_overlap = overlap_sq / max(1, min_dim)

        results[lid] = {
            "overlap_frob": overlap_frob,
            "overlap_sq": overlap_sq,
            "normalized_overlap": normalized_overlap,
            "is_orthogonal": normalized_overlap < 0.1,
        }

    return results


# ============================================================================
# Step 7: Generation utilities
# ============================================================================

def generate_text(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    max_length: int,
) -> str:
    """Generate text from a prompt (greedy)."""
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_injection(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    inject_deltas: dict[int, np.ndarray],
    max_new_tokens: int,
    max_length: int,
) -> str:
    """Generate text with delta injected at specified layers.

    inject_deltas: {layer_idx: delta_array[d_model]}
    """
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Pre-create delta tensors lazily in hooks
    delta_cache: dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int, delta_np: np.ndarray):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            key = layer_id
            if key not in delta_cache or delta_cache[key].device != hs.device or delta_cache[key].dtype != hs.dtype:
                delta_cache[key] = torch.tensor(delta_np, dtype=hs.dtype, device=hs.device)
            hs[:, -1, :] += delta_cache[key]
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        return hook

    hooks = []
    for lid, delta in inject_deltas.items():
        hooks.append(layers[lid].register_forward_hook(make_hook(lid, delta)))

    try:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        for h in hooks:
            h.remove()
        delta_cache.clear()

    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# ============================================================================
# Output detection
# ============================================================================

def detect_category(text: str, categories: list[str]) -> str:
    """Detect which category appears in the output."""
    text_lower = text.lower()
    found = [c for c in categories if c.lower() in text_lower]
    if len(found) == 1:
        return found[0]
    elif len(found) == 0:
        return "none"
    else:
        return "ambiguous"


def detect_object(text: str, objects: list[str]) -> str:
    """Detect which object appears in the output."""
    text_lower = text.lower()
    found = [o for o in objects if o.lower() in text_lower]
    if len(found) == 1:
        return found[0]
    elif len(found) == 0:
        return "none"
    else:
        return "ambiguous"


# ============================================================================
# Step 8: Category subspace swap with strength scan
# ============================================================================

def compute_swap_delta(
    U_C: np.ndarray,
    h_A: np.ndarray,
    h_B: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute category subspace swap delta.

    delta = alpha * P_C @ (h_A - h_B) = alpha * U_C @ (U_C^T @ (h_A - h_B))
    """
    diff = h_A - h_B
    return alpha * (U_C @ (U_C.T @ diff))


def select_swap_pairs(
    truth_table: dict[str, str],
    categories: list[str],
    n_pairs: int = 2,
    seed: int = 42,
) -> list[tuple[str, str, str, str]]:
    """Select pairs of objects with different categories.

    Returns [(obj_a, cat_a, obj_b, cat_b), ...]
    """
    rng = random.Random(seed)
    cat_objs: dict[str, list[str]] = {}
    for obj, cat in truth_table.items():
        cat_objs.setdefault(cat, []).append(obj)

    cat_list = [c for c in categories if c in cat_objs and len(cat_objs[c]) > 0]
    rng.shuffle(cat_list)

    pairs: list[tuple[str, str, str, str]] = []
    for i in range(min(n_pairs, len(cat_list))):
        cat_a = cat_list[i % len(cat_list)]
        cat_b = cat_list[(i + 1) % len(cat_list)]
        if cat_a != cat_b and cat_objs.get(cat_a) and cat_objs.get(cat_b):
            obj_a = rng.choice(cat_objs[cat_a])
            obj_b = rng.choice(cat_objs[cat_b])
            pairs.append((obj_a, cat_a, obj_b, cat_b))

    return pairs


def category_swap_experiment(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    truth_tables: list[dict[str, str]],
    objects: list[str],
    categories: list[str],
    hidden_by_layer: dict[int, np.ndarray],
    cat_subspaces: dict[int, np.ndarray],
    peak_layer: int,
    prompt_table_map: list[int],
    prompt_obj_map: list[str],
    prompt_cat_map: list[str],
    max_length: int = 384,
    max_new_tokens: int = 5,
    n_pairs_per_table: int = 2,
) -> dict[str, Any]:
    """Category subspace swap with strength scan.

    For each pair (A, B) with different categories:
    - Compute delta = alpha * P_C @ (h_A - h_B) at peak layer
    - Inject delta into B's generation
    - Check if output changes to cat_A, object stays obj_B
    """
    results: dict[str, Any] = {"swaps": [], "summary": {}}
    n_swaps = 0
    n_cat_changed = 0
    n_obj_preserved = 0
    n_both = 0

    U_C = cat_subspaces.get(peak_layer)
    if U_C is None:
        log(f"  WARNING: No category subspace at peak layer {peak_layer}")
        return results

    # Build prompt index lookup
    prompt_index: dict[tuple[int, str], int] = {}
    for i, (tt_idx, obj) in enumerate(zip(prompt_table_map, prompt_obj_map)):
        prompt_index[(tt_idx, obj)] = i

    total_pairs = 0
    for tt_idx, tt in enumerate(truth_tables):
        pairs = select_swap_pairs(tt, categories, n_pairs_per_table, seed=tt_idx * 100)
        total_pairs += len(pairs)
    log(f"  Total swap pairs: {total_pairs}, alphas: {ALPHAS}")

    pair_count = 0
    for tt_idx, tt in enumerate(truth_tables):
        pairs = select_swap_pairs(tt, categories, n_pairs_per_table, seed=tt_idx * 100)

        for obj_a, cat_a, obj_b, cat_b in pairs:
            pair_count += 1
            idx_a = prompt_index.get((tt_idx, obj_a))
            idx_b = prompt_index.get((tt_idx, obj_b))

            if idx_a is None or idx_b is None:
                continue

            h_A = hidden_by_layer[peak_layer][idx_a]
            h_B = hidden_by_layer[peak_layer][idx_b]

            prompt_b = build_prompt(tt, obj_b, seed=tt_idx * 100 + hash(obj_b) % 1000)

            # Baseline generation
            baseline_b = generate_text(model, tokenizer, device, prompt_b, max_new_tokens, max_length)
            baseline_cat = detect_category(baseline_b, categories)
            baseline_obj = detect_object(baseline_b, objects)

            swap_record: dict[str, Any] = {
                "table_idx": tt_idx,
                "obj_a": obj_a, "cat_a": cat_a,
                "obj_b": obj_b, "cat_b": cat_b,
                "baseline_b": baseline_b,
                "baseline_cat": baseline_cat,
                "alphas": [],
            }

            for alpha in ALPHAS:
                delta = compute_swap_delta(U_C, h_A, h_B, alpha)

                swapped = generate_with_injection(
                    model, tokenizer, device, layers, prompt_b,
                    {peak_layer: delta}, max_new_tokens, max_length,
                )

                swapped_cat = detect_category(swapped, categories)
                swapped_obj = detect_object(swapped, objects)

                cat_changed = swapped_cat == cat_a and swapped_cat != cat_b
                obj_preserved = swapped_obj == obj_b or swapped_obj == "none"

                n_swaps += 1
                if cat_changed:
                    n_cat_changed += 1
                if obj_preserved:
                    n_obj_preserved += 1
                if cat_changed and obj_preserved:
                    n_both += 1

                swap_record["alphas"].append({
                    "alpha": alpha,
                    "swapped_b": swapped,
                    "swapped_cat": swapped_cat,
                    "swapped_obj": swapped_obj,
                    "cat_changed": cat_changed,
                    "obj_preserved": obj_preserved,
                })

            results["swaps"].append(swap_record)

            if pair_count % 5 == 0:
                log(f"    Swap pair {pair_count}/{total_pairs}: "
                    f"baseline_cat={baseline_cat}, "
                    f"swapped_cats={[a['swapped_cat'] for a in swap_record['alphas']]}")

    results["summary"] = {
        "n_swaps": n_swaps,
        "n_cat_changed": n_cat_changed,
        "n_obj_preserved": n_obj_preserved,
        "n_both": n_both,
        "cat_change_rate": n_cat_changed / max(1, n_swaps),
        "obj_preserve_rate": n_obj_preserved / max(1, n_swaps),
        "both_rate": n_both / max(1, n_swaps),
    }

    return results


# ============================================================================
# Step 9: Cross-table generalization
# ============================================================================

def cross_table_swap(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    truth_tables: list[dict[str, str]],
    objects: list[str],
    categories: list[str],
    hidden_by_layer: dict[int, np.ndarray],
    cat_subspaces: dict[int, np.ndarray],
    peak_layer: int,
    prompt_table_map: list[int],
    prompt_obj_map: list[str],
    max_length: int = 384,
    max_new_tokens: int = 5,
    n_pairs: int = 4,
) -> dict[str, Any]:
    """Cross-table swap: subspace from all tables, swap on held-out pairs."""
    results: dict[str, Any] = {"swaps": [], "summary": {}}

    U_C = cat_subspaces.get(peak_layer)
    if U_C is None:
        return results

    prompt_index: dict[tuple[int, str], int] = {}
    for i, (tt_idx, obj) in enumerate(zip(prompt_table_map, prompt_obj_map)):
        prompt_index[(tt_idx, obj)] = i

    n_tables = len(truth_tables)
    n_swaps = 0
    n_cat_changed = 0
    n_obj_preserved = 0

    # Pick cross-table pairs: A from table 0, B from table n-1
    rng = random.Random(99)
    tested = 0

    for _ in range(n_pairs * 3):
        if tested >= n_pairs:
            break
        tt_a = rng.randint(0, n_tables // 2)
        tt_b = rng.randint(n_tables // 2 + 1, n_tables - 1)
        if tt_a == tt_b:
            continue

        tt_A = truth_tables[tt_a]
        tt_B = truth_tables[tt_b]

        # Pick obj_a from tt_A, obj_b from tt_B with different cats
        obj_a = rng.choice(list(tt_A.keys()))
        cat_a = tt_A[obj_a]
        obj_b = rng.choice(list(tt_B.keys()))
        cat_b = tt_B[obj_b]

        if cat_a == cat_b:
            continue

        idx_a = prompt_index.get((tt_a, obj_a))
        idx_b = prompt_index.get((tt_b, obj_b))
        if idx_a is None or idx_b is None:
            continue

        h_A = hidden_by_layer[peak_layer][idx_a]
        h_B = hidden_by_layer[peak_layer][idx_b]

        prompt_b = build_prompt(tt_B, obj_b, seed=tt_b * 100 + hash(obj_b) % 1000)
        baseline_b = generate_text(model, tokenizer, device, prompt_b, max_new_tokens, max_length)

        delta = compute_swap_delta(U_C, h_A, h_B, 1.0)
        swapped = generate_with_injection(
            model, tokenizer, device, layers, prompt_b,
            {peak_layer: delta}, max_new_tokens, max_length,
        )

        swapped_cat = detect_category(swapped, categories)
        swapped_obj = detect_object(swapped, objects)

        cat_changed = swapped_cat == cat_a
        obj_preserved = swapped_obj == obj_b or swapped_obj == "none"

        n_swaps += 1
        if cat_changed:
            n_cat_changed += 1
        if obj_preserved:
            n_obj_preserved += 1

        results["swaps"].append({
            "tt_a": tt_a, "tt_b": tt_b,
            "obj_a": obj_a, "cat_a": cat_a,
            "obj_b": obj_b, "cat_b": cat_b,
            "baseline_b": baseline_b,
            "swapped_b": swapped,
            "swapped_cat": swapped_cat,
            "swapped_obj": swapped_obj,
            "cat_changed": cat_changed,
            "obj_preserved": obj_preserved,
        })
        tested += 1

        log(f"    Cross-table {tested}: {obj_a}({cat_a},T{tt_a}) → {obj_b}({cat_b},T{tt_b}): "
            f"swapped_cat={swapped_cat}")

    results["summary"] = {
        "n_swaps": n_swaps,
        "n_cat_changed": n_cat_changed,
        "n_obj_preserved": n_obj_preserved,
        "cat_change_rate": n_cat_changed / max(1, n_swaps),
        "obj_preserve_rate": n_obj_preserved / max(1, n_swaps),
    }

    return results


# ============================================================================
# Step 10: Multi-layer combination intervention
# ============================================================================

def multi_layer_intervention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    truth_tables: list[dict[str, str]],
    objects: list[str],
    categories: list[str],
    hidden_by_layer: dict[int, np.ndarray],
    cat_subspaces: dict[int, np.ndarray],
    peak_layer: int,
    prompt_table_map: list[int],
    prompt_obj_map: list[str],
    max_length: int = 384,
    max_new_tokens: int = 5,
    n_pairs: int = 4,
) -> dict[str, Any]:
    """Multi-layer combination intervention: single, two-layer, range."""
    results: dict[str, Any] = {"configs": [], "summary": {}}
    n_layers = len(layers)

    # Define configs relative to peak
    configs = [
        {"name": "single", "layers": [peak_layer]},
        {"name": "two_layer", "layers": [peak_layer, min(peak_layer + 2, n_layers - 1)]},
        {"name": "range", "layers": sorted(set([
            max(0, peak_layer - 2), peak_layer, min(peak_layer + 2, n_layers - 1)
        ]))},
    ]

    prompt_index: dict[tuple[int, str], int] = {}
    for i, (tt_idx, obj) in enumerate(zip(prompt_table_map, prompt_obj_map)):
        prompt_index[(tt_idx, obj)] = i

    rng = random.Random(77)
    total = 0

    for cfg in configs:
        cfg_result: dict[str, Any] = {"name": cfg["name"], "layers": cfg["layers"], "pairs": []}
        n_cat_changed = 0
        n_total = 0

        for _ in range(n_pairs):
            tt_idx = rng.randint(0, len(truth_tables) - 1)
            tt = truth_tables[tt_idx]
            pairs = select_swap_pairs(tt, categories, 1, seed=tt_idx * 200 + total)
            if not pairs:
                continue
            obj_a, cat_a, obj_b, cat_b = pairs[0]

            idx_a = prompt_index.get((tt_idx, obj_a))
            idx_b = prompt_index.get((tt_idx, obj_b))
            if idx_a is None or idx_b is None:
                continue

            prompt_b = build_prompt(tt, obj_b, seed=tt_idx * 200 + hash(obj_b) % 1000)

            # Compute deltas at each layer in config
            inject_deltas: dict[int, np.ndarray] = {}
            for lid in cfg["layers"]:
                if lid in cat_subspaces and lid in hidden_by_layer:
                    U_C_l = cat_subspaces[lid]
                    h_A_l = hidden_by_layer[lid][idx_a]
                    h_B_l = hidden_by_layer[lid][idx_b]
                    inject_deltas[lid] = compute_swap_delta(U_C_l, h_A_l, h_B_l, 1.0)

            if not inject_deltas:
                continue

            swapped = generate_with_injection(
                model, tokenizer, device, layers, prompt_b,
                inject_deltas, max_new_tokens, max_length,
            )
            swapped_cat = detect_category(swapped, categories)

            cat_changed = swapped_cat == cat_a and swapped_cat != cat_b
            n_total += 1
            if cat_changed:
                n_cat_changed += 1

            cfg_result["pairs"].append({
                "obj_a": obj_a, "cat_a": cat_a,
                "obj_b": obj_b, "cat_b": cat_b,
                "swapped_b": swapped,
                "swapped_cat": swapped_cat,
                "cat_changed": cat_changed,
            })
            total += 1

        cfg_result["cat_change_rate"] = n_cat_changed / max(1, n_total)
        cfg_result["n_total"] = n_total
        cfg_result["n_cat_changed"] = n_cat_changed
        results["configs"].append(cfg_result)
        log(f"    Multi-layer [{cfg['name']}] layers={cfg['layers']}: "
            f"cat_change={n_cat_changed}/{n_total}")

    results["summary"] = {
        c["name"]: c["cat_change_rate"] for c in results["configs"]
    }
    return results


# ============================================================================
# Step 11: Retrieval ablation
# ============================================================================

def retrieval_ablation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    truth_tables: list[dict[str, str]],
    objects: list[str],
    categories: list[str],
    hidden_by_layer: dict[int, np.ndarray],
    cat_subspaces: dict[int, np.ndarray],
    peak_layer: int,
    prompt_table_map: list[int],
    prompt_obj_map: list[str],
    max_length: int = 384,
    max_new_tokens: int = 5,
    n_tests: int = 4,
) -> dict[str, Any]:
    """Retrieval ablation: masked rules and no-rules transplant."""
    results: dict[str, Any] = {"tests": [], "summary": {}}

    U_C = cat_subspaces.get(peak_layer)
    if U_C is None:
        return results

    prompt_index: dict[tuple[int, str], int] = {}
    for i, (tt_idx, obj) in enumerate(zip(prompt_table_map, prompt_obj_map)):
        prompt_index[(tt_idx, obj)] = i

    rng = random.Random(88)
    n_masked_changed = 0
    n_transplant_changed = 0
    n_total = 0

    for test_i in range(n_tests):
        tt_idx = rng.randint(0, len(truth_tables) - 1)
        tt = truth_tables[tt_idx]
        pairs = select_swap_pairs(tt, categories, 1, seed=tt_idx * 300 + test_i)
        if not pairs:
            continue
        obj_a, cat_a, obj_b, cat_b = pairs[0]

        idx_a = prompt_index.get((tt_idx, obj_a))
        idx_b = prompt_index.get((tt_idx, obj_b))
        if idx_a is None or idx_b is None:
            continue

        h_A = hidden_by_layer[peak_layer][idx_a]
        h_B = hidden_by_layer[peak_layer][idx_b]
        delta = compute_swap_delta(U_C, h_A, h_B, 1.0)

        # Test 1: Masked rules — categories hidden
        masked_prompt = build_masked_prompt(tt, obj_b, seed=tt_idx * 300 + test_i)
        masked_baseline = generate_text(model, tokenizer, device, masked_prompt, max_new_tokens, max_length)
        masked_swapped = generate_with_injection(
            model, tokenizer, device, layers, masked_prompt,
            {peak_layer: delta}, max_new_tokens, max_length,
        )
        masked_cat = detect_category(masked_swapped, categories)
        masked_changed = masked_cat == cat_a

        # Test 2: No-rules transplant
        no_rules_prompt = build_no_rules_prompt(obj_b)
        no_rules_baseline = generate_text(model, tokenizer, device, no_rules_prompt, max_new_tokens, max_length)
        no_rules_swapped = generate_with_injection(
            model, tokenizer, device, layers, no_rules_prompt,
            {peak_layer: delta}, max_new_tokens, max_length,
        )
        no_rules_cat = detect_category(no_rules_swapped, categories)
        transplant_changed = no_rules_cat == cat_a

        n_total += 1
        if masked_changed:
            n_masked_changed += 1
        if transplant_changed:
            n_transplant_changed += 1

        results["tests"].append({
            "table_idx": tt_idx,
            "obj_a": obj_a, "cat_a": cat_a,
            "obj_b": obj_b, "cat_b": cat_b,
            "masked_baseline": masked_baseline,
            "masked_swapped": masked_swapped,
            "masked_cat": masked_cat,
            "masked_changed": masked_changed,
            "no_rules_baseline": no_rules_baseline,
            "no_rules_swapped": no_rules_swapped,
            "no_rules_cat": no_rules_cat,
            "transplant_changed": transplant_changed,
        })

        log(f"    Ablation {test_i + 1}: masked_cat={masked_cat}({cat_a}), "
            f"transplant_cat={no_rules_cat}({cat_a})")

    results["summary"] = {
        "n_total": n_total,
        "masked_change_rate": n_masked_changed / max(1, n_total),
        "transplant_change_rate": n_transplant_changed / max(1, n_total),
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

        log(f"{args.model}: phase576 n_layers={n_layers}, probe_layers={probe_layers}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "phase": 576,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "n_layers": n_layers,
            "d_model": info.d_model,
        }

        # === Step 0: Tokenizer audit ===
        log("=== Step 0: Tokenizer audit ===")
        audit = tokenizer_audit(tokenizer, model, CANDIDATE_OBJECTS, CANDIDATE_CATEGORIES)
        result["step0_audit"] = audit
        selected = audit["selected"]
        sel_objects = selected["objects"]
        sel_categories = selected["categories"]
        n_cats = len(sel_categories)
        n_objs = len(sel_objects)

        # === Step 1: Build truth tables ===
        log("=== Step 1: Build truth tables ===")
        n_tables = args.n_tables
        truth_tables = build_truth_tables(sel_objects, sel_categories, n_tables)
        result["step1_truth_tables"] = truth_tables
        for i, tt in enumerate(truth_tables):
            log(f"  Table {i}: {tt}")

        # === Step 2: Build prompts and labels ===
        log("=== Step 2: Build prompts ===")
        all_prompts: list[str] = []
        object_labels_list: list[int] = []
        category_labels_list: list[int] = []
        table_labels_list: list[int] = []
        prompt_obj_map: list[str] = []
        prompt_cat_map: list[str] = []
        prompt_table_map: list[int] = []

        for tt_idx, tt in enumerate(truth_tables):
            for obj in sel_objects:
                prompt = build_prompt(tt, obj, seed=tt_idx * 100 + hash(obj) % 1000)
                all_prompts.append(prompt)
                object_labels_list.append(sel_objects.index(obj))
                cat = tt[obj]
                category_labels_list.append(sel_categories.index(cat))
                table_labels_list.append(tt_idx)
                prompt_obj_map.append(obj)
                prompt_cat_map.append(cat)
                prompt_table_map.append(tt_idx)

        object_labels = np.array(object_labels_list)
        category_labels = np.array(category_labels_list)
        table_labels = np.array(table_labels_list)

        log(f"  {len(all_prompts)} prompts: {n_tables} tables x {n_objs} objects, {n_cats} categories")

        # === Step 3: Collect hidden states ===
        log("=== Step 3: Collect hidden states ===")
        t0 = time.time()
        hidden_by_layer = collect_hidden_states(
            model, tokenizer, device, all_prompts, layers, probe_layers,
            args.max_length, args.batch_size,
        )
        log(f"  Collected hidden states at {len(hidden_by_layer)} layers in {time.time() - t0:.1f}s")

        # === Step 4: Train probes ===
        log("=== Step 4: Train probes ===")
        object_probes = train_probes(hidden_by_layer, object_labels)
        category_probes = train_probes(hidden_by_layer, category_labels)

        # Find peak layers
        peak_cat_layer = max(category_probes.keys(),
                             key=lambda l: category_probes[l].get("accuracy_mean", 0))
        peak_obj_layer = max(object_probes.keys(),
                             key=lambda l: object_probes[l].get("accuracy_mean", 0))
        peak_cat_acc = category_probes[peak_cat_layer].get("accuracy_mean", 0)
        peak_obj_acc = object_probes[peak_obj_layer].get("accuracy_mean", 0)

        log(f"  Object probe peak: L{peak_obj_layer} acc={peak_obj_acc:.3f}")
        log(f"  Category probe peak: L{peak_cat_layer} acc={peak_cat_acc:.3f}")

        for lid in sorted(object_probes.keys()):
            op = object_probes[lid].get("accuracy_mean", 0)
            cp = category_probes[lid].get("accuracy_mean", 0)
            log(f"    L{lid}: object={op:.3f} category={cp:.3f}")

        result["step4_probes"] = {
            "object_probes": object_probes,
            "category_probes": category_probes,
            "peak_cat_layer": peak_cat_layer,
            "peak_cat_acc": peak_cat_acc,
            "peak_obj_layer": peak_obj_layer,
            "peak_obj_acc": peak_obj_acc,
        }

        # === Step 5: Extract subspaces ===
        log("=== Step 5: Extract subspaces ===")
        cat_subspaces, obj_subspaces, sing_vals = extract_all_subspaces(
            hidden_by_layer, category_labels, object_labels, n_cats, n_objs,
        )
        result["step5_subspaces"] = {
            "cat_subspace_dims": {str(lid): U.shape[1] for lid, U in cat_subspaces.items()},
            "obj_subspace_dims": {str(lid): U.shape[1] for lid, U in obj_subspaces.items()},
            "cat_singular_values": {str(lid): S.tolist() for lid, S in sing_vals.items()},
        }
        log(f"  Category subspaces at {len(cat_subspaces)} layers, "
            f"object subspaces at {len(obj_subspaces)} layers")

        # === Step 6: Orthogonality test ===
        log("=== Step 6: Orthogonality test ===")
        ortho_results = orthogonality_test(cat_subspaces, obj_subspaces)
        result["step6_orthogonality"] = ortho_results
        for lid in sorted(ortho_results.keys()):
            r = ortho_results[lid]
            log(f"    L{lid}: overlap={r['normalized_overlap']:.4f} "
                f"orthogonal={r['is_orthogonal']}")

        # Use peak_cat_layer for swap experiments
        peak_layer = peak_cat_layer

        # === Step 7: Category subspace swap with strength scan ===
        log(f"=== Step 7: Category swap at L{peak_layer} (strength scan) ===")
        t0 = time.time()
        swap_results = category_swap_experiment(
            model, tokenizer, device, layers,
            truth_tables, sel_objects, sel_categories,
            hidden_by_layer, cat_subspaces, peak_layer,
            prompt_table_map, prompt_obj_map, prompt_cat_map,
            args.max_length, args.max_new_tokens,
            n_pairs_per_table=args.n_pairs_per_table,
        )
        result["step7_swap"] = swap_results
        s = swap_results["summary"]
        log(f"  Swap: cat_change={s['cat_change_rate']:.3f} "
            f"obj_preserve={s['obj_preserve_rate']:.3f} "
            f"both={s['both_rate']:.3f} ({time.time() - t0:.1f}s)")

        # === Step 8: Cross-table generalization ===
        log("=== Step 8: Cross-table generalization ===")
        t0 = time.time()
        cross_results = cross_table_swap(
            model, tokenizer, device, layers,
            truth_tables, sel_objects, sel_categories,
            hidden_by_layer, cat_subspaces, peak_layer,
            prompt_table_map, prompt_obj_map,
            args.max_length, args.max_new_tokens,
            n_pairs=args.n_cross_pairs,
        )
        result["step8_cross_table"] = cross_results
        cs = cross_results["summary"]
        log(f"  Cross-table: cat_change={cs['cat_change_rate']:.3f} "
            f"obj_preserve={cs['obj_preserve_rate']:.3f} ({time.time() - t0:.1f}s)")

        # === Step 9: Multi-layer intervention ===
        log("=== Step 9: Multi-layer intervention ===")
        t0 = time.time()
        multi_results = multi_layer_intervention(
            model, tokenizer, device, layers,
            truth_tables, sel_objects, sel_categories,
            hidden_by_layer, cat_subspaces, peak_layer,
            prompt_table_map, prompt_obj_map,
            args.max_length, args.max_new_tokens,
            n_pairs=args.n_multi_pairs,
        )
        result["step9_multi_layer"] = multi_results
        log(f"  Multi-layer: {multi_results['summary']} ({time.time() - t0:.1f}s)")

        # === Step 10: Retrieval ablation ===
        log("=== Step 10: Retrieval ablation ===")
        t0 = time.time()
        ablation_results = retrieval_ablation(
            model, tokenizer, device, layers,
            truth_tables, sel_objects, sel_categories,
            hidden_by_layer, cat_subspaces, peak_layer,
            prompt_table_map, prompt_obj_map,
            args.max_length, args.max_new_tokens,
            n_tests=args.n_ablation_tests,
        )
        result["step10_ablation"] = ablation_results
        ab = ablation_results["summary"]
        log(f"  Ablation: masked={ab['masked_change_rate']:.3f} "
            f"transplant={ab['transplant_change_rate']:.3f} ({time.time() - t0:.1f}s)")

        return result

    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=10)
    parser.add_argument("--n-pairs-per-table", type=int, default=2)
    parser.add_argument("--n-cross-pairs", type=int, default=4)
    parser.add_argument("--n-multi-pairs", type=int, default=4)
    parser.add_argument("--n-ablation-tests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 6
        args.n_pairs_per_table = 1
        args.n_cross_pairs = 2
        args.n_multi_pairs = 2
        args.n_ablation_tests = 2
        log("SMOKE TEST MODE: n_tables=6, reduced pairs")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase576_{args.model}_isolated_category{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
