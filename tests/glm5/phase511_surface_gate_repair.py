#!/usr/bin/env python3
"""
Phase 511: Surface Gate Direct Repair after Category Margin Shift
=================================================================
Phase 510 proved support/release subspaces can change category-vs-competitor
margin, but hit rate barely moves. This script tests whether suppressing
surface competitors (punctuation/generic/object-copy) can convert margin
improvement into actual category token hits.

Experiments:
  Exp1: Baseline surface competition profiling (3-step trace, all groups)
  Exp2: Surface gate direction construction
  Exp3: Combined semantic + surface intervention
  Exp4: Bottleneck type classification per category
  Exp5: Cross-model validation

All models use BF16 + device_map="auto" with flash attention (sdpa).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
from phase507_orthogonal_field import (  # noqa: E402
    ALL_CLASS,
    CATEGORIES,
    get_norm_g,
    get_token_ids,
    load_bf16_auto,
)
from phase508_orthogonal_field_basis_decomposition import (  # noqa: E402
    NEUTRAL_TEMPLATES,
    RICH_TEMPLATES,
    batched_hidden,
    batched_logits_with_delta,
    build_cat_meta,
    build_examples,
    cos,
    delta_summary,
    label_effect,
    max_abs_basis_cos,
    orthonormal_rows,
    project_remove_deltas,
    random_basis,
    score_logits,
    summarize_scores,
    svd_basis,
)
from phase509_rotation_stable_orthogonal_field import FOCUS_CATEGORIES, make_candidate_axes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase511_surface_gate_repair")

# Generation templates that don't contain the answer category
GEN_TEMPLATES = [
    ("category_of", "The {obj} belongs to the category of"),
    ("taxonomy_as", "In taxonomy, {obj} is classified as"),
    ("classify_colon", "Classify {obj}:"),
]

PUNCTUATION_TOKENS = [".", ",", ":", ";", "!", "?", "\n", " the", " a", " an"]
GENERIC_TOKENS = [" thing", " item", " type", " kind", " object", " entity", " one", " it", " that", " something"]
OBJECT_COPY_PREFIX = ""  # Will be filled per-category


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids: list[int] = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def build_generation_examples(
    cat: str, train_n: int, test_n: int
) -> list[dict[str, Any]]:
    objs = CATEGORIES[cat]["objects"][train_n: train_n + test_n]
    rows = []
    for obj in objs:
        for tid, (name, tpl) in enumerate(GEN_TEMPLATES):
            rows.append({
                "cat": cat,
                "obj": obj,
                "template_id": tid,
                "template_name": name,
                "prompt": tpl.format(obj=obj),
            })
    return rows


def build_token_groups(
    tokenizer: Any, cat: str, objects: list[str], competitor_cats: list[str]
) -> dict[str, list[int]]:
    """Build token ID groups for each competitor type."""
    return {
        "category": token_ids(tokenizer, [cat, " " + cat]),
        "competitor_category": token_ids(
            tokenizer, competitor_cats + [" " + c for c in competitor_cats]
        ),
        "punctuation": token_ids(tokenizer, PUNCTUATION_TOKENS),
        "generic": token_ids(tokenizer, GENERIC_TOKENS),
        "object_copy": token_ids(
            tokenizer, objects + [" " + o for o in objects]
        ),
    }


def max_for_ids(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.full(
            (logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype
        )
    valid = [i for i in ids if 0 <= i < logits.shape[-1]]
    if not valid:
        return torch.full(
            (logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype
        )
    return logits[:, valid].max(dim=1).values


def mean_for_ids_np(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    valid = [i for i in ids if 0 <= i < logits.shape[1]]
    if not valid:
        return np.zeros(logits.shape[0], dtype=np.float32)
    return logits[:, valid].mean(axis=1)


def logits_with_axis_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    axis: np.ndarray | None,
    mode: str,
    scale: float,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Run forward pass with axis projection removal/addition at a specific layer."""
    outs = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))
    axis_t = None if axis is None else torch.tensor(axis, device=device, dtype=torch.float32)
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handle = None
        if axis_t is not None and mode != "clean":
            pos_t = pos.to(device)

            def hook(_module, _inp, output):
                sign = -1.0 if mode.startswith("remove") else 1.0
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    cur = hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        pos_t.to(hs.device)
                    ].float()
                    a = axis_t.to(hs.device)
                    proj = (cur @ a)[:, None] * a[None, :]
                    hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        pos_t.to(hs.device)
                    ] += (sign * scale * proj).to(hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                cur = hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    pos_t.to(hs.device)
                ].float()
                a = axis_t.to(hs.device)
                proj = (cur @ a)[:, None] * a[None, :]
                hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    pos_t.to(hs.device)
                ] += (sign * scale * proj).to(hs.dtype)
                return hs

            handle = layers[module_index].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**batch, return_dict=True, use_cache=False)
        if handle is not None:
            handle.remove()
        logits = out.logits[
            torch.arange(out.logits.shape[0], device=out.logits.device),
            pos.to(out.logits.device)
        ]
        outs.append(logits.float().cpu().numpy().astype(np.float32))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def stepwise_trace_with_deltas(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    layer_id: int,
    deltas: np.ndarray | None,
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Stepwise trace with a batch of deltas applied at the intervention layer.

    deltas: shape [n_prompts, d_model] or None (clean)
    """
    prompts = [x["prompt"] for x in examples]
    cur = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)
    generated: list[list[int]] = [[] for _ in prompts]
    step_rows = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))

    for step in range(steps):
        all_logits = []
        all_next = []
        input_ids = cur["input_ids"]
        attn = cur["attention_mask"]
        for start in range(0, input_ids.shape[0], batch_size):
            sub_ids = input_ids[start:start + batch_size]
            sub_attn = attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1
            handle = None
            if deltas is not None:
                delta_np = deltas[start:start + sub_ids.shape[0]]
                delta_t = torch.tensor(delta_np, device=device, dtype=torch.bfloat16)
                pos_t = pos.to(device)

                def hook(_module, _inp, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[
                            torch.arange(hs.shape[0], device=hs.device),
                            pos_t.to(hs.device)
                        ] += delta_t.to(hs.device, hs.dtype)
                        return (hs,) + output[1:]
                    hs = output.clone()
                    hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        pos_t.to(hs.device)
                    ] += delta_t.to(hs.device, hs.dtype)
                    return hs

                handle = layers[module_index].register_forward_hook(hook)
            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )
            if handle is not None:
                handle.remove()
            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ]
            next_ids = logits.argmax(dim=1)
            all_logits.append(logits.detach())
            all_next.append(next_ids.detach())
            del out
        logits_step = torch.cat(all_logits, dim=0)
        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            generated[i].append(int(tid))
        cat = max_for_ids(logits_step, token_groups["category"])
        comp = max_for_ids(logits_step, token_groups["competitor_category"])
        punct = max_for_ids(logits_step, token_groups["punctuation"])
        generic = max_for_ids(logits_step, token_groups["generic"])
        obj = max_for_ids(logits_step, token_groups["object_copy"])
        step_rows.append({
            "step": step + 1,
            "category_mean": float(cat.mean().item()),
            "competitor_mean": float(comp.mean().item()),
            "punctuation_mean": float(punct.mean().item()),
            "generic_mean": float(generic.mean().item()),
            "object_copy_mean": float(obj.mean().item()),
            "category_vs_competitor": float((cat - comp).mean().item()),
            "category_vs_punctuation": float((cat - punct).mean().item()),
            "category_vs_generic": float((cat - generic).mean().item()),
            "category_vs_object_copy": float((cat - obj).mean().item()),
            "category_top1_rate": float(
                torch.isin(
                    next_step,
                    torch.tensor(token_groups["category"], device=next_step.device)
                ).float().mean().item()
            ) if token_groups["category"] else 0.0,
            "punctuation_top1_rate": float(
                torch.isin(
                    next_step,
                    torch.tensor(token_groups["punctuation"], device=next_step.device)
                ).float().mean().item()
            ) if token_groups["punctuation"] else 0.0,
            "generic_top1_rate": float(
                torch.isin(
                    next_step,
                    torch.tensor(token_groups["generic"], device=next_step.device)
                ).float().mean().item()
            ) if token_groups["generic"] else 0.0,
            "object_copy_top1_rate": float(
                torch.isin(
                    next_step,
                    torch.tensor(token_groups["object_copy"], device=next_step.device)
                ).float().mean().item()
            ) if token_groups["object_copy"] else 0.0,
            "top_tokens": top_token_counts(
                tokenizer, next_step.detach().cpu().tolist(), 8
            ),
        })
        input_ids = torch.cat([input_ids, next_step[:, None].to(input_ids.device)], dim=1)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)], dim=1
        )
        cur = {"input_ids": input_ids, "attention_mask": attn}
        del logits_step
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    return {
        "step_metrics": step_rows,
        "generated_samples": decoded[:10],
        "category_hit_rate": float(np.mean([
            contains_category(x, examples[i]["cat"])
            for i, x in enumerate(decoded)
        ])),
    }


def contains_category(text: str, cat: str) -> float:
    return 1.0 if cat.lower() in text.lower() else 0.0


def top_token_counts(tokenizer: Any, ids: list[int], k: int) -> list[list[Any]]:
    counts: dict[str, int] = {}
    for tid in ids:
        tok = tokenizer.decode([int(tid)])
        counts[tok] = counts.get(tok, 0) + 1
    return [[a, b] for a, b in sorted(counts.items(), key=lambda x: -x[1])[:k]]


def find_surface_directions(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    gen_examples: list[dict[str, Any]],
    hidden_layers: list[int],
    bases: dict[int, np.ndarray],
    token_groups: dict[str, list[int]],
    batch_size: int,
    max_length: int,
    scale: float,
    candidate_random_axes: int,
) -> dict[str, Any]:
    """Find semantic support/release axes and surface suppression axes."""
    prompts = [x["prompt"] for x in gen_examples]
    clean_logits_cache: dict[int, np.ndarray] = {}
    candidates = []

    for layer_id in hidden_layers:
        clean_logits = logits_with_axis_condition(
            model, tokenizer, device, layers, prompts, layer_id, None, "clean",
            scale, batch_size, max_length
        )
        clean_logits_cache[layer_id] = clean_logits

        cat_mean = mean_for_ids_np(clean_logits, token_groups["category"])
        comp_mean = mean_for_ids_np(clean_logits, token_groups["competitor_category"])
        punct_mean = mean_for_ids_np(clean_logits, token_groups["punctuation"])
        generic_mean = mean_for_ids_np(clean_logits, token_groups["generic"])
        obj_mean = mean_for_ids_np(clean_logits, token_groups["object_copy"])

        base_D = cat_mean - comp_mean
        base_cat_punct = cat_mean - punct_mean
        base_cat_generic = cat_mean - generic_mean
        base_cat_obj = cat_mean - obj_mean

        axes = make_candidate_axes(bases[layer_id], 51100 + layer_id, candidate_random_axes)
        for ax in axes:
            logits = logits_with_axis_condition(
                model, tokenizer, device, layers, prompts, layer_id, ax["vec"],
                "remove", scale, batch_size, max_length
            )
            d_cat = mean_for_ids_np(logits, token_groups["category"])
            d_comp = mean_for_ids_np(logits, token_groups["competitor_category"])
            d_punct = mean_for_ids_np(logits, token_groups["punctuation"])
            d_generic = mean_for_ids_np(logits, token_groups["generic"])
            d_obj = mean_for_ids_np(logits, token_groups["object_copy"])

            d_D = d_cat - d_comp
            d_cat_punct = d_cat - d_punct
            d_cat_generic = d_cat - d_generic
            d_cat_obj = d_cat - d_obj

            candidates.append({
                "layer": layer_id,
                "name": ax["name"],
                "axis": ax["vec"].astype(np.float32),
                "delta_D": float(np.mean(d_D - base_D)),
                "delta_cat_punct": float(np.mean(d_cat_punct - base_cat_punct)),
                "delta_cat_generic": float(np.mean(d_cat_generic - base_cat_generic)),
                "delta_cat_obj": float(np.mean(d_cat_obj - base_cat_obj)),
            })

    # Semantic axes: best for category-vs-competitor margin
    support = min(candidates, key=lambda x: x["delta_D"])
    release = max(candidates, key=lambda x: x["delta_D"])

    # Surface suppression axes:
    # - punct_suppressor: axes that lower punct relative to category (raise cat-punct)
    punct_suppressor = max(candidates, key=lambda x: x["delta_cat_punct"])
    # - generic_suppressor: axes that lower generic relative to category
    generic_suppressor = max(candidates, key=lambda x: x["delta_cat_generic"])
    # - obj_suppressor: axes that lower object-copy relative to category
    obj_suppressor = max(candidates, key=lambda x: x["delta_cat_obj"])

    return {
        "support": support,
        "release": release,
        "punct_suppressor": punct_suppressor,
        "generic_suppressor": generic_suppressor,
        "obj_suppressor": obj_suppressor,
        "candidate_count": len(candidates),
        "clean_logits_cache": clean_logits_cache,
    }


def run_category(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    cat: str,
    hidden_layers: list[int],
    args: argparse.Namespace,
    W_U: np.ndarray,
    g: np.ndarray,
    cat_meta: dict[str, Any],
) -> dict[str, Any]:
    """Run all experiments for a single category."""
    log(f"  Building examples for {cat}...")
    train_ex, _ = build_examples(cat, args.train_objects, args.test_objects)
    rich_train = [x["rich"] for x in train_ex]
    neutral_train = [x["neutral"] for x in train_ex]
    gen_examples = build_generation_examples(cat, args.train_objects, args.test_objects)
    q_hat = cat_meta[cat]["q_hat"]

    # Build Phi_perp SVD basis
    log(f"  Collecting hidden states for {cat}...")
    train_r = batched_hidden(
        model, tokenizer, device, rich_train, hidden_layers,
        args.batch_size, args.max_length
    )
    train_n = batched_hidden(
        model, tokenizer, device, neutral_train, hidden_layers,
        args.batch_size, args.max_length
    )
    bases = {}
    for layer_id in hidden_layers:
        phi_train = train_r[layer_id] - train_n[layer_id]
        para_train = (phi_train @ q_hat)[:, None] * q_hat[None, :]
        perp_train = (phi_train - para_train).astype(np.float32)
        basis, _sv, _vr = svd_basis(perp_train, args.rank)
        bases[layer_id] = basis

    # Token groups
    objects = [x["obj"] for x in gen_examples]
    competitor_cats = [c for c in CATEGORIES if c != cat]
    token_groups = build_token_groups(tokenizer, cat, objects, competitor_cats)

    # Find axes
    log(f"  Finding semantic + surface axes for {cat}...")
    chosen = find_surface_directions(
        model, tokenizer, device, layers, gen_examples, hidden_layers, bases,
        token_groups, args.batch_size, args.max_length, args.scale,
        args.candidate_random_axes
    )
    support = chosen["support"]
    release = chosen["release"]
    punct_supp = chosen["punct_suppressor"]
    generic_supp = chosen["generic_suppressor"]
    obj_supp = chosen["obj_suppressor"]

    log(
        f"  {cat}: support L{support['layer']} ΔD={support['delta_D']:+.3f}; "
        f"release L{release['layer']} ΔD={release['delta_D']:+.3f}; "
        f"punct_supp Δcat-punct={punct_supp['delta_cat_punct']:+.3f}; "
        f"generic_supp Δcat-gen={generic_supp['delta_cat_generic']:+.3f}; "
        f"obj_supp Δcat-obj={obj_supp['delta_cat_obj']:+.3f}"
    )

    # ===== Exp1: Baseline surface competition profiling =====
    log(f"  Exp1: Baseline profiling for {cat}...")
    clean_trace = stepwise_trace_with_deltas(
        model, tokenizer, device, layers, gen_examples,
        hidden_layers[-1], None, token_groups, args.steps,
        args.batch_size, args.max_length
    )

    # ===== Exp2+3: Combined interventions =====
    # Build delta vectors for each intervention type
    prompts = [x["prompt"] for x in gen_examples]
    intervention_results = {}

    # Define intervention specs
    intervention_specs = [
        # Semantic only
        ("add_support", support["layer"], support["axis"], "add_support"),
        ("remove_release", release["layer"], release["axis"], "remove_release"),
        # Surface only
        ("add_punct_supp", punct_supp["layer"], punct_supp["axis"], "add_punct_supp"),
        ("add_generic_supp", generic_supp["layer"], generic_supp["axis"], "add_generic_supp"),
        ("add_obj_supp", obj_supp["layer"], obj_supp["axis"], "add_obj_supp"),
    ]

    # Compute deltas for each single intervention
    single_deltas = {}
    for name, layer_id, axis, mode in intervention_specs:
        # Get hidden states at intervention layer for computing projection
        test_r = batched_hidden(
            model, tokenizer, device, prompts, [layer_id],
            args.batch_size, args.max_length
        )
        h_test = test_r[layer_id]
        if mode.startswith("add_"):
            # Add: h' = h + scale * proj(h, axis) * axis
            coeff = h_test @ axis
            proj = np.outer(coeff, axis)
            delta = (args.scale * proj).astype(np.float32)
        elif mode.startswith("remove_"):
            # Remove: h' = h - scale * proj(h, axis) * axis
            coeff = h_test @ axis
            proj = np.outer(coeff, axis)
            delta = (-args.scale * proj).astype(np.float32)
        else:
            delta = None
        single_deltas[name] = (layer_id, delta)
        del test_r
        gc.collect()
        torch.cuda.empty_cache()

    # Combined deltas: best semantic + best surface
    # We try: add_support + add_punct_supp, add_support + add_generic_supp, etc.
    # Also: remove_release + add_punct_supp, etc.
    combined_specs = []

    # Best semantic + each surface suppression
    best_semantic_name = "add_support" if abs(support["delta_D"]) > abs(release["delta_D"]) else "remove_release"

    for surface_name in ["add_punct_supp", "add_generic_supp", "add_obj_supp"]:
        combo_name = f"{best_semantic_name}+{surface_name}"
        combined_specs.append((combo_name, best_semantic_name, surface_name))

    # Also try remove_release + surface
    if best_semantic_name != "remove_release":
        for surface_name in ["add_punct_supp", "add_generic_supp", "add_obj_supp"]:
            combo_name = f"remove_release+{surface_name}"
            combined_specs.append((combo_name, "remove_release", surface_name))

    # Run all interventions (single + combined)
    all_interventions = list(single_deltas.keys()) + [s[0] for s in combined_specs]

    for interv_name in all_interventions:
        log(f"  Running intervention: {interv_name}...")
        if interv_name in single_deltas:
            layer_id, delta = single_deltas[interv_name]
            trace = stepwise_trace_with_deltas(
                model, tokenizer, device, layers, gen_examples,
                layer_id, delta, token_groups, args.steps,
                args.batch_size, args.max_length
            )
        else:
            # Combined: find the spec
            spec = None
            for s in combined_specs:
                if s[0] == interv_name:
                    spec = s
                    break
            if spec is None:
                continue
            _, sem_name, surf_name = spec
            sem_layer, sem_delta = single_deltas[sem_name]
            surf_layer, surf_delta = single_deltas[surf_name]

            # If same layer, combine deltas directly
            if sem_layer == surf_layer:
                combined_delta = (sem_delta + surf_delta).astype(np.float32)
                trace = stepwise_trace_with_deltas(
                    model, tokenizer, device, layers, gen_examples,
                    sem_layer, combined_delta, token_groups, args.steps,
                    args.batch_size, args.max_length
                )
            else:
                # Different layers: need two hooks
                trace = stepwise_trace_with_deltas_two_layers(
                    model, tokenizer, device, layers, gen_examples,
                    sem_layer, sem_delta, surf_layer, surf_delta,
                    token_groups, args.steps, args.batch_size, args.max_length
                )

        intervention_results[interv_name] = trace

    # ===== Exp4: Bottleneck classification =====
    clean_metrics = clean_trace["step_metrics"]
    bottleneck_info = classify_bottleneck(clean_metrics, token_groups)

    cat_out = {
        "n_generation_prompts": len(gen_examples),
        "chosen_axes": {
            "support": {k: v for k, v in support.items() if k != "axis"},
            "release": {k: v for k, v in release.items() if k != "axis"},
            "punct_suppressor": {k: v for k, v in punct_supp.items() if k != "axis"},
            "generic_suppressor": {k: v for k, v in generic_supp.items() if k != "axis"},
            "obj_suppressor": {k: v for k, v in obj_supp.items() if k != "axis"},
            "candidate_count": chosen["candidate_count"],
        },
        "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
        "clean_trace": clean_trace,
        "interventions": intervention_results,
        "bottleneck": bottleneck_info,
    }

    del train_r, train_n
    gc.collect()
    torch.cuda.empty_cache()

    return cat_out


def stepwise_trace_with_deltas_two_layers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    layer_id_1: int,
    delta_1: np.ndarray,
    layer_id_2: int,
    delta_2: np.ndarray,
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Stepwise trace with deltas applied at two different layers."""
    prompts = [x["prompt"] for x in examples]
    cur = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)
    generated: list[list[int]] = [[] for _ in prompts]
    step_rows = []
    module_index_1 = max(0, min(layer_id_1 - 1, len(layers) - 1))
    module_index_2 = max(0, min(layer_id_2 - 1, len(layers) - 1))

    for step in range(steps):
        all_logits = []
        all_next = []
        input_ids = cur["input_ids"]
        attn = cur["attention_mask"]
        for start in range(0, input_ids.shape[0], batch_size):
            sub_ids = input_ids[start:start + batch_size]
            sub_attn = attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1

            handles = []
            # Hook 1
            d1_np = delta_1[start:start + sub_ids.shape[0]]
            d1_t = torch.tensor(d1_np, device=device, dtype=torch.bfloat16)
            pos_t = pos.to(device)

            def hook1(_module, _inp, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        pos_t.to(hs.device)
                    ] += d1_t.to(hs.device, hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    pos_t.to(hs.device)
                ] += d1_t.to(hs.device, hs.dtype)
                return hs

            handles.append(layers[module_index_1].register_forward_hook(hook1))

            # Hook 2
            d2_np = delta_2[start:start + sub_ids.shape[0]]
            d2_t = torch.tensor(d2_np, device=device, dtype=torch.bfloat16)

            def hook2(_module, _inp, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        pos_t.to(hs.device)
                    ] += d2_t.to(hs.device, hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                hs[
                    torch.arange(hs.shape[0], device=hs.device),
                    pos_t.to(hs.device)
                ] += d2_t.to(hs.device, hs.dtype)
                return hs

            if module_index_2 != module_index_1:
                handles.append(layers[module_index_2].register_forward_hook(hook2))

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )
            for h in handles:
                h.remove()

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ]
            next_ids = logits.argmax(dim=1)
            all_logits.append(logits.detach())
            all_next.append(next_ids.detach())
            del out

        logits_step = torch.cat(all_logits, dim=0)
        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            generated[i].append(int(tid))

        cat = max_for_ids(logits_step, token_groups["category"])
        comp = max_for_ids(logits_step, token_groups["competitor_category"])
        punct = max_for_ids(logits_step, token_groups["punctuation"])
        generic = max_for_ids(logits_step, token_groups["generic"])
        obj = max_for_ids(logits_step, token_groups["object_copy"])

        step_rows.append({
            "step": step + 1,
            "category_mean": float(cat.mean().item()),
            "competitor_mean": float(comp.mean().item()),
            "punctuation_mean": float(punct.mean().item()),
            "generic_mean": float(generic.mean().item()),
            "object_copy_mean": float(obj.mean().item()),
            "category_vs_competitor": float((cat - comp).mean().item()),
            "category_vs_punctuation": float((cat - punct).mean().item()),
            "category_vs_generic": float((cat - generic).mean().item()),
            "category_vs_object_copy": float((cat - obj).mean().item()),
            "category_top1_rate": float(
                torch.isin(
                    next_step,
                    torch.tensor(token_groups["category"], device=next_step.device)
                ).float().mean().item()
            ) if token_groups["category"] else 0.0,
        })

        input_ids = torch.cat(
            [input_ids, next_step[:, None].to(input_ids.device)], dim=1
        )
        attn = torch.cat(
            [attn, torch.ones(
                (attn.shape[0], 1), device=attn.device, dtype=attn.dtype
            )], dim=1
        )
        cur = {"input_ids": input_ids, "attention_mask": attn}
        del logits_step
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    return {
        "step_metrics": step_rows,
        "generated_samples": decoded[:10],
        "category_hit_rate": float(np.mean([
            contains_category(x, examples[i]["cat"])
            for i, x in enumerate(decoded)
        ])),
    }


def classify_bottleneck(
    clean_metrics: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
) -> dict[str, Any]:
    """Classify the dominant bottleneck type based on baseline generation."""
    # Use step 1 metrics for classification
    if not clean_metrics:
        return {"type": "unknown", "details": {}}

    m = clean_metrics[0]  # Step 1
    cat_comp = m.get("category_vs_competitor", 0)
    cat_punct = m.get("category_vs_punctuation", 0)
    cat_generic = m.get("category_vs_generic", 0)
    cat_obj = m.get("category_vs_object_copy", 0)
    cat_top1 = m.get("category_top1_rate", 0)

    # Identify which surface competitor is strongest (most negative margin)
    margins = {
        "punctuation": cat_punct,
        "generic": cat_generic,
        "object_copy": cat_obj,
        "competitor": cat_comp,
    }
    worst_surface = min(margins, key=lambda k: margins[k])

    if cat_comp < 0:
        btype = "semantic"
    elif cat_punct < cat_generic and cat_punct < cat_obj:
        btype = "punctuation"
    elif cat_generic < cat_punct and cat_generic < cat_obj:
        btype = "generic"
    elif cat_obj < cat_punct and cat_obj < cat_generic:
        btype = "object_copy"
    else:
        btype = "mixed"

    return {
        "type": btype,
        "worst_surface": worst_surface,
        "details": {
            "category_vs_competitor": round(cat_comp, 3),
            "category_vs_punctuation": round(cat_punct, 3),
            "category_vs_generic": round(cat_generic, 3),
            "category_vs_object_copy": round(cat_obj, 3),
            "category_top1_rate": round(cat_top1, 3),
        },
        "step1_top_tokens": m.get("top_tokens", []),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        L, d = info.n_layers, info.d_model
        hidden_layers = sorted(set(
            [max(1, min(L, int(x))) for x in [L // 2, 3 * L // 4, L - 3]]
        ))
        W_U = get_W_U(model, args.model).astype(np.float32)
        g = get_norm_g(model, args.model)
        if g is None:
            raise RuntimeError("cannot read final norm gain")
        cat_meta = build_cat_meta(tokenizer, W_U, g.astype(np.float32), d)
        categories = (
            args.categories.split(",")
            if args.categories
            else FOCUS_CATEGORIES[args.model]
        )
        log(f"{args.model}: L={L}, d={d}, categories={categories}, layers={hidden_layers}")

        result = {
            "phase": 511,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "templates": [x[0] for x in GEN_TEMPLATES],
            "basis_templates": [x[0] for x in RICH_TEMPLATES],
            "layers": hidden_layers,
            "rank": args.rank,
            "steps": args.steps,
            "scale": args.scale,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            cat_out = run_category(
                model, tokenizer, device, layers, cat, hidden_layers,
                args, W_U, g, cat_meta
            )
            result["category_results"][cat] = cat_out

            # Print summary
            bn = cat_out["bottleneck"]
            clean_hit = cat_out["clean_trace"]["category_hit_rate"]
            log(
                f"  {cat} bottleneck={bn['type']}, "
                f"clean_hit={clean_hit:.3f}, "
                f"cat-punct={bn['details']['category_vs_punctuation']:.3f}, "
                f"cat-generic={bn['details']['category_vs_generic']:.3f}"
            )
            # Print intervention hit deltas
            for iname, itrace in cat_out["interventions"].items():
                hit_delta = itrace["category_hit_rate"] - clean_hit
                if hit_delta > 0.01 or hit_delta < -0.01:
                    log(f"    {iname}: hit_delta={hit_delta:+.3f}")

        return result
    finally:
        release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=20)
    parser.add_argument("--test-objects", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--candidate-random-axes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase511_{args.model}_surface_gate_repair.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
