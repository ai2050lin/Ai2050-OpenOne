#!/usr/bin/env python3
"""
Phase 512: Surface Gate Sufficiency and Residual Source Tracing
================================================================
Phase 511 proved category margin can be changed but hit doesn't follow.
This is because Phi_perp candidate axes cannot separate semantic support
from surface suppression — they converge to the same axis.

Phase 512 takes a fundamentally different approach:
  Exp1: Direct logit patch — suppress punctuation/generic/object-copy logits
        at the final output to test if surface competition is the bottleneck.
  Exp2: Per-sample bottleneck classification (P/G/O/S/M).
  Exp3: Surface readout direction layer tracing — when does each surface
        competitor become dominant?
  Exp4: Cross-layer combination — middle-layer semantic + late-layer surface.
  Exp5: Action category with natural templates.

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


OUT_ROOT = Path("results/glm5_phase512_surface_gate_sufficiency")

# Generation templates that don't contain the answer category
GEN_TEMPLATES = [
    ("category_of", "The {obj} belongs to the category of"),
    ("taxonomy_as", "In taxonomy, {obj} is classified as"),
    ("classify_colon", "Classify {obj}:"),
]

# More natural action templates (Exp5)
ACTION_TEMPLATES = [
    ("doing_what", "The person is doing"),
    ("can_do", "This is an example of"),
    ("action_type", "The action of running is a type of"),
]

PUNCTUATION_TOKENS = [".", ",", ":", ";", "!", "?", "\n", " the", " a", " an", " is", " are"]
GENERIC_TOKENS = [" thing", " item", " type", " kind", " object", " entity", " one", " it", " that", " something",
                  " of", " which", " that", " and"]
OBJECT_COPY_PREFIX = ""


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
    cat: str, train_n: int, test_n: int, templates: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if templates is None:
        templates = GEN_TEMPLATES
    objs = CATEGORIES[cat]["objects"][train_n: train_n + test_n]
    rows = []
    for obj in objs:
        for tid, (name, tpl) in enumerate(templates):
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


def contains_category(text: str, cat: str) -> float:
    return 1.0 if cat.lower() in text.lower() else 0.0


def top_token_counts(tokenizer: Any, ids: list[int], k: int) -> list[list[Any]]:
    counts: dict[str, int] = {}
    for tid in ids:
        tok = tokenizer.decode([int(tid)])
        counts[tok] = counts.get(tok, 0) + 1
    return [[a, b] for a, b in sorted(counts.items(), key=lambda x: -x[1])[:k]]


# ============================================================================
# Exp1: Direct logit patch — the most critical experiment
# ============================================================================
def direct_logit_patch_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
    logit_suppression_configs: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """
    Stepwise trace with direct logit patching.

    For each config, we run greedy decoding and at each step:
    - Get the logits at the last position
    - Suppress specified token groups by subtracting a value
    - Pick the argmax token

    logit_suppression_configs: {
        config_name: {
            "group_name": suppression_value,  # e.g. {"punctuation": 5.0, "generic": 5.0}
        }
    }

    Returns: {config_name: {step_metrics, generated_samples, category_hit_rate, per_sample_bottleneck}}
    """
    prompts = [x["prompt"] for x in examples]
    n = len(prompts)
    results = {}

    for config_name, suppress_cfg in logit_suppression_configs.items():
        log(f"    Logit patch config: {config_name} = {suppress_cfg}")
        cur = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        generated: list[list[int]] = [[] for _ in prompts]
        step_rows = []
        per_sample_bottlenecks = []

        for step in range(steps):
            all_logits = []
            all_next = []
            input_ids = cur["input_ids"]
            attn = cur["attention_mask"]

            for start in range(0, input_ids.shape[0], batch_size):
                sub_ids = input_ids[start:start + batch_size]
                sub_attn = attn[start:start + batch_size]
                pos = sub_attn.sum(dim=1) - 1

                with torch.no_grad():
                    out = model(
                        input_ids=sub_ids, attention_mask=sub_attn,
                        return_dict=True, use_cache=False
                    )

                logits = out.logits[
                    torch.arange(out.logits.shape[0], device=out.logits.device),
                    pos.to(out.logits.device)
                ].clone().float()

                # Apply logit suppression
                for group_name, suppress_val in suppress_cfg.items():
                    group_ids = token_groups.get(group_name, [])
                    for tid in group_ids:
                        if 0 <= tid < logits.shape[-1]:
                            logits[:, tid] -= suppress_val

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
                "top_tokens": top_token_counts(
                    tokenizer, next_step.detach().cpu().tolist(), 8
                ),
            })

            # Per-sample bottleneck classification at step 1
            if step == 0:
                for i in range(n):
                    cat_logit = max_for_ids(logits_step[i:i+1], token_groups["category"]).item()
                    comp_logit = max_for_ids(logits_step[i:i+1], token_groups["competitor_category"]).item()
                    punct_logit = max_for_ids(logits_step[i:i+1], token_groups["punctuation"]).item()
                    gen_logit = max_for_ids(logits_step[i:i+1], token_groups["generic"]).item()
                    obj_logit = max_for_ids(logits_step[i:i+1], token_groups["object_copy"]).item()
                    margins = {
                        "competitor": cat_logit - comp_logit,
                        "punctuation": cat_logit - punct_logit,
                        "generic": cat_logit - gen_logit,
                        "object_copy": cat_logit - obj_logit,
                    }
                    worst = min(margins, key=lambda k: margins[k])
                    btype = worst if margins[worst] < 0 else "category_wins"
                    per_sample_bottlenecks.append({
                        "bottleneck_type": btype,
                        "margins": {k: round(v, 3) for k, v in margins.items()},
                    })

            input_ids = torch.cat([input_ids, next_step[:, None].to(input_ids.device)], dim=1)
            attn = torch.cat(
                [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)], dim=1
            )
            cur = {"input_ids": input_ids, "attention_mask": attn}
            del logits_step
            torch.cuda.empty_cache()

        decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
        results[config_name] = {
            "step_metrics": step_rows,
            "generated_samples": decoded[:10],
            "category_hit_rate": float(np.mean([
                contains_category(x, examples[i]["cat"])
                for i, x in enumerate(decoded)
            ])),
            "per_sample_bottleneck": per_sample_bottlenecks,
        }

    return results


# ============================================================================
# Exp3: Surface readout direction layer tracing
# ============================================================================
def surface_readout_layer_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    all_layer_ids: list[int],
    token_groups: dict[str, list[int]],
    W_U: np.ndarray,
    g: np.ndarray,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """
    For each layer, compute <h_l, q_surface> for each surface readout direction.
    q_punct = g ⊙ (W_U[punct_tokens] - W_U[category_tokens])
    q_generic = g ⊙ (W_U[generic_tokens] - W_U[category_tokens])
    q_obj = g ⊙ (W_U[obj_tokens] - W_U[category_tokens])

    This traces when each surface competitor becomes dominant.
    """
    prompts = [x["prompt"] for x in examples]
    cat = examples[0]["cat"]

    # Construct readout directions
    cat_ids = token_groups["category"]
    punct_ids = token_groups["punctuation"]
    generic_ids = token_groups["generic"]
    obj_ids = token_groups["object_copy"]

    def make_q_direction(target_ids: list[int], ref_ids: list[int]) -> np.ndarray:
        """q = g ⊙ (mean(W_U[target]) - mean(W_U[ref]))"""
        # W_U shape: [vocab_size, d_model]
        vocab_size, d_model = W_U.shape
        valid_t = [i for i in target_ids if 0 <= i < vocab_size]
        valid_r = [i for i in ref_ids if 0 <= i < vocab_size]
        if not valid_t or not valid_r:
            return np.zeros(d_model, dtype=np.float32)
        w_target = W_U[valid_t, :].mean(axis=0)
        w_ref = W_U[valid_r, :].mean(axis=0)
        q = g.astype(np.float32) * (w_target - w_ref)
        n = np.linalg.norm(q)
        if n > 1e-7:
            q /= n
        return q.astype(np.float32)  # shape: [d_model]

    q_punct = make_q_direction(punct_ids, cat_ids)
    q_generic = make_q_direction(generic_ids, cat_ids)
    q_obj = make_q_direction(obj_ids, cat_ids)
    q_comp = make_q_direction(token_groups["competitor_category"], cat_ids)

    # Collect hidden states at all layers
    log(f"    Collecting hidden states at {len(all_layer_ids)} layers for surface trace...")
    all_hidden = batched_hidden(
        model, tokenizer, device, prompts, all_layer_ids,
        batch_size, max_length
    )

    layer_results = []
    for layer_id in all_layer_ids:
        h = all_hidden[layer_id]  # [n, d_model]

        proj_punct = (h @ q_punct).mean()
        proj_generic = (h @ q_generic).mean()
        proj_obj = (h @ q_obj).mean()
        proj_comp = (h @ q_comp).mean()

        # Also compute direct logit projections
        # h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        # logits_approx = h_norm @ W_U  # This would be too expensive, skip

        layer_results.append({
            "layer": layer_id,
            "q_punct_projection_mean": float(proj_punct),
            "q_generic_projection_mean": float(proj_generic),
            "q_obj_projection_mean": float(proj_obj),
            "q_comp_projection_mean": float(proj_comp),
            "q_punct_projection_std": float((h @ q_punct).std()),
            "q_generic_projection_std": float((h @ q_generic).std()),
            "q_obj_projection_std": float((h @ q_obj).std()),
        })

    del all_hidden
    gc.collect()
    torch.cuda.empty_cache()

    return layer_results, {
        "q_punct_norm": float(np.linalg.norm(q_punct)),
        "q_generic_norm": float(np.linalg.norm(q_generic)),
        "q_obj_norm": float(np.linalg.norm(q_obj)),
        "q_comp_norm": float(np.linalg.norm(q_comp)),
    }


# ============================================================================
# Exp4: Cross-layer combination — semantic mid + surface late
# ============================================================================
def stepwise_trace_two_layer_intervention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    sem_layer: int,
    sem_axis: np.ndarray,
    sem_mode: str,  # "remove" or "add"
    surf_layer: int,
    surf_axis: np.ndarray,
    surf_mode: str,  # "remove" or "add"
    scale: float,
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Stepwise trace with two different layer interventions."""
    prompts = [x["prompt"] for x in examples]
    cur = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)
    generated: list[list[int]] = [[] for _ in prompts]
    step_rows = []
    sem_module = max(0, min(sem_layer - 1, len(layers) - 1))
    surf_module = max(0, min(surf_layer - 1, len(layers) - 1))

    sem_axis_t = torch.tensor(sem_axis, device=device, dtype=torch.float32)
    surf_axis_t = torch.tensor(surf_axis, device=device, dtype=torch.float32)

    for step in range(steps):
        all_logits = []
        all_next = []
        input_ids = cur["input_ids"]
        attn = cur["attention_mask"]

        for start in range(0, input_ids.shape[0], batch_size):
            sub_ids = input_ids[start:start + batch_size]
            sub_attn = attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1
            pos_t = pos.to(device)

            handles = []

            # Semantic hook
            def make_sem_hook(sa, sm, pt):
                def hook(_module, _inp, output):
                    sign = -1.0 if sm.startswith("remove") else 1.0
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)].float()
                        proj = (cur_h @ sa)[:, None] * sa[None, :]
                        hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                        return (hs,) + output[1:]
                    hs = output.clone()
                    cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)].float()
                    proj = (cur_h @ sa)[:, None] * sa[None, :]
                    hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                    return hs
                return hook

            handles.append(layers[sem_module].register_forward_hook(
                make_sem_hook(sem_axis_t.to(device), sem_mode, pos_t)
            ))

            # Surface hook (if different layer)
            if surf_module != sem_module:
                def make_surf_hook(sa, sm, pt):
                    def hook(_module, _inp, output):
                        sign = -1.0 if sm.startswith("remove") else 1.0
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)].float()
                            proj = (cur_h @ sa)[:, None] * sa[None, :]
                            hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                            return (hs,) + output[1:]
                        hs = output.clone()
                        cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)].float()
                        proj = (cur_h @ sa)[:, None] * sa[None, :]
                        hs[torch.arange(hs.shape[0], device=hs.device), pt.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                        return hs
                    return hook

                handles.append(layers[surf_module].register_forward_hook(
                    make_surf_hook(surf_axis_t.to(device), surf_mode, pos_t)
                ))

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


def find_best_axes(
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
    """Find semantic support/release axes across layers."""
    prompts = [x["prompt"] for x in gen_examples]
    candidates = []

    for layer_id in hidden_layers:
        clean_logits = logits_with_axis_condition(
            model, tokenizer, device, layers, prompts, layer_id, None, "clean",
            scale, batch_size, max_length
        )
        cat_mean = mean_for_ids_np(clean_logits, token_groups["category"])
        comp_mean = mean_for_ids_np(clean_logits, token_groups["competitor_category"])
        base_D = cat_mean - comp_mean

        axes = make_candidate_axes(bases[layer_id], 51200 + layer_id, candidate_random_axes)
        for ax in axes:
            logits = logits_with_axis_condition(
                model, tokenizer, device, layers, prompts, layer_id, ax["vec"],
                "remove", scale, batch_size, max_length
            )
            d_cat = mean_for_ids_np(logits, token_groups["category"])
            d_comp = mean_for_ids_np(logits, token_groups["competitor_category"])
            d_D = d_cat - d_comp

            candidates.append({
                "layer": layer_id,
                "name": ax["name"],
                "axis": ax["vec"].astype(np.float32),
                "delta_D": float(np.mean(d_D - base_D)),
            })

    support = min(candidates, key=lambda x: x["delta_D"])
    release = max(candidates, key=lambda x: x["delta_D"])

    return {"support": support, "release": release, "candidate_count": len(candidates)}


# ============================================================================
# Main runner
# ============================================================================
def run_category(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    cat: str,
    hidden_layers: list[int],
    all_layer_ids: list[int],
    args: argparse.Namespace,
    W_U: np.ndarray,
    g: np.ndarray,
    cat_meta: dict[str, Any],
) -> dict[str, Any]:
    """Run all Phase 512 experiments for a single category."""
    is_action = (cat == "action")
    templates = ACTION_TEMPLATES if is_action else GEN_TEMPLATES

    log(f"  Building examples for {cat} (action={is_action})...")
    train_ex, _ = build_examples(cat, args.train_objects, args.test_objects)
    rich_train = [x["rich"] for x in train_ex]
    neutral_train = [x["neutral"] for x in train_ex]
    gen_examples = build_generation_examples(
        cat, args.train_objects, args.test_objects, templates
    )
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

    # ===== Exp1: Direct logit patch =====
    log(f"  Exp1: Direct logit patch for {cat}...")
    logit_configs = {
        "clean": {},  # No suppression
        "suppress_punct_3": {"punctuation": 3.0},
        "suppress_punct_5": {"punctuation": 5.0},
        "suppress_punct_10": {"punctuation": 10.0},
        "suppress_generic_3": {"generic": 3.0},
        "suppress_generic_5": {"generic": 5.0},
        "suppress_obj_3": {"object_copy": 3.0},
        "suppress_obj_5": {"object_copy": 5.0},
        "suppress_punct_generic_5": {"punctuation": 5.0, "generic": 5.0},
        "suppress_punct_generic_obj_5": {"punctuation": 5.0, "generic": 5.0, "object_copy": 5.0},
        "suppress_all_10": {"punctuation": 10.0, "generic": 10.0, "object_copy": 10.0, "competitor_category": 10.0},
        "oracle_surface": {},  # Will be special-cased
    }

    logit_patch_results = direct_logit_patch_trace(
        model, tokenizer, device, layers, gen_examples,
        token_groups, args.steps, args.batch_size, args.max_length,
        logit_configs
    )

    # Oracle surface: suppress everything except category to max
    # This directly tests: if we suppress ALL competition, does category win?
    log(f"  Exp1b: Oracle surface patch for {cat}...")
    oracle_configs = {
        "oracle_surface": {
            "punctuation": 20.0, "generic": 20.0, "object_copy": 20.0,
            "competitor_category": 20.0,
        },
    }
    oracle_results = direct_logit_patch_trace(
        model, tokenizer, device, layers, gen_examples,
        token_groups, args.steps, args.batch_size, args.max_length,
        oracle_configs
    )
    logit_patch_results["oracle_surface"] = oracle_results["oracle_surface"]

    # Print logit patch summary
    clean_hit = logit_patch_results["clean"]["category_hit_rate"]
    log(f"  {cat} logit patch summary (clean hit={clean_hit:.3f}):")
    for cfg_name, cfg_result in logit_patch_results.items():
        hit_delta = cfg_result["category_hit_rate"] - clean_hit
        s1 = cfg_result["step_metrics"][0] if cfg_result["step_metrics"] else {}
        log(f"    {cfg_name}: hit={cfg_result['category_hit_rate']:.3f} (delta={hit_delta:+.3f}), "
            f"cat-punct={s1.get('category_vs_punctuation', 0):.3f}")

    # ===== Exp2: Per-sample bottleneck classification =====
    log(f"  Exp2: Bottleneck classification for {cat}...")
    per_sample = logit_patch_results["clean"]["per_sample_bottleneck"]
    bottleneck_counts = {}
    for ps in per_sample:
        bt = ps["bottleneck_type"]
        bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1

    # ===== Exp3: Surface readout direction layer tracing =====
    log(f"  Exp3: Surface readout direction tracing for {cat}...")
    layer_trace, q_info = surface_readout_layer_trace(
        model, tokenizer, device, layers, gen_examples,
        all_layer_ids, token_groups, W_U, g,
        args.batch_size, args.max_length
    )

    # Find the layer where each surface competitor becomes dominant
    surf_emergence = {}
    for qtype in ["q_punct", "q_generic", "q_obj", "q_comp"]:
        key = f"{qtype}_projection_mean"
        for lr in layer_trace:
            if lr[key] > 0:
                surf_emergence[qtype] = lr["layer"]
                break

    log(f"  {cat} surface emergence: {surf_emergence}")

    # ===== Exp4: Cross-layer combination =====
    log(f"  Exp4: Cross-layer combination for {cat}...")
    chosen = find_best_axes(
        model, tokenizer, device, layers, gen_examples, hidden_layers,
        bases, token_groups, args.batch_size, args.max_length,
        args.scale, args.candidate_random_axes
    )
    support = chosen["support"]
    release = chosen["release"]
    log(f"  {cat}: support L{support['layer']} dD={support['delta_D']:+.3f}, "
        f"release L{release['layer']} dD={release['delta_D']:+.3f}")

    cross_layer_results = {}

    # Semantic intervention only (baseline for comparison)
    for axis_name, axis_info in [("support", support), ("release", release)]:
        mode = f"remove_{axis_name}" if axis_name == "release" else f"add_{axis_name}"
        log(f"    Running {mode} at L{axis_info['layer']}...")
        trace = stepwise_trace_two_layer_intervention(
            model, tokenizer, device, layers, gen_examples,
            axis_info["layer"], axis_info["axis"], mode,
            axis_info["layer"], axis_info["axis"], "clean",  # dummy second layer
            args.scale, token_groups, args.steps, args.batch_size, args.max_length
        )
        cross_layer_results[mode] = trace

    # Cross-layer: semantic at mid + surface suppression from logit patch
    # Use the deepest available layer for surface suppression
    late_layer = hidden_layers[-1]
    # Find best surface suppression axis at the late layer
    prompts = [x["prompt"] for x in gen_examples]

    # Search for surface suppression direction at late layer
    # Using a different approach: find the axis that maximally lowers punctuation
    # relative to category, orthogonalized against support/release
    late_basis = bases[late_layer]
    late_candidates = make_candidate_axes(late_basis, 51200 + late_layer * 100, 8)

    # Get clean logits at late layer
    clean_logits_late = logits_with_axis_condition(
        model, tokenizer, device, layers, prompts, late_layer, None, "clean",
        args.scale, args.batch_size, args.max_length
    )
    cat_mean_clean = mean_for_ids_np(clean_logits_late, token_groups["category"])
    punct_mean_clean = mean_for_ids_np(clean_logits_late, token_groups["punctuation"])
    generic_mean_clean = mean_for_ids_np(clean_logits_late, token_groups["generic"])
    base_cat_punct = cat_mean_clean - punct_mean_clean
    base_cat_generic = cat_mean_clean - generic_mean_clean

    best_surf_axis = None
    best_surf_delta = -1e9
    best_surf_name = ""

    for ax in late_candidates:
        logits = logits_with_axis_condition(
            model, tokenizer, device, layers, prompts, late_layer, ax["vec"],
            "remove", args.scale, args.batch_size, args.max_length
        )
        d_cat = mean_for_ids_np(logits, token_groups["category"])
        d_punct = mean_for_ids_np(logits, token_groups["punctuation"])
        d_generic = mean_for_ids_np(logits, token_groups["generic"])
        # We want axis that increases cat-punct margin AND cat-generic margin
        d_cat_punct = d_cat - d_punct
        d_cat_generic = d_cat - d_generic
        combined_delta = np.mean(d_cat_punct - base_cat_punct) + np.mean(d_cat_generic - base_cat_generic)

        if combined_delta > best_surf_delta:
            best_surf_delta = combined_delta
            best_surf_axis = ax["vec"].astype(np.float32)
            best_surf_name = ax["name"]

    log(f"  Best surface axis at L{late_layer}: {best_surf_name}, combined_delta={best_surf_delta:+.3f}")

    # Cross-layer: support at mid + surface suppression at late
    if best_surf_axis is not None:
        sem_layer = support["layer"]
        sem_axis = support["axis"]
        sem_mode = "add"

        # Same layer combination
        if sem_layer == late_layer:
            log(f"    Same layer L{sem_layer}: add_support + remove_surface...")
            # Run as single-layer combined
            trace = stepwise_trace_two_layer_intervention(
                model, tokenizer, device, layers, gen_examples,
                sem_layer, sem_axis, "add",
                sem_layer, best_surf_axis, "remove",
                args.scale, token_groups, args.steps, args.batch_size, args.max_length
            )
            cross_layer_results["add_support+remove_surface_same_layer"] = trace
        else:
            # Different layers
            log(f"    Cross-layer: add_support L{sem_layer} + remove_surface L{late_layer}...")
            trace = stepwise_trace_two_layer_intervention(
                model, tokenizer, device, layers, gen_examples,
                sem_layer, sem_axis, "add",
                late_layer, best_surf_axis, "remove",
                args.scale, token_groups, args.steps, args.batch_size, args.max_length
            )
            cross_layer_results[f"add_support_L{sem_layer}+remove_surface_L{late_layer}"] = trace

        # Also try release
        sem_layer_r = release["layer"]
        sem_axis_r = release["axis"]
        if sem_layer_r != late_layer:
            log(f"    Cross-layer: remove_release L{sem_layer_r} + remove_surface L{late_layer}...")
            trace = stepwise_trace_two_layer_intervention(
                model, tokenizer, device, layers, gen_examples,
                sem_layer_r, sem_axis_r, "remove",
                late_layer, best_surf_axis, "remove",
                args.scale, token_groups, args.steps, args.batch_size, args.max_length
            )
            cross_layer_results[f"remove_release_L{sem_layer_r}+remove_surface_L{late_layer}"] = trace
        else:
            trace = stepwise_trace_two_layer_intervention(
                model, tokenizer, device, layers, gen_examples,
                sem_layer_r, sem_axis_r, "remove",
                sem_layer_r, best_surf_axis, "remove",
                args.scale, token_groups, args.steps, args.batch_size, args.max_length
            )
            cross_layer_results["remove_release+remove_surface_same_layer"] = trace

    # ===== Exp4b: Logit patch + semantic axis =====
    log(f"  Exp4b: Logit patch + semantic axis for {cat}...")
    # This is the key test: semantic intervention + direct logit suppression
    # We combine the best semantic axis with logit-level surface suppression
    for axis_name, axis_info in [("support", support), ("release", release)]:
        mode = f"remove_{axis_name}" if axis_name == "release" else f"add_{axis_name}"
        layer_id = axis_info["layer"]
        axis = axis_info["axis"]

        # Run stepwise trace with axis intervention AND logit patching
        log(f"    Running {mode} + logit_punct_5 for {cat}...")
        trace = stepwise_trace_with_axis_and_logit_patch(
            model, tokenizer, device, layers, gen_examples,
            layer_id, axis, mode, args.scale,
            token_groups, args.steps, args.batch_size, args.max_length,
            {"punctuation": 5.0, "generic": 5.0}
        )
        cross_layer_results[f"{mode}+logit_punct_generic_5"] = trace

    # Print cross-layer summary
    for iname, itrace in cross_layer_results.items():
        hit_delta = itrace["category_hit_rate"] - clean_hit
        log(f"    {iname}: hit={itrace['category_hit_rate']:.3f} (delta={hit_delta:+.3f})")

    # ===== Assemble output =====
    cat_out = {
        "n_generation_prompts": len(gen_examples),
        "is_action": is_action,
        "templates": [t[0] for t in templates],
        "chosen_axes": {
            "support": {k: v for k, v in support.items() if k != "axis"},
            "release": {k: v for k, v in release.items() if k != "axis"},
            "best_surface": {
                "layer": late_layer,
                "name": best_surf_name,
                "combined_delta": float(best_surf_delta),
            },
        },
        "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
        # Exp1
        "logit_patch_results": logit_patch_results,
        # Exp2
        "bottleneck_counts": bottleneck_counts,
        "per_sample_bottlenecks": per_sample,
        # Exp3
        "surface_layer_trace": layer_trace,
        "surface_emergence": surf_emergence,
        "q_direction_info": q_info,
        # Exp4
        "cross_layer_results": cross_layer_results,
    }

    del train_r, train_n
    gc.collect()
    torch.cuda.empty_cache()

    return cat_out


def stepwise_trace_with_axis_and_logit_patch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    layer_id: int,
    axis: np.ndarray,
    mode: str,
    scale: float,
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
    logit_suppress: dict[str, float],
) -> dict[str, Any]:
    """Stepwise trace with hidden state axis intervention AND logit suppression."""
    prompts = [x["prompt"] for x in examples]
    cur = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length
    ).to(device)
    generated: list[list[int]] = [[] for _ in prompts]
    step_rows = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))
    axis_t = torch.tensor(axis, device=device, dtype=torch.float32)

    for step in range(steps):
        all_logits = []
        all_next = []
        input_ids = cur["input_ids"]
        attn = cur["attention_mask"]

        for start in range(0, input_ids.shape[0], batch_size):
            sub_ids = input_ids[start:start + batch_size]
            sub_attn = attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1
            pos_t = pos.to(device)

            def hook(_module, _inp, output):
                sign = -1.0 if mode.startswith("remove") else 1.0
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                    proj = (cur_h @ axis_t.to(hs.device))[:, None] * axis_t.to(hs.device)[None, :]
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                proj = (cur_h @ axis_t.to(hs.device))[:, None] * axis_t.to(hs.device)[None, :]
                hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                return hs

            handle = layers[module_index].register_forward_hook(hook)

            with torch.no_grad():
                out = model(
                    input_ids=sub_ids, attention_mask=sub_attn,
                    return_dict=True, use_cache=False
                )
            handle.remove()

            logits = out.logits[
                torch.arange(out.logits.shape[0], device=out.logits.device),
                pos.to(out.logits.device)
            ].clone().float()

            # Apply logit suppression
            for group_name, suppress_val in logit_suppress.items():
                group_ids = token_groups.get(group_name, [])
                for tid in group_ids:
                    if 0 <= tid < logits.shape[-1]:
                        logits[:, tid] -= suppress_val

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


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        L, d = info.n_layers, info.d_model
        hidden_layers = sorted(set(
            [max(1, min(L, int(x))) for x in [L // 2, 3 * L // 4, L - 3]]
        ))
        # For Exp3, trace every N-th layer
        all_layer_ids = sorted(set(
            [max(1, min(L, i)) for i in range(1, L + 1, max(1, L // 8))]
            + [L - 3, L - 1]
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
        log(f"{args.model}: L={L}, d={d}, categories={categories}")
        log(f"  hidden_layers={hidden_layers}, all_trace_layers={all_layer_ids}")

        result = {
            "phase": 512,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "hidden_layers": hidden_layers,
            "all_trace_layers": all_layer_ids,
            "rank": args.rank,
            "steps": args.steps,
            "scale": args.scale,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            cat_out = run_category(
                model, tokenizer, device, layers, cat, hidden_layers,
                all_layer_ids, args, W_U, g, cat_meta
            )
            result["category_results"][cat] = cat_out

            # Print summary
            lp = cat_out["logit_patch_results"]
            clean_hit = lp["clean"]["category_hit_rate"]
            oracle_hit = lp.get("oracle_surface", {}).get("category_hit_rate", 0)
            bn = cat_out["bottleneck_counts"]
            log(
                f"  {cat}: clean_hit={clean_hit:.3f}, "
                f"oracle_hit={oracle_hit:.3f}, "
                f"bottlenecks={bn}"
            )

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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase512_{args.model}_surface_gate_sufficiency.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
