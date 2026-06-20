#!/usr/bin/env python3
"""
Phase 556: donor specificity decomposition.

Phase555 showed that GLM4 vehicle donors are usually stronger than wrong
donors, but tool donors can still restore the all-layer sentence route. This
phase decomposes restore into category, task-interface, and generic gate
components by adding unrelated categories and shuffled donors.
"""
from __future__ import annotations

import argparse
import gc
import itertools
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
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402


OUT_ROOT = Path("results/glm5_phase556_donor_specificity_decomposition")
DEFAULT_PAIR = "vehicle_tool"
DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_sentence_completion",
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DEFAULT_CONDITIONS = [
    "baseline",
    "add_perp",
    "resid_remove_perp",
    "resid_remove_random_perp",
    "resid_donor_vehicle_add",
    "resid_donor_tool_add",
    "resid_donor_furniture_add",
    "resid_donor_animal_add",
    "resid_donor_fruit_add",
    "resid_donor_vehicle_shuffle_add",
    "resid_donor_tool_shuffle_add",
    "resid_donor_furniture_shuffle_add",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_scaffold_modes(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in parse_csv(text):
        if ":" not in item:
            raise ValueError(f"scaffold mode must be scaffold:mode, got {item}")
        a, b = item.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    return pairs


def parse_routes(text: str) -> list[dict[str, str]]:
    routes = []
    for item in parse_csv(text):
        if "<-" not in item or ":" not in item:
            raise ValueError(f"route must be recipient_scaffold:mode<-donor_scaffold, got {item}")
        left, donor_scaffold = item.split("<-", 1)
        recipient_scaffold, mode = left.split(":", 1)
        routes.append({
            "recipient_scaffold": recipient_scaffold.strip(),
            "mode": mode.strip(),
            "donor_scaffold": donor_scaffold.strip(),
            "name": f"{recipient_scaffold.strip()}:{mode.strip()}<-{donor_scaffold.strip()}",
        })
    return routes


def combo_layers(window: list[int], spec: str) -> dict[str, list[int]]:
    if spec:
        out = {}
        for item in parse_csv(spec):
            if item == "all":
                out["all"] = list(window)
            elif "+" in item:
                vals = [int(x.strip().lstrip("L")) for x in item.split("+")]
                out[item] = vals
            else:
                val = int(item.strip().lstrip("L"))
                out[f"L{val}"] = [val]
        return out
    first, mid, last = window[0], window[len(window) // 2], window[-1]
    return {
        f"L{first}": [first],
        f"L{last}": [last],
        f"L{first}+L{last}": [first, last],
        "all": [first, mid, last],
    }


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return arr
    return arr / norm


def build_components_by_layer(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pair: str,
    layers_to_collect: list[int],
    train_n: int,
    batch_size: int,
    max_length: int,
    W_U: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    candidates = p548.build_candidates(pair, train_n)
    components_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for layer_id in layers_to_collect:
        log(f"  collect L{layer_id}")
        dirs = {}
        for name, meta in candidates.items():
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, batch_size, max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, batch_size, max_length)
            dirs[name] = mean_dir(pos_h, neg_h)
        components_by_layer[str(layer_id)] = p548.build_components(pair, dirs, W_U, tokenizer, layer_id)
    return components_by_layer


def tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def project_remove(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, scale: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    vecs = out[bidx, pos, :].float()
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    coeff = (vecs * d).sum(dim=-1, keepdim=True)
    proj = coeff * d.unsqueeze(0)
    out[bidx, pos, :] = out[bidx, pos, :] - float(scale) * proj.to(out.dtype)
    return out


def add_direction(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    out[bidx, pos, :] = out[bidx, pos, :] + (float(alpha) * d).to(out.dtype)
    return out


def module_for_site(layer: Any, site: str) -> Any:
    if site == "resid":
        return layer
    if site == "attn":
        return layer.self_attn
    if site == "mlp":
        return layer.mlp
    raise ValueError(f"unknown site: {site}")


def collect_restore_cache(
    model: Any,
    layers: list[Any],
    batch: dict[str, torch.Tensor],
    pos: torch.Tensor,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    site_name: str,
    donor_condition: str,
    add_alpha: float,
) -> tuple[dict[int, torch.Tensor], np.ndarray]:
    """Cache donor activations at the target site and last active token."""
    cache: dict[int, torch.Tensor] = {}
    handles = []
    donor_add = donor_condition == "add_perp"

    for layer_id in layer_ids:
        layer = layers[layer_id]
        site = module_for_site(layer, site_name)
        layer_device = next(site.parameters()).device
        pos_local = pos.to(layer_device)

        if donor_add:
            direction_np = components_by_layer[str(layer_id)]["residual_perp"]
            direction = torch.tensor(normalize_vec(direction_np), dtype=torch.float32, device=layer_device)
        else:
            direction = None

        if site_name == "resid":
            def make_resid_cache_hook(lid: int, d_vec: torch.Tensor | None, p_vec: torch.Tensor):
                def hook(_module: Any, _inp: Any, output: Any):
                    hs = tensor_from_output(output)
                    out = hs
                    if donor_add and d_vec is not None:
                        out = add_direction(out, p_vec.to(out.device), d_vec.to(out.device), add_alpha)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    cache[lid] = out[bidx, p_vec.to(out.device), :].detach()
                    return replace_output(output, out)
                return hook

            handles.append(layer.register_forward_hook(make_resid_cache_hook(layer_id, direction, pos_local)))
        else:
            def make_site_cache_hook(lid: int, p_vec: torch.Tensor):
                def hook(_module: Any, _inp: Any, output: Any):
                    hs = tensor_from_output(output)
                    bidx = torch.arange(hs.shape[0], device=hs.device)
                    cache[lid] = hs[bidx, p_vec.to(hs.device), :].detach()
                    return output
                return hook

            handles.append(site.register_forward_hook(make_site_cache_hook(layer_id, pos_local)))

            if donor_add:
                def make_add_hook(d_vec: torch.Tensor, p_vec: torch.Tensor):
                    def hook(_module: Any, _inp: Any, output: Any):
                        hs = tensor_from_output(output)
                        out = add_direction(hs, p_vec.to(hs.device), d_vec.to(hs.device), add_alpha)
                        return replace_output(output, out)
                    return hook

                handles.append(layer.register_forward_hook(make_add_hook(direction, pos_local)))

    with torch.inference_mode():
        out = model(**batch, return_dict=True, use_cache=False)
        idx = pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), idx]
        logits_np = logits.float().cpu().numpy().astype(np.float32)
    for handle in handles:
        handle.remove()
    return cache, logits_np


def condition_plan(condition: str) -> dict[str, Any]:
    if condition == "baseline":
        return {"site": "none", "component": None, "remove": False, "add": False, "restore": None, "donor_category": None}
    if condition == "add_perp":
        return {"site": "resid", "component": "residual_perp", "remove": False, "add": True, "restore": None, "donor_category": None}
    if condition == "resid_remove_perp":
        return {"site": "resid", "component": "residual_perp", "remove": True, "add": False, "restore": None, "donor_category": None}
    if condition == "resid_remove_random_perp":
        return {"site": "resid", "component": "random_perp", "remove": True, "add": False, "restore": None, "donor_category": None}
    prefix = "resid_donor_"
    if condition.startswith(prefix):
        tail = condition[len(prefix):]
        parts = tail.split("_")
        donor_state = parts[-1]
        donor_variant = "aligned"
        donor_category = "_".join(parts[:-1])
        if len(parts) >= 3 and parts[-2] == "shuffle":
            donor_variant = "shuffle"
            donor_category = "_".join(parts[:-2])
        if donor_state not in {"base", "add"}:
            raise ValueError(f"unknown donor state in {condition}")
        return {
            "site": "resid",
            "component": "residual_perp",
            "remove": True,
            "add": False,
            "restore": "baseline" if donor_state == "base" else "add_perp",
            "donor_category": donor_category,
            "donor_state": donor_state,
            "donor_variant": donor_variant,
        }
    raise ValueError(f"unknown condition: {condition}")


def batched_next_logits_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    texts: list[str],
    donor_texts: list[str] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    batch_size: int,
    max_length: int,
    remove_scale: float,
    add_alpha: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    plan = condition_plan(condition)
    outs = []
    donor_outs = []
    for start in range(0, len(texts), batch_size):
        batch = p544.encode_batch(tokenizer, texts[start:start + batch_size], device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handles = []
        restore_cache: dict[int, torch.Tensor] = {}
        donor_logits: np.ndarray | None = None
        if plan["site"] != "none":
            if plan["restore"] is not None:
                if donor_texts is None:
                    raise ValueError(f"{condition} requires donor_texts")
                donor_batch = p544.encode_batch(tokenizer, donor_texts[start:start + batch_size], device, max_length)
                donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
                restore_cache, donor_logits = collect_restore_cache(
                    model, layers, donor_batch, donor_pos, components_by_layer, layer_ids,
                    plan["site"], plan["restore"], add_alpha
                )
                donor_outs.append(donor_logits)
            for layer_id in layer_ids:
                layer = layers[layer_id]
                site = module_for_site(layer, plan["site"])
                layer_device = next(site.parameters()).device
                pos_local = pos.to(layer_device)
                direction_np = components_by_layer[str(layer_id)][plan["component"]]
                direction = torch.tensor(normalize_vec(direction_np), dtype=torch.float32, device=layer_device)

                def make_hook(site_name: str, d_vec: torch.Tensor, p_vec: torch.Tensor):
                    def hook(_module: Any, _inp: Any, output: Any):
                        hs = tensor_from_output(output)
                        out = hs
                        if plan["remove"]:
                            out = project_remove(out, p_vec.to(out.device), d_vec.to(out.device), remove_scale)
                        if plan["add"] and site_name == "resid":
                            out = add_direction(out, p_vec.to(out.device), d_vec.to(out.device), add_alpha)
                        if plan["restore"] is not None:
                            cached = restore_cache.get(layer_id)
                            if cached is None:
                                raise RuntimeError(f"missing restore cache for L{layer_id}")
                            bidx = torch.arange(out.shape[0], device=out.device)
                            out = out.clone()
                            out[bidx, p_vec.to(out.device), :] = cached.to(out.device, dtype=out.dtype)
                        return replace_output(output, out)
                    return hook

                handles.append(site.register_forward_hook(make_hook(plan["site"], direction, pos_local)))
        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)
            idx = pos.to(out.logits.device)
            rows = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), idx]
            outs.append(rows.float().cpu().numpy().astype(np.float32))
        for handle in handles:
            handle.remove()
        del out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    donor_np = np.concatenate(donor_outs, axis=0) if donor_outs else None
    return np.concatenate(outs, axis=0), donor_np


def run_linear_decode_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    donor_prompts: list[str] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    texts = list(prompts)
    donor_texts = list(donor_prompts) if donor_prompts is not None else None
    generated: list[list[int]] = [[] for _ in prompts]
    first_types, target_ranks, competitor_ranks = [], [], []
    rng = np.random.default_rng(seed)
    for step in range(max_new_tokens):
        logits, donor_logits = batched_next_logits_surgery(
            model, tokenizer, device, layers, texts, donor_texts, components_by_layer, layer_ids, condition,
            batch_size, max_length, remove_scale, add_alpha
        )
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        donor_toks: list[int] | None = None
        if donor_logits is not None:
            donor_toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in donor_logits]
        if step == 0:
            for row, tok in zip(logits, toks):
                first_types.append(p544.token_type(tok, groups))
                target_ranks.append(p544.best_rank(row, groups["target"]))
                competitor_ranks.append(p544.best_rank(row, groups["competitor"]))
        for i, tok in enumerate(toks):
            generated[i].append(tok)
            texts[i] += tokenizer.decode([tok], skip_special_tokens=False)
            if donor_texts is not None and donor_toks is not None:
                donor_texts[i] += tokenizer.decode([donor_toks[i]], skip_special_tokens=False)
    suffixes = [texts[i][len(prompts[i]):] for i in range(len(prompts))]
    return generated, suffixes, first_types, target_ranks, competitor_ranks


def decode_and_classify_surgery(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt_rows: list[dict[str, str]],
    donor_rows: list[dict[str, str]] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [r["prompt"] for r in prompt_rows]
    donor_prompts = [r["prompt"] for r in donor_rows] if donor_rows is not None else None
    generated, suffixes, first_types, target_ranks, competitor_ranks = run_linear_decode_surgery(
        model, tokenizer, device, layers, prompts, donor_prompts, components_by_layer, layer_ids, condition,
        groups, mode, max_new_tokens, batch_size, max_length, seed, temperature, top_p,
        remove_scale, add_alpha
    )
    pos_label, neg_label = PAIR_SPECS[pair]
    records = []
    for i, (row, suffix, ids, first_type, target_rank, competitor_rank) in enumerate(
        zip(prompt_rows, suffixes, generated, first_types, target_ranks, competitor_ranks)
    ):
        cls = p548.classify_suffix(suffix, row["object"], pos_label, neg_label)
        records.append({
            "prompt_index": i,
            "object": row["object"],
            "prompt": row["prompt"],
            "donor_object": donor_rows[i]["object"] if donor_rows is not None else "",
            "donor_prompt": donor_rows[i]["prompt"] if donor_rows is not None else "",
            "generated_suffix": suffix,
            "generated_ids": ids,
            "first_type": first_type,
            "first_target_rank": float(target_rank),
            "first_competitor_rank": float(competitor_rank),
            **cls,
        })
    return p548.aggregate(records), records


def compact_metrics(
    row: dict[str, Any],
    base: dict[str, Any],
    add_ref: dict[str, Any],
    random_ref: dict[str, Any],
    remove_ref: dict[str, Any],
    condition: str,
) -> dict[str, float | str]:
    clean_delta = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    score_delta = row["clean_non_object_score"] - base["clean_non_object_score"]
    label_delta = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    add_gain = add_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    random_drop = random_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    remove_delta = remove_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
    restore_gain = row["clean_non_object_rate"] - remove_ref["clean_non_object_rate"]
    is_restore = "_donor_" in condition
    if is_restore:
        if remove_delta <= -0.06 and restore_gain >= 0.08 and label_delta <= 0.05:
            cls = "restore_success"
        elif remove_delta <= -0.06 and restore_gain >= 0.04 and label_delta <= 0.08:
            cls = "weak_restore"
        elif restore_gain >= 0.08:
            cls = "restore_without_drop_or_leaky"
        else:
            cls = "restore_fail"
    elif clean_delta <= -0.10 and score_delta <= -0.08 and label_delta <= 0.05:
        cls = "necessity_drop"
    elif clean_delta <= -0.06 and score_delta <= -0.04:
        cls = "weak_drop"
    elif label_delta >= 0.12:
        cls = "label_leak_or_noise"
    elif clean_delta >= 0.08:
        cls = "positive_add_or_release"
    else:
        cls = "flat"
    return {
        "clean_delta": float(clean_delta),
        "score_delta": float(score_delta),
        "label_delta": float(label_delta),
        "add_gain": float(add_gain),
        "random_delta": float(random_drop),
        "drop_vs_random": float(clean_delta - random_drop),
        "remove_delta": float(remove_delta),
        "restore_gain": float(restore_gain),
        "class": cls,
    }


def shifted_rows(rows: list[dict[str, str]], shift: int = 1) -> list[dict[str, str]]:
    if not rows:
        return rows
    n = len(rows)
    return [rows[(i + shift) % n] for i in range(n)]


def build_category_donor_rows(
    pair: str,
    donor_scaffold: str,
    donor_category: str,
    test_n: int,
    shift: int,
) -> list[dict[str, str]]:
    pos_label, neg_label = PAIR_SPECS[pair]
    if donor_category not in p548.CATEGORY_BANK:
        raise ValueError(f"unknown donor category: {donor_category}")
    objects = p548.CATEGORY_BANK[donor_category][-test_n:]
    if objects:
        n = len(objects)
        objects = [objects[(i + shift) % n] for i in range(n)]
    return [
        {
            "object": obj,
            "prompt": p548.forbidden_prompt(donor_scaffold, obj, pos_label, neg_label),
            "donor_category": donor_category,
        }
        for obj in objects
    ]


def donor_rows_for(
    pair: str,
    donor_scaffold: str,
    donor_category: str | None,
    donor_variant: str | None,
    test_n: int,
) -> list[dict[str, str]] | None:
    if donor_category is None:
        return None
    if donor_variant == "shuffle":
        shift = max(2, test_n // 2)
    else:
        shift = 1 if donor_category == PAIR_SPECS[pair][0] else 0
    return build_category_donor_rows(pair, donor_scaffold, donor_category, test_n, shift)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = parse_csv(args.conditions)
    sample_seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase556 expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase556 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")

        audit: dict[str, Any] = {}
        compact = []
        saved_samples: list[dict[str, Any]] = []
        all_tsv: list[dict[str, Any]] = []
        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                recipient_scaffold = route["recipient_scaffold"]
                donor_scaffold = route["donor_scaffold"]
                mode = route["mode"]
                route_name = route["name"]
                key = route_name
                audit[combo_name]["rows"][key] = {}
                prompt_rows = prompt_sets[recipient_scaffold]
                for condition in conditions:
                    plan = condition_plan(condition)
                    donor_rows = donor_rows_for(
                        pair, donor_scaffold, plan.get("donor_category"), plan.get("donor_variant"), args.test_n
                    )
                    all_records = []
                    seed_rows = []
                    for seed in sample_seeds:
                        agg, records = decode_and_classify_surgery(
                            model, tokenizer, device, layers, prompt_rows, donor_rows, components_by_layer,
                            layer_ids, condition, groups, pair, mode, args.max_new_tokens,
                            args.batch_size, args.max_length, seed, args.temperature, args.top_p,
                            args.remove_scale, args.add_alpha,
                        )
                        seed_rows.append({"seed": seed, **agg})
                        for rec in records:
                            rec2 = {
                                "combo": combo_name,
                                "layers": layer_ids,
                                "pair": pair,
                                "route": route_name,
                                "recipient_scaffold": recipient_scaffold,
                                "donor_scaffold": donor_scaffold,
                                "mode": mode,
                                "condition": condition,
                                "donor_category": plan.get("donor_category") or "",
                                "donor_state": plan.get("donor_state") or "",
                                "donor_variant": plan.get("donor_variant") or "",
                                "seed": seed,
                                **rec,
                            }
                            all_records.append(rec2)
                    row = p548.aggregate(all_records)
                    row["seed_aggregates"] = seed_rows
                    audit[combo_name]["rows"][key][condition] = row
                    saved_samples.extend(all_records[: args.samples_per_row])
                    all_tsv.extend(all_records)
                rows = audit[combo_name]["rows"][key]
                base = rows["baseline"]
                add_ref = rows.get("add_perp", base)
                random_ref = rows.get("resid_remove_random_perp", base)
                for condition, row in rows.items():
                    if condition == "baseline":
                        continue
                    remove_ref = rows.get("resid_remove_perp", base)
                    compact.append({
                        "combo": combo_name,
                        "layers": layer_ids,
                        "route": route_name,
                        "recipient_scaffold": recipient_scaffold,
                        "donor_scaffold": donor_scaffold,
                        "mode": mode,
                        "condition": condition,
                        "donor_category": condition_plan(condition).get("donor_category") or "",
                        "donor_state": condition_plan(condition).get("donor_state") or "",
                        "donor_variant": condition_plan(condition).get("donor_variant") or "",
                        "base_clean_non_object_rate": base["clean_non_object_rate"],
                        "clean_non_object_rate": row["clean_non_object_rate"],
                        "base_label_violation_rate": base["any_label_violation_rate"],
                        "label_violation_rate": row["any_label_violation_rate"],
                        "object_echo_rate": row["object_echo_rate"],
                        "prompt_echo_rate": row["prompt_echo_rate"],
                        "clean_non_object_score": row["clean_non_object_score"],
                        **compact_metrics(row, base, add_ref, random_ref, remove_ref, condition),
                    })
                rp = rows.get("resid_remove_perp", base)
                ap = rows.get("add_perp", base)
                log(
                    f"    {combo_name} {key}: base={base['clean_non_object_rate']:.2f}; "
                    f"add={ap['clean_non_object_rate']:.2f}; "
                    f"resid_rm={rp['clean_non_object_rate']:.2f}; "
                    f"veh={rows.get('resid_donor_vehicle_add', base)['clean_non_object_rate']:.2f}; "
                    f"tool={rows.get('resid_donor_tool_add', base)['clean_non_object_rate']:.2f}; "
                    f"furn={rows.get('resid_donor_furniture_add', base)['clean_non_object_rate']:.2f}; "
                    f"animal={rows.get('resid_donor_animal_add', base)['clean_non_object_rate']:.2f}; "
                    f"veh_shuf={rows.get('resid_donor_vehicle_shuffle_add', base)['clean_non_object_rate']:.2f}"
                )

        return {
            "phase": 556,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": sample_seeds,
            "restore_note": "donor decomposition writes aligned or shuffled donor activations from same/competitor/unrelated categories after projection removal",
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[: args.max_saved_samples],
            "all_records_for_tsv": all_tsv[: args.max_tsv_records],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_tsv(result: dict[str, Any], out_dir: Path, model_name: str) -> None:
    fields = [
        "combo", "layers", "pair", "route", "recipient_scaffold", "donor_scaffold", "mode",
        "condition", "donor_category", "donor_state", "donor_variant", "seed", "object", "donor_object", "quality",
        "clean_non_object", "any_label_violation", "object_echo", "prompt_echo",
        "target_non_object_matches", "target_label_matches", "competitor_synonym_matches",
        "prompt", "donor_prompt", "generated_suffix",
    ]
    lines = ["\t".join(fields)]
    for rec in result.get("all_records_for_tsv", []):
        vals = []
        for field in fields:
            val = rec.get(field, "")
            if isinstance(val, list):
                val = ",".join(str(x) for x in val)
            vals.append(str(val).replace("\t", " ").replace("\n", " "))
        lines.append("\t".join(vals))
    path = out_dir / f"phase556_{model_name}_readable_samples.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--max-tsv-records", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase556_{args.model}_donor_specificity_decomposition.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(result, out_dir, args.model)
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
