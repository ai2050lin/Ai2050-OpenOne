#!/usr/bin/env python3
"""
Phase 153: format-syntax subspace localization and joint steering.

Locate a format/syntax contrast basis, compare it with semantic answer bases,
and test semantic-only, format-only, and joint additive steering under true
short iterative generation. The format path is audited through punctuation /
whitespace / option-label token groups.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase128_final_block_gateway_cuda import get_final_norm  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis, project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import TEMPLATE_FAMILIES, split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records, get_site_module, target_token_ids  # noqa: E402
from phase147_train_router_format_token_cuda import format_prompt  # noqa: E402
from phase148_router_feature_lmhead_alignment_cuda import lm_head_direction  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import classify_text, first_token_set, rank_for_ids, surface_strings  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase153_format_syntax_subspace_joint_steering")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
GOOD_CLASSES = {"canonical", "synonym", "object_near", "option_like"}
BASE_FORMATS = {"plain", "label_colon", "answer_one_word", "multiple_choice"}
FORMAT_GROUP_STRINGS = {
    "whitespace": [" ", "\t"],
    "newline": ["\n", "\n\n"],
    "colon": [":", ": "],
    "period": [".", ". "],
    "quote": ['"', " '", "'", ' "'],
    "option_label": ["A", " A", "A.", " A.", "(A", " (A"],
    "list_marker": ["-", " -", "1", "1.", " 1."],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extended_format_prompt(base: str, fmt: str, options: list[str]) -> str:
    if fmt in BASE_FORMATS:
        return format_prompt(base, fmt, options)
    stem = base.rstrip(" :")
    if fmt == "quoted_answer":
        return stem + ' Answer with one category word in quotes: "'
    if fmt == "list_answer":
        return stem + " Answer as a short list item:\n-"
    raise ValueError(fmt)


def route_format(fmt: str) -> str:
    if fmt in BASE_FORMATS:
        return fmt
    if fmt == "quoted_answer":
        return "answer_one_word"
    if fmt == "list_answer":
        return "answer_one_word"
    return "label_colon"


def build_items_ext(cat: str, family: str, template_ids: list[int], object_indices: list[int], fmt: str, options: list[str]) -> list[dict[str, Any]]:
    items = []
    templates = TEMPLATE_FAMILIES[family]
    objects = CATEGORY_OBJECTS[cat]
    for local_ti, ti in enumerate(template_ids):
        tpl = templates[ti % len(templates)]
        for oi in object_indices:
            obj = objects[oi % len(objects)]
            base = tpl["prefix"] + obj + tpl["relation"] + tpl["bridge"] + tpl["tail"]
            items.append({
                "ti": local_ti,
                "cat": cat,
                "obj": obj,
                "prompt": extended_format_prompt(base, fmt, options),
                "template": tpl,
                "format": fmt,
            })
    return items


def capture_records_with_format(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    layer_id: int,
) -> list[dict[str, Any]]:
    records = capture_records(model, tokenizer, device, layers, items, cat_local_ids, categories, batch_size, max_length, layer_id)
    for rec, item in zip(records, items):
        rec["format"] = item["format"]
    return records


def output_head(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is not None:
        return emb(hidden.to(dtype=emb.weight.dtype))
    if hasattr(model, "lm_head"):
        return model.lm_head(hidden.to(dtype=model.lm_head.weight.dtype))
    raise TypeError("Cannot locate output embedding / lm_head")


def decode_token(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)


def format_token_sets(tokenizer: Any) -> dict[str, list[int]]:
    return {name: first_token_set(tokenizer, vals) for name, vals in FORMAT_GROUP_STRINGS.items()}


def generation_class(text: str, surfaces: dict[str, list[str]]) -> str:
    cls = classify_text(text, surfaces)
    if cls in GOOD_CLASSES:
        return cls
    if re.fullmatch(r"[\s\W_]+", text) or not text.strip():
        return "format_only"
    return cls


def orthonormal_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    q, _ = np.linalg.qr(mat.astype(np.float64).T)
    return q.T.astype(np.float32)


def basis_overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = orthonormal_rows(normalize_basis(a))
    bb = orthonormal_rows(normalize_basis(b))
    if aa.size == 0 or bb.size == 0:
        return {"mean_abs_cos": 0.0, "max_abs_cos": 0.0, "fro_overlap": 0.0}
    cos = aa @ bb.T
    return {
        "mean_abs_cos": float(np.mean(np.abs(cos))),
        "max_abs_cos": float(np.max(np.abs(cos))),
        "fro_overlap": float(np.linalg.norm(cos, ord="fro") / max(1, min(aa.shape[0], bb.shape[0]))),
    }


def format_contrast_basis(records: list[dict[str, Any]], formats: list[str], target_fmt: str, rank: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    d_model = int(records[0]["answer_vec"].shape[0])
    centers = []
    for fmt in formats:
        vecs = [r["answer_vec"] for r in records if r["format"] == fmt]
        centers.append(np.mean(np.stack(vecs), axis=0) if vecs else np.zeros(d_model, dtype=np.float32))
    centers_np = np.stack(centers).astype(np.float32)
    target = centers_np[formats.index(target_fmt)]
    others = np.mean(np.delete(centers_np, formats.index(target_fmt), axis=0), axis=0)
    rows.append(target - others)
    for i, fmt in enumerate(formats):
        if fmt != target_fmt:
            rows.append(target - centers_np[i])
    basis, _ = svd_basis(np.stack(rows).astype(np.float32), rank)
    direction = target - others
    return basis, direction.astype(np.float32)


def format_lm_direction(model: Any, token_ids: list[int]) -> np.ndarray | None:
    if not token_ids:
        return None
    return lm_head_direction(model, token_ids)


def classify_token_group(token_id: int, group_ids: dict[str, list[int]]) -> str:
    for name, ids in group_ids.items():
        if int(token_id) in set(ids):
            return name
    return "other"


def run_logits_variant(
    model: Any,
    tokenizer: Any,
    layers: list[Any],
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layer_id: int,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    semantic_scale: float,
    fmt_basis: np.ndarray,
    fmt_direction: np.ndarray,
    format_scale: float,
    sem_lm_dir: np.ndarray | None,
    fmt_lm_dir: np.ndarray | None,
    lm_scale: float,
    variant: str,
) -> torch.Tensor:
    logits_rows = []
    layer = layers[layer_id - 1]
    site_module, site_kind = get_site_module(layers, layer_id, site)
    final_norm = get_final_norm(model)
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    fmt_b = torch.tensor(normalize_basis(fmt_basis), dtype=torch.float32)
    fmt_dir = torch.tensor(fmt_direction.reshape(1, -1), dtype=torch.float32)
    sem_lm = None if sem_lm_dir is None else torch.tensor(sem_lm_dir.reshape(1, -1), dtype=torch.float32)
    fmt_lm = None if fmt_lm_dir is None else torch.tensor(fmt_lm_dir.reshape(1, -1), dtype=torch.float32)

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        handles = []
        store: dict[str, torch.Tensor] = {}

        if variant in {"semantic_additive", "joint_internal", "joint_lm"}:
            def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                x = inputs[0]
                pb = pre_b.to(x.device)
                coeff_rows = []
                for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                    pos = torch.tensor(positions, device=x.device, dtype=torch.long)
                    coeff_rows.append(x[bi, pos, :].float().mean(dim=0) @ pb.T)
                coeff = torch.stack(coeff_rows, dim=0)
                store["semantic_add"] = (coeff @ w.to(x.device)) @ ans_b.to(x.device)
                return None
            handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        if variant in {"format_internal", "joint_internal"}:
            base = fmt_dir.to(device)
            fb = fmt_b.to(device)
            if fb.numel() > 0:
                base = (base @ fb.T) @ fb
            store["format_add"] = base.repeat(len(batch_items), 1)

        if variant in {"semantic_additive", "format_internal", "joint_internal", "joint_lm"}:
            def apply_site(x: torch.Tensor) -> torch.Tensor:
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                add = torch.zeros((out.shape[0], out.shape[-1]), device=out.device, dtype=torch.float32)
                if variant in {"semantic_additive", "joint_internal", "joint_lm"}:
                    add = add + semantic_scale * store["semantic_add"].to(out.device)
                if variant in {"format_internal", "joint_internal"}:
                    add = add + format_scale * store["format_add"].to(out.device)
                out[bidx, apos, :] = out[bidx, apos, :] + add.to(out.dtype)
                return out

            if site_kind == "pre":
                def site_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                    return (apply_site(inputs[0]),) + inputs[1:]
                handles.append(site_module.register_forward_pre_hook(site_pre_hook))
            else:
                def site_post_hook(_module: Any, _inputs: Any, output: Any):
                    out = apply_site(tensor_from_output(output))
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                handles.append(site_module.register_forward_hook(site_post_hook))

        if final_norm is not None and lm_scale != 0 and variant in {"format_lm", "joint_lm"}:
            def final_post_hook(_module: Any, _inputs: Any, output: Any):
                x = tensor_from_output(output)
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                add = torch.zeros((out.shape[0], out.shape[-1]), device=out.device, dtype=torch.float32)
                if variant == "joint_lm" and sem_lm is not None:
                    add = add + sem_lm.to(out.device).repeat(out.shape[0], 1)
                if fmt_lm is not None:
                    add = add + fmt_lm.to(out.device).repeat(out.shape[0], 1)
                out[bidx, apos, :] = out[bidx, apos, :] + lm_scale * add.to(out.dtype)
                if isinstance(output, tuple):
                    return (out,) + output[1:]
                return out
            handles.append(final_norm.register_forward_hook(final_post_hook))

        with torch.no_grad():
            out = model(**batch, use_cache=False)
        for h in handles:
            h.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        logits_rows.append(logits.detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(logits_rows, dim=0)


def token_rank_summary(logits: torch.Tensor, tokenizer: Any, surfaces: dict[str, list[str]], group_ids: dict[str, list[int]]) -> dict[str, Any]:
    expanded_ids = first_token_set(tokenizer, surfaces["expanded"])
    fmt_all = sorted(set(x for ids in group_ids.values() for x in ids))
    ranks = {
        "expanded_answer": rank_for_ids(logits, expanded_ids),
        "all_format": rank_for_ids(logits, fmt_all),
    }
    for name, ids in group_ids.items():
        ranks[name] = rank_for_ids(logits, ids)
    argmax = logits.argmax(dim=-1).detach().cpu().tolist()
    group_counts = Counter(classify_token_group(t, group_ids) for t in argmax)
    token_counts = Counter(int(t) for t in argmax)
    return {
        "ranks": ranks,
        "argmax_group_rates": {k: float(v / max(1, len(argmax))) for k, v in group_counts.items()},
        "top_tokens": [
            {"token_id": t, "token": decode_token(tokenizer, t), "rate": float(c / max(1, len(argmax)))}
            for t, c in token_counts.most_common(6)
        ],
    }


def iterative_generate(
    model: Any,
    tokenizer: Any,
    layers: list[Any],
    base_items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layer_id: int,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    semantic_scale: float,
    fmt_basis: np.ndarray,
    fmt_direction: np.ndarray,
    format_scale: float,
    sem_lm_dir: np.ndarray | None,
    fmt_lm_dir: np.ndarray | None,
    lm_scale: float,
    variant: str,
    steps: int,
    surfaces: dict[str, list[str]],
    group_ids: dict[str, list[int]],
) -> dict[str, Any]:
    items = [dict(x) for x in base_items]
    generated = ["" for _ in items]
    first_audit = None
    step_classes: list[list[str]] = []
    for step in range(steps):
        logits = run_logits_variant(
            model, tokenizer, layers, items, batch_size, max_length, layer_id, site,
            pre_basis, ans_basis, transfer, semantic_scale, fmt_basis, fmt_direction,
            format_scale, sem_lm_dir, fmt_lm_dir, lm_scale, variant,
        )
        if step == 0:
            first_audit = token_rank_summary(logits, tokenizer, surfaces, group_ids)
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for i, tid in enumerate(ids):
            tok = decode_token(tokenizer, int(tid))
            generated[i] += tok
            items[i]["prompt"] += tok
        step_classes.append([generation_class(x, surfaces) for x in generated])
    final_classes = [generation_class(x, surfaces) for x in generated]
    first_classes = step_classes[0] if step_classes else []
    hit = [c in GOOD_CLASSES for c in final_classes]
    fmt_first_later = []
    for i in range(len(items)):
        later_good = any(step_classes[s][i] in GOOD_CLASSES for s in range(1, len(step_classes)))
        fmt_first_later.append(first_classes[i] == "format_only" and later_good)
    return {
        "hit_rate": float(np.mean(hit)) if hit else 0.0,
        "format_first_answer_later_rate": float(np.mean(fmt_first_later)) if fmt_first_later else 0.0,
        "final_class_rates": {k: float(v / max(1, len(final_classes))) for k, v in Counter(final_classes).items()},
        "first_audit": first_audit or {},
        "examples": generated[:8],
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    phase147_path = PHASE147_ROOT / f"phase147_{args.model}_train_router_format_token.json"
    if not phase147_path.exists():
        raise SystemExit(f"Missing Phase147 result: {phase147_path}")
    phase147 = json.loads(phase147_path.read_text(encoding="utf-8"))
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        format_scales = parse_float_list(args.format_scales)
        group_ids = format_token_sets(tokenizer)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase153 format subspace and joint steering, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 153,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "formats": formats,
            "format_scales": format_scales,
            "semantic_scale": args.semantic_scale,
            "format_token_groups": {k: [int(x) for x in v] for k, v in group_ids.items()},
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        train_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        fmt_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                for fmt in formats:
                    rfmt = route_format(fmt)
                    for cat in test_categories:
                        prev_key = f"{split}:{family}:{rfmt}:{cat}"
                        if prev_key not in phase147["results"]:
                            continue
                        best = phase147["results"][prev_key]["train_best"]
                        layer_id = int(best["layer_id"])
                        site = best["site"]
                        train_cache_key = (split, family, rfmt, layer_id)
                        if train_cache_key not in train_cache:
                            train_all = []
                            for c in categories:
                                train_all.extend(build_items_ext(c, family, train_tpl, train_idx, rfmt, options))
                            recs = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                            train_cache[train_cache_key] = {
                                "records": recs,
                                "pre_centers": centers_from_records(recs, categories, "pre_vec", len(train_tpl)),
                                "ans_centers": centers_from_records(recs, categories, "answer_vec", len(train_tpl)),
                            }
                        fmt_cache_key = (split, family, layer_id)
                        if fmt_cache_key not in fmt_cache:
                            fmt_items = []
                            for f in formats:
                                for c in test_categories:
                                    fmt_items.extend(build_items_ext(c, family, train_tpl, train_idx, f, options))
                            fmt_recs = capture_records_with_format(model, tokenizer, device, layers, fmt_items, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                            fmt_cache[fmt_cache_key] = {"records": fmt_recs}

                        cached = train_cache[train_cache_key]
                        cat_recs = [r for r in cached["records"] if r["cat"] == cat]
                        pre_basis, _ = svd_basis(build_category_contrast_matrix(cached["pre_centers"], categories, cat), args.rank)
                        ans_basis, _ = svd_basis(build_category_contrast_matrix(cached["ans_centers"], categories, cat), args.rank)
                        x_train = project_np(np.stack([r["pre_vec"] for r in cat_recs]), pre_basis)
                        y_train = project_np(np.stack([r["answer_vec"] for r in cat_recs]), ans_basis)
                        transfer = ridge_map(x_train, y_train, args.ridge)
                        fmt_basis, fmt_dir = format_contrast_basis(fmt_cache[fmt_cache_key]["records"], formats, fmt, args.format_rank)
                        overlap = basis_overlap(ans_basis, fmt_basis)
                        target_ids = target_token_ids(tokenizer, cat)
                        sem_lm_dir = lm_head_direction(model, target_ids)
                        fmt_ids = sorted(set(group_ids.get("quote" if fmt == "quoted_answer" else "list_marker" if fmt == "list_answer" else "option_label" if fmt == "multiple_choice" else "colon", [])))
                        fmt_lm_dir = format_lm_direction(model, fmt_ids)
                        held_items = build_items_ext(cat, family, heldout_tpl, test_idx, fmt, options)
                        surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")

                        variants: dict[str, Any] = {}
                        for variant in ["clean", "semantic_additive", "format_internal", "format_lm"]:
                            scale = format_scales[0]
                            variants[variant] = iterative_generate(
                                model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                                layer_id, site, pre_basis, ans_basis, transfer, args.semantic_scale,
                                fmt_basis, fmt_dir, scale, sem_lm_dir, fmt_lm_dir, args.lm_scale,
                                variant, args.steps, surfaces, group_ids,
                            )
                        joint_rows = []
                        for scale in format_scales:
                            for variant in ["joint_internal", "joint_lm"]:
                                row = iterative_generate(
                                    model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                                    layer_id, site, pre_basis, ans_basis, transfer, args.semantic_scale,
                                    fmt_basis, fmt_dir, scale, sem_lm_dir, fmt_lm_dir, args.lm_scale,
                                    variant, args.steps, surfaces, group_ids,
                                )
                                row["format_scale"] = scale
                                row["variant"] = variant
                                joint_rows.append(row)
                        variants["best_joint"] = max(joint_rows, key=lambda r: (r["hit_rate"], -r["first_audit"]["ranks"]["expanded_answer"]["rank"]))
                        key = f"{split}:{family}:{fmt}:{cat}"
                        result["results"][key] = {
                            "path": {"route_format": rfmt, "layer_id": layer_id, "site": site},
                            "semantic_format_overlap": overlap,
                            "format_token_target_ids": fmt_ids,
                            "generation": variants,
                            "joint_rows": joint_rows,
                        }
        return result
    finally:
        release_loaded(loaded)


def top_cls(row: dict[str, Any]) -> str:
    rates = row.get("final_class_rates", {})
    return max(rates.items(), key=lambda x: x[1])[0] if rates else ""


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 153 Format-Syntax Subspace Joint Steering: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | overlap max | clean | sem | fmt_int | fmt_lm | best_joint | joint | fmt_group | answer_rank | examples |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        gen = item["generation"]
        bj = gen["best_joint"]
        audit = bj.get("first_audit", {})
        fmt_groups = audit.get("argmax_group_rates", {})
        fmt_group = max(fmt_groups.items(), key=lambda x: x[1])[0] if fmt_groups else ""
        ans_rank = audit.get("ranks", {}).get("expanded_answer", {}).get("rank", 0.0)
        examples = " ".join(x.replace("\n", "\\n") for x in bj.get("examples", [])[:3])
        lines.append(
            f"| {key} | {item['semantic_format_overlap']['max_abs_cos']:.3f} | "
            f"{gen['clean']['hit_rate']:.2f} | {gen['semantic_additive']['hit_rate']:.2f} | "
            f"{gen['format_internal']['hit_rate']:.2f} | {gen['format_lm']['hit_rate']:.2f} | "
            f"{bj['hit_rate']:.2f} | {bj.get('variant')}:{bj.get('format_scale')} | {fmt_group} | "
            f"{ans_rank:.1f} | {examples} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--format-rank", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--semantic-scale", type=float, default=0.05)
    parser.add_argument("--format-scales", default="0.05,0.2")
    parser.add_argument("--lm-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase153_{args.model}_format_syntax_subspace_joint_steering.json"
    md_path = out_dir / f"phase153_{args.model}_format_syntax_subspace_joint_steering.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
