#!/usr/bin/env python3
"""
Phase 152: natural surface path preservation and iterative generation closure.

Compare remove+restore against additive support while running true iterative
greedy generation for 1-3 tokens.
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
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records, get_site_module, target_token_ids  # noqa: E402
from phase147_train_router_format_token_cuda import build_items  # noqa: E402
from phase148_router_feature_lmhead_alignment_cuda import lm_head_direction  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import classify_text, surface_strings  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase152_natural_surface_iterative_generation")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
GOOD_CLASSES = {"canonical", "synonym", "object_near", "option_like"}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def output_head(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is not None:
        return emb(hidden.to(dtype=emb.weight.dtype))
    if hasattr(model, "lm_head"):
        return model.lm_head(hidden.to(dtype=model.lm_head.weight.dtype))
    raise TypeError("Cannot locate output embedding / lm_head")


def decode_token(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)


def generation_class(text: str, surfaces: dict[str, list[str]]) -> str:
    cls = classify_text(text, surfaces)
    if cls in GOOD_CLASSES:
        return cls
    if re.fullmatch(r"[\s\W_]+", text) or not text.strip():
        return "format_only"
    return cls


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
    support_scale: float,
    lm_dir: np.ndarray | None,
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
    lm_b = None if lm_dir is None else torch.tensor(lm_dir.reshape(1, -1), dtype=torch.float32)

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        handles = []
        store: dict[str, torch.Tensor] = {}

        if variant != "clean":
            def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                x = inputs[0]
                out = x.clone()
                pb = pre_b.to(out.device)
                coeff_rows = []
                for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                    pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                    vecs = out[bi, pos, :].float()
                    proj = (vecs @ pb.T) @ pb
                    if variant in {"remove", "remove_restore"}:
                        out[bi, pos, :] = out[bi, pos, :] - proj.to(out.dtype)
                    coeff_rows.append(vecs.mean(dim=0) @ pb.T)
                coeff = torch.stack(coeff_rows, dim=0)
                store["support_add"] = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
                return (out,) + inputs[1:]
            handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        if variant in {"remove_restore", "additive_support", "additive_support_lm"}:
            def apply_site(x: torch.Tensor) -> torch.Tensor:
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                out[bidx, apos, :] = out[bidx, apos, :] + support_scale * store["support_add"].to(out.device, dtype=out.dtype)
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

        if variant == "additive_support_lm" and final_norm is not None and lm_b is not None and lm_scale != 0:
            def final_post_hook(_module: Any, _inputs: Any, output: Any):
                x = tensor_from_output(output)
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                out[bidx, apos, :] = out[bidx, apos, :] + lm_scale * lm_b.to(out.device, dtype=out.dtype)
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
    support_scale: float,
    lm_dir: np.ndarray | None,
    lm_scale: float,
    variant: str,
    steps: int,
    surfaces: dict[str, list[str]],
) -> dict[str, Any]:
    items = [dict(x) for x in base_items]
    generated = ["" for _ in items]
    step_classes: list[list[str]] = []
    step_tokens: list[list[str]] = []
    for _step in range(steps):
        logits = run_logits_variant(
            model, tokenizer, layers, items, batch_size, max_length, layer_id, site,
            pre_basis, ans_basis, transfer, support_scale, lm_dir, lm_scale, variant,
        )
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        tokens = [decode_token(tokenizer, int(t)) for t in ids]
        step_tokens.append(tokens)
        cur_classes = []
        for i, tok in enumerate(tokens):
            generated[i] += tok
            items[i]["prompt"] += tok
            cur_classes.append(generation_class(generated[i], surfaces))
        step_classes.append(cur_classes)
    final_classes = [generation_class(x, surfaces) for x in generated]
    first_classes = step_classes[0] if step_classes else []
    hit = [c in GOOD_CLASSES for c in final_classes]
    format_first_answer_later = []
    for i in range(len(items)):
        first = first_classes[i] if first_classes else ""
        later_good = any(step_classes[s][i] in GOOD_CLASSES for s in range(1, len(step_classes)))
        format_first_answer_later.append(first == "format_only" and later_good)
    return {
        "hit_rate": float(np.mean(hit)) if hit else 0.0,
        "format_first_answer_later_rate": float(np.mean(format_first_answer_later)) if format_first_answer_later else 0.0,
        "final_class_rates": {k: float(v / max(1, len(final_classes))) for k, v in Counter(final_classes).items()},
        "first_class_rates": {k: float(v / max(1, len(first_classes))) for k, v in Counter(first_classes).items()},
        "examples": generated[:8],
        "step_tokens_examples": [[row[i] for row in step_tokens] for i in range(min(4, len(items)))] if step_tokens else [],
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
        families = set(parse_str_list(args.template_families))
        splits = set(parse_str_list(args.splits))
        formats = set(parse_str_list(args.formats))
        add_scales = parse_float_list(args.add_scales)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase152 iterative generation, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 152,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": test_categories,
            "variants": ["clean", "remove", "remove_restore", "additive_support", "additive_support_lm"],
            "add_scales": add_scales,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        train_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for key, prev in phase147["results"].items():
            split, family, fmt, cat = key.split(":")
            if cat not in test_categories or family not in families or split not in splits or fmt not in formats:
                continue
            best = prev["train_best"]
            layer_id = int(best["layer_id"])
            site = best["site"]
            route_scale = float(best["scale"])
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            held_items = build_items(cat, family, heldout_tpl, test_idx, fmt, options)
            cache_key = (split, family, fmt, layer_id)
            if cache_key not in train_cache:
                train_all = []
                for c in categories:
                    train_all.extend(build_items(c, family, train_tpl, train_idx, fmt, options))
                recs = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                train_cache[cache_key] = {
                    "records": recs,
                    "pre_centers": centers_from_records(recs, categories, "pre_vec", len(train_tpl)),
                    "ans_centers": centers_from_records(recs, categories, "answer_vec", len(train_tpl)),
                }
            cached = train_cache[cache_key]
            cat_recs = [r for r in cached["records"] if r["cat"] == cat]
            pre_basis, _ = svd_basis(build_category_contrast_matrix(cached["pre_centers"], categories, cat), args.rank)
            ans_basis, _ = svd_basis(build_category_contrast_matrix(cached["ans_centers"], categories, cat), args.rank)
            x_train = project_np(np.stack([r["pre_vec"] for r in cat_recs]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in cat_recs]), ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            tids = target_token_ids(tokenizer, cat)
            lm_dir = lm_head_direction(model, tids)
            surfaces = surface_strings(cat, fmt)
            rows = {}
            for variant in ["clean", "remove", "remove_restore"]:
                rows[variant] = iterative_generate(
                    model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                    layer_id, site, pre_basis, ans_basis, transfer, route_scale, lm_dir,
                    args.lm_scale, variant, args.steps, surfaces,
                )
            add_rows = []
            for scale in add_scales:
                for variant in ["additive_support", "additive_support_lm"]:
                    row = iterative_generate(
                        model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                        layer_id, site, pre_basis, ans_basis, transfer, scale, lm_dir,
                        args.lm_scale, variant, args.steps, surfaces,
                    )
                    row["scale"] = scale
                    row["variant"] = variant
                    add_rows.append(row)
            best_add = max(add_rows, key=lambda r: (r["hit_rate"], r["format_first_answer_later_rate"]))
            rows["best_additive"] = best_add
            result["results"][key] = {
                "path": {"layer_id": layer_id, "site": site, "route_scale": route_scale},
                "surface_strings": surfaces,
                "generation": rows,
                "additive_rows": add_rows,
            }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 152 Natural Surface Iterative Generation: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | clean hit | remove_restore hit | best_add hit | best_add | fmt_first_later | clean class | best class | examples |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        gen = item["generation"]
        clean = gen["clean"]
        rr = gen["remove_restore"]
        ba = gen["best_additive"]
        def top_cls(row: dict[str, Any]) -> str:
            rates = row.get("final_class_rates", {})
            return max(rates.items(), key=lambda x: x[1])[0] if rates else ""
        examples = " | ".join(x.replace("\n", "\\n") for x in ba.get("examples", [])[:3])
        lines.append(
            f"| {key} | {clean['hit_rate']:.2f} | {rr['hit_rate']:.2f} | {ba['hit_rate']:.2f} | "
            f"{ba.get('variant')}:{ba.get('scale')} | {ba['format_first_answer_later_rate']:.2f} | "
            f"{top_cls(clean)} | {top_cls(ba)} | {examples} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--lm-scale", type=float, default=4.0)
    parser.add_argument("--add-scales", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase152_{args.model}_natural_surface_iterative_generation.json"
    md_path = out_dir / f"phase152_{args.model}_natural_surface_iterative_generation.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
