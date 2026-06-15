#!/usr/bin/env python3
"""
Phase 154: format writer localization and surface gate closure.

For each Phase153 format condition, scan final layers and ablate the semantic
projection, format projection, or their joint projection from attention/MLP
outputs at the answer site. Measure first-token answer and format rank damage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import first_token_set, rank_for_ids, surface_strings  # noqa: E402
from phase153_format_syntax_subspace_joint_steering_cuda import (  # noqa: E402
    build_items_ext,
    capture_records_with_format,
    format_contrast_basis,
    format_token_sets,
    route_format,
)


OUT_ROOT = Path("results/gpt5_phase154_format_writer_surface_gate")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
COMPONENTS = ["attention_output", "mlp_output"]
MODES = ["semantic_proj", "format_proj", "joint_proj"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def orthonormal_rows(mat: np.ndarray) -> np.ndarray:
    mat = normalize_basis(mat)
    if mat.size == 0:
        return mat.astype(np.float32)
    q, _ = np.linalg.qr(mat.astype(np.float64).T)
    return q.T.astype(np.float32)


def joint_basis(a: np.ndarray, b: np.ndarray, rank: int) -> np.ndarray:
    rows = np.concatenate([normalize_basis(a), normalize_basis(b)], axis=0)
    return orthonormal_rows(rows)[:rank].astype(np.float32)


def component_module(layers: list[Any], layer_id: int, component: str) -> Any:
    layer = layers[layer_id - 1]
    if component == "attention_output":
        return get_attention_module(layer)
    if component == "mlp_output":
        return get_mlp_module(layer)
    raise ValueError(component)


def ablate_component_projection_hook(answer_positions: list[int], basis: np.ndarray, scale: float):
    b = torch.tensor(orthonormal_rows(basis), dtype=torch.float32)

    def hook(_module: Any, _inputs: Any, output: Any):
        x = tensor_from_output(output)
        out = x.clone()
        bb = b.to(out.device)
        if bb.numel() > 0:
            bidx = torch.arange(out.shape[0], device=out.device)
            pos = torch.tensor(answer_positions, device=out.device, dtype=torch.long)
            vec = out[bidx, pos, :].float()
            proj = (vec @ bb.T) @ bb
            out[bidx, pos, :] = out[bidx, pos, :] - scale * proj.to(out.dtype)
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    return hook


def clean_logits_for_items(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = batch_context(tokenizer, batch, batch_items)
            out = model(**batch, use_cache=False)
            pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
            rows.append(logits.detach().float().cpu())
            del out, batch
            torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def patched_logits_for_items(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    patch_layer: int,
    component: str,
    basis: np.ndarray,
    scale: float,
) -> torch.Tensor:
    rows = []
    module = component_module(layers, patch_layer, component)
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        handle = module.register_forward_hook(ablate_component_projection_hook(ctx["last_pos"], basis, scale))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        handle.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        rows.append(logits.detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def metric_row(logits: torch.Tensor, clean: torch.Tensor, tokenizer: Any, surfaces: dict[str, list[str]], fmt_ids: list[int]) -> dict[str, float]:
    answer_ids = first_token_set(tokenizer, surfaces["expanded"])
    clean_answer = rank_for_ids(clean, answer_ids)
    patched_answer = rank_for_ids(logits, answer_ids)
    clean_format = rank_for_ids(clean, fmt_ids)
    patched_format = rank_for_ids(logits, fmt_ids)
    return {
        "answer_rank": patched_answer["rank"],
        "answer_argmax": patched_answer["argmax"],
        "answer_rank_delta": patched_answer["rank"] - clean_answer["rank"],
        "answer_argmax_delta": patched_answer["argmax"] - clean_answer["argmax"],
        "format_rank": patched_format["rank"],
        "format_argmax": patched_format["argmax"],
        "format_rank_delta": patched_format["rank"] - clean_format["rank"],
        "format_argmax_delta": patched_format["argmax"] - clean_format["argmax"],
    }


def clean_metric_row(clean: torch.Tensor, tokenizer: Any, surfaces: dict[str, list[str]], fmt_ids: list[int]) -> dict[str, float]:
    answer_ids = first_token_set(tokenizer, surfaces["expanded"])
    ans = rank_for_ids(clean, answer_ids)
    fmt = rank_for_ids(clean, fmt_ids)
    return {
        "answer_rank": ans["rank"],
        "answer_argmax": ans["argmax"],
        "format_rank": fmt["rank"],
        "format_argmax": fmt["argmax"],
    }


def format_target_ids(fmt: str, group_ids: dict[str, list[int]]) -> list[int]:
    if fmt == "multiple_choice":
        return group_ids.get("option_label", [])
    if fmt == "quoted_answer":
        return group_ids.get("quote", [])
    if fmt == "list_answer":
        return group_ids.get("list_marker", []) + group_ids.get("newline", [])
    if fmt == "label_colon":
        return group_ids.get("colon", []) + group_ids.get("whitespace", [])
    if fmt == "answer_one_word":
        return group_ids.get("whitespace", [])
    return sorted(set(x for ids in group_ids.values() for x in ids))


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
        last_layer = len(layers)
        patch_layers = list(range(max(1, last_layer - args.layer_back + 1), last_layer + 1))
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        group_ids = format_token_sets(tokenizer)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase154 writer scan, L{last_layer}, patch={patch_layers}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 154,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "patch_layers": patch_layers,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "formats": formats,
            "components": COMPONENTS,
            "modes": MODES,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        semantic_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        format_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                for patch_layer in patch_layers:
                    fmt_key = (split, family, patch_layer)
                    if fmt_key not in format_cache:
                        fmt_items = []
                        for fmt in formats:
                            for c in test_categories:
                                fmt_items.extend(build_items_ext(c, family, train_tpl, train_idx, fmt, options))
                        fmt_records = capture_records_with_format(
                            model, tokenizer, device, layers, fmt_items, cat_local_ids,
                            categories, args.batch_size, args.max_length, patch_layer,
                        )
                        format_cache[fmt_key] = {"records": fmt_records}
                    for fmt in formats:
                        rfmt = route_format(fmt)
                        sem_key = (split, family, rfmt, patch_layer)
                        if sem_key not in semantic_cache:
                            train_all = []
                            for c in categories:
                                train_all.extend(build_items_ext(c, family, train_tpl, train_idx, rfmt, options))
                            recs = capture_records(
                                model, tokenizer, device, layers, train_all, cat_local_ids,
                                categories, args.batch_size, args.max_length, patch_layer,
                            )
                            semantic_cache[sem_key] = {
                                "records": recs,
                                "ans_centers": centers_from_records(recs, categories, "answer_vec", len(train_tpl)),
                            }
                        for cat in test_categories:
                            prev_key = f"{split}:{family}:{rfmt}:{cat}"
                            if prev_key not in phase147["results"]:
                                continue
                            sem_basis, _ = svd_basis(
                                build_category_contrast_matrix(semantic_cache[sem_key]["ans_centers"], categories, cat),
                                args.rank,
                            )
                            fmt_basis, _fmt_dir = format_contrast_basis(format_cache[fmt_key]["records"], formats, fmt, args.format_rank)
                            bases = {
                                "semantic_proj": sem_basis,
                                "format_proj": fmt_basis,
                                "joint_proj": joint_basis(sem_basis, fmt_basis, args.rank + args.format_rank),
                            }
                            held_items = build_items_ext(cat, family, heldout_tpl, test_idx, fmt, options)
                            surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")
                            fmt_ids = sorted(set(format_target_ids(fmt, group_ids)))
                            case_key = f"{split}:{family}:{fmt}:{cat}"
                            case = result["results"].setdefault(case_key, {
                                "clean": None,
                                "conditions": [],
                            })
                            clean = clean_logits_for_items(model, tokenizer, device, held_items, args.batch_size, args.max_length)
                            if case["clean"] is None:
                                case["clean"] = clean_metric_row(clean, tokenizer, surfaces, fmt_ids)
                                case["n_prompts"] = len(held_items)
                            for component in COMPONENTS:
                                for mode in MODES:
                                    patched = patched_logits_for_items(
                                        model, tokenizer, device, layers, held_items,
                                        args.batch_size, args.max_length, patch_layer,
                                        component, bases[mode], args.ablate_scale,
                                    )
                                    row = metric_row(patched, clean, tokenizer, surfaces, fmt_ids)
                                    row.update({
                                        "patch_layer": patch_layer,
                                        "component": component,
                                        "mode": mode,
                                    })
                                    case["conditions"].append(row)
        return result
    finally:
        release_loaded(loaded)


def best_rows(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    conds = case.get("conditions", [])
    out = {}
    for mode in MODES:
        rows = [r for r in conds if r["mode"] == mode]
        out[f"{mode}_answer"] = max(rows, key=lambda r: r["answer_rank_delta"]) if rows else {}
        out[f"{mode}_format"] = max(rows, key=lambda r: r["format_rank_delta"]) if rows else {}
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 154 Format Writer Surface Gate: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | clean answer rank | clean fmt rank | sem answer damage | fmt damage | joint answer damage | joint fmt damage |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, case in sorted(result["results"].items()):
        best = best_rows(case)

        def fmt(row: dict[str, Any], field: str) -> str:
            if not row:
                return ""
            return f"L{row['patch_layer']} {row['component']} {row[field]:+.1f}"

        lines.append(
            f"| {key} | {case['clean']['answer_rank']:.1f} | {case['clean']['format_rank']:.1f} | "
            f"{fmt(best['semantic_proj_answer'], 'answer_rank_delta')} | "
            f"{fmt(best['format_proj_format'], 'format_rank_delta')} | "
            f"{fmt(best['joint_proj_answer'], 'answer_rank_delta')} | "
            f"{fmt(best['joint_proj_format'], 'format_rank_delta')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--format-rank", type=int, default=4)
    parser.add_argument("--layer-back", type=int, default=4)
    parser.add_argument("--ablate-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase154_{args.model}_format_writer_surface_gate.json"
    md_path = out_dir / f"phase154_{args.model}_format_writer_surface_gate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
