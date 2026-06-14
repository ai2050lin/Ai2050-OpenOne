#!/usr/bin/env python3
"""
Phase 125: joint head-set + MLP-subspace closure with cross-heldout validation.

Objects are split into train / selection / evaluation sets. Candidate heads and
MLP/residual layers are selected only on the selection split, then evaluated on a
disjoint object split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase120_post_object_token_localization_cuda import capture_local_centers, select_local_varimax_axis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import scan_attention_groups, site_positions_for_condition  # noqa: E402
from phase124_writer_set_value_alignment_cuda import (  # noqa: E402
    capture_head_value_alignment,
    deterministic_random_heads,
    get_mlp_module,
    make_head_set_pre_hook,
    make_mlp_subspace_hook,
    unique_heads,
)
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase125_joint_closure_crossheldout")
TEST_CATEGORIES = ["number", "container", "plant"]
SET_TYPES = [
    "attention_mass",
    "value_aligned",
    "target_discovered",
    "projection_discovered",
    "low_pre_value_control",
    "object_control",
    "random_control",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_offset_prompts(cat: str, offset: int, n_objects: int) -> list[dict[str, Any]]:
    prompts = []
    objs = CATEGORY_OBJECTS[cat][offset:offset + n_objects]
    for tpl in TEMPLATES:
        for obj in objs:
            prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj), "template": tpl["name"]})
    return prompts


def make_residual_subspace_hook(basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        b = basis.to(out.device).float()
        for bi, positions in enumerate(batch_positions):
            if not positions:
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            vecs = out[bi, pos, :].float()
            proj = (vecs @ b.T) @ b
            out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    num_heads: int,
    head_specs: list[dict[str, Any]] | None = None,
    mlp_specs: list[tuple[int, str, np.ndarray]] | None = None,
    residual_specs: list[tuple[int, str, np.ndarray]] | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    head_specs = head_specs or []
    mlp_specs = mlp_specs or []
    residual_specs = residual_specs or []
    by_layer: dict[int, list[int]] = {}
    for spec in head_specs:
        by_layer.setdefault(int(spec["patch_layer"]), []).append(int(spec["head_id"]))

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        handles = []
        for layer_id, head_ids in by_layer.items():
            handles.append(get_o_proj(get_attention_module(layers[layer_id - 1])).register_forward_pre_hook(
                make_head_set_pre_hook(num_heads, head_ids, answer_pos)
            ))
        for layer_id, site, basis_np in mlp_specs:
            positions = [site_positions_for_condition(tokenizer, item, int(answer_pos[bi]), site) for bi, item in enumerate(items)]
            basis = torch.tensor(basis_np, device=device, dtype=torch.float32)
            handles.append(get_mlp_module(layers[layer_id - 1]).register_forward_hook(
                make_mlp_subspace_hook(basis, positions, scale)
            ))
        for layer_id, site, basis_np in residual_specs:
            positions = [site_positions_for_condition(tokenizer, item, int(answer_pos[bi]), site) for bi, item in enumerate(items)]
            basis = torch.tensor(basis_np, device=device, dtype=torch.float32)
            handles.append(layers[layer_id - 1].register_forward_hook(
                make_residual_subspace_hook(basis, positions, scale)
            ))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for handle in handles:
            handle.remove()
        pos_gpu = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def summarize_condition(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def rank_heads(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    baseline: dict[str, np.ndarray],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    target_idx: int,
    patch_layers: list[int],
    num_heads: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    batch_size: int,
    max_length: int,
    candidate_pool: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    attn_scan = scan_attention_groups(model, tokenizer, device, prompts, patch_layers, num_heads, batch_size, max_length)
    align = capture_head_value_alignment(model, tokenizer, device, layers, prompts, patch_layers, num_heads, monitor_basis, batch_size, max_length)
    all_heads = []
    for layer_id in patch_layers:
        li = patch_layers.index(layer_id)
        for head_id in range(num_heads):
            row = {"patch_layer": int(layer_id), "head_id": int(head_id)}
            row.update(align[(layer_id, head_id)])
            row["post_object_mass"] = float(attn_scan["post_object"][li, head_id])
            row["object_mass"] = float(attn_scan["object_span"][li, head_id] + attn_scan["object_last"][li, head_id])
            row["self_mass"] = float(attn_scan["self"][li, head_id])
            all_heads.append(row)

    attention_rank = unique_heads(sorted(all_heads, key=lambda x: x["post_object_mass"], reverse=True))
    value_rank = unique_heads(sorted(all_heads, key=lambda x: x["value_alignment_mean"], reverse=True))
    projection_seed = unique_heads(sorted(all_heads, key=lambda x: x["value_alignment_abs_mean"], reverse=True))
    object_rank = unique_heads(sorted(all_heads, key=lambda x: x["object_mass"], reverse=True))
    median_pre = float(np.median([x["post_object_mass"] for x in all_heads]))
    low_pre_value_rank = unique_heads(sorted(
        [x for x in all_heads if x["post_object_mass"] <= median_pre],
        key=lambda x: x["value_alignment_mean"],
        reverse=True,
    ))
    random_rank = deterministic_random_heads(patch_layers, num_heads, candidate_pool, 25100 + target_idx)
    pool = unique_heads(
        attention_rank[:candidate_pool]
        + value_rank[:candidate_pool]
        + projection_seed[:candidate_pool]
        + object_rank[:candidate_pool]
        + low_pre_value_rank[:candidate_pool]
        + random_rank[:candidate_pool]
    )
    single_rows = []
    for spec in pool:
        patched = run_condition(
            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
            batch_size, max_length, monitor_layer, monitor_basis, num_heads,
            head_specs=[spec],
        )
        single_rows.append({**spec, **summarize_condition(patched, baseline, target_idx, categories)})
    ranks = {
        "attention_mass": attention_rank,
        "value_aligned": value_rank,
        "target_discovered": unique_heads(sorted(single_rows, key=lambda x: x["target_delta"])),
        "projection_discovered": unique_heads(sorted(single_rows, key=lambda x: x["answer_proj_delta"])),
        "low_pre_value_control": low_pre_value_rank,
        "object_control": object_rank,
        "random_control": random_rank,
    }
    return ranks, single_rows


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        patch_layers = list(range(max(1, peak_layer - args.layer_back), peak_layer + 1))
        set_sizes = [int(x) for x in args.set_sizes.split(",") if x.strip()]
        num_heads = get_num_heads(model, get_attention_module(layers[peak_layer - 1]))
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        selection_offset = args.train_objects
        eval_offset = args.train_objects + args.selection_objects
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, layers={patch_layers}, heads={num_heads}, "
            f"split={args.train_objects}/{args.selection_objects}/{args.eval_objects}, vram={alloc:.2f}/{reserved:.2f}GB"
        )

        result: dict[str, Any] = {
            "phase": 125,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "monitor_layer": peak_layer,
            "patch_layers": patch_layers,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "selection_objects_per_category": args.selection_objects,
            "evaluation_objects_per_category": args.eval_objects,
            "selection_offset": selection_offset,
            "evaluation_offset": eval_offset,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "set_sizes": set_sizes,
            "candidate_pool": args.candidate_pool,
            "scale": args.scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log("Building train-split answer/pre centers")
        answer_centers = capture_local_centers(model, tokenizer, device, categories, peak_layer, "answer_last", args.train_objects, args.batch_size, args.max_length)
        pre_centers_by_layer = {
            layer_id: capture_local_centers(model, tokenizer, device, categories, layer_id, "post_object_excluding_answer", args.train_objects, args.batch_size, args.max_length)
            for layer_id in patch_layers
        }

        for ci, cat in enumerate(test_categories, 1):
            if eval_offset + args.eval_objects > len(CATEGORY_OBJECTS[cat]):
                raise ValueError(f"{cat} has only {len(CATEGORY_OBJECTS[cat])} objects; split exceeds list")
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            selection_prompts = build_offset_prompts(cat, selection_offset, args.selection_objects)
            eval_prompts = build_offset_prompts(cat, eval_offset, args.eval_objects)
            answer_basis, answer_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            selection_basis_baseline = run_condition(
                model, tokenizer, device, layers, selection_prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, answer_basis, num_heads
            )
            answer_choice = select_local_varimax_axis(
                model, tokenizer, device, layers, selection_prompts, selection_basis_baseline["scores"],
                peak_layer, "answer_last", cat_local_ids, categories, target_idx,
                args.batch_size, args.max_length, args.scale, answer_basis
            )
            monitor_basis = answer_choice["axis"] if args.monitor_axis == "varimax" else answer_basis
            selection_baseline = run_condition(
                model, tokenizer, device, layers, selection_prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads
            )
            eval_baseline = run_condition(
                model, tokenizer, device, layers, eval_prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads
            )
            ranks, single_rows = rank_heads(
                model, tokenizer, device, layers, selection_prompts, selection_baseline,
                cat_local_ids, categories, target_idx, patch_layers, num_heads, peak_layer,
                monitor_basis, args.batch_size, args.max_length, args.candidate_pool,
            )
            pre_bases = {}
            for layer_id in patch_layers:
                basis, _sv = svd_basis(build_category_contrast_matrix(pre_centers_by_layer[layer_id], categories, cat), args.rank)
                pre_bases[layer_id] = basis

            selection_mlp_rows = []
            selection_residual_rows = []
            for layer_id in patch_layers:
                mlp_sel = run_condition(
                    model, tokenizer, device, layers, selection_prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                    mlp_specs=[(layer_id, "pre_answer", pre_bases[layer_id])],
                    scale=args.scale,
                )
                selection_mlp_rows.append({
                    "patch_layer": int(layer_id),
                    "site": "pre_answer",
                    **summarize_condition(mlp_sel, selection_baseline, target_idx, categories),
                })
                residual_sel = run_condition(
                    model, tokenizer, device, layers, selection_prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                    residual_specs=[(layer_id, "pre_answer", pre_bases[layer_id])],
                    scale=args.scale,
                )
                selection_residual_rows.append({
                    "patch_layer": int(layer_id),
                    "site": "pre_answer",
                    **summarize_condition(residual_sel, selection_baseline, target_idx, categories),
                })
            best_mlp_sel = min(selection_mlp_rows, key=lambda x: x["target_delta"])
            best_residual_sel = min(selection_residual_rows, key=lambda x: x["target_delta"])
            best_mlp_spec = (int(best_mlp_sel["patch_layer"]), "pre_answer", pre_bases[int(best_mlp_sel["patch_layer"])])
            best_residual_spec = (int(best_residual_sel["patch_layer"]), "pre_answer", pre_bases[int(best_residual_sel["patch_layer"])])

            cat_out: dict[str, Any] = {
                "n_selection_prompts": len(selection_prompts),
                "n_evaluation_prompts": len(eval_prompts),
                "selection_objects": CATEGORY_OBJECTS[cat][selection_offset:selection_offset + args.selection_objects],
                "evaluation_objects": CATEGORY_OBJECTS[cat][eval_offset:eval_offset + args.eval_objects],
                "baseline_eval_target_mean": float(eval_baseline["scores"][:, target_idx].mean()),
                "baseline_eval_answer_proj_mean": float(eval_baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                },
                "selection_single_head_pool": single_rows,
                "selection_mlp_rows": selection_mlp_rows,
                "selection_residual_rows": selection_residual_rows,
                "selected_mlp": best_mlp_sel,
                "selected_residual_reference": best_residual_sel,
                "evaluation_conditions": [],
            }

            residual_eval = run_condition(
                model, tokenizer, device, layers, eval_prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                residual_specs=[best_residual_spec],
                scale=args.scale,
            )
            residual_eval_summary = summarize_condition(residual_eval, eval_baseline, target_idx, categories)
            cat_out["evaluation_conditions"].append({
                "condition": "residual_pre_reference",
                "set_name": "residual_pre_reference",
                "set_size": 0,
                "heads": [],
                "mlp_layer": None,
                "residual_layer": int(best_residual_sel["patch_layer"]),
                **residual_eval_summary,
                "effect_ratio_vs_residual_ref": 1.0,
            })

            mlp_eval = run_condition(
                model, tokenizer, device, layers, eval_prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                mlp_specs=[best_mlp_spec],
                scale=args.scale,
            )
            mlp_summary = summarize_condition(mlp_eval, eval_baseline, target_idx, categories)
            denom = residual_eval_summary["target_delta"]
            cat_out["evaluation_conditions"].append({
                "condition": "pre_mlp_subspace_only",
                "set_name": "pre_mlp_subspace",
                "set_size": 0,
                "heads": [],
                "mlp_layer": int(best_mlp_sel["patch_layer"]),
                "residual_layer": None,
                **mlp_summary,
                "effect_ratio_vs_residual_ref": float(mlp_summary["target_delta"] / denom) if abs(denom) > 1e-6 else 0.0,
            })

            for set_name in SET_TYPES:
                ranking = ranks[set_name]
                for size in set_sizes:
                    selected = unique_heads(ranking)[:size]
                    if not selected:
                        continue
                    head_eval = run_condition(
                        model, tokenizer, device, layers, eval_prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                        head_specs=selected,
                        scale=args.scale,
                    )
                    head_summary = summarize_condition(head_eval, eval_baseline, target_idx, categories)
                    cat_out["evaluation_conditions"].append({
                        "condition": "head_set_only",
                        "set_name": set_name,
                        "set_size": int(size),
                        "heads": [{"patch_layer": int(x["patch_layer"]), "head_id": int(x["head_id"])} for x in selected],
                        "mlp_layer": None,
                        "residual_layer": None,
                        **head_summary,
                        "effect_ratio_vs_residual_ref": float(head_summary["target_delta"] / denom) if abs(denom) > 1e-6 else 0.0,
                    })
                    if set_name in {"value_aligned", "target_discovered", "projection_discovered"} and size == max(set_sizes):
                        combo_eval = run_condition(
                            model, tokenizer, device, layers, eval_prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                            head_specs=selected,
                            mlp_specs=[best_mlp_spec],
                            scale=args.scale,
                        )
                        combo_summary = summarize_condition(combo_eval, eval_baseline, target_idx, categories)
                        cat_out["evaluation_conditions"].append({
                            "condition": "head_set_plus_pre_mlp",
                            "set_name": set_name,
                            "set_size": int(size),
                            "heads": [{"patch_layer": int(x["patch_layer"]), "head_id": int(x["head_id"])} for x in selected],
                            "mlp_layer": int(best_mlp_sel["patch_layer"]),
                            "residual_layer": None,
                            **combo_summary,
                            "effect_ratio_vs_residual_ref": float(combo_summary["target_delta"] / denom) if abs(denom) > 1e-6 else 0.0,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return (
        f"{row.get('condition','')} {row.get('set_name','')} k{row.get('set_size','')} "
        f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} "
        f"A{row['answer_proj_delta']:+.2f} ratio{row.get('effect_ratio_vs_residual_ref',0.0):+.2f}"
    )


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 125 Joint Closure Cross-heldout: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | residual ref | best head only | best combo | best control | pre-MLP only |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        rows = item["evaluation_conditions"]

        def best(xs: list[dict[str, Any]]) -> dict[str, Any] | None:
            return min(xs, key=lambda x: x["target_delta"]) if xs else None

        residual = best([x for x in rows if x["condition"] == "residual_pre_reference"])
        head = best([x for x in rows if x["condition"] == "head_set_only" and x["set_name"] not in {"object_control", "random_control", "low_pre_value_control"}])
        combo = best([x for x in rows if x["condition"] == "head_set_plus_pre_mlp"])
        control = best([x for x in rows if x["set_name"] in {"object_control", "random_control", "low_pre_value_control"}])
        mlp = best([x for x in rows if x["condition"] == "pre_mlp_subspace_only"])
        lines.append(f"| {cat} | {_fmt(residual)} | {_fmt(head)} | {_fmt(combo)} | {_fmt(control)} | {_fmt(mlp)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--selection-objects", type=int, default=8)
    parser.add_argument("--eval-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--set-sizes", default="4,8,16")
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase125_{args.model}_joint_closure_crossheldout.json"
    md_path = out_dir / f"phase125_{args.model}_joint_closure_crossheldout.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
