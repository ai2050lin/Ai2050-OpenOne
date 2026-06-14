#!/usr/bin/env python3
"""
Phase 123: attention/MLP writer localization for the pre-answer to answer path.

This phase tests whether the Phase122 pre-answer -> answer projection closure can
be localized to specific attention heads or to MLP output at pre-answer/answer
positions.
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
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase120_post_object_token_localization_cuda import capture_local_centers, item_positions, select_local_varimax_axis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import (  # noqa: E402
    get_attention_module,
    get_num_heads,
    get_o_proj,
    group_indices,
    make_head_ablation_pre_hook,
    object_span_positions,
)


OUT_ROOT = Path("results/gpt5_phase123_attention_mlp_writer_localization")
TEST_CATEGORIES = ["number", "container", "plant"]
SOURCE_GROUPS = ["object_span", "object_last", "pre_object", "post_object", "self"]
MLP_SITES = ["pre_answer", "answer_last"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_mlp_module(layer: Any) -> Any:
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise TypeError(f"Cannot find MLP module for {type(layer).__name__}")


def make_mlp_zero_multi_hook(batch_positions: list[list[int]]):
    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        for bi, positions in enumerate(batch_positions):
            if not positions:
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            out[bi, pos, :] = 0
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def deterministic_random_heads(patch_layers: list[int], num_heads: int, n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    all_heads = [(layer_id, head_id) for layer_id in patch_layers for head_id in range(num_heads)]
    picks = rng.permutation(len(all_heads))[:n]
    return [
        {
            "patch_layer": int(all_heads[int(i)][0]),
            "head_id": int(all_heads[int(i)][1]),
            "selection_group": "random",
            "post_object_mass": 0.0,
            "object_mass": 0.0,
            "self_mass": 0.0,
        }
        for i in picks
    ]


def scan_attention_groups(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[dict[str, Any]],
    patch_layers: list[int],
    num_heads: int,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    sums = {g: np.zeros((len(patch_layers), num_heads), dtype=np.float64) for g in SOURCE_GROUPS}
    counts = 0
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            items = prompts[start:start + batch_size]
            texts = [x["prompt"] for x in items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            spans = [
                object_span_positions(tokenizer, item["prompt"], item["obj"], answer_pos[bi])
                for bi, item in enumerate(items)
            ]
            out = model(**batch, output_attentions=True, use_cache=False)
            if out.attentions is None:
                raise RuntimeError("Model did not return attentions")
            for li, layer_id in enumerate(patch_layers):
                attn = out.attentions[layer_id - 1].detach().float().cpu().numpy()
                for bi, ans in enumerate(answer_pos):
                    groups = group_indices(spans[bi], ans)
                    row = attn[bi, :, ans, :]
                    for group_name, idxs in groups.items():
                        if idxs:
                            sums[group_name][li] += row[:, idxs].sum(axis=1)
                counts += len(items)
            del out, batch
            torch.cuda.empty_cache()
    return {g: (v / max(counts, 1)).astype(np.float32) for g, v in sums.items()}


def select_head_groups(
    attn_scan: dict[str, np.ndarray],
    patch_layers: list[int],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    rows = []
    pre_mass = attn_scan["post_object"]
    object_mass = attn_scan["object_span"] + attn_scan["object_last"]
    for li, layer_id in enumerate(patch_layers):
        for head_id in range(pre_mass.shape[1]):
            rows.append({
                "patch_layer": int(layer_id),
                "head_id": int(head_id),
                "post_object_mass": float(pre_mass[li, head_id]),
                "object_mass": float(object_mass[li, head_id]),
                "object_span_mass": float(attn_scan["object_span"][li, head_id]),
                "object_last_mass": float(attn_scan["object_last"][li, head_id]),
                "pre_object_mass": float(attn_scan["pre_object"][li, head_id]),
                "self_mass": float(attn_scan["self"][li, head_id]),
            })

    def unique(items: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            key = (item["patch_layer"], item["head_id"])
            if key in seen:
                continue
            seen.add(key)
            out.append({**item, "selection_group": group})
            if len(out) >= top_k:
                break
        return out

    return {
        "pre_answer_top": unique(sorted(rows, key=lambda x: x["post_object_mass"], reverse=True), "pre_answer_top"),
        "object_top": unique(sorted(rows, key=lambda x: x["object_mass"], reverse=True), "object_top"),
        "self_top": unique(sorted(rows, key=lambda x: x["self_mass"], reverse=True), "self_top"),
    }


def site_positions_for_condition(tokenizer: Any, item: dict[str, Any], answer_pos: int, site: str) -> list[int]:
    pos = item_positions(tokenizer, item, answer_pos)
    if site == "pre_answer":
        return pos["post_object_excluding_answer"]
    if site == "answer_last":
        return pos["answer_last"]
    raise ValueError(f"Unknown MLP site: {site}")


def run_monitor_condition(
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
    mlp_specs: list[tuple[int, str]] | None = None,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    head_specs = head_specs or []
    mlp_specs = mlp_specs or []
    heads_by_layer: dict[int, list[int]] = {}
    for spec in head_specs:
        heads_by_layer.setdefault(int(spec["patch_layer"]), []).append(int(spec["head_id"]))

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        handles = []
        for layer_id, head_ids in heads_by_layer.items():
            attn = get_attention_module(layers[layer_id - 1])
            o_proj = get_o_proj(attn)
            for head_id in sorted(set(head_ids)):
                handles.append(o_proj.register_forward_pre_hook(
                    make_head_ablation_pre_hook(num_heads, head_id, answer_pos)
                ))
        for layer_id, site in mlp_specs:
            batch_positions = [
                site_positions_for_condition(tokenizer, item, int(answer_pos[bi]), site)
                for bi, item in enumerate(items)
            ]
            handles.append(get_mlp_module(layers[layer_id - 1]).register_forward_hook(
                make_mlp_zero_multi_hook(batch_positions)
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


def condition_summary(
    patched: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    target_idx: int,
    categories: list[str],
) -> dict[str, Any]:
    summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    summary["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return summary


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
        num_heads = get_num_heads(model, get_attention_module(layers[peak_layer - 1]))
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, layers={patch_layers}, heads={num_heads}, "
            f"rank={args.rank}, train/test={args.train_objects}/{args.test_objects}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        result: dict[str, Any] = {
            "phase": 123,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "patch_layers": patch_layers,
            "monitor_layer": peak_layer,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "top_k_heads": args.top_k_heads,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log(f"Building peak answer centers L{peak_layer}")
        answer_centers = capture_local_centers(
            model, tokenizer, device, categories, peak_layer, "answer_last",
            args.train_objects, args.batch_size, args.max_length
        )

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            contrast = build_category_contrast_matrix(answer_centers, categories, cat)
            answer_basis, singular_values = svd_basis(contrast, args.rank)

            baseline_for_selection = run_monitor_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, answer_basis, num_heads
            )
            answer_choice = select_local_varimax_axis(
                model, tokenizer, device, layers, prompts, baseline_for_selection["scores"],
                peak_layer, "answer_last", cat_local_ids, categories, target_idx,
                args.batch_size, args.max_length, args.scale, answer_basis
            )
            monitor_basis = answer_choice["axis"] if args.monitor_axis == "varimax" else answer_basis
            baseline = run_monitor_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads
            )

            attn_scan = scan_attention_groups(
                model, tokenizer, device, prompts, patch_layers, num_heads,
                args.batch_size, args.max_length
            )
            selected = select_head_groups(attn_scan, patch_layers, args.top_k_heads)
            selected["random"] = deterministic_random_heads(
                patch_layers, num_heads, args.top_k_heads, 12300 + target_idx
            )

            cat_out: dict[str, Any] = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in singular_values],
                "monitor_axis": args.monitor_axis,
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                    "selection_max_other_delta": float(answer_choice["selection_max_other_delta"]),
                },
                "head_groups": selected,
                "head_conditions": [],
                "mlp_conditions": [],
            }

            for group_name, heads in selected.items():
                for rank, spec in enumerate(heads, 1):
                    patched = run_monitor_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                        head_specs=[spec]
                    )
                    cat_out["head_conditions"].append({
                        **spec,
                        "selection_rank": rank,
                        **condition_summary(patched, baseline, target_idx, categories),
                    })

            for layer_id in patch_layers:
                for site in MLP_SITES:
                    patched = run_monitor_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                        mlp_specs=[(layer_id, site)]
                    )
                    cat_out["mlp_conditions"].append({
                        "patch_layer": int(layer_id),
                        "site": site,
                        **condition_summary(patched, baseline, target_idx, categories),
                    })

            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    layer = row.get("patch_layer", "NA")
    head = "" if "head_id" not in row else f" H{row['head_id']}"
    site = row.get("site", row.get("selection_group", ""))
    return f"L{layer}{head} {site} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} Aproj{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 123 Attention MLP Writer Localization: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak/monitor layer: L{result['peak_layer']}; patch layers: {result['patch_layers']}; heads: {result['num_heads']}")
    lines.append("")
    lines.append("| category | best pre-head | best object-head | best random-head | best pre-MLP | best answer-MLP |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        heads = item["head_conditions"]
        mlps = item["mlp_conditions"]

        def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
            return min(rows, key=lambda x: x["target_delta"]) if rows else None

        lines.append(
            f"| {cat} | "
            f"{_fmt(best([x for x in heads if x['selection_group'] == 'pre_answer_top']))} | "
            f"{_fmt(best([x for x in heads if x['selection_group'] == 'object_top']))} | "
            f"{_fmt(best([x for x in heads if x['selection_group'] == 'random']))} | "
            f"{_fmt(best([x for x in mlps if x['site'] == 'pre_answer']))} | "
            f"{_fmt(best([x for x in mlps if x['site'] == 'answer_last']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--top-k-heads", type=int, default=4)
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase123_{args.model}_attention_mlp_writer_localization.json"
    md_path = out_dir / f"phase123_{args.model}_attention_mlp_writer_localization.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
