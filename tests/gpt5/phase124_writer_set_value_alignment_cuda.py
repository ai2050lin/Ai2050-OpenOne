#!/usr/bin/env python3
"""
Phase 124: pre-answer writer set and value-alignment sweep.

This phase moves from single heads to cumulative head sets. It ranks candidate
heads by pre-answer attention mass, by head-output alignment with the answer
monitor axis, and by measured single-head ablation effect.
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
from phase123_attention_mlp_writer_localization_cuda import (  # noqa: E402
    get_mlp_module,
    scan_attention_groups,
    select_head_groups,
    site_positions_for_condition,
)
from phase112_attention_transport_head_mapping_cuda import (  # noqa: E402
    get_attention_module,
    get_num_heads,
    get_o_proj,
)


OUT_ROOT = Path("results/gpt5_phase124_writer_set_value_alignment")
TEST_CATEGORIES = ["number", "container", "plant"]
SET_SIZES = [1, 2, 4, 8, 16]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unique_heads(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (int(item["patch_layer"]), int(item["head_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def deterministic_random_heads(patch_layers: list[int], num_heads: int, n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    all_heads = [(layer_id, head_id) for layer_id in patch_layers for head_id in range(num_heads)]
    picks = rng.permutation(len(all_heads))[:n]
    return [{"patch_layer": int(all_heads[int(i)][0]), "head_id": int(all_heads[int(i)][1])} for i in picks]


def make_head_set_pre_hook(num_heads: int, head_ids: list[int], positions: torch.Tensor):
    head_ids = sorted(set(int(h) for h in head_ids))

    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        batch_idx = torch.arange(y.shape[0], device=y.device)
        pos = positions.to(y.device)
        head_tensor = torch.tensor(head_ids, device=y.device, dtype=torch.long)
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        y_view[batch_idx[:, None], pos[:, None], head_tensor[None, :], :] = 0
        return (y,) + inputs[1:]

    return hook


def make_mlp_subspace_hook(basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
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


def capture_head_value_alignment(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    patch_layers: list[int],
    num_heads: int,
    monitor_basis: np.ndarray,
    batch_size: int,
    max_length: int,
) -> dict[tuple[int, int], dict[str, float]]:
    basis = torch.tensor(monitor_basis, device=device, dtype=torch.float32)
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)
    sums: dict[tuple[int, int], float] = {(l, h): 0.0 for l in patch_layers for h in range(num_heads)}
    abs_sums: dict[tuple[int, int], float] = {(l, h): 0.0 for l in patch_layers for h in range(num_heads)}
    counts = 0

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        handles = []
        captured: dict[int, torch.Tensor] = {}

        def make_capture(layer_id: int):
            def hook(_module: Any, inputs: tuple[Any, ...]):
                captured[layer_id] = inputs[0].detach()
            return hook

        for layer_id in patch_layers:
            handles.append(get_o_proj(get_attention_module(layers[layer_id - 1])).register_forward_pre_hook(make_capture(layer_id)))
        with torch.no_grad():
            _ = model(**batch, use_cache=False)
        for h in handles:
            h.remove()

        for layer_id in patch_layers:
            x = captured[layer_id]
            o_proj = get_o_proj(get_attention_module(layers[layer_id - 1]))
            weight = o_proj.weight.detach().float()
            head_dim = x.shape[-1] // num_heads
            pos = answer_pos.to(x.device)
            x_ans = x[torch.arange(x.shape[0], device=x.device), pos, :].float()
            for head_id in range(num_heads):
                lo = head_id * head_dim
                hi = lo + head_dim
                contrib = x_ans[:, lo:hi] @ weight[:, lo:hi].T
                coeff = contrib @ basis.T.to(contrib.device)
                if basis.shape[0] == 1:
                    vals = coeff[:, 0]
                else:
                    vals = coeff.norm(dim=1)
                key = (layer_id, head_id)
                sums[key] += float(vals.sum().detach().cpu())
                abs_sums[key] += float(vals.abs().sum().detach().cpu())
        counts += len(items)
        del batch
        torch.cuda.empty_cache()

    return {
        key: {
            "value_alignment_mean": sums[key] / max(counts, 1),
            "value_alignment_abs_mean": abs_sums[key] / max(counts, 1),
        }
        for key in sums
    }


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
    mlp_scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    head_specs = head_specs or []
    mlp_specs = mlp_specs or []
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
                make_mlp_subspace_hook(basis, positions, mlp_scale)
            ))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
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
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, layers={patch_layers}, heads={num_heads}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 124,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "monitor_layer": peak_layer,
            "patch_layers": patch_layers,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "candidate_pool": args.candidate_pool,
            "set_sizes": set_sizes,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log(f"Building peak answer and pre-answer centers")
        answer_centers = capture_local_centers(model, tokenizer, device, categories, peak_layer, "answer_last", args.train_objects, args.batch_size, args.max_length)
        pre_centers_by_layer = {
            layer_id: capture_local_centers(model, tokenizer, device, categories, layer_id, "post_object_excluding_answer", args.train_objects, args.batch_size, args.max_length)
            for layer_id in patch_layers
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            answer_basis, answer_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            baseline_for_selection = run_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, answer_basis, num_heads)
            answer_choice = select_local_varimax_axis(model, tokenizer, device, layers, prompts, baseline_for_selection["scores"], peak_layer, "answer_last", cat_local_ids, categories, target_idx, args.batch_size, args.max_length, args.scale, answer_basis)
            monitor_basis = answer_choice["axis"] if args.monitor_axis == "varimax" else answer_basis
            baseline = run_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads)

            attn_scan = scan_attention_groups(model, tokenizer, device, prompts, patch_layers, num_heads, args.batch_size, args.max_length)
            head_groups = select_head_groups(attn_scan, patch_layers, max(args.candidate_pool, max(set_sizes)))
            align = capture_head_value_alignment(model, tokenizer, device, layers, prompts, patch_layers, num_heads, monitor_basis, args.batch_size, args.max_length)
            all_heads = []
            for layer_id in patch_layers:
                for head_id in range(num_heads):
                    item = {"patch_layer": int(layer_id), "head_id": int(head_id)}
                    item.update(align[(layer_id, head_id)])
                    li = patch_layers.index(layer_id)
                    item["post_object_mass"] = float(attn_scan["post_object"][li, head_id])
                    item["object_mass"] = float(attn_scan["object_span"][li, head_id] + attn_scan["object_last"][li, head_id])
                    item["self_mass"] = float(attn_scan["self"][li, head_id])
                    all_heads.append(item)

            attention_rank = unique_heads(sorted(all_heads, key=lambda x: x["post_object_mass"], reverse=True))
            value_rank = unique_heads(sorted(all_heads, key=lambda x: x["value_alignment_mean"], reverse=True))
            abs_value_rank = unique_heads(sorted(all_heads, key=lambda x: x["value_alignment_abs_mean"], reverse=True))
            object_rank = unique_heads(sorted(all_heads, key=lambda x: x["object_mass"], reverse=True))
            random_rank = deterministic_random_heads(patch_layers, num_heads, max(args.candidate_pool, max(set_sizes)), 24000 + target_idx)
            pool = unique_heads(attention_rank[:args.candidate_pool] + value_rank[:args.candidate_pool] + abs_value_rank[:args.candidate_pool] + object_rank[:args.candidate_pool] + random_rank[:args.candidate_pool])

            single_rows = []
            for spec in pool:
                patched = run_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads, head_specs=[spec])
                single_rows.append({**spec, **summarize_condition(patched, baseline, target_idx, categories)})
            target_rank = unique_heads(sorted(single_rows, key=lambda x: x["target_delta"]))
            proj_rank = unique_heads(sorted(single_rows, key=lambda x: x["answer_proj_delta"]))

            pre_bases = {}
            for layer_id in patch_layers:
                basis, _sv = svd_basis(build_category_contrast_matrix(pre_centers_by_layer[layer_id], categories, cat), args.rank)
                pre_bases[layer_id] = basis

            cat_out: dict[str, Any] = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                },
                "top_heads": {
                    "attention_mass": attention_rank[:args.candidate_pool],
                    "value_aligned": value_rank[:args.candidate_pool],
                    "abs_value_aligned": abs_value_rank[:args.candidate_pool],
                    "target_discovered": target_rank[:args.candidate_pool],
                    "projection_discovered": proj_rank[:args.candidate_pool],
                    "object_control": object_rank[:args.candidate_pool],
                    "random_control": random_rank[:args.candidate_pool],
                },
                "single_head_pool": single_rows,
                "set_conditions": [],
                "mlp_subspace_conditions": [],
            }

            set_ranks = {
                "attention_mass": attention_rank,
                "value_aligned": value_rank,
                "abs_value_aligned": abs_value_rank,
                "target_discovered": target_rank,
                "projection_discovered": proj_rank,
                "object_control": object_rank,
                "random_control": random_rank,
            }
            for set_name, ranking in set_ranks.items():
                for size in set_sizes:
                    selected = unique_heads(ranking)[:size]
                    if not selected:
                        continue
                    patched = run_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads, head_specs=selected)
                    cat_out["set_conditions"].append({
                        "set_name": set_name,
                        "set_size": int(size),
                        "heads": [{"patch_layer": int(x["patch_layer"]), "head_id": int(x["head_id"])} for x in selected],
                        **summarize_condition(patched, baseline, target_idx, categories),
                    })

            for layer_id in patch_layers:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, peak_layer, monitor_basis, num_heads,
                    mlp_specs=[(layer_id, "pre_answer", pre_bases[layer_id])],
                    mlp_scale=args.scale,
                )
                cat_out["mlp_subspace_conditions"].append({
                    "patch_layer": int(layer_id),
                    "site": "pre_answer",
                    "rank": int(pre_bases[layer_id].shape[0]),
                    **summarize_condition(patched, baseline, target_idx, categories),
                })

            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row.get('set_name', row.get('site',''))} k{row.get('set_size','')} L{row.get('patch_layer','')} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 124 Writer Set Value Alignment: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | best attention set | best value set | best target-discovered set | best object control | best random control | best pre-MLP subspace |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        sets = item["set_conditions"]
        mlps = item["mlp_subspace_conditions"]

        def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
            return min(rows, key=lambda x: x["target_delta"]) if rows else None

        lines.append(
            f"| {cat} | "
            f"{_fmt(best([x for x in sets if x['set_name']=='attention_mass']))} | "
            f"{_fmt(best([x for x in sets if x['set_name'] in ('value_aligned','abs_value_aligned')]))} | "
            f"{_fmt(best([x for x in sets if x['set_name']=='target_discovered']))} | "
            f"{_fmt(best([x for x in sets if x['set_name']=='object_control']))} | "
            f"{_fmt(best([x for x in sets if x['set_name']=='random_control']))} | "
            f"{_fmt(best(mlps))} |"
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
    parser.add_argument("--set-sizes", default="1,2,4,8,16")
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase124_{args.model}_writer_set_value_alignment.json"
    md_path = out_dir / f"phase124_{args.model}_writer_set_value_alignment.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
