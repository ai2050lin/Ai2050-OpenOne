#!/usr/bin/env python3
"""
Phase 118: causal axis transport and source-to-answer closure.

Build answer-site causal axes, then patch the same axes at object_last,
answer_last, or both across a short layer sweep. Monitor final DCF logits and
answer-layer axis projection to test whether axes are written upstream or
assembled mainly at the answer site.
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
from phase106_multitemplate_residual_cuda import TEMPLATES, object_last_position  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase115_causal_subspace_robustness_cuda import capture_centers_template_subset  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, run_condition, svd_basis  # noqa: E402
from phase117_basis_rotation_causal_axis_cuda import orthonormalize_rows, varimax_basis  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase118_causal_axis_transport_closure")
TEST_CATEGORIES = ["number", "container", "plant"]
PATCH_SITES = ["object_last", "answer_last", "both"]
AXIS_TYPES = ["varimax_best", "svd_subspace", "random_in_subspace"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def random_in_subspace(basis: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeff = rng.standard_normal(basis.shape[0]).astype(np.float32)
    coeff /= np.linalg.norm(coeff) + 1e-8
    vec = (coeff @ basis).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec[None, :]


def make_multi_site_subspace_hook(
    basis: torch.Tensor,
    answer_pos: torch.Tensor,
    object_pos: torch.Tensor,
    patch_site: str,
    scale: float,
):
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        bidx = torch.arange(out.shape[0], device=out.device)
        b = basis.to(out.device).float()
        positions = [answer_pos] if patch_site == "answer_last" else [object_pos]
        if patch_site == "both":
            positions = [object_pos, answer_pos]
        for pos_cpu in positions:
            pos = pos_cpu.to(out.device)
            vecs = out[bidx, pos, :].float()
            proj = (vecs @ b.T) @ b
            out[bidx, pos, :] = out[bidx, pos, :] - scale * proj.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def run_path_condition(
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
    monitor_axis: np.ndarray,
    patch_layer: int | None = None,
    patch_site: str = "answer_last",
    patch_basis: np.ndarray | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    score_chunks = []
    answer_proj_chunks = []
    object_proj_chunks = []
    axis = torch.tensor(monitor_axis, device=device, dtype=torch.float32)
    axis = axis / (axis.norm() + 1e-8)
    module_index = None if patch_layer is None else patch_layer - 1

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        object_pos = torch.tensor([
            object_last_position(tokenizer, item["prompt"], item["obj"], int(answer_pos[bi]))
            for bi, item in enumerate(items)
        ], dtype=torch.long)

        handle = None
        if patch_basis is not None and module_index is not None:
            b = torch.tensor(patch_basis, device=device, dtype=torch.float32)
            handle = layers[module_index].register_forward_hook(
                make_multi_site_subspace_hook(b, answer_pos, object_pos, patch_site, scale)
            )
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        if handle is not None:
            handle.remove()

        pos_gpu = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        score_chunks.append(score_logits(logits, cat_local_ids, categories))

        hs = out.hidden_states[monitor_layer]
        bidx = torch.arange(hs.shape[0], device=hs.device)
        ans = hs[bidx, answer_pos.to(hs.device), :].float()
        obj = hs[bidx, object_pos.to(hs.device), :].float()
        monitor = axis.to(hs.device)
        answer_proj_chunks.append((ans @ monitor).detach().float().cpu().numpy())
        object_proj_chunks.append((obj @ monitor).detach().float().cpu().numpy())

        del out, batch
        torch.cuda.empty_cache()

    return {
        "scores": np.concatenate(score_chunks, axis=0),
        "answer_axis_proj": np.concatenate(answer_proj_chunks, axis=0),
        "object_axis_proj": np.concatenate(object_proj_chunks, axis=0),
    }


def pick_varimax_axis(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    baseline_scores: np.ndarray,
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    target_idx: int,
    batch_size: int,
    max_length: int,
    scale: float,
    basis: np.ndarray,
) -> dict[str, Any]:
    vbasis = varimax_basis(basis)
    rows = []
    for bi, vec in enumerate(vbasis):
        patched = run_condition(
            model, tokenizer, device, layers, prompts, layer_id,
            cat_local_ids, categories, batch_size, max_length, vec[None, :], scale
        )
        summary = summarize_delta(patched - baseline_scores, target_idx, categories)
        rows.append({"basis_index": bi, "axis": vec[None, :], **summary})
    best = min(rows, key=lambda r: r["target_delta"])
    return {
        "basis_index": int(best["basis_index"]),
        "axis": best["axis"].astype(np.float32),
        "selection_target_delta": float(best["target_delta"]),
        "selection_max_other_delta": float(best["max_other_delta"]),
        "selection_top_releases": best.get("top_releases", []),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        monitor_layer = args.monitor_layer if args.monitor_layer is not None else BOUNDARY_LAYER[args.model]
        first_layer = max(1, monitor_layer - args.layer_back)
        patch_layers = list(range(first_layer, monitor_layer + 1))
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)

        alloc, reserved = vram_gb()
        log(
            f"{args.model}: monitor=L{monitor_layer}, patch_layers={patch_layers}, rank={args.rank}, "
            f"train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB"
        )

        centers = capture_centers_template_subset(
            model, tokenizer, device, categories, monitor_layer,
            args.train_objects, args.batch_size, args.max_length,
            list(range(len(TEMPLATES)))
        )

        result: dict[str, Any] = {
            "phase": 118,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitor_layer": monitor_layer,
            "patch_layers": patch_layers,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "patch_sites": PATCH_SITES,
            "axis_types": AXIS_TYPES,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            contrast = build_category_contrast_matrix(centers, categories, cat)
            basis, singular_values = svd_basis(contrast, args.rank)

            seed = 18000 + target_idx * 811 + args.rank
            random_axis = random_in_subspace(basis, seed)

            baseline = run_path_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, monitor_layer, basis[0],
            )
            baseline_scores = baseline["scores"]
            varimax_choice = pick_varimax_axis(
                model, tokenizer, device, layers, prompts, baseline_scores,
                monitor_layer, cat_local_ids, categories, target_idx,
                args.batch_size, args.max_length, args.scale, basis
            )
            varimax_axis = varimax_choice["axis"]

            axis_map = {
                "varimax_best": varimax_axis,
                "svd_subspace": basis,
                "random_in_subspace": random_axis,
            }
            monitor_axis_map = {
                "varimax_best": varimax_axis[0],
                "svd_subspace": varimax_axis[0],
                "random_in_subspace": random_axis[0],
            }

            cat_out: dict[str, Any] = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline_scores[:, target_idx].mean()),
                "baseline_answer_axis_proj_mean": float(
                    run_path_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, monitor_layer, varimax_axis[0],
                    )["answer_axis_proj"].mean()
                ),
                "singular_values": [float(x) for x in singular_values],
                "varimax_best_selection": {k: v for k, v in varimax_choice.items() if k != "axis"},
                "conditions": [],
            }

            base_for_axis: dict[str, dict[str, np.ndarray]] = {}
            for axis_name, monitor_axis in monitor_axis_map.items():
                base_for_axis[axis_name] = run_path_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, monitor_layer, monitor_axis,
                )

            for axis_name, patch_basis in axis_map.items():
                log(f"  {cat}: axis={axis_name}")
                base_axis = base_for_axis[axis_name]
                for layer_id in patch_layers:
                    for patch_site in PATCH_SITES:
                        patched = run_path_condition(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, monitor_layer, monitor_axis_map[axis_name],
                            patch_layer=layer_id, patch_site=patch_site,
                            patch_basis=patch_basis, scale=args.scale
                        )
                        summary = summarize_delta(patched["scores"] - base_axis["scores"], target_idx, categories)
                        cat_out["conditions"].append({
                            "axis_type": axis_name,
                            "axis_rank": int(patch_basis.shape[0]),
                            "patch_layer": layer_id,
                            "patch_site": patch_site,
                            "answer_axis_proj_delta": float(
                                patched["answer_axis_proj"].mean() - base_axis["answer_axis_proj"].mean()
                            ),
                            "object_axis_proj_delta": float(
                                patched["object_axis_proj"].mean() - base_axis["object_axis_proj"].mean()
                            ),
                            **summary,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 118 Causal Axis Transport Closure: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | axis | best object_last | best answer_last | best both |")
    lines.append("|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        for axis in result["axis_types"]:
            conds = [c for c in item["conditions"] if c["axis_type"] == axis]

            def fmt(site: str) -> str:
                rows = [c for c in conds if c["patch_site"] == site]
                if not rows:
                    return "NA"
                r = min(rows, key=lambda x: x["target_delta"])
                return f"L{r['patch_layer']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f} Aproj{r['answer_axis_proj_delta']:+.2f}"

            lines.append(f"| {cat} | {axis} | {fmt('object_last')} | {fmt('answer_last')} | {fmt('both')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--monitor-layer", type=int, default=None)
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase118_{args.model}_causal_axis_transport_closure.json"
    md_path = out_dir / f"phase118_{args.model}_causal_axis_transport_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
