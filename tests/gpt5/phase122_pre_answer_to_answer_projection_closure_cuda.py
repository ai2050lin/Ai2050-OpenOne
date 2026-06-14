#!/usr/bin/env python3
"""
Phase 122: pre-answer to answer projection closure.

Remove local pre-answer fields at nearby layers, then monitor final logits and
the peak-layer answer_last projection on the answer-site causal axis.
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


OUT_ROOT = Path("results/gpt5_phase122_pre_answer_to_answer_projection_closure")
TEST_CATEGORIES = ["number", "container", "plant"]
PRE_SITE = "post_object_excluding_answer"
ANSWER_SITE = "answer_last"
AXIS_TYPES = ["local_varimax_best", "local_svd_subspace"]
PATCH_MODES = ["pre_remove", "answer_remove", "pre_plus_answer"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_multi_site_hook(specs: list[tuple[torch.Tensor, list[list[int]]]], scale: float):
    prepared = [(basis / (basis.norm(dim=1, keepdim=True) + 1e-8), positions) for basis, positions in specs]

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        for basis, batch_positions in prepared:
            b = basis.to(out.device).float()
            for bi, positions in enumerate(batch_positions):
                pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                vecs = out[bi, pos, :].float()
                proj = (vecs @ b.T) @ b
                out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def projection_values(vecs: torch.Tensor, basis_np: np.ndarray) -> np.ndarray:
    basis = torch.tensor(basis_np, device=vecs.device, dtype=torch.float32)
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)
    coeff = vecs.float() @ basis.T
    if basis.shape[0] == 1:
        return coeff[:, 0].detach().float().cpu().numpy()
    return coeff.norm(dim=1).detach().float().cpu().numpy()


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
    patch_layer: int | None = None,
    patches: list[tuple[str, np.ndarray]] | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    module_index = None if patch_layer is None else patch_layer - 1
    patches = patches or []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
        specs = []
        for site, basis in patches:
            batch_positions = [item_positions(tokenizer, item, answer_pos[bi])[site] for bi, item in enumerate(items)]
            specs.append((torch.tensor(basis, device=device, dtype=torch.float32), batch_positions))
        handle = None
        if specs and module_index is not None:
            handle = layers[module_index].register_forward_hook(make_multi_site_hook(specs, scale))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        if handle is not None:
            handle.remove()
        pos_gpu = torch.tensor(answer_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        patch_layers = list(range(max(1, peak_layer - args.layer_back), peak_layer + 1))
        axis_types = [x.strip() for x in args.axis_types.split(",") if x.strip()]
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, patch_layers={patch_layers}, rank={args.rank}, vram={alloc:.2f}/{reserved:.2f}GB")

        result = {
            "phase": 122,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "patch_layers": patch_layers,
            "monitor_layer": peak_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "axis_types": axis_types,
            "patch_modes": PATCH_MODES,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        basis_cache: dict[tuple[int, str, str], tuple[np.ndarray, list[float]]] = {}
        for layer_id in patch_layers:
            for site in [PRE_SITE, ANSWER_SITE]:
                log(f"Building local centers L{layer_id} {site}")
                centers = capture_local_centers(model, tokenizer, device, categories, layer_id, site, args.train_objects, args.batch_size, args.max_length)
                for cat in test_categories:
                    contrast = build_category_contrast_matrix(centers, categories, cat)
                    basis, singular_values = svd_basis(contrast, args.rank)
                    basis_cache[(layer_id, site, cat)] = (basis, [float(x) for x in singular_values])

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            peak_pre_basis, _ = basis_cache[(peak_layer, PRE_SITE, cat)]
            peak_ans_basis, _ = basis_cache[(peak_layer, ANSWER_SITE, cat)]
            baseline_scores_only = run_monitor_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, peak_ans_basis
            )
            baseline_scores = baseline_scores_only["scores"]
            peak_pre_choice = select_local_varimax_axis(model, tokenizer, device, layers, prompts, baseline_scores, peak_layer, PRE_SITE, cat_local_ids, categories, target_idx, args.batch_size, args.max_length, args.scale, peak_pre_basis)
            peak_ans_choice = select_local_varimax_axis(model, tokenizer, device, layers, prompts, baseline_scores, peak_layer, ANSWER_SITE, cat_local_ids, categories, target_idx, args.batch_size, args.max_length, args.scale, peak_ans_basis)

            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline_scores[:, target_idx].mean()),
                "baseline_answer_varimax_proj_mean": float(run_monitor_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, peak_ans_choice["axis"])["answer_proj"].mean()),
                "baseline_answer_subspace_proj_mean": float(baseline_scores_only["answer_proj"].mean()),
                "conditions": [],
            }
            for layer_id in patch_layers:
                log(f"  {cat}: L{layer_id}")
                pre_basis, pre_sv = basis_cache[(layer_id, PRE_SITE, cat)]
                ans_basis, ans_sv = basis_cache[(layer_id, ANSWER_SITE, cat)]
                pre_choice = select_local_varimax_axis(model, tokenizer, device, layers, prompts, baseline_scores, layer_id, PRE_SITE, cat_local_ids, categories, target_idx, args.batch_size, args.max_length, args.scale, pre_basis)
                ans_choice = select_local_varimax_axis(model, tokenizer, device, layers, prompts, baseline_scores, layer_id, ANSWER_SITE, cat_local_ids, categories, target_idx, args.batch_size, args.max_length, args.scale, ans_basis)
                axes = {
                    "local_varimax_best": (pre_choice["axis"], ans_choice["axis"], peak_ans_choice["axis"]),
                    "local_svd_subspace": (pre_basis, ans_basis, peak_ans_basis),
                }
                for axis_type in axis_types:
                    pre_axis, ans_axis, monitor_basis = axes[axis_type]
                    baseline_monitor = run_monitor_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, monitor_basis)
                    mode_patches = {
                        "pre_remove": [(PRE_SITE, pre_axis)],
                        "answer_remove": [(ANSWER_SITE, ans_axis)],
                        "pre_plus_answer": [(PRE_SITE, pre_axis), (ANSWER_SITE, ans_axis)],
                    }
                    for mode, patches in mode_patches.items():
                        patched = run_monitor_condition(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, peak_layer, monitor_basis, layer_id, patches, args.scale)
                        summary = summarize_delta(patched["scores"] - baseline_scores, target_idx, categories)
                        cat_out["conditions"].append({
                            "patch_layer": layer_id,
                            "axis_type": axis_type,
                            "patch_mode": mode,
                            "answer_proj_delta": float(patched["answer_proj"].mean() - baseline_monitor["answer_proj"].mean()),
                            "pre_axis_rank": int(pre_axis.shape[0]),
                            "answer_axis_rank": int(ans_axis.shape[0]),
                            "pre_singular_values": pre_sv,
                            "answer_singular_values": ans_sv,
                            **summary,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 122 Pre-answer to Answer Projection Closure: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| category | axis | best pre remove | best answer remove | best combined | strongest answer proj drop |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        for axis in result["axis_types"]:
            conds = [c for c in item["conditions"] if c["axis_type"] == axis]

            def fmt(rows: list[dict[str, Any]]) -> str:
                if not rows:
                    return "NA"
                r = min(rows, key=lambda x: x["target_delta"])
                return f"L{r['patch_layer']} {r['patch_mode']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f} Aproj{r['answer_proj_delta']:+.2f}"

            proj = min(conds, key=lambda x: x["answer_proj_delta"]) if conds else None
            lines.append(
                f"| {cat} | {axis} | {fmt([c for c in conds if c['patch_mode']=='pre_remove'])} | "
                f"{fmt([c for c in conds if c['patch_mode']=='answer_remove'])} | "
                f"{fmt([c for c in conds if c['patch_mode']=='pre_plus_answer'])} | "
                f"{fmt([proj] if proj else [])} |"
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
    parser.add_argument("--axis-types", default="local_varimax_best,local_svd_subspace")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase122_{args.model}_pre_answer_to_answer_projection_closure.json"
    md_path = out_dir / f"phase122_{args.model}_pre_answer_to_answer_projection_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
