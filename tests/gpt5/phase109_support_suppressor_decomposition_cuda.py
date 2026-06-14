#!/usr/bin/env python3
"""
Phase 109: support/suppressor decomposition.

Decompose each category boundary B into:
  B_parallel: component aligned with category readout direction W_c
  B_orth:     residual component orthogonal to W_c

Then run real forward interventions and measure final DCF logits.
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
from phase107_causal_boundary_removal_cuda import (  # noqa: E402
    BOUNDARY_LAYER,
    build_boundaries,
    capture_centers,
    deterministic_random_direction,
    score_logits,
    summarize_delta,
)


OUT_ROOT = Path("results/gpt5_phase109_support_suppressor_decomposition")
CATEGORIES = ["number", "time", "container", "clothing", "furniture", "plant"]
SCALES = [0.5, 1.0, 1.5]
POSITIONS = ["answer_last", "both"]
KINDS = ["full_boundary", "readout_parallel", "orthogonal", "random_same_norm", "neighbor_boundary"]
NEIGHBOR_CONTROL = {
    "number": "animal",
    "time": "animal",
    "container": "fruit",
    "clothing": "tool",
    "furniture": "building",
    "plant": "color",
}
PHASE108_BEST_LAYERS = {
    "qwen3": {"number": 35, "time": 35, "container": 32, "clothing": 33, "furniture": 35, "plant": 32},
    "glm4": {"number": 15, "time": 16, "container": 15, "clothing": 17, "furniture": 15, "plant": 16},
    "deepseek7b": {"number": 27, "time": 26, "container": 27, "clothing": 25, "furniture": 26, "plant": 25},
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = float(np.dot(b, b))
    if denom < 1e-10:
        return np.zeros_like(a)
    return (float(np.dot(a, b)) / denom) * b


def build_readout_directions(
    readout_rows: np.ndarray,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
) -> dict[str, np.ndarray]:
    dirs: dict[str, np.ndarray] = {}
    all_by_cat = {}
    for cat in categories:
        ids = cat_local_ids.get(cat, [])
        if ids:
            all_by_cat[cat] = readout_rows[ids].mean(axis=0).astype(np.float32)
    for cat in categories:
        own = all_by_cat.get(cat)
        if own is None:
            continue
        others = [v for k, v in all_by_cat.items() if k != cat]
        other_mean = np.mean(np.stack(others), axis=0).astype(np.float32)
        w = own - other_mean
        dirs[cat] = w / (np.linalg.norm(w) + 1e-8)
    return dirs


def make_patch_hook(direction: torch.Tensor, pos_map: dict[str, torch.Tensor], position_mode: str, scale: float):
    direction = direction / (direction.norm() + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        batch_idx = torch.arange(out.shape[0], device=out.device)
        patch_positions = [pos_map["answer_last"], pos_map["object_last"]] if position_mode == "both" else [pos_map[position_mode]]
        for pos_cpu in patch_positions:
            pos = pos_cpu.to(out.device)
            vecs = out[batch_idx, pos, :].float()
            coeff = (vecs @ direction.float()).to(out.dtype)
            out[batch_idx, pos, :] = out[batch_idx, pos, :] - scale * coeff[:, None] * direction.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def run_prompts(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, str]],
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    direction: np.ndarray | None = None,
    position_mode: str = "answer_last",
    scale: float = 1.0,
) -> np.ndarray:
    scores = []
    module_index = layer_id - 1
    for start in range(0, len(prompts), batch_size):
        batch_items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in batch_items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        object_pos = torch.tensor([
            object_last_position(tokenizer, item["prompt"], item["obj"], int(answer_pos[bi]))
            for bi, item in enumerate(batch_items)
        ], dtype=torch.long)
        handle = None
        if direction is not None:
            d = torch.tensor(direction, device=device, dtype=torch.float32)
            handle = layers[module_index].register_forward_hook(
                make_patch_hook(d, {"answer_last": answer_pos, "object_last": object_pos}, position_mode, scale)
            )
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        if handle is not None:
            handle.remove()
        pos = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        scores.append(score_logits(logits, cat_local_ids, categories))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0)


def direction_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.input_device
    layers = get_layers(model)
    categories = list(CATEGORY_OBJECTS.keys())
    cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)
    center_layer = args.center_layer if args.center_layer is not None else BOUNDARY_LAYER[args.model]
    test_categories = args.categories.split(",") if args.categories else CATEGORIES

    alloc, reserved = vram_gb()
    log(f"{args.model}: center_layer=L{center_layer}, vram={alloc:.2f}/{reserved:.2f}GB")
    centers = capture_centers(model, tokenizer, device, categories, center_layer, args.train_objects, args.batch_size, args.max_length)
    boundaries = build_boundaries(centers, categories)

    result = {
        "phase": 109,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "center_layer_for_boundaries": center_layer,
        "train_objects_per_category": args.train_objects,
        "test_objects_per_category": args.test_objects,
        "templates": [t["name"] for t in TEMPLATES],
        "positions": POSITIONS,
        "scales": SCALES,
        "kinds": KINDS,
        "test_categories": test_categories,
        "readout_token_labels": token_labels,
        "category_results": {},
    }

    test_start = args.train_objects
    test_end = args.train_objects + args.test_objects
    for idx, cat in enumerate(test_categories, 1):
        log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
        prompts = []
        for tpl in TEMPLATES:
            for obj in CATEGORY_OBJECTS[cat][test_start:test_end]:
                prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj)})
        target_idx = categories.index(cat)
        b = boundaries[cat]
        w = readout_dirs[cat]
        b_parallel = projection(b, w).astype(np.float32)
        b_orth = (b - b_parallel).astype(np.float32)
        neighbor = boundaries[NEIGHBOR_CONTROL[cat]]
        random_dir = deterministic_random_direction(b.shape[0], 5000 + target_idx)
        directions = {
            "full_boundary": b,
            "readout_parallel": b_parallel,
            "orthogonal": b_orth,
            "random_same_norm": random_dir,
            "neighbor_boundary": neighbor,
        }
        baseline = run_prompts(
            model, tokenizer, device, layers, prompts, center_layer, cat_local_ids, categories,
            args.batch_size, args.max_length
        )
        cat_out = {
            "n_prompts": len(prompts),
            "baseline_target_mean": float(baseline[:, target_idx].mean()),
            "boundary_readout_cos": direction_cos(b, w),
            "parallel_norm_fraction": float(np.linalg.norm(b_parallel) / (np.linalg.norm(b) + 1e-8)),
            "conditions": [],
        }
        layers_to_test = sorted(set([center_layer, PHASE108_BEST_LAYERS[args.model][cat]]))
        for layer_id in layers_to_test:
            for pos in POSITIONS:
                for scale in SCALES:
                    for kind in KINDS:
                        patched = run_prompts(
                            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                            args.batch_size, args.max_length, direction=directions[kind],
                            position_mode=pos, scale=scale
                        )
                        summary = summarize_delta(patched - baseline, target_idx, categories)
                        cat_out["conditions"].append({
                            "layer": layer_id,
                            "position": pos,
                            "scale": scale,
                            "kind": kind,
                            **summary,
                        })
        result["category_results"][cat] = cat_out
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 109 Support/Suppressor Decomposition: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| category | cos(B,W) | parallel norm | strongest parallel down | strongest orthogonal release | strongest full down |")
    lines.append("|---|---:|---:|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]
        par = [c for c in conds if c["kind"] == "readout_parallel"]
        orth = [c for c in conds if c["kind"] == "orthogonal"]
        full = [c for c in conds if c["kind"] == "full_boundary"]
        par_down = min(par, key=lambda x: x["target_delta"])
        full_down = min(full, key=lambda x: x["target_delta"])
        orth_rel = max(orth, key=lambda x: x["max_other_delta"])
        top = orth_rel["top_releases"][0] if orth_rel["top_releases"] else {"category": "none", "delta": 0.0}
        lines.append(
            f"| {cat} | {item['boundary_readout_cos']:.3f} | {item['parallel_norm_fraction']:.3f} | "
            f"L{par_down['layer']} {par_down['position']} s{par_down['scale']} T{par_down['target_delta']:.3f} | "
            f"L{orth_rel['layer']} {orth_rel['position']} s{orth_rel['scale']} {top['category']}+{top['delta']:.3f} | "
            f"L{full_down['layer']} {full_down['position']} s{full_down['scale']} T{full_down['target_delta']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--center-layer", type=int, default=None)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    loaded = None
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_model(args)
        json_path = out_dir / f"phase109_{args.model}_support_suppressor_decomposition.json"
        md_path = out_dir / f"phase109_{args.model}_support_suppressor_decomposition.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(result, md_path)
        log(f"Wrote {json_path}")
        log(f"Wrote {md_path}")
    finally:
        release_loaded(loaded)
        if args.hard_exit_after_model:
            os._exit(0)


if __name__ == "__main__":
    main()
