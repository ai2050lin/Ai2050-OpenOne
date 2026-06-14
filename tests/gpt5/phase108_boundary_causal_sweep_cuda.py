#!/usr/bin/env python3
"""
Phase 108: boundary causal sweep.

Sweep:
  categories: number,time,container,clothing,furniture,plant
  layers: boundary_layer-3 .. boundary_layer
  positions: answer_last, object_last, both
  scales: 0.25,0.5,1.0,1.5
  controls: boundary, random_same_norm, neighbor_boundary

Run one model per process with --hard-exit-after-model.
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


OUT_ROOT = Path("results/gpt5_phase108_boundary_causal_sweep")
SWEEP_CATEGORIES = ["number", "time", "container", "clothing", "furniture", "plant"]
SCALES = [0.25, 0.5, 1.0, 1.5]
POSITIONS = ["answer_last", "object_last", "both"]
CONTROL_KINDS = ["boundary", "random_same_norm", "neighbor_boundary"]
NEIGHBOR_CONTROL = {
    "number": "animal",
    "time": "animal",
    "container": "fruit",
    "clothing": "tool",
    "furniture": "building",
    "plant": "color",
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_multi_pos_patch_hook(direction: torch.Tensor, pos_map: dict[str, torch.Tensor], position_mode: str, scale: float):
    direction = direction / (direction.norm() + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        batch_idx = torch.arange(out.shape[0], device=out.device)
        if position_mode == "both":
            patch_positions = [pos_map["answer_last"], pos_map["object_last"]]
        else:
            patch_positions = [pos_map[position_mode]]
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
                make_multi_pos_patch_hook(
                    d,
                    {"answer_last": answer_pos, "object_last": object_pos},
                    position_mode,
                    scale,
                )
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


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.input_device
    layers = get_layers(model)
    categories = list(CATEGORY_OBJECTS.keys())
    cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    center_layer = args.center_layer if args.center_layer is not None else BOUNDARY_LAYER[args.model]
    layer_start = max(1, center_layer - args.layer_back)
    layer_ids = list(range(layer_start, center_layer + 1))
    test_categories = args.categories.split(",") if args.categories else SWEEP_CATEGORIES

    alloc, reserved = vram_gb()
    log(f"{args.model}: center_layer=L{center_layer}, sweep_layers={layer_ids}, vram={alloc:.2f}/{reserved:.2f}GB")
    centers = capture_centers(
        model, tokenizer, device, categories, center_layer, args.train_objects, args.batch_size, args.max_length
    )
    boundaries = build_boundaries(centers, categories)
    random_dirs = {
        cat: deterministic_random_direction(boundaries[cat].shape[0], 3000 + categories.index(cat))
        for cat in test_categories
    }

    result = {
        "phase": 108,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "center_layer_for_boundaries": center_layer,
        "sweep_layers": layer_ids,
        "train_objects_per_category": args.train_objects,
        "test_objects_per_category": args.test_objects,
        "templates": [t["name"] for t in TEMPLATES],
        "scales": SCALES,
        "positions": POSITIONS,
        "control_kinds": CONTROL_KINDS,
        "test_categories": test_categories,
        "readout_token_labels": token_labels,
        "category_results": {},
    }

    test_start = args.train_objects
    test_end = args.train_objects + args.test_objects
    for ci, cat in enumerate(test_categories, 1):
        log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
        prompts = []
        for tpl in TEMPLATES:
            for obj in CATEGORY_OBJECTS[cat][test_start:test_end]:
                prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj)})
        target_idx = categories.index(cat)
        baseline = run_prompts(
            model, tokenizer, device, layers, prompts, center_layer, cat_local_ids, categories,
            args.batch_size, args.max_length
        )
        cat_out = {"n_prompts": len(prompts), "baseline_target_mean": float(baseline[:, target_idx].mean()), "conditions": []}
        for layer_id in layer_ids:
            for pos in POSITIONS:
                for scale in SCALES:
                    for kind in CONTROL_KINDS:
                        if kind == "boundary":
                            direction = boundaries[cat]
                        elif kind == "random_same_norm":
                            direction = random_dirs[cat]
                        else:
                            direction = boundaries[NEIGHBOR_CONTROL[cat]]
                        patched = run_prompts(
                            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                            args.batch_size, args.max_length, direction=direction, position_mode=pos, scale=scale
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
    lines = [f"# Phase 108 Boundary Causal Sweep: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Boundary center layer: L{result['center_layer_for_boundaries']}")
    lines.append("")
    lines.append("| category | baseline | strongest target down | strongest target up | strongest release |")
    lines.append("|---|---:|---|---|---|")
    for cat, item in result["category_results"].items():
        boundary = [c for c in item["conditions"] if c["kind"] == "boundary"]
        down = min(boundary, key=lambda x: x["target_delta"])
        up = max(boundary, key=lambda x: x["target_delta"])
        rel = max(boundary, key=lambda x: x["max_other_delta"])
        top = rel["top_releases"][0] if rel["top_releases"] else {"category": "none", "delta": 0.0}
        lines.append(
            f"| {cat} | {item['baseline_target_mean']:.3f} | "
            f"L{down['layer']} {down['position']} s{down['scale']} T{down['target_delta']:.3f} | "
            f"L{up['layer']} {up['position']} s{up['scale']} T{up['target_delta']:.3f} | "
            f"L{rel['layer']} {rel['position']} s{rel['scale']} {top['category']}+{top['delta']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--layer-back", type=int, default=3)
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
        json_path = out_dir / f"phase108_{args.model}_boundary_causal_sweep.json"
        md_path = out_dir / f"phase108_{args.model}_boundary_causal_sweep.md"
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
