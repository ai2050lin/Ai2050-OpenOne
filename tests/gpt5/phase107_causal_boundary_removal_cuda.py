#!/usr/bin/env python3
"""
Phase 107: downstream causal boundary removal on CUDA.

This phase moves beyond logit-lens atlas evidence. It estimates category
boundary vectors from train objects at the model-specific boundary layer, then
removes the natural projection along that boundary during a real forward pass
on heldout objects. Final logits are measured with the same 32-category DCF
readout and compared to a random same-norm direction control.
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


OUT_ROOT = Path("results/gpt5_phase107_causal_boundary_removal")
MODELS = ["qwen3", "glm4", "deepseek7b"]
BOUNDARY_LAYER = {"qwen3": 35, "glm4": 18, "deepseek7b": 27}
TEST_CATEGORIES = [
    "fruit",
    "vehicle",
    "clothing",
    "furniture",
    "plant",
    "body",
    "place",
    "building",
    "time",
    "number",
    "weather",
    "container",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_logits(logits: torch.Tensor, cat_local_ids: dict[str, list[int]], categories: list[str]) -> np.ndarray:
    arr = logits.detach().float().cpu().numpy()
    scores = np.zeros((arr.shape[0], len(categories)), dtype=np.float32)
    for ci, cat in enumerate(categories):
        ids = cat_local_ids.get(cat, [])
        if ids:
            scores[:, ci] = arr[:, ids].mean(axis=1)
    return scores


def make_patch_hook(direction: torch.Tensor, positions: torch.Tensor, scale: float = 1.0):
    """Remove each sample's natural projection along direction at answer_last."""
    direction = direction / (direction.norm() + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        b = torch.arange(out.shape[0], device=out.device)
        pos = positions.to(out.device)
        vecs = out[b, pos, :].float()
        coeff = (vecs @ direction.float()).to(out.dtype)
        out[b, pos, :] = out[b, pos, :] - scale * coeff[:, None] * direction.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def deterministic_random_direction(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def capture_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    categories: list[str],
    layer_id: int,
    train_n: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    n_tpl = len(TEMPLATES)
    n_cat = len(categories)
    sums = np.zeros((n_tpl, n_cat, d_model), dtype=np.float64)
    counts = np.zeros((n_tpl, n_cat), dtype=np.int64)
    items = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_n]:
                items.append((ti, ci, tpl["text"].format(obj=obj)))

    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            prompts = [x[2] for x in batch_items]
            batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            pos = batch["attention_mask"].sum(dim=1) - 1
            hs = out.hidden_states[layer_id]
            picked = hs[torch.arange(hs.shape[0], device=hs.device), pos].detach().float().cpu().numpy()
            for bi, (ti, ci, _prompt) in enumerate(batch_items):
                sums[ti, ci] += picked[bi].astype(np.float32)
                counts[ti, ci] += 1
            del out, batch
            torch.cuda.empty_cache()
    return (sums / counts[:, :, None]).astype(np.float32)


def build_boundaries(centers: np.ndarray, categories: list[str]) -> dict[str, np.ndarray]:
    out = {}
    n_tpl, n_cat, _dim = centers.shape
    for ci, cat in enumerate(categories):
        vecs = []
        for ti in range(n_tpl):
            other = np.mean(np.delete(centers[ti], ci, axis=0), axis=0)
            vecs.append(centers[ti, ci] - other)
        b = np.mean(np.stack(vecs), axis=0).astype(np.float32)
        out[cat] = b / (np.linalg.norm(b) + 1e-8)
    return out


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    direction: np.ndarray | None,
) -> np.ndarray:
    scores_all = []
    module_index = layer_id - 1
    if module_index < 0:
        raise ValueError("Cannot patch embedding layer with this script")
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        pos = batch["attention_mask"].sum(dim=1) - 1
        handle = None
        if direction is not None:
            d = torch.tensor(direction, device=device, dtype=torch.float32)
            handle = layers[module_index].register_forward_hook(make_patch_hook(d, pos))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        if handle is not None:
            handle.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        scores_all.append(score_logits(logits, cat_local_ids, categories))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(scores_all, axis=0)


def summarize_delta(delta: np.ndarray, target_idx: int, categories: list[str]) -> dict[str, Any]:
    mean_delta = delta.mean(axis=0)
    releases = []
    for ci, cat in enumerate(categories):
        if ci != target_idx and mean_delta[ci] > 0:
            releases.append({"category": cat, "delta": float(mean_delta[ci])})
    releases.sort(key=lambda x: x["delta"], reverse=True)
    return {
        "target_delta": float(mean_delta[target_idx]),
        "max_other_delta": float(max([x["delta"] for x in releases], default=0.0)),
        "top_releases": releases[:8],
        "mean_delta_by_category": {cat: float(mean_delta[i]) for i, cat in enumerate(categories)},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.input_device
    layers = get_layers(model)
    categories = list(CATEGORY_OBJECTS.keys())
    cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    layer_id = args.layer if args.layer is not None else BOUNDARY_LAYER[args.model]
    train_n = args.train_objects
    test_start = train_n
    test_end = train_n + args.test_objects
    test_categories = TEST_CATEGORIES if not args.categories else args.categories.split(",")

    alloc, reserved = vram_gb()
    log(f"{args.model}: L={len(layers)}, boundary_layer={layer_id}, vram={alloc:.2f}/{reserved:.2f}GB")
    log("Building train centers")
    centers = capture_centers(model, tokenizer, device, categories, layer_id, train_n, args.batch_size, args.max_length)
    boundaries = build_boundaries(centers, categories)

    result = {
        "phase": 107,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boundary_layer": layer_id,
        "train_objects_per_category": train_n,
        "test_objects_per_category": args.test_objects,
        "templates": [t["name"] for t in TEMPLATES],
        "test_categories": test_categories,
        "readout_token_labels": token_labels,
        "category_results": {},
        "notes": [
            "Final logits are measured after real forward intervention.",
            "remove_boundary subtracts the natural projection along the train-estimated category boundary at answer_last.",
            "random_same_norm removes a projection along a deterministic random unit direction.",
        ],
    }

    for idx, cat in enumerate(test_categories, 1):
        log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
        prompts = []
        for tpl in TEMPLATES:
            for obj in CATEGORY_OBJECTS[cat][test_start:test_end]:
                prompts.append(tpl["text"].format(obj=obj))
        target_idx = categories.index(cat)
        b = boundaries[cat]
        random_dir = deterministic_random_direction(b.shape[0], seed=1000 + categories.index(cat))
        baseline = run_condition(
            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
            args.batch_size, args.max_length, None
        )
        removed = run_condition(
            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
            args.batch_size, args.max_length, b
        )
        control = run_condition(
            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
            args.batch_size, args.max_length, random_dir
        )
        remove_delta = removed - baseline
        control_delta = control - baseline
        result["category_results"][cat] = {
            "n_prompts": len(prompts),
            "baseline_target_mean": float(baseline[:, target_idx].mean()),
            "remove_boundary": summarize_delta(remove_delta, target_idx, categories),
            "random_same_norm": summarize_delta(control_delta, target_idx, categories),
        }
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 107 Causal Boundary Removal: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Boundary layer: L{result['boundary_layer']}")
    lines.append("")
    lines.append("| category | n | baseline target | remove target Δ | random target Δ | top release | control top release |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for cat, item in result["category_results"].items():
        rem = item["remove_boundary"]
        ctl = item["random_same_norm"]
        top = rem["top_releases"][0] if rem["top_releases"] else {"category": "none", "delta": 0.0}
        ctop = ctl["top_releases"][0] if ctl["top_releases"] else {"category": "none", "delta": 0.0}
        lines.append(
            f"| {cat} | {item['n_prompts']} | {item['baseline_target_mean']:.3f} | "
            f"{rem['target_delta']:.3f} | {ctl['target_delta']:.3f} | "
            f"{top['category']} {top['delta']:.3f} | {ctop['category']} {ctop['delta']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    loaded = None
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_model(args)
        json_path = out_dir / f"phase107_{args.model}_causal_boundary_removal.json"
        md_path = out_dir / f"phase107_{args.model}_causal_boundary_removal.md"
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
