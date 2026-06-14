#!/usr/bin/env python3
"""
Phase 110: split readout-orthogonal category boundary.

For each category:
  B_orth = B - proj(B, W_readout)
  B_neighbor = projection of B_orth onto a neighbor-boundary basis
  B_transport = projection of residual onto object_last->answer_last transport
  B_residual = remaining component

Run real forward interventions for each component.
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
from phase109_support_suppressor_decomposition_cuda import build_readout_directions, projection  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase110_orthogonal_subspace_split")
TEST_CATEGORIES = ["number", "time", "container", "clothing", "furniture", "plant"]
NEIGHBOR_SETS = {
    "number": ["time", "shape", "property", "animal"],
    "time": ["number", "event", "weather", "animal"],
    "container": ["clothing", "furniture", "tool", "fruit"],
    "clothing": ["tool", "furniture", "container", "plant"],
    "furniture": ["clothing", "building", "container", "tool"],
    "plant": ["fruit", "color", "animal", "food"],
}
SCALES = [1.0, 1.5]
POSITIONS = ["answer_last", "both"]
KINDS = ["orthogonal_full", "neighbor_aligned", "transport_aligned", "residual", "random_same_norm"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def basis_project(v: np.ndarray, basis_vecs: list[np.ndarray]) -> np.ndarray:
    valid = [b / (np.linalg.norm(b) + 1e-8) for b in basis_vecs if np.linalg.norm(b) > 1e-8]
    if not valid:
        return np.zeros_like(v)
    B = np.stack(valid, axis=1).astype(np.float32)
    q, _ = np.linalg.qr(B)
    return (q @ (q.T @ v)).astype(np.float32)


def capture_transport_dirs(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    categories: list[str],
    layer_id: int,
    train_n: int,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    sums = {cat: np.zeros((d_model,), dtype=np.float64) for cat in categories}
    counts = {cat: 0 for cat in categories}
    items = []
    for tpl in TEMPLATES:
        for cat in categories:
            for obj in CATEGORY_OBJECTS[cat][:train_n]:
                items.append({"cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x["prompt"] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            object_pos = [
                object_last_position(tokenizer, item["prompt"], item["obj"], answer_pos[bi])
                for bi, item in enumerate(batch_items)
            ]
            hs = out.hidden_states[layer_id]
            ans = hs[torch.arange(hs.shape[0], device=hs.device), torch.tensor(answer_pos, device=hs.device)]
            obj = hs[torch.arange(hs.shape[0], device=hs.device), torch.tensor(object_pos, device=hs.device)]
            diff = (ans - obj).detach().float().cpu().numpy()
            for bi, item in enumerate(batch_items):
                sums[item["cat"]] += diff[bi].astype(np.float32)
                counts[item["cat"]] += 1
            del out, batch
            torch.cuda.empty_cache()
    return {cat: (sums[cat] / max(counts[cat], 1)).astype(np.float32) for cat in categories}


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


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.input_device
    layers = get_layers(model)
    categories = list(CATEGORY_OBJECTS.keys())
    test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
    cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)
    layer_id = args.layer if args.layer is not None else BOUNDARY_LAYER[args.model]

    alloc, reserved = vram_gb()
    log(f"{args.model}: layer=L{layer_id}, vram={alloc:.2f}/{reserved:.2f}GB")
    centers = capture_centers(model, tokenizer, device, categories, layer_id, args.train_objects, args.batch_size, args.max_length)
    boundaries = build_boundaries(centers, categories)
    transport_dirs = capture_transport_dirs(
        model, tokenizer, device, categories, layer_id, args.train_objects, args.batch_size, args.max_length
    )

    result = {
        "phase": 110,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layer": layer_id,
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
        neighbor_basis = [boundaries[n] for n in NEIGHBOR_SETS[cat] if n in boundaries]
        neighbor_component = basis_project(b_orth, neighbor_basis)
        after_neighbor = (b_orth - neighbor_component).astype(np.float32)
        transport_component = projection(after_neighbor, transport_dirs[cat]).astype(np.float32)
        residual_component = (after_neighbor - transport_component).astype(np.float32)
        random_dir = deterministic_random_direction(b.shape[0], 7000 + target_idx)
        directions = {
            "orthogonal_full": b_orth,
            "neighbor_aligned": neighbor_component,
            "transport_aligned": transport_component,
            "residual": residual_component,
            "random_same_norm": random_dir,
        }
        baseline = run_prompts(
            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
            args.batch_size, args.max_length
        )
        cat_out = {
            "n_prompts": len(prompts),
            "baseline_target_mean": float(baseline[:, target_idx].mean()),
            "cos_orth_neighbor": cos(b_orth, neighbor_component),
            "cos_after_neighbor_transport": cos(after_neighbor, transport_component),
            "norm_fractions": {
                "neighbor": float(np.linalg.norm(neighbor_component) / (np.linalg.norm(b_orth) + 1e-8)),
                "transport": float(np.linalg.norm(transport_component) / (np.linalg.norm(b_orth) + 1e-8)),
                "residual": float(np.linalg.norm(residual_component) / (np.linalg.norm(b_orth) + 1e-8)),
            },
            "conditions": [],
        }
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
                        "position": pos,
                        "scale": scale,
                        "kind": kind,
                        **summary,
                    })
        result["category_results"][cat] = cat_out
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 110 Orthogonal Subspace Split: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layer: L{result['layer']}")
    lines.append("")
    lines.append("| category | fractions N/T/R | best neighbor | best transport | best residual | best orth full |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]
        def min_kind(kind: str):
            xs = [c for c in conds if c["kind"] == kind]
            return min(xs, key=lambda x: x["target_delta"])
        n = min_kind("neighbor_aligned")
        t = min_kind("transport_aligned")
        r = min_kind("residual")
        o = min_kind("orthogonal_full")
        fr = item["norm_fractions"]
        lines.append(
            f"| {cat} | {fr['neighbor']:.2f}/{fr['transport']:.2f}/{fr['residual']:.2f} | "
            f"{n['position']} s{n['scale']} T{n['target_delta']:.2f} | "
            f"{t['position']} s{t['scale']} T{t['target_delta']:.2f} | "
            f"{r['position']} s{r['scale']} T{r['target_delta']:.2f} | "
            f"{o['position']} s{o['scale']} T{o['target_delta']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
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
        json_path = out_dir / f"phase110_{args.model}_orthogonal_subspace_split.json"
        md_path = out_dir / f"phase110_{args.model}_orthogonal_subspace_split.md"
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
