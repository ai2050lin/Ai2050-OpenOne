#!/usr/bin/env python3
"""
Phase 114: answer-site causal subspace expansion.

Build target-category answer-site contrast subspaces from template/category
centers at the causal peak layer, then compare rank-k subspace removal against
the single transport direction T_c and random same-rank controls.
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
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, capture_centers, score_logits, summarize_delta  # noqa: E402
from phase109_support_suppressor_decomposition_cuda import build_readout_directions  # noqa: E402
from phase110_orthogonal_subspace_split_cuda import capture_transport_dirs  # noqa: E402
from phase111_transport_path_causal_mapping_cuda import build_transport_components  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase114_answer_site_causal_subspace")
TEST_CATEGORIES = ["number", "container", "clothing", "plant"]
RANKS = [1, 2, 4, 8, 16]
SCALES = [1.0, 1.5]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompts(cat: str, train_n: int, test_n: int) -> list[dict[str, str]]:
    prompts = []
    for tpl in TEMPLATES:
        for obj in CATEGORY_OBJECTS[cat][train_n:train_n + test_n]:
            prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj)})
    return prompts


def orthonormal_rows(mat: np.ndarray, rank: int) -> np.ndarray:
    mat = mat.astype(np.float32)
    mat = mat - mat.mean(axis=0, keepdims=True)
    if mat.shape[0] == 0:
        raise ValueError("empty matrix")
    _u, _s, vt = np.linalg.svd(mat, full_matrices=False)
    k = min(rank, vt.shape[0])
    basis = vt[:k].astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
    return basis


def build_category_contrast_matrix(centers: np.ndarray, categories: list[str], cat: str) -> np.ndarray:
    target_idx = categories.index(cat)
    rows = []
    for ti in range(centers.shape[0]):
        target = centers[ti, target_idx]
        other_mean = np.mean(np.delete(centers[ti], target_idx, axis=0), axis=0)
        rows.append(target - other_mean)
        for ci, other_cat in enumerate(categories):
            if other_cat == cat:
                continue
            rows.append(target - centers[ti, ci])
    return np.stack(rows).astype(np.float32)


def deterministic_random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((rank, dim)).astype(np.float32)
    q, _r = np.linalg.qr(mat.T)
    basis = q[:, :rank].T.astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
    return basis


def make_subspace_hook(basis: torch.Tensor, positions: torch.Tensor, scale: float):
    # basis shape: [rank, d], rows orthonormal-ish
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        bidx = torch.arange(out.shape[0], device=out.device)
        pos = positions.to(out.device)
        vecs = out[bidx, pos, :].float()
        b = basis.to(out.device).float()
        proj = (vecs @ b.T) @ b
        out[bidx, pos, :] = out[bidx, pos, :] - scale * proj.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def run_condition(
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
    basis: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    scores = []
    module_index = layer_id - 1
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        handle = None
        if basis is not None:
            b = torch.tensor(basis, device=device, dtype=torch.float32)
            handle = layers[module_index].register_forward_hook(make_subspace_hook(b, answer_pos, scale))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        if handle is not None:
            handle.remove()
        pos_gpu = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
        scales = [float(x) for x in args.scales.split(",") if x.strip()]
        layer_id = args.layer if args.layer is not None else BOUNDARY_LAYER[args.model]
        cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)

        alloc, reserved = vram_gb()
        log(f"{args.model}: layer=L{layer_id}, ranks={ranks}, vram={alloc:.2f}/{reserved:.2f}GB")
        centers = capture_centers(model, tokenizer, device, categories, layer_id, args.train_objects, args.batch_size, args.max_length)
        transport_dirs = capture_transport_dirs(
            model, tokenizer, device, categories, layer_id,
            args.train_objects, args.batch_size, args.max_length
        )
        components = build_transport_components(centers, transport_dirs, readout_dirs, categories)

        result = {
            "phase": 114,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer": layer_id,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "ranks": ranks,
            "scales": scales,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        dim = int(model.get_input_embeddings().weight.shape[1])
        for idx, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            target_idx = categories.index(cat)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                args.batch_size, args.max_length
            )
            contrast = build_category_contrast_matrix(centers, categories, cat)
            tc = components[cat]["transport"].astype(np.float32)
            if np.linalg.norm(tc) < 1e-8:
                tc = components[cat]["raw_transport"].astype(np.float32)
            tc_basis = tc[None, :] / (np.linalg.norm(tc) + 1e-8)
            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline[:, target_idx].mean()),
                "contrast_matrix_rows": int(contrast.shape[0]),
                "conditions": [],
            }
            for scale in scales:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                    args.batch_size, args.max_length, tc_basis, scale
                )
                summary = summarize_delta(patched - baseline, target_idx, categories)
                cat_out["conditions"].append({
                    "kind": "transport_direction",
                    "rank": 1,
                    "scale": scale,
                    **summary,
                })
            for rank in ranks:
                basis = orthonormal_rows(contrast, rank)
                random_basis = deterministic_random_basis(dim, basis.shape[0], 10000 + target_idx * 97 + rank)
                for scale in scales:
                    for kind, active_basis in [
                        ("answer_contrast_subspace", basis),
                        ("random_subspace", random_basis),
                    ]:
                        patched = run_condition(
                            model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                            args.batch_size, args.max_length, active_basis, scale
                        )
                        summary = summarize_delta(patched - baseline, target_idx, categories)
                        cat_out["conditions"].append({
                            "kind": kind,
                            "rank": int(active_basis.shape[0]),
                            "scale": scale,
                            **summary,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 114 Answer-site Causal Subspace: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layer: L{result['layer']}")
    lines.append("")
    lines.append("| category | best T_c | best contrast subspace | best random subspace |")
    lines.append("|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]

        def pick(kind: str):
            xs = [c for c in conds if c["kind"] == kind]
            return min(xs, key=lambda c: c["target_delta"]) if xs else None

        def fmt(c):
            if c is None:
                return "NA"
            return f"r{c['rank']} s{c['scale']} T{c['target_delta']:+.2f} R{c['max_other_delta']:+.2f}"

        lines.append(
            f"| {cat} | {fmt(pick('transport_direction'))} | "
            f"{fmt(pick('answer_contrast_subspace'))} | {fmt(pick('random_subspace'))} |"
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
    parser.add_argument("--ranks", default="1,2,4,8,16")
    parser.add_argument("--scales", default="1.0,1.5")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase114_{args.model}_answer_site_causal_subspace.json"
    md_path = out_dir / f"phase114_{args.model}_answer_site_causal_subspace.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
