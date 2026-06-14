#!/usr/bin/env python3
"""
Phase 117: basis rotation and causal axis stabilization.

Test whether Phase116 basis-level support/release labels are stable under
orthogonal rotations inside the same answer-site causal subspace.
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
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, summarize_delta  # noqa: E402
from phase109_support_suppressor_decomposition_cuda import build_readout_directions  # noqa: E402
from phase110_orthogonal_subspace_split_cuda import capture_transport_dirs  # noqa: E402
from phase111_transport_path_causal_mapping_cuda import build_transport_components  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase115_causal_subspace_robustness_cuda import capture_centers_template_subset  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import (  # noqa: E402
    build_prompts,
    cos,
    label_component,
    max_abs_basis_cos,
    run_condition,
    svd_basis,
    template_basis,
)


OUT_ROOT = Path("results/gpt5_phase117_basis_rotation_causal_axis")
TEST_CATEGORIES = ["number", "container", "clothing", "plant"]
SET_SIZES = [1, 2, 4, 8, 16]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def orthonormalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float32)
    if rows.shape[0] == 0:
        return rows
    q, _ = np.linalg.qr(rows.T)
    out = q[:, : rows.shape[0]].T.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def random_rotation(rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((rank, rank)).astype(np.float32)
    q, _ = np.linalg.qr(mat)
    return q.astype(np.float32)


def varimax_rotation(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 40, tol: float = 1e-6) -> np.ndarray:
    # loadings shape: features x factors. Returns rotated loadings.
    p, k = loadings.shape
    rot = np.eye(k, dtype=np.float32)
    old = 0.0
    for _ in range(max_iter):
        lam = loadings @ rot
        inner = lam**3 - (gamma / float(p)) * lam @ np.diag(np.diag(lam.T @ lam))
        u, s, vh = np.linalg.svd(loadings.T @ inner, full_matrices=False)
        rot = (u @ vh).astype(np.float32)
        score = float(s.sum())
        if old > 0.0 and score <= old * (1.0 + tol):
            break
        old = score
    return (loadings @ rot).astype(np.float32)


def varimax_basis(basis: np.ndarray) -> np.ndarray:
    rotated_loadings = varimax_rotation(basis.T)
    return orthonormalize_rows(rotated_loadings.T)


def rotated_basis_variants(basis: np.ndarray, n_random: int, seed_base: int) -> list[dict[str, Any]]:
    variants = [
        {"basis_name": "svd", "basis": basis},
        {"basis_name": "varimax", "basis": varimax_basis(basis)},
    ]
    for ri in range(n_random):
        rot = random_rotation(basis.shape[0], seed_base + ri * 17)
        variants.append({"basis_name": f"random_rot_{ri}", "basis": orthonormalize_rows(rot @ basis)})
    return variants


def component_row(
    vec: np.ndarray,
    idx: int,
    summary: dict[str, Any],
    readout: np.ndarray,
    transport: np.ndarray,
    tpl_basis: np.ndarray,
) -> dict[str, Any]:
    row = {
        "basis_index": idx,
        "readout_cos": cos(vec, readout),
        "transport_cos": cos(vec, transport),
        "template_abs_cos": max_abs_basis_cos(vec, tpl_basis),
        **summary,
    }
    row["component_label"] = label_component(row)
    return row


def audit_basis(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    baseline: np.ndarray,
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    target_idx: int,
    batch_size: int,
    max_length: int,
    scale: float,
    basis: np.ndarray,
    set_sizes: list[int],
    readout: np.ndarray,
    transport: np.ndarray,
    tpl_basis: np.ndarray,
) -> dict[str, Any]:
    basis = orthonormalize_rows(basis)
    out: dict[str, Any] = {"components": [], "cumulative": [], "split_sets": []}
    for bi, vec in enumerate(basis):
        patched = run_condition(
            model, tokenizer, device, layers, prompts, layer_id,
            cat_local_ids, categories, batch_size, max_length, vec[None, :], scale
        )
        summary = summarize_delta(patched - baseline, target_idx, categories)
        out["components"].append(component_row(vec, bi, summary, readout, transport, tpl_basis))

    sorted_components = sorted(out["components"], key=lambda r: r["target_delta"])
    for size in set_sizes:
        active = sorted_components[: min(size, len(sorted_components))]
        if not active:
            continue
        active_basis = basis[[int(r["basis_index"]) for r in active]]
        patched = run_condition(
            model, tokenizer, device, layers, prompts, layer_id,
            cat_local_ids, categories, batch_size, max_length, active_basis, scale
        )
        summary = summarize_delta(patched - baseline, target_idx, categories)
        out["cumulative"].append({
            "set_name": "target_sorted",
            "set_size": int(active_basis.shape[0]),
            "basis_indices": [int(r["basis_index"]) for r in active],
            **summary,
        })

    for label in ["support", "release", "mixed", "weak"]:
        ids = [int(r["basis_index"]) for r in out["components"] if r["component_label"] == label]
        if not ids:
            continue
        active_basis = basis[ids]
        patched = run_condition(
            model, tokenizer, device, layers, prompts, layer_id,
            cat_local_ids, categories, batch_size, max_length, active_basis, scale
        )
        summary = summarize_delta(patched - baseline, target_idx, categories)
        out["split_sets"].append({
            "set_name": label,
            "set_size": int(active_basis.shape[0]),
            "basis_indices": ids,
            **summary,
        })
    return out


def greedy_select(candidates: list[dict[str, Any]], dim: int, rank: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for row in sorted(candidates, key=lambda r: r["target_delta"]):
        vec = row["vector"].copy()
        for prev in selected:
            vec = vec - float(np.dot(vec, prev)) * prev
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        selected.append((vec / norm).astype(np.float32))
        if len(selected) >= rank:
            break
    if not selected:
        return np.zeros((0, dim), dtype=np.float32)
    return orthonormalize_rows(np.stack(selected).astype(np.float32))


def causal_greedy_basis(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    baseline: np.ndarray,
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    target_idx: int,
    batch_size: int,
    max_length: int,
    scale: float,
    basis: np.ndarray,
    candidates_n: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    for ci in range(candidates_n):
        coeff = rng.standard_normal(basis.shape[0]).astype(np.float32)
        coeff /= np.linalg.norm(coeff) + 1e-8
        vec = (coeff @ basis).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        patched = run_condition(
            model, tokenizer, device, layers, prompts, layer_id,
            cat_local_ids, categories, batch_size, max_length, vec[None, :], scale
        )
        summary = summarize_delta(patched - baseline, target_idx, categories)
        candidates.append({"candidate_index": ci, "vector": vec, **summary})
    selected = greedy_select(candidates, basis.shape[1], basis.shape[0])
    public = [{k: v for k, v in row.items() if k != "vector"} for row in candidates]
    return selected, public


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        set_sizes = [int(x) for x in args.set_sizes.split(",") if x.strip()]
        layer_id = args.layer if args.layer is not None else BOUNDARY_LAYER[args.model]
        cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)

        alloc, reserved = vram_gb()
        log(
            f"{args.model}: layer=L{layer_id}, rank={args.rank}, rotations={args.random_rotations}, "
            f"candidates={args.causal_candidates}, train/test={args.train_objects}/{args.test_objects}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        centers = capture_centers_template_subset(
            model, tokenizer, device, categories, layer_id,
            args.train_objects, args.batch_size, args.max_length,
            list(range(len(TEMPLATES)))
        )
        transport_dirs = capture_transport_dirs(
            model, tokenizer, device, categories, layer_id,
            args.train_objects, args.batch_size, args.max_length
        )
        components = build_transport_components(centers, transport_dirs, readout_dirs, categories)
        tpl_basis = template_basis(centers)

        result = {
            "phase": 117,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer": layer_id,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "set_sizes": set_sizes,
            "scale": args.scale,
            "random_rotations": args.random_rotations,
            "causal_candidates": args.causal_candidates,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, layer_id,
                cat_local_ids, categories, args.batch_size, args.max_length
            )
            contrast = build_category_contrast_matrix(centers, categories, cat)
            basis, singular_values = svd_basis(contrast, args.rank)
            transport = components[cat]["transport"]
            if np.linalg.norm(transport) < 1e-8:
                transport = components[cat]["raw_transport"]

            cat_out: dict[str, Any] = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline[:, target_idx].mean()),
                "singular_values": [float(x) for x in singular_values],
                "basis_variants": {},
                "causal_candidates": [],
            }
            seed_base = 17000 + target_idx * 503 + args.rank
            variants = rotated_basis_variants(basis, args.random_rotations, seed_base)
            for variant in variants:
                name = variant["basis_name"]
                log(f"  {cat}: audit {name}")
                cat_out["basis_variants"][name] = audit_basis(
                    model, tokenizer, device, layers, prompts, baseline, layer_id,
                    cat_local_ids, categories, target_idx, args.batch_size,
                    args.max_length, args.scale, variant["basis"], set_sizes,
                    readout_dirs[cat], transport, tpl_basis
                )

            log(f"  {cat}: causal_greedy")
            greedy_basis, candidates = causal_greedy_basis(
                model, tokenizer, device, layers, prompts, baseline, layer_id,
                cat_local_ids, categories, target_idx, args.batch_size,
                args.max_length, args.scale, basis, args.causal_candidates,
                seed_base + 9000
            )
            cat_out["causal_candidates"] = candidates
            cat_out["basis_variants"]["causal_greedy"] = audit_basis(
                model, tokenizer, device, layers, prompts, baseline, layer_id,
                cat_local_ids, categories, target_idx, args.batch_size,
                args.max_length, args.scale, greedy_basis, set_sizes,
                readout_dirs[cat], transport, tpl_basis
            )
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def fmt_best(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "NA"
    row = min(rows, key=lambda r: r["target_delta"])
    idx = row.get("basis_index", row.get("set_size", ""))
    label = row.get("component_label", row.get("set_name", ""))
    return f"{label}{idx} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 117 Basis Rotation and Causal Axis Stabilization: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layer: L{result['layer']}; rank: {result['rank']}; scale: {result['scale']}")
    lines.append("")
    lines.append("| category | basis | best single | best cumulative | support set | release set |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        for name, rr in item["basis_variants"].items():
            support = next((r for r in rr["split_sets"] if r["set_name"] == "support"), None)
            release = next((r for r in rr["split_sets"] if r["set_name"] == "release"), None)
            lines.append(
                f"| {cat} | {name} | {fmt_best(rr['components'])} | {fmt_best(rr['cumulative'])} | "
                f"{fmt_best([support] if support else [])} | {fmt_best([release] if release else [])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--set-sizes", default="1,2,4,8,16")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--random-rotations", type=int, default=2)
    parser.add_argument("--causal-candidates", type=int, default=24)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase117_{args.model}_basis_rotation_causal_axis.json"
    md_path = out_dir / f"phase117_{args.model}_basis_rotation_causal_axis.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
