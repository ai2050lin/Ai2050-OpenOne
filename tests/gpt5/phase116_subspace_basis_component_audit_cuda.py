#!/usr/bin/env python3
"""
Phase 116: subspace basis component audit.

Audit each basis vector in the robust answer-site causal subspace:
  - single basis ablation
  - cumulative target-sorted ablation
  - support/release/mixed split
  - random basis component controls
  - readout/transport/template cosine diagnostics
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
from phase109_support_suppressor_decomposition_cuda import build_readout_directions  # noqa: E402
from phase110_orthogonal_subspace_split_cuda import capture_transport_dirs  # noqa: E402
from phase111_transport_path_causal_mapping_cuda import build_transport_components  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import (  # noqa: E402
    build_category_contrast_matrix,
    make_subspace_hook,
)
from phase115_causal_subspace_robustness_cuda import capture_centers_template_subset  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase116_subspace_basis_component_audit")
TEST_CATEGORIES = ["number", "container", "clothing", "plant"]
RANKS = [8, 16]
SET_SIZES = [1, 2, 4, 8, 16]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompts(cat: str, train_n: int, test_n: int) -> list[dict[str, Any]]:
    prompts = []
    for ti, tpl in enumerate(TEMPLATES):
        for obj in CATEGORY_OBJECTS[cat][train_n:train_n + test_n]:
            prompts.append({"obj": obj, "template_id": ti, "prompt": tpl["text"].format(obj=obj)})
    return prompts


def svd_basis(mat: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    x = mat.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(rank, vt.shape[0])
    basis = vt[:k].astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
    return basis, s[:k].astype(np.float32)


def deterministic_random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((rank, dim)).astype(np.float32)
    q, _ = np.linalg.qr(mat.T)
    basis = q[:, :rank].T.astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
    return basis


def template_basis(centers: np.ndarray) -> np.ndarray:
    # Template directions at answer site: template mean minus grand mean.
    tpl_mean = centers.mean(axis=1)
    grand = tpl_mean.mean(axis=0, keepdims=True)
    rows = (tpl_mean - grand).astype(np.float32)
    rows = rows[np.linalg.norm(rows, axis=1) > 1e-8]
    if rows.shape[0] == 0:
        return rows
    q, _ = np.linalg.qr(rows.T)
    out = q[:, : rows.shape[0]].T.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def max_abs_basis_cos(v: np.ndarray, basis: np.ndarray) -> float:
    if basis.size == 0:
        return 0.0
    return float(np.max(np.abs(basis @ (v / (np.linalg.norm(v) + 1e-8)))))


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    layer_id: int,
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    basis: np.ndarray | None = None,
    scale: float = 1.5,
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
        if basis is not None and basis.shape[0] > 0:
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


def label_component(row: dict[str, Any]) -> str:
    target = row["target_delta"]
    release = row["max_other_delta"]
    if target <= -0.25 and release <= max(0.25, abs(target) * 0.5):
        return "support"
    if release >= 0.5 and target > -0.25:
        return "release"
    if target <= -0.25 and release >= max(0.25, abs(target) * 0.5):
        return "mixed"
    return "weak"


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
        set_sizes = [int(x) for x in args.set_sizes.split(",") if x.strip()]
        layer_id = args.layer if args.layer is not None else BOUNDARY_LAYER[args.model]
        cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: layer=L{layer_id}, ranks={ranks}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

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
            "phase": 116,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer": layer_id,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "ranks": ranks,
            "set_sizes": set_sizes,
            "scale": args.scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        dim = int(model.get_input_embeddings().weight.shape[1])
        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, layer_id,
                cat_local_ids, categories, args.batch_size, args.max_length
            )
            contrast = build_category_contrast_matrix(centers, categories, cat)
            transport = components[cat]["transport"]
            if np.linalg.norm(transport) < 1e-8:
                transport = components[cat]["raw_transport"]
            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline[:, target_idx].mean()),
                "rank_results": {},
            }
            for rank in ranks:
                basis, singular_values = svd_basis(contrast, rank)
                random_basis = deterministic_random_basis(dim, basis.shape[0], 16000 + target_idx * 101 + rank)
                rank_out = {
                    "singular_values": [float(x) for x in singular_values],
                    "basis_components": [],
                    "random_components": [],
                    "cumulative": [],
                    "split_sets": [],
                }
                for bi, vec in enumerate(basis):
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, layer_id,
                        cat_local_ids, categories, args.batch_size, args.max_length,
                        vec[None, :], args.scale
                    )
                    summary = summarize_delta(patched - baseline, target_idx, categories)
                    row = {
                        "basis_index": bi,
                        "singular_value": float(singular_values[bi]),
                        "readout_cos": cos(vec, readout_dirs[cat]),
                        "transport_cos": cos(vec, transport),
                        "template_abs_cos": max_abs_basis_cos(vec, tpl_basis),
                        **summary,
                    }
                    row["component_label"] = label_component(row)
                    rank_out["basis_components"].append(row)
                for bi, vec in enumerate(random_basis):
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, layer_id,
                        cat_local_ids, categories, args.batch_size, args.max_length,
                        vec[None, :], args.scale
                    )
                    summary = summarize_delta(patched - baseline, target_idx, categories)
                    rank_out["random_components"].append({
                        "basis_index": bi,
                        "readout_cos": cos(vec, readout_dirs[cat]),
                        "transport_cos": cos(vec, transport),
                        "template_abs_cos": max_abs_basis_cos(vec, tpl_basis),
                        **summary,
                    })

                sorted_components = sorted(rank_out["basis_components"], key=lambda r: r["target_delta"])
                for size in set_sizes:
                    active = sorted_components[: min(size, len(sorted_components))]
                    if not active:
                        continue
                    active_basis = basis[[int(r["basis_index"]) for r in active]]
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, layer_id,
                        cat_local_ids, categories, args.batch_size, args.max_length,
                        active_basis, args.scale
                    )
                    summary = summarize_delta(patched - baseline, target_idx, categories)
                    rank_out["cumulative"].append({
                        "set_name": "target_sorted",
                        "set_size": int(active_basis.shape[0]),
                        "basis_indices": [int(r["basis_index"]) for r in active],
                        **summary,
                    })

                for label in ["support", "release", "mixed", "weak"]:
                    ids = [int(r["basis_index"]) for r in rank_out["basis_components"] if r["component_label"] == label]
                    if not ids:
                        continue
                    active_basis = basis[ids]
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, layer_id,
                        cat_local_ids, categories, args.batch_size, args.max_length,
                        active_basis, args.scale
                    )
                    summary = summarize_delta(patched - baseline, target_idx, categories)
                    rank_out["split_sets"].append({
                        "set_name": label,
                        "set_size": int(active_basis.shape[0]),
                        "basis_indices": ids,
                        **summary,
                    })
                cat_out["rank_results"][str(rank)] = rank_out
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 116 Subspace Basis Component Audit: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layer: L{result['layer']}")
    lines.append("")
    lines.append("| category | rank | best single | best cumulative | support set | release set |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        for rank, rr in item["rank_results"].items():
            comps = rr["basis_components"]
            best_single = min(comps, key=lambda r: r["target_delta"]) if comps else None
            best_cum = min(rr["cumulative"], key=lambda r: r["target_delta"]) if rr["cumulative"] else None
            support = next((r for r in rr["split_sets"] if r["set_name"] == "support"), None)
            release = next((r for r in rr["split_sets"] if r["set_name"] == "release"), None)

            def fmt(r):
                if r is None:
                    return "NA"
                idx = r.get("basis_index", r.get("set_size", ""))
                return f"{idx} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"

            lines.append(f"| {cat} | {rank} | {fmt(best_single)} | {fmt(best_cum)} | {fmt(support)} | {fmt(release)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--ranks", default="8,16")
    parser.add_argument("--set-sizes", default="1,2,4,8,16")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase116_{args.model}_subspace_basis_component_audit.json"
    md_path = out_dir / f"phase116_{args.model}_subspace_basis_component_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
