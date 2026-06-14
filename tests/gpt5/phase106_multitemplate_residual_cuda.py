#!/usr/bin/env python3
"""
Phase 106: multi-template residual atlas on CUDA.

Objective:
  Verify whether Phase105 category-layer observations survive:
  - multiple prompt templates
  - answer-slot vs object-token positions
  - subtracting a same-template common residual vector

Run one model per process:
  python tests/gpt5/phase106_multitemplate_residual_cuda.py qwen3 --hard-exit-after-model
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
from phase105_global_category_atlas_cuda import (  # noqa: E402
    CATEGORY_OBJECTS,
    CATEGORY_READOUT_WORDS,
    category_scores,
    collect_readout_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "gpt5_phase106_multitemplate_residual"

TEMPLATES = [
    {"name": "kind_of", "text": "The {obj} is a kind of"},
    {"name": "belongs", "text": "A {obj} belongs to the category of"},
    {"name": "word_refers", "text": "The word {obj} refers to a type of"},
    {"name": "talk_about", "text": "People use the word {obj} when talking about"},
]

POSITIONS = ["answer_last", "object_last"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def object_last_position(tokenizer: Any, prompt: str, obj: str, fallback: int) -> int:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    obj_ids = tokenizer(obj, add_special_tokens=False)["input_ids"]
    start = find_subsequence(full_ids, obj_ids)
    if start is None:
        obj_ids = tokenizer(" " + obj, add_special_tokens=False)["input_ids"]
        start = find_subsequence(full_ids, obj_ids)
    if start is None:
        return fallback
    return min(start + len(obj_ids) - 1, fallback)


def summarize_curve(xs: list[float]) -> dict[str, Any]:
    arr = np.array(xs, dtype=np.float64)
    return {
        "max_layer": int(np.argmax(arr)),
        "max": float(arr.max()),
        "final": float(arr[-1]),
        "mean": float(arr.mean()),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    categories = list(CATEGORY_OBJECTS.keys())
    n_cat = len(categories)
    n_obj = min(args.objects_per_category, min(len(CATEGORY_OBJECTS[c]) for c in categories))
    templates = TEMPLATES[: args.templates]
    n_tpl = len(templates)

    loaded = load_probe_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    layers = get_layers(model)
    n_layers = len(layers)
    d_model = int(model.get_input_embeddings().weight.shape[1])
    alloc, reserved = vram_gb()
    log(f"Loaded {args.model}: L={n_layers}, d={d_model}, vram={alloc:.2f}/{reserved:.2f}GB")

    cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
    readout_rows = readout_rows.astype(np.float32)

    # sums[position][raw/template_residual basis]
    sums = {
        pos: np.zeros((n_layers + 1, n_tpl, n_cat, d_model), dtype=np.float64)
        for pos in POSITIONS
    }
    tpl_sums = {
        pos: np.zeros((n_layers + 1, n_tpl, d_model), dtype=np.float64)
        for pos in POSITIONS
    }
    counts = np.zeros((n_tpl, n_cat), dtype=np.int64)
    tpl_counts = np.zeros((n_tpl,), dtype=np.int64)

    items: list[dict[str, Any]] = []
    for ti, tpl in enumerate(templates):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:n_obj]:
                prompt = tpl["text"].format(obj=obj)
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": prompt})

    log(f"Running {args.model}: prompts={len(items)}, templates={n_tpl}, categories={n_cat}, objects/category={n_obj}")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            prompts = [x["prompt"] for x in batch_items]
            batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            batch = {k: v.to(loaded.input_device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            object_pos = [
                object_last_position(tokenizer, item["prompt"], item["obj"], answer_pos[bi])
                for bi, item in enumerate(batch_items)
            ]
            pos_by_name = {"answer_last": answer_pos, "object_last": object_pos}

            for li in range(n_layers + 1):
                hs = out.hidden_states[li]
                for pos_name, pos_list in pos_by_name.items():
                    picked = hs[
                        torch.arange(hs.shape[0], device=hs.device),
                        torch.tensor(pos_list, device=hs.device),
                    ].detach().float().cpu().numpy()
                    for bi, item in enumerate(batch_items):
                        ti = item["ti"]
                        ci = item["ci"]
                        vec = picked[bi].astype(np.float32)
                        sums[pos_name][li, ti, ci] += vec
                        tpl_sums[pos_name][li, ti] += vec

            for item in batch_items:
                counts[item["ti"], item["ci"]] += 1
                tpl_counts[item["ti"]] += 1

            if (start // args.batch_size) % args.progress_every == 0:
                alloc, reserved = vram_gb()
                log(f"  {start + len(batch_items)}/{len(items)} vram={alloc:.2f}/{reserved:.2f}GB")
            del out, batch
            torch.cuda.empty_cache()

    log(f"Captured in {(time.time() - t0) / 60:.2f} min")

    result: dict[str, Any] = {
        "phase": 106,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_layers": n_layers,
        "d_model": d_model,
        "categories": categories,
        "templates": [x["name"] for x in templates],
        "objects_per_category": n_obj,
        "positions": POSITIONS,
        "readout_token_labels": token_labels,
        "global": {},
        "category_summary": {},
        "notes": [
            "template_residual means subtracting the same-template mean vector at each layer and position.",
            "answer_last is the category-answer slot; object_last is the object token position.",
            "This phase is still atlas/readout evidence, not downstream causal patching.",
        ],
    }

    for pos_name in POSITIONS:
        result["global"][pos_name] = {}
        result["category_summary"][pos_name] = {}
        centers = sums[pos_name] / counts.reshape(1, n_tpl, n_cat, 1)
        tpl_means = tpl_sums[pos_name] / tpl_counts.reshape(1, n_tpl, 1)
        residual_centers = centers - tpl_means[:, :, None, :]

        for basis_name, basis_centers in [("raw", centers), ("template_residual", residual_centers)]:
            global_layers = []
            cat_summary: dict[str, Any] = {}
            for li in range(n_layers + 1):
                # Average scores over templates, keeping category identity.
                scores_by_tpl = []
                boundary_norm_by_tpl = []
                neighbor_cos_by_tpl = []
                for ti in range(n_tpl):
                    C = basis_centers[li, ti].astype(np.float32)
                    scores_by_tpl.append(category_scores(C, readout_rows, cat_local_ids, categories))
                    other_means = np.array([
                        np.mean(np.delete(C, ci, axis=0), axis=0)
                        for ci in range(n_cat)
                    ])
                    boundary_norm_by_tpl.append(np.linalg.norm(C - other_means, axis=1))
                    norms = np.linalg.norm(C, axis=1) + 1e-8
                    neighbor_cos_by_tpl.append((C @ C.T) / (norms[:, None] * norms[None, :]))
                scores = np.mean(np.stack(scores_by_tpl), axis=0)
                boundary_norms = np.mean(np.stack(boundary_norm_by_tpl), axis=0)
                target = np.diag(scores)
                other = scores.copy()
                np.fill_diagonal(other, -1e9)
                margins = target - other.max(axis=1)
                ranks = 1 + np.sum(scores > target[:, None], axis=1)
                global_layers.append({
                    "layer": li,
                    "mean_margin": float(np.mean(margins)),
                    "median_margin": float(np.median(margins)),
                    "top1_count": int(np.sum(ranks == 1)),
                    "mean_boundary_norm": float(np.mean(boundary_norms)),
                })

            for ci, cat in enumerate(categories):
                margin_curve = []
                rank_curve = []
                boundary_curve = []
                for li in range(n_layers + 1):
                    scores_by_tpl = []
                    bnorms = []
                    for ti in range(n_tpl):
                        C = basis_centers[li, ti].astype(np.float32)
                        scores_by_tpl.append(category_scores(C, readout_rows, cat_local_ids, categories))
                        other_mean = np.mean(np.delete(C, ci, axis=0), axis=0)
                        bnorms.append(float(np.linalg.norm(C[ci] - other_mean)))
                    scores = np.mean(np.stack(scores_by_tpl), axis=0)
                    own = scores[ci, ci]
                    other_scores = np.delete(scores[ci], ci)
                    margin_curve.append(float(own - np.max(other_scores)))
                    rank_curve.append(int(1 + np.sum(scores[ci] > own)))
                    boundary_curve.append(float(np.mean(bnorms)))

                best_layer = int(np.argmax(np.array(margin_curve)))
                # Neighbors at the best margin layer, averaged over templates.
                cos_avg = np.zeros((n_cat,), dtype=np.float64)
                for ti in range(n_tpl):
                    C = basis_centers[best_layer, ti].astype(np.float32)
                    norms = np.linalg.norm(C, axis=1) + 1e-8
                    cos = (C[ci] @ C.T) / ((np.linalg.norm(C[ci]) + 1e-8) * norms)
                    cos_avg += cos
                cos_avg /= n_tpl
                neigh_ids = [j for j in np.argsort(cos_avg)[::-1] if j != ci][:5]

                cat_summary[cat] = {
                    "best_margin_layer": best_layer,
                    "best_boundary_layer": int(np.argmax(np.array(boundary_curve))),
                    "best_margin": float(max(margin_curve)),
                    "final_margin": float(margin_curve[-1]),
                    "best_rank": int(min(rank_curve)),
                    "final_rank": int(rank_curve[-1]),
                    "margin_curve": summarize_curve(margin_curve),
                    "boundary_curve": summarize_curve(boundary_curve),
                    "neighbors_at_best_margin_layer": [
                        {"category": categories[j], "cos": float(cos_avg[j])}
                        for j in neigh_ids
                    ],
                }

            result["global"][pos_name][basis_name] = global_layers
            result["category_summary"][pos_name][basis_name] = cat_summary

    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 106 Multi-template Residual Atlas: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    for pos in result["positions"]:
        lines.append(f"## {pos}")
        for basis in ["raw", "template_residual"]:
            glob = result["global"][pos][basis]
            top1 = max(glob, key=lambda x: x["top1_count"])
            margin = max(glob, key=lambda x: x["mean_margin"])
            boundary = max(glob, key=lambda x: x["mean_boundary_norm"])
            lines.append(
                f"- {basis}: top1 L{top1['layer']} {top1['top1_count']}/32; "
                f"margin L{margin['layer']} {margin['mean_margin']:.3f}; "
                f"boundary L{boundary['layer']} {boundary['mean_boundary_norm']:.3f}"
            )
        lines.append("")
    lines.append("## Category Rows")
    cats = result["categories"]
    for cat in cats:
        row = [f"- {cat}:"]
        for pos in result["positions"]:
            for basis in ["raw", "template_residual"]:
                x = result["category_summary"][pos][basis][cat]
                row.append(
                    f"{pos}/{basis}=M{x['best_margin_layer']} B{x['best_boundary_layer']} "
                    f"margin{x['best_margin']:.2f} rank{x['best_rank']}"
                )
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--objects-per-category", type=int, default=24)
    parser.add_argument("--templates", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    loaded = None
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_model(args)
        json_path = out_dir / f"phase106_{args.model}_multitemplate_residual.json"
        md_path = out_dir / f"phase106_{args.model}_multitemplate_residual.md"
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
