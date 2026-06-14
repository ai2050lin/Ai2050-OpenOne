#!/usr/bin/env python3
"""
Phase 115: causal subspace robustness and release decomposition.

Focus:
  - larger heldout objects: train 8 / test 16
  - scale sweep for rank8/rank16 answer-site subspaces
  - leave-template-out subspace construction
  - matched-spectrum random controls
  - release-aware ablations for mixed categories
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
from phase114_answer_site_causal_subspace_cuda import (  # noqa: E402
    build_category_contrast_matrix,
    make_subspace_hook,
    orthonormal_rows,
)


OUT_ROOT = Path("results/gpt5_phase115_causal_subspace_robustness")
TEST_CATEGORIES = ["number", "container", "clothing", "plant"]
RANKS = [8, 16]
SCALES = [0.25, 0.5, 1.0, 1.5]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompts(cat: str, train_n: int, test_n: int, template_ids: list[int] | None = None) -> list[dict[str, Any]]:
    tids = template_ids if template_ids is not None else list(range(len(TEMPLATES)))
    prompts = []
    for ti in tids:
        tpl = TEMPLATES[ti]
        for obj in CATEGORY_OBJECTS[cat][train_n:train_n + test_n]:
            prompts.append({"obj": obj, "template_id": ti, "prompt": tpl["text"].format(obj=obj)})
    return prompts


def capture_centers_template_subset(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    categories: list[str],
    layer_id: int,
    train_n: int,
    batch_size: int,
    max_length: int,
    template_ids: list[int],
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    sums = np.zeros((len(template_ids), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(template_ids), len(categories)), dtype=np.int64)
    items = []
    for local_ti, ti in enumerate(template_ids):
        tpl = TEMPLATES[ti]
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_n]:
                items.append((local_ti, ci, tpl["text"].format(obj=obj)))
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x[2] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            pos = batch["attention_mask"].sum(dim=1) - 1
            hs = out.hidden_states[layer_id]
            picked = hs[torch.arange(hs.shape[0], device=hs.device), pos].detach().float().cpu().numpy()
            for bi, (local_ti, ci, _text) in enumerate(batch_items):
                sums[local_ti, ci] += picked[bi].astype(np.float32)
                counts[local_ti, ci] += 1
            del out, batch
            torch.cuda.empty_cache()
    return (sums / counts[:, :, None]).astype(np.float32)


def matched_spectrum_random_basis(contrast: np.ndarray, rank: int, seed: int) -> np.ndarray:
    x = contrast.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(x, full_matrices=False)
    k = min(rank, len(s))
    rng = np.random.default_rng(seed)
    rand_left = rng.standard_normal((x.shape[0], k)).astype(np.float32)
    ql, _ = np.linalg.qr(rand_left)
    rand_right = rng.standard_normal((x.shape[1], k)).astype(np.float32)
    qr, _ = np.linalg.qr(rand_right)
    random_matrix = (ql[:, :k] * s[:k][None, :]) @ qr[:, :k].T
    return orthonormal_rows(random_matrix, k)


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


def release_category(summary: dict[str, Any]) -> str | None:
    releases = summary.get("top_releases", [])
    return releases[0]["category"] if releases else None


def build_excluding_category_contrast(
    centers: np.ndarray,
    categories: list[str],
    cat: str,
    excluded: str | None,
) -> np.ndarray:
    if excluded is None or excluded not in categories or excluded == cat:
        return build_category_contrast_matrix(centers, categories, cat)
    target_idx = categories.index(cat)
    exclude_idx = categories.index(excluded)
    rows = []
    for ti in range(centers.shape[0]):
        target = centers[ti, target_idx]
        keep = [i for i in range(len(categories)) if i not in {target_idx, exclude_idx}]
        rows.append(target - centers[ti, keep].mean(axis=0))
        for ci in keep:
            rows.append(target - centers[ti, ci])
    return np.stack(rows).astype(np.float32)


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
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: layer=L{layer_id}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        all_templates = list(range(len(TEMPLATES)))
        centers_all = capture_centers_template_subset(
            model, tokenizer, device, categories, layer_id,
            args.train_objects, args.batch_size, args.max_length, all_templates
        )
        centers_lto = {
            heldout: capture_centers_template_subset(
                model, tokenizer, device, categories, layer_id,
                args.train_objects, args.batch_size, args.max_length,
                [ti for ti in all_templates if ti != heldout]
            )
            for heldout in all_templates
        }

        result = {
            "phase": 115,
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

        for idx, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts_all = build_prompts(cat, args.train_objects, args.test_objects)
            baseline_all = run_condition(
                model, tokenizer, device, layers, prompts_all, layer_id, cat_local_ids, categories,
                args.batch_size, args.max_length
            )
            contrast_all = build_category_contrast_matrix(centers_all, categories, cat)
            cat_out = {
                "n_prompts_all_templates": len(prompts_all),
                "baseline_target_mean": float(baseline_all[:, target_idx].mean()),
                "conditions": [],
                "leave_template_out": [],
            }
            best_summary = None
            best_release = None
            for rank in ranks:
                basis = orthonormal_rows(contrast_all, rank)
                random_basis = matched_spectrum_random_basis(contrast_all, basis.shape[0], 12000 + target_idx * 101 + rank)
                for scale in scales:
                    for kind, active_basis in [
                        ("answer_contrast_subspace", basis),
                        ("matched_spectrum_random", random_basis),
                    ]:
                        patched = run_condition(
                            model, tokenizer, device, layers, prompts_all, layer_id, cat_local_ids, categories,
                            args.batch_size, args.max_length, active_basis, scale
                        )
                        summary = summarize_delta(patched - baseline_all, target_idx, categories)
                        row = {"kind": kind, "rank": int(active_basis.shape[0]), "scale": scale, **summary}
                        cat_out["conditions"].append(row)
                        if kind == "answer_contrast_subspace" and (best_summary is None or row["target_delta"] < best_summary["target_delta"]):
                            best_summary = row
                            best_release = release_category(row)

            # Release-aware decomposition: remove the strongest release category from contrast construction.
            if best_release:
                contrast_ex = build_excluding_category_contrast(centers_all, categories, cat, best_release)
                for rank in ranks:
                    basis = orthonormal_rows(contrast_ex, rank)
                    for scale in scales:
                        patched = run_condition(
                            model, tokenizer, device, layers, prompts_all, layer_id, cat_local_ids, categories,
                            args.batch_size, args.max_length, basis, scale
                        )
                        summary = summarize_delta(patched - baseline_all, target_idx, categories)
                        cat_out["conditions"].append({
                            "kind": "release_excluded_subspace",
                            "excluded_release_category": best_release,
                            "rank": int(basis.shape[0]),
                            "scale": scale,
                            **summary,
                        })

            for heldout in all_templates:
                prompts = build_prompts(cat, args.train_objects, args.test_objects, [heldout])
                baseline = run_condition(
                    model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                    args.batch_size, args.max_length
                )
                contrast = build_category_contrast_matrix(centers_lto[heldout], categories, cat)
                for rank in ranks:
                    basis = orthonormal_rows(contrast, rank)
                    random_basis = matched_spectrum_random_basis(contrast, basis.shape[0], 14000 + heldout * 509 + target_idx * 101 + rank)
                    for scale in scales:
                        for kind, active_basis in [
                            ("lto_answer_contrast_subspace", basis),
                            ("lto_matched_spectrum_random", random_basis),
                        ]:
                            patched = run_condition(
                                model, tokenizer, device, layers, prompts, layer_id, cat_local_ids, categories,
                                args.batch_size, args.max_length, active_basis, scale
                            )
                            summary = summarize_delta(patched - baseline, target_idx, categories)
                            cat_out["leave_template_out"].append({
                                "heldout_template_id": heldout,
                                "heldout_template": TEMPLATES[heldout]["name"],
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
    lines = [f"# Phase 115 Causal Subspace Robustness: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layer: L{result['layer']}")
    lines.append("")
    lines.append("| category | best full | best matched random | best release-excluded | best LTO | best LTO random |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]
        lto = item["leave_template_out"]

        def pick(rows, kind):
            xs = [r for r in rows if r["kind"] == kind]
            return min(xs, key=lambda r: r["target_delta"]) if xs else None

        def fmt(r):
            if r is None:
                return "NA"
            return f"r{r['rank']} s{r['scale']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"

        lines.append(
            f"| {cat} | {fmt(pick(conds, 'answer_contrast_subspace'))} | "
            f"{fmt(pick(conds, 'matched_spectrum_random'))} | "
            f"{fmt(pick(conds, 'release_excluded_subspace'))} | "
            f"{fmt(pick(lto, 'lto_answer_contrast_subspace'))} | "
            f"{fmt(pick(lto, 'lto_matched_spectrum_random'))} |"
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
    parser.add_argument("--ranks", default="8,16")
    parser.add_argument("--scales", default="0.25,0.5,1.0,1.5")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase115_{args.model}_causal_subspace_robustness.json"
    md_path = out_dir / f"phase115_{args.model}_causal_subspace_robustness.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
