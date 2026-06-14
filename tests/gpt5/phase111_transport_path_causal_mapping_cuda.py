#!/usr/bin/env python3
"""
Phase 111: test whether transport-aligned components behave like a real
object_last -> answer_last category path.

The script patches target transport directions at object_last or answer_last
across a short layer sweep, then monitors both final DCF logits and the
answer_last transport projection at the peak layer.
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
from phase110_orthogonal_subspace_split_cuda import NEIGHBOR_SETS, basis_project, capture_transport_dirs  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase111_transport_path_causal_mapping")
TEST_CATEGORIES = ["number", "time", "container", "clothing", "furniture", "plant"]
PATCH_SITES = ["object_last", "answer_last"]
PATCH_MODES = ["remove_target", "amplify_target", "wrong_inject_abs", "random_remove"]
DEFAULT_SCALES = [0.25, 0.5, 1.0]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def build_transport_components(
    centers: np.ndarray,
    transport_dirs: dict[str, np.ndarray],
    readout_dirs: dict[str, np.ndarray],
    categories: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    boundaries = build_boundaries(centers, categories)
    out: dict[str, dict[str, np.ndarray]] = {}
    for cat in categories:
        b = boundaries[cat]
        b_parallel = projection(b, readout_dirs[cat]).astype(np.float32)
        b_orth = (b - b_parallel).astype(np.float32)
        neighbor_basis = [boundaries[n] for n in NEIGHBOR_SETS.get(cat, []) if n in boundaries]
        neighbor_component = basis_project(b_orth, neighbor_basis)
        after_neighbor = (b_orth - neighbor_component).astype(np.float32)
        transport_component = projection(after_neighbor, transport_dirs[cat]).astype(np.float32)
        residual_component = (after_neighbor - transport_component).astype(np.float32)
        out[cat] = {
            "boundary": b,
            "orthogonal": b_orth,
            "neighbor": neighbor_component,
            "transport": transport_component,
            "residual": residual_component,
            "raw_transport": transport_dirs[cat],
        }
    return out


def choose_wrong_category(cat: str, test_categories: list[str], categories: list[str]) -> str:
    for item in NEIGHBOR_SETS.get(cat, []):
        if item in categories and item != cat:
            return item
    for item in test_categories:
        if item != cat:
            return item
    return categories[(categories.index(cat) + 1) % len(categories)]


def make_transport_hook(
    direction: torch.Tensor,
    positions: torch.Tensor,
    mode: str,
    scale: float,
):
    direction = direction / (direction.norm() + 1e-8)

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
        coeff = (vecs @ direction.float()).to(out.dtype)
        d = direction.to(out.dtype)
        if mode in {"remove_target", "random_remove"}:
            out[bidx, pos, :] = out[bidx, pos, :] - scale * coeff[:, None] * d
        elif mode == "amplify_target":
            out[bidx, pos, :] = out[bidx, pos, :] + scale * coeff[:, None] * d
        elif mode == "wrong_inject_abs":
            mag = coeff.abs().mean().clamp_min(torch.tensor(1e-4, device=out.device, dtype=out.dtype))
            out[bidx, pos, :] = out[bidx, pos, :] + scale * mag * d
        else:
            raise ValueError(f"Unknown patch mode: {mode}")
        if rest is not None:
            return (out,) + rest
        return out

    return hook


def run_with_monitor(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, str]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_direction: np.ndarray,
    patch_layer: int | None = None,
    patch_site: str = "answer_last",
    patch_mode: str = "remove_target",
    patch_direction: np.ndarray | None = None,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    score_chunks = []
    answer_proj_chunks = []
    object_proj_chunks = []
    monitor_dir = torch.tensor(monitor_direction, device=device, dtype=torch.float32)
    monitor_dir = monitor_dir / (monitor_dir.norm() + 1e-8)
    module_index = None if patch_layer is None else patch_layer - 1

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
        if patch_direction is not None and module_index is not None:
            patch_pos = object_pos if patch_site == "object_last" else answer_pos
            d = torch.tensor(patch_direction, device=device, dtype=torch.float32)
            handle = layers[module_index].register_forward_hook(
                make_transport_hook(d, patch_pos, patch_mode, scale)
            )
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        if handle is not None:
            handle.remove()

        pos_gpu = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        score_chunks.append(score_logits(logits, cat_local_ids, categories))

        hs = out.hidden_states[monitor_layer]
        bidx = torch.arange(hs.shape[0], device=hs.device)
        ans_pos = answer_pos.to(hs.device)
        obj_pos = object_pos.to(hs.device)
        ans = hs[bidx, ans_pos, :].float()
        obj = hs[bidx, obj_pos, :].float()
        answer_proj_chunks.append((ans @ monitor_dir.to(hs.device)).detach().float().cpu().numpy())
        object_proj_chunks.append((obj @ monitor_dir.to(hs.device)).detach().float().cpu().numpy())

        del out, batch
        torch.cuda.empty_cache()

    return {
        "scores": np.concatenate(score_chunks, axis=0),
        "answer_proj": np.concatenate(answer_proj_chunks, axis=0),
        "object_proj": np.concatenate(object_proj_chunks, axis=0),
    }


def parse_scales(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)
        peak_layer = args.monitor_layer if args.monitor_layer is not None else BOUNDARY_LAYER[args.model]
        first_layer = max(1, peak_layer - args.layer_back)
        patch_layers = list(range(first_layer, peak_layer + 1))
        scales = parse_scales(args.scales)

        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, patch_layers={patch_layers}, vram={alloc:.2f}/{reserved:.2f}GB")

        components_by_layer: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for layer_id in patch_layers:
            log(f"Building layer L{layer_id} components")
            centers = capture_centers(
                model, tokenizer, device, categories, layer_id,
                args.train_objects, args.batch_size, args.max_length
            )
            transport_dirs = capture_transport_dirs(
                model, tokenizer, device, categories, layer_id,
                args.train_objects, args.batch_size, args.max_length
            )
            components_by_layer[layer_id] = build_transport_components(
                centers, transport_dirs, readout_dirs, categories
            )

        result = {
            "phase": 111,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitor_layer": peak_layer,
            "patch_layers": patch_layers,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "patch_sites": PATCH_SITES,
            "patch_modes": PATCH_MODES,
            "scales": scales,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        test_start = args.train_objects
        test_end = args.train_objects + args.test_objects
        monitor_components = components_by_layer[peak_layer]
        for idx, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
            prompts = []
            for tpl in TEMPLATES:
                for obj in CATEGORY_OBJECTS[cat][test_start:test_end]:
                    prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj)})
            target_idx = categories.index(cat)
            wrong_cat = choose_wrong_category(cat, test_categories, categories)
            monitor_dir = monitor_components[cat]["transport"]
            if np.linalg.norm(monitor_dir) < 1e-8:
                monitor_dir = monitor_components[cat]["raw_transport"]
            baseline = run_with_monitor(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_dir
            )
            cat_out = {
                "n_prompts": len(prompts),
                "wrong_category": wrong_cat,
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_transport_proj": float(baseline["answer_proj"].mean()),
                "baseline_object_transport_proj": float(baseline["object_proj"].mean()),
                "conditions": [],
            }
            for layer_id in patch_layers:
                layer_components = components_by_layer[layer_id]
                target_dir = layer_components[cat]["transport"]
                wrong_dir = layer_components[wrong_cat]["transport"]
                random_dir = deterministic_random_direction(target_dir.shape[0], 8100 + layer_id * 37 + target_idx)
                layer_info = {
                    "transport_norm": float(np.linalg.norm(target_dir)),
                    "raw_transport_norm": float(np.linalg.norm(layer_components[cat]["raw_transport"])),
                    "monitor_transport_cos": cos(target_dir, monitor_dir),
                    "wrong_transport_cos": cos(target_dir, wrong_dir),
                }
                for site in PATCH_SITES:
                    for scale in scales:
                        for mode in PATCH_MODES:
                            if mode == "wrong_inject_abs":
                                patch_dir = wrong_dir
                            elif mode == "random_remove":
                                patch_dir = random_dir
                            else:
                                patch_dir = target_dir
                            patched = run_with_monitor(
                                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                                args.batch_size, args.max_length, peak_layer, monitor_dir,
                                patch_layer=layer_id, patch_site=site, patch_mode=mode,
                                patch_direction=patch_dir, scale=scale
                            )
                            delta = patched["scores"] - baseline["scores"]
                            summary = summarize_delta(delta, target_idx, categories)
                            cat_out["conditions"].append({
                                "patch_layer": layer_id,
                                "patch_site": site,
                                "scale": scale,
                                "patch_mode": mode,
                                **layer_info,
                                **summary,
                                "answer_transport_proj_delta": float((patched["answer_proj"] - baseline["answer_proj"]).mean()),
                                "object_transport_proj_delta": float((patched["object_proj"] - baseline["object_proj"]).mean()),
                            })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 111 Transport Path Causal Mapping: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}")
    lines.append(f"Patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | best object remove | best answer remove | best object answer-proj down | strongest wrong inject | best random |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]

        def pick(filter_fn, key):
            xs = [c for c in conds if filter_fn(c)]
            return min(xs, key=key) if xs else None

        object_remove = pick(lambda c: c["patch_site"] == "object_last" and c["patch_mode"] == "remove_target", lambda c: c["target_delta"])
        answer_remove = pick(lambda c: c["patch_site"] == "answer_last" and c["patch_mode"] == "remove_target", lambda c: c["target_delta"])
        object_proj = pick(lambda c: c["patch_site"] == "object_last" and c["patch_mode"] == "remove_target", lambda c: c["answer_transport_proj_delta"])
        wrong = pick(lambda c: c["patch_mode"] == "wrong_inject_abs", lambda c: c["target_delta"])
        random = pick(lambda c: c["patch_mode"] == "random_remove", lambda c: c["target_delta"])

        def fmt(c):
            if c is None:
                return "NA"
            return (
                f"L{c['patch_layer']} {c['patch_site']} s{c['scale']} "
                f"T{c['target_delta']:+.2f} Aproj{c['answer_transport_proj_delta']:+.2f}"
            )

        lines.append(
            f"| {cat} | {fmt(object_remove)} | {fmt(answer_remove)} | {fmt(object_proj)} | "
            f"{fmt(wrong)} | {fmt(random)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--monitor-layer", type=int, default=None)
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--categories", default="")
    parser.add_argument("--scales", default="0.25,0.5,1.0")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase111_{args.model}_transport_path_causal_mapping.json"
    md_path = out_dir / f"phase111_{args.model}_transport_path_causal_mapping.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
