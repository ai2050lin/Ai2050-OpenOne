#!/usr/bin/env python3
"""
Phase 129: position-corrected gateway audit.

Audit whether boundary-peak pre-answer effects survive when positions are
computed in the actual padded token grid, then compare peak layer, true last
layer, and final norm sites.
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
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from model_registry import get_model_spec  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES, find_subsequence  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase126_residual_gap_decomposition_cuda import capture_answer_centers, replace_output_tensor, tensor_from_output  # noqa: E402
from phase128_final_block_gateway_cuda import get_final_norm  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase129_position_corrected_gateway_audit")
TEST_CATEGORIES = ["number", "container", "plant"]
SITES = [
    "peak_block_input",
    "peak_block_output",
    "last_block_input",
    "last_block_output",
    "final_norm_input",
    "final_norm_output",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def last_nonpad_positions(attention_mask: torch.Tensor) -> list[int]:
    out = []
    for row in attention_mask.detach().cpu():
        nz = row.nonzero(as_tuple=False).flatten()
        out.append(int(nz[-1].item()))
    return out


def first_nonpad_positions(attention_mask: torch.Tensor) -> list[int]:
    out = []
    for row in attention_mask.detach().cpu():
        nz = row.nonzero(as_tuple=False).flatten()
        out.append(int(nz[0].item()))
    return out


def object_span_in_batch(tokenizer: Any, item: dict[str, Any], token_ids: list[int], first_nonpad: int, last_nonpad: int) -> list[int]:
    active_ids = token_ids[first_nonpad:last_nonpad + 1]
    obj_ids = tokenizer(item["obj"], add_special_tokens=False)["input_ids"]
    start = find_subsequence(active_ids, obj_ids)
    if start is None:
        obj_ids = tokenizer(" " + item["obj"], add_special_tokens=False)["input_ids"]
        start = find_subsequence(active_ids, obj_ids)
    if start is None:
        return [max(first_nonpad, last_nonpad - 1)]
    begin = first_nonpad + start
    end = min(begin + len(obj_ids), last_nonpad + 1)
    return list(range(begin, end)) or [max(first_nonpad, last_nonpad - 1)]


def corrected_positions_for_site(tokenizer: Any, item: dict[str, Any], token_ids: list[int], first_nonpad: int, last_nonpad: int, site: str) -> list[int]:
    if site == "answer_last":
        return [last_nonpad]
    span = object_span_in_batch(tokenizer, item, token_ids, first_nonpad, last_nonpad)
    start = min(max(span) + 1, last_nonpad)
    excluding = list(range(start, last_nonpad))
    if not excluding:
        excluding = [max(first_nonpad, last_nonpad - 1)]
    return excluding


def get_site_module(model: Any, layers: list[Any], peak_layer: int, last_layer: int, site: str) -> tuple[Any, str]:
    if site == "peak_block_input":
        return layers[peak_layer - 1], "pre"
    if site == "peak_block_output":
        return layers[peak_layer - 1], "post"
    if site == "last_block_input":
        return layers[last_layer - 1], "pre"
    if site == "last_block_output":
        return layers[last_layer - 1], "post"
    if site == "final_norm_input":
        module = get_final_norm(model)
        if module is None:
            raise RuntimeError("final norm unavailable")
        return module, "pre"
    if site == "final_norm_output":
        module = get_final_norm(model)
        if module is None:
            raise RuntimeError("final norm unavailable")
        return module, "post"
    raise ValueError(site)


def make_capture_hook(kind: str, store: dict[str, torch.Tensor]):
    if kind == "pre":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            store["value"] = inputs[0].detach()
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        store["value"] = tensor_from_output(output).detach()
    return hook, False


def make_patch_hook(kind: str, basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def patch_tensor(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        b = basis.to(out.device).float()
        for bi, positions in enumerate(batch_positions):
            if not positions:
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            vecs = out[bi, pos, :].float()
            proj = (vecs @ b.T) @ b
            out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        return out

    if kind == "pre":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            return (patch_tensor(inputs[0]),) + inputs[1:]
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        return replace_output_tensor(output, patch_tensor(tensor_from_output(output)))
    return hook, False


def capture_site_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    categories: list[str],
    peak_layer: int,
    last_layer: int,
    site: str,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})
    module, kind = get_site_module(model, layers, peak_layer, last_layer, site)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x["prompt"] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            first_pos = first_nonpad_positions(batch["attention_mask"])
            last_pos = last_nonpad_positions(batch["attention_mask"])
            token_rows = batch["input_ids"].detach().cpu().tolist()
            store: dict[str, torch.Tensor] = {}
            hook_fn, is_pre = make_capture_hook(kind, store)
            handle = module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn)
            _ = model(**batch, use_cache=False)
            handle.remove()
            value = store["value"]
            for bi, item in enumerate(batch_items):
                positions = corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
                pos = torch.tensor(positions, device=value.device, dtype=torch.long)
                vec = value[bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del batch
            torch.cuda.empty_cache()
    return (centers / counts[:, :, None]).astype(np.float32)


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    peak_layer: int,
    last_layer: int,
    patch_site: str | None = None,
    patch_basis: np.ndarray | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        first_pos = first_nonpad_positions(batch["attention_mask"])
        last_pos = last_nonpad_positions(batch["attention_mask"])
        token_rows = batch["input_ids"].detach().cpu().tolist()
        handles = []
        if patch_site is not None and patch_basis is not None:
            positions = [
                corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
                for bi, item in enumerate(items)
            ]
            basis = torch.tensor(patch_basis, device=device, dtype=torch.float32)
            module, kind = get_site_module(model, layers, peak_layer, last_layer, patch_site)
            hook_fn, is_pre = make_patch_hook(kind, basis, positions, scale)
            handles.append(module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for handle in handles:
            handle.remove()
        pos_gpu = torch.tensor(last_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def position_audit(tokenizer: Any, prompts: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    batch = tokenizer([x["prompt"] for x in prompts], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    token_rows = batch["input_ids"].tolist()
    answer_in_pre = 0
    lengths = []
    for bi, item in enumerate(prompts):
        positions = corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
        answer_in_pre += int(last_pos[bi] in positions)
        lengths.append(len(positions))
    old_answer = (batch["attention_mask"].sum(dim=1) - 1).tolist()
    mismatches = sum(int(old_answer[i] != last_pos[i]) for i in range(len(last_pos)))
    return {
        "n_prompts": len(prompts),
        "answer_in_pre_count": int(answer_in_pre),
        "old_answer_pos_mismatch_count": int(mismatches),
        "mean_pre_len": float(np.mean(lengths)) if lengths else 0.0,
        "min_pre_len": int(min(lengths)) if lengths else 0,
        "max_pre_len": int(max(lengths)) if lengths else 0,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        spec = get_model_spec(args.model)
        cfg = AutoConfig.from_pretrained(str(spec.local_dir), trust_remote_code=spec.trust_remote_code, local_files_only=True)
        cfg_layers = getattr(cfg, "num_hidden_layers", None) or len(layers)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        last_layer = len(layers)
        sites = [x.strip() for x in args.sites.split(",") if x.strip()]
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, cfg_layers={cfg_layers}, sites={sites}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 129,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "config_num_hidden_layers": int(cfg_layers),
            "sites": sites,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log("Building answer monitor centers")
        answer_centers = capture_answer_centers(model, tokenizer, device, categories, peak_layer, args.train_objects, args.batch_size, args.max_length)
        center_cache = {}
        for site in sites:
            log(f"Capturing centers {site}")
            center_cache[site] = capture_site_centers(
                model, tokenizer, device, layers, categories, peak_layer, last_layer, site,
                args.train_objects, args.batch_size, args.max_length,
            )

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            answer_basis, answer_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, answer_basis, peak_layer, last_layer,
            )
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "conditions": [],
            }
            for site in sites:
                basis, sv = svd_basis(build_category_contrast_matrix(center_cache[site], categories, cat), args.rank)
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, peak_layer, answer_basis, peak_layer, last_layer,
                    patch_site=site, patch_basis=basis, scale=args.scale,
                )
                summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
                summary["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
                cat_out["conditions"].append({
                    "site": site,
                    "singular_values": [float(x) for x in sv],
                    **summary,
                })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 129 Position-corrected Gateway Audit: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}")
    lines.append("")
    lines.append("| category | audit | " + " | ".join(result["sites"]) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(result["sites"])) + "|")
    for cat, item in result["category_results"].items():
        by_site = {x["site"]: x for x in item["conditions"]}
        audit = item["position_audit"]
        audit_text = f"answer_in_pre={audit['answer_in_pre_count']}, old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        lines.append(f"| {cat} | {audit_text} | " + " | ".join(_fmt(by_site.get(site)) for site in result["sites"]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--sites", default=",".join(SITES))
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase129_{args.model}_position_corrected_gateway_audit.json"
    md_path = out_dir / f"phase129_{args.model}_position_corrected_gateway_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
