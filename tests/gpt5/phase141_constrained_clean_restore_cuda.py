#!/usr/bin/env python3
"""
Phase 141: constrained clean restore and time failure localization.

Use hard clean-restore constraints and a wider answer-site restore sweep to
separate support recovery, competitor release, and token-level behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase128_final_block_gateway_cuda import get_final_norm  # noqa: E402
from phase135_long_template_source_field_cuda import LONG_TEMPLATES, batch_context, build_long_prompts, position_audit_long  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import (  # noqa: E402
    build_items,
    capture_records,
    centers_from_records,
    cosine_mean,
    normalize_basis,
    project_np,
    r2_score,
    ridge_map,
)
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase141_constrained_clean_restore")
TEST_CATEGORIES = ["number", "container", "plant", "time", "clothing", "furniture"]
RESTORE_SITES = ["input_answer", "attention_output", "mlp_input", "mlp_output", "block_output", "final_norm_input"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_summary(tokenizer: Any, token_ids: np.ndarray, limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter(int(x) for x in token_ids.tolist())
    total = max(1, len(token_ids))
    return [
        {"token_id": tok, "token": tokenizer.decode([tok]), "count": int(count), "rate": float(count / total)}
        for tok, count in counts.most_common(limit)
    ]


def summarize_condition(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def add_restore_metrics(row: dict[str, Any], remove_target_delta: float, release_threshold: float) -> None:
    recovery = (row["target_delta"] - remove_target_delta) / (abs(remove_target_delta) + 1e-8)
    row["recovery_ratio"] = float(recovery)
    row["support_component"] = float(row["target_delta"])
    row["release_component"] = float(row["max_other_delta"])
    row["suppressor_component"] = float(row["target_delta"] - row["max_other_delta"])
    row["is_constrained_clean"] = bool(recovery >= 0.5 and row["max_other_delta"] <= release_threshold)


def get_site_module(model: Any, layers: list[Any], last_layer: int, site: str) -> tuple[Any, str] | None:
    layer = layers[last_layer - 1]
    if site == "attention_output":
        return get_attention_module(layer), "post"
    if site == "mlp_input":
        return get_mlp_module(layer), "pre"
    if site == "mlp_output":
        return get_mlp_module(layer), "post"
    if site == "block_output":
        return layer, "post"
    if site == "final_norm_input":
        norm = get_final_norm(model)
        if norm is None:
            return None
        return norm, "pre"
    if site == "input_answer":
        return layer, "pre"
    raise ValueError(site)


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
    last_layer: int,
    mode: str,
    restore_site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    remove_scale: float,
    restore_scale: float,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    first_token_ids = []
    layer = layers[last_layer - 1]
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    site_module = None if mode != "restore" else get_site_module(model, layers, last_layer, restore_site)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        handles = []
        store: dict[str, torch.Tensor] = {}

        if mode in {"remove", "restore"}:
            def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                x = inputs[0]
                out = x.clone()
                pb = pre_b.to(out.device)
                coeff_rows = []
                for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                    pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                    vecs = out[bi, pos, :].float()
                    proj = (vecs @ pb.T) @ pb
                    out[bi, pos, :] = out[bi, pos, :] - remove_scale * proj.to(out.dtype)
                    coeff_rows.append(vecs.mean(dim=0) @ pb.T)
                if mode == "restore":
                    coeff = torch.stack(coeff_rows, dim=0)
                    add = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
                    if restore_site == "input_answer":
                        bidx = torch.arange(out.shape[0], device=out.device)
                        pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                        out[bidx, pos, :] = out[bidx, pos, :] + restore_scale * add.to(out.dtype)
                    else:
                        store["answer_add"] = add
                return (out,) + inputs[1:]

            handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        if mode == "restore" and restore_site != "input_answer" and site_module is not None:
            module, kind = site_module

            def add_to_tensor(x: torch.Tensor) -> torch.Tensor:
                out = x.clone()
                add = store["answer_add"].to(out.device, dtype=out.dtype)
                bidx = torch.arange(out.shape[0], device=out.device)
                pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                out[bidx, pos, :] = out[bidx, pos, :] + restore_scale * add
                return out

            if kind == "pre":
                def pre_inject_hook(_module: Any, inputs: tuple[Any, ...]):
                    return (add_to_tensor(inputs[0]),) + inputs[1:]

                handles.append(module.register_forward_pre_hook(pre_inject_hook))
            else:
                def post_inject_hook(_module: Any, _inputs: Any, output: Any):
                    out = add_to_tensor(tensor_from_output(output))
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out

                handles.append(module.register_forward_hook(post_inject_hook))

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        first_token_ids.append(logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64))
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[last_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, ans_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {
        "scores": np.concatenate(scores, axis=0),
        "answer_proj": np.concatenate(answer_proj, axis=0),
        "first_token_ids": np.concatenate(first_token_ids, axis=0),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories) or TEST_CATEGORIES
        restore_sites = parse_str_list(args.restore_sites) or RESTORE_SITES
        restore_scales = parse_float_list(args.restore_scales)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, train/test="
            f"{args.train_objects}/{args.test_objects}, cats={test_categories}, sites={restore_sites}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        train_items = build_items(categories, args.train_objects, args.test_objects, "train")
        log(f"Capturing train records: {len(train_items)}")
        train_records = capture_records(
            model, tokenizer, device, layers, train_items, cat_local_ids, categories,
            args.batch_size, args.max_length, last_layer,
        )
        pre_centers = centers_from_records(train_records, categories, "pre_vec")
        ans_centers = centers_from_records(train_records, categories, "answer_vec")

        cache: dict[str, dict[str, Any]] = {}
        for cat in test_categories:
            pre_basis, pre_sv = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
            ans_basis, ans_sv = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
            cat_train = [r for r in train_records if r["cat"] == cat]
            x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
            cache[cat] = {
                "pre_basis": pre_basis,
                "ans_basis": ans_basis,
                "transfer": ridge_map(x_train, y_train, args.ridge),
                "pre_sv": pre_sv,
                "ans_sv": ans_sv,
            }

        result: dict[str, Any] = {
            "phase": 141,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "restore_sites": restore_sites,
            "restore_scales": restore_scales,
            "release_threshold": args.release_threshold,
            "rank": args.rank,
            "ridge": args.ridge,
            "remove_scale": args.remove_scale,
            "templates": [x["name"] for x in LONG_TEMPLATES],
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_long_prompts(cat, args.train_objects, args.test_objects)
            records = capture_records(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer,
            )
            cc = cache[cat]
            x_test = project_np(np.stack([r["pre_vec"] for r in records]), cc["pre_basis"])
            y_test = project_np(np.stack([r["answer_vec"] for r in records]), cc["ans_basis"])
            pred_test = x_test @ cc["transfer"]
            clean = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "clean", "input_answer",
                cc["pre_basis"], cc["ans_basis"], cc["transfer"], args.remove_scale, 0.0,
            )
            remove = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "remove", "input_answer",
                cc["pre_basis"], cc["ans_basis"], cc["transfer"], args.remove_scale, 0.0,
            )
            remove_summary = summarize_condition(remove, clean, target_idx, categories)
            remove_summary["token_audit"] = token_summary(tokenizer, remove["first_token_ids"])
            restore_rows = []
            for site in restore_sites:
                for scale in restore_scales:
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, last_layer, "restore", site,
                        cc["pre_basis"], cc["ans_basis"], cc["transfer"], args.remove_scale, scale,
                    )
                    row = {
                        "site": site,
                        "scale": scale,
                        **summarize_condition(patched, clean, target_idx, categories),
                    }
                    add_restore_metrics(row, remove_summary["target_delta"], args.release_threshold)
                    row["token_audit"] = token_summary(tokenizer, patched["first_token_ids"])
                    row["top_competitor"] = max(
                        ((k, v) for k, v in row["mean_delta_by_category"].items() if k != cat),
                        key=lambda x: x[1],
                    )
                    restore_rows.append(row)
            constrained = [r for r in restore_rows if r["is_constrained_clean"]]
            best_constrained = max(constrained, key=lambda x: (x["recovery_ratio"], x["target_delta"])) if constrained else None
            best_min_release = max(restore_rows, key=lambda x: (-x["max_other_delta"], x["recovery_ratio"]))
            best_target = max(restore_rows, key=lambda x: (x["recovery_ratio"], x["target_delta"]))
            result["category_results"][cat] = {
                "n_prompts": len(prompts),
                "position_audit": position_audit_long(tokenizer, prompts, args.max_length),
                "transfer_r2": r2_score(y_test, pred_test),
                "transfer_cosine": cosine_mean(y_test, pred_test),
                "clean_token_audit": token_summary(tokenizer, clean["first_token_ids"]),
                "remove": remove_summary,
                "restore_sweep": restore_rows,
                "constrained_clean_count": len(constrained),
                "best_constrained_clean": best_constrained,
                "best_min_release": best_min_release,
                "best_target": best_target,
            }
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NONE"
    comp = f" comp={row['top_competitor'][0]}:{row['top_competitor'][1]:+.2f}" if "top_competitor" in row else ""
    return (
        f"{row['site']} s{row['scale']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} "
        f"rec{row['recovery_ratio']:+.2f} clean={row['is_constrained_clean']}{comp}"
    )


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 141 Constrained Clean Restore: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(
        f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
        f"rank: {result['rank']}; train/test: {result['train_objects_per_category']}/{result['test_objects_per_category']}; "
        f"threshold: {result['release_threshold']}"
    )
    lines.append("")
    lines.append("| category | transfer | remove | constrained | min release | best target |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
        rem = f"T{item['remove']['target_delta']:+.2f} R{item['remove']['max_other_delta']:+.2f}"
        lines.append(
            f"| {cat} | {transfer} | {rem} | {_fmt(item['best_constrained_clean'])} | "
            f"{_fmt(item['best_min_release'])} | {_fmt(item['best_target'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=10)
    parser.add_argument("--test-objects", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--restore-scales", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--restore-sites", default="input_answer,attention_output,mlp_input,mlp_output,block_output,final_norm_input")
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase141_{args.model}_constrained_clean_restore.json"
    md_path = out_dir / f"phase141_{args.model}_constrained_clean_restore.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
