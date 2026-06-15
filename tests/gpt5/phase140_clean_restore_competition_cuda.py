#!/usr/bin/env python3
"""
Phase 140: clean restore criterion and competition decomposition.

Re-run transfer-map restore sweeps with a clean-restore score, competitor
decomposition, and first-token argmax audit. This tests whether restore is
selective rather than merely amplifying target and competitors together.
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
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
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


OUT_ROOT = Path("results/gpt5_phase140_clean_restore_competition")
TEST_CATEGORIES = ["number", "container", "plant", "time", "clothing", "furniture"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_summary(tokenizer: Any, token_ids: np.ndarray, limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter(int(x) for x in token_ids.tolist())
    rows = []
    total = max(1, len(token_ids))
    for tok_id, count in counts.most_common(limit):
        rows.append({
            "token_id": tok_id,
            "token": tokenizer.decode([tok_id]),
            "count": int(count),
            "rate": float(count / total),
        })
    return rows


def category_top_stats(scores: np.ndarray, categories: list[str], target: str) -> dict[str, Any]:
    argmax = scores.argmax(axis=1)
    target_idx = categories.index(target)
    counts = Counter(int(x) for x in argmax.tolist())
    return {
        "target_top_rate": float(np.mean(argmax == target_idx)),
        "top_categories": [
            {"category": categories[i], "count": int(c), "rate": float(c / max(1, len(argmax)))}
            for i, c in counts.most_common(8)
        ],
    }


def add_competition_fields(row: dict[str, Any], remove_target_delta: float, lambda_release: float) -> dict[str, Any]:
    recovery = (row["target_delta"] - remove_target_delta) / (abs(remove_target_delta) + 1e-8)
    row["recovery_ratio"] = float(recovery)
    row["support_component"] = float(row["target_delta"])
    row["release_component"] = float(row["max_other_delta"])
    row["suppressor_component"] = float(row["target_delta"] - row["max_other_delta"])
    row["clean_restore_score"] = float(recovery - lambda_release * max(0.0, row["max_other_delta"]))
    return row


def summarize_condition(
    patched: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    target_idx: int,
    categories: list[str],
) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


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
    remove_pre_basis: np.ndarray,
    inject_ans_basis: np.ndarray,
    monitor_basis: np.ndarray,
    transfer: np.ndarray,
    remove_scale: float,
    inject_scale: float,
    inject_site: str,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    first_token_ids = []
    module = layers[last_layer - 1]
    pre_b = torch.tensor(normalize_basis(remove_pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(inject_ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        handles = []
        store: dict[str, torch.Tensor] = {}

        if mode in {"remove", "restore"}:
            def pre_hook(_module: Any, inputs: tuple[Any, ...]):
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
                    pred = coeff @ w.to(out.device)
                    add = pred @ ans_b.to(out.device)
                    if inject_site == "input_answer":
                        bidx = torch.arange(out.shape[0], device=out.device)
                        pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                        out[bidx, pos, :] = out[bidx, pos, :] + inject_scale * add.to(out.dtype)
                    else:
                        store["answer_add"] = add
                return (out,) + inputs[1:]

            handles.append(module.register_forward_pre_hook(pre_hook))

            if mode == "restore" and inject_site == "block_output":
                def post_hook(_module: Any, _inputs: Any, output: Any):
                    x = tensor_from_output(output)
                    out = x.clone()
                    add = store["answer_add"].to(out.device, dtype=out.dtype)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                    out[bidx, pos, :] = out[bidx, pos, :] + inject_scale * add
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out

                handles.append(module.register_forward_hook(post_hook))
        elif mode != "clean":
            raise ValueError(mode)

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
        answer_proj.append(projection_values(ans, monitor_basis))
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
        restore_scales = parse_float_list(args.restore_scales)
        restore_sites = parse_str_list(args.restore_sites)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, train/test="
            f"{args.train_objects}/{args.test_objects}, rank={args.rank}, cats={test_categories}, "
            f"lambda={args.lambda_release}, vram={alloc:.2f}/{reserved:.2f}GB"
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
            "phase": 140,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "templates": [x["name"] for x in LONG_TEMPLATES],
            "rank": args.rank,
            "ridge": args.ridge,
            "remove_scale": args.remove_scale,
            "restore_scales": restore_scales,
            "restore_sites": restore_sites,
            "lambda_release": args.lambda_release,
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
                args.batch_size, args.max_length, last_layer, "clean",
                cc["pre_basis"], cc["ans_basis"], cc["ans_basis"], cc["transfer"],
                args.remove_scale, 0.0, "input_answer",
            )
            remove = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "remove",
                cc["pre_basis"], cc["ans_basis"], cc["ans_basis"], cc["transfer"],
                args.remove_scale, 0.0, "input_answer",
            )
            remove_summary = summarize_condition(remove, clean, target_idx, categories)
            remove_summary["token_audit"] = token_summary(tokenizer, remove["first_token_ids"])
            remove_summary["category_top_stats"] = category_top_stats(remove["scores"], categories, cat)
            restore_rows = []
            for site in restore_sites:
                for scale in restore_scales:
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, last_layer, "restore",
                        cc["pre_basis"], cc["ans_basis"], cc["ans_basis"], cc["transfer"],
                        args.remove_scale, scale, site,
                    )
                    row = {
                        "site": site,
                        "scale": scale,
                        **summarize_condition(patched, clean, target_idx, categories),
                    }
                    add_competition_fields(row, remove_summary["target_delta"], args.lambda_release)
                    row["token_audit"] = token_summary(tokenizer, patched["first_token_ids"])
                    row["category_top_stats"] = category_top_stats(patched["scores"], categories, cat)
                    row["top_competitor_after_restore"] = max(
                        ((k, v) for k, v in row["mean_delta_by_category"].items() if k != cat),
                        key=lambda x: x[1],
                    )
                    restore_rows.append(row)
            best_by_target = max(restore_rows, key=lambda x: (x["recovery_ratio"], x["target_delta"]))
            best_clean = max(restore_rows, key=lambda x: (x["clean_restore_score"], x["recovery_ratio"]))
            result["category_results"][cat] = {
                "n_prompts": len(prompts),
                "position_audit": position_audit_long(tokenizer, prompts, args.max_length),
                "transfer_r2": r2_score(y_test, pred_test),
                "transfer_cosine": cosine_mean(y_test, pred_test),
                "clean_token_audit": token_summary(tokenizer, clean["first_token_ids"]),
                "clean_category_top_stats": category_top_stats(clean["scores"], categories, cat),
                "remove": remove_summary,
                "restore_sweep": restore_rows,
                "best_by_target": best_by_target,
                "best_clean_restore": best_clean,
            }
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    prefix = f"{row['site']} s{row['scale']} " if "site" in row else ""
    clean = f" clean{row['clean_restore_score']:+.2f}" if "clean_restore_score" in row else ""
    rec = f" rec{row['recovery_ratio']:+.2f}" if "recovery_ratio" in row else ""
    comp = ""
    if "top_competitor_after_restore" in row:
        comp = f" comp={row['top_competitor_after_restore'][0]}:{row['top_competitor_after_restore'][1]:+.2f}"
    return f"{prefix}T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f}{rec}{clean}{comp}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 140 Clean Restore Competition: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(
        f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
        f"rank: {result['rank']}; train/test: {result['train_objects_per_category']}/{result['test_objects_per_category']}; "
        f"lambda: {result['lambda_release']}"
    )
    lines.append("")
    lines.append("| category | transfer | remove | best target restore | best clean restore | clean first tokens |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
        tokens = ", ".join(f"{x['token']}:{x['rate']:.2f}" for x in item["best_clean_restore"]["token_audit"][:3])
        lines.append(
            f"| {cat} | {transfer} | {_fmt(item['remove'])} | "
            f"{_fmt(item['best_by_target'])} | {_fmt(item['best_clean_restore'])} | {tokens} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--restore-scales", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--restore-sites", default="input_answer,block_output")
    parser.add_argument("--lambda-release", type=float, default=0.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase140_{args.model}_clean_restore_competition.json"
    md_path = out_dir / f"phase140_{args.model}_clean_restore_competition.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
