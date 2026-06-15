#!/usr/bin/env python3
"""
Phase 139: restore/swap calibration for transfer-map closure.

Calibrate restore scale/site and replace prototype swap with sample-conditioned
swap under the Phase138 transfer-map framework.
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


OUT_ROOT = Path("results/gpt5_phase139_restore_swap_calibration")
TEST_CATEGORIES = ["number", "container", "plant", "time"]
SWAP_PAIRS = {
    "number": "container",
    "container": "plant",
    "plant": "time",
    "time": "number",
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def top_category_stats(scores: np.ndarray, categories: list[str], target: str, swap_cat: str | None = None) -> dict[str, Any]:
    argmax = scores.argmax(axis=1)
    target_idx = categories.index(target)
    out = {
        "target_top_rate": float(np.mean(argmax == target_idx)),
        "mean_target_score": float(scores[:, target_idx].mean()),
    }
    if swap_cat is not None:
        swap_idx = categories.index(swap_cat)
        out["swap_top_rate"] = float(np.mean(argmax == swap_idx))
        out["mean_swap_score"] = float(scores[:, swap_idx].mean())
    return out


def summarize_condition(
    patched: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    target_idx: int,
    categories: list[str],
) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def run_calibrated_condition(
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
    donor_pre_coeffs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    module = layers[last_layer - 1]
    pre_b = torch.tensor(normalize_basis(remove_pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(inject_ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    donor_coeffs_t = None if donor_pre_coeffs is None else torch.tensor(donor_pre_coeffs, dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        handles = []
        store: dict[str, torch.Tensor] = {}

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
            elif mode == "sample_swap":
                if donor_coeffs_t is None:
                    raise RuntimeError("sample_swap requires donor_pre_coeffs")
                pred = donor_coeffs_t[start:start + len(items)].to(out.device) @ w.to(out.device)
            else:
                pred = None
            if pred is not None:
                add = pred @ ans_b.to(out.device)
                if inject_site == "input_answer":
                    bidx = torch.arange(out.shape[0], device=out.device)
                    pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                    out[bidx, pos, :] = out[bidx, pos, :] + inject_scale * add.to(out.dtype)
                else:
                    store["answer_add"] = add
            return (out,) + inputs[1:]

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

        handles.append(module.register_forward_pre_hook(pre_hook))
        if mode in {"restore", "sample_swap"} and inject_site == "block_output":
            handles.append(module.register_forward_hook(post_hook))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[last_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


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
        swap_scales = parse_float_list(args.swap_scales)
        restore_sites = parse_str_list(args.restore_sites)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, train/test="
            f"{args.train_objects}/{args.test_objects}, rank={args.rank}, cats={test_categories}, "
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
            "phase": 139,
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
            "swap_scales": swap_scales,
            "restore_sites": restore_sites,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        test_record_cache: dict[str, list[dict[str, Any]]] = {}
        prompt_cache: dict[str, list[dict[str, Any]]] = {}
        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_long_prompts(cat, args.train_objects, args.test_objects)
            prompt_cache[cat] = prompts
            test_records = capture_records(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer,
            )
            test_record_cache[cat] = test_records
            cc = cache[cat]
            x_test = project_np(np.stack([r["pre_vec"] for r in test_records]), cc["pre_basis"])
            y_test = project_np(np.stack([r["answer_vec"] for r in test_records]), cc["ans_basis"])
            pred_test = x_test @ cc["transfer"]
            baseline = {
                "scores": np.stack([r["scores"] for r in test_records]),
                "answer_proj": projection_values(
                    torch.tensor(np.stack([r["answer_vec"] for r in test_records]), device=device),
                    cc["ans_basis"],
                ),
            }
            remove = run_calibrated_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "remove",
                cc["pre_basis"], cc["ans_basis"], cc["ans_basis"], cc["transfer"],
                args.remove_scale, 0.0, "input_answer",
            )
            remove_summary = summarize_condition(remove, baseline, target_idx, categories)
            restore_rows = []
            for site in restore_sites:
                for scale in restore_scales:
                    patched = run_calibrated_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, last_layer, "restore",
                        cc["pre_basis"], cc["ans_basis"], cc["ans_basis"], cc["transfer"],
                        args.remove_scale, scale, site,
                    )
                    row = {
                        "site": site,
                        "scale": scale,
                        **summarize_condition(patched, baseline, target_idx, categories),
                    }
                    row["recovery_ratio"] = float((row["target_delta"] - remove_summary["target_delta"]) / (abs(remove_summary["target_delta"]) + 1e-8))
                    row["top_stats"] = top_category_stats(patched["scores"], categories, cat)
                    restore_rows.append(row)

            swap_cat = SWAP_PAIRS.get(cat, test_categories[(ci % len(test_categories))])
            if swap_cat not in test_categories:
                swap_cat = test_categories[(ci % len(test_categories))]
            if swap_cat not in test_record_cache:
                donor_prompts = build_long_prompts(swap_cat, args.train_objects, args.test_objects)
                prompt_cache[swap_cat] = donor_prompts
                test_record_cache[swap_cat] = capture_records(
                    model, tokenizer, device, layers, donor_prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer,
                )
            donor_cache = cache[swap_cat]
            donor_coeff = project_np(np.stack([r["pre_vec"] for r in test_record_cache[swap_cat]]), donor_cache["pre_basis"])
            swap_rows = []
            for site in restore_sites:
                for scale in swap_scales:
                    patched = run_calibrated_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, last_layer, "sample_swap",
                        cc["pre_basis"], donor_cache["ans_basis"], cc["ans_basis"], donor_cache["transfer"],
                        args.remove_scale, scale, site, donor_pre_coeffs=donor_coeff,
                    )
                    delta = patched["scores"] - baseline["scores"]
                    row = {
                        "site": site,
                        "scale": scale,
                        "swap_category": swap_cat,
                        **summarize_condition(patched, baseline, target_idx, categories),
                        "swap_category_delta": float(delta[:, categories.index(swap_cat)].mean()),
                        "top_stats": top_category_stats(patched["scores"], categories, cat, swap_cat),
                    }
                    swap_rows.append(row)

            best_restore = max(restore_rows, key=lambda x: (x["recovery_ratio"], x["target_delta"]))
            best_swap = max(swap_rows, key=lambda x: (x["swap_category_delta"], -x["target_delta"]))
            result["category_results"][cat] = {
                "n_prompts": len(prompts),
                "position_audit": position_audit_long(tokenizer, prompts, args.max_length),
                "transfer_r2": r2_score(y_test, pred_test),
                "transfer_cosine": cosine_mean(y_test, pred_test),
                "pre_singular_values": [float(x) for x in cc["pre_sv"]],
                "answer_singular_values": [float(x) for x in cc["ans_sv"]],
                "baseline_top_stats": top_category_stats(baseline["scores"], categories, cat),
                "remove": remove_summary,
                "restore_sweep": restore_rows,
                "sample_swap_sweep": swap_rows,
                "best_restore": best_restore,
                "best_sample_swap": best_swap,
            }
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    prefix = ""
    if "site" in row:
        prefix = f"{row['site']} s{row['scale']} "
    extra = ""
    if "swap_category" in row:
        extra = f" swap={row['swap_category']} SΔ{row['swap_category_delta']:+.2f}"
    rec = f" rec{row['recovery_ratio']:+.2f}" if "recovery_ratio" in row else ""
    return f"{prefix}T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f}{rec}{extra}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 139 Restore/Swap Calibration: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(
        f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
        f"rank: {result['rank']}; train/test: {result['train_objects_per_category']}/{result['test_objects_per_category']}"
    )
    lines.append("")
    lines.append("| category | transfer | remove | best restore | best sample swap |")
    lines.append("|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
        lines.append(
            f"| {cat} | {transfer} | {_fmt(item['remove'])} | "
            f"{_fmt(item['best_restore'])} | {_fmt(item['best_sample_swap'])} |"
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
    parser.add_argument("--swap-scales", default="0.5,1.0,1.5")
    parser.add_argument("--restore-sites", default="input_answer,block_output")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase139_{args.model}_restore_swap_calibration.json"
    md_path = out_dir / f"phase139_{args.model}_restore_swap_calibration.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
