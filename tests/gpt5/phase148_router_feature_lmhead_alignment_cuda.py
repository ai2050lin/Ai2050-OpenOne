#!/usr/bin/env python3
"""
Phase 148: router feature audit and LM-head alignment.

Reuse Phase147 train-selected routers, audit why they fail on heldout data, and
test whether a small LM-head aligned steering term closes the token-rank gap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase107_causal_boundary_removal_cuda import score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis, project_np, r2_score, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import (  # noqa: E402
    capture_records,
    centers_from_records,
    get_site_module,
    layer_from_offset,
    target_token_ids,
    token_audit,
)
from phase147_train_router_format_token_cuda import build_items, clean_baseline, score_row  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase148_router_feature_lmhead_alignment")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa = normalize_basis(a)
    qb = normalize_basis(b)
    s = np.linalg.svd(qa @ qb.T, compute_uv=False)
    return float(np.mean(s)) if s.size else 0.0


def lm_head_direction(model: Any, target_ids: list[int]) -> np.ndarray:
    emb = model.get_output_embeddings()
    if emb is None and hasattr(model, "lm_head"):
        emb = model.lm_head
    weight = emb.weight.detach().float().cpu().numpy()
    vec = weight[target_ids].mean(axis=0).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def compute_support_direction(records: list[dict[str, Any]], pre_basis: np.ndarray, ans_basis: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    x = project_np(np.stack([r["pre_vec"] for r in records]), pre_basis)
    y = (x @ transfer) @ normalize_basis(ans_basis)
    vec = y.mean(axis=0).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def run_condition_lmsteer(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    layer_id: int,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    support_scale: float,
    lm_dir: np.ndarray | None,
    lm_scale: float,
    target_ids: list[int],
) -> dict[str, Any]:
    scores = []
    answer_proj = []
    token_metrics = []
    layer = layers[layer_id - 1]
    site_module, site_kind = get_site_module(layers, layer_id, site)
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    lm_b = None if lm_dir is None else torch.tensor(lm_dir.reshape(1, -1), dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        with torch.no_grad():
            clean_out = model(**batch, use_cache=False)
        pos_gpu = torch.tensor(ctx["last_pos"], device=clean_out.logits.device, dtype=torch.long)
        clean_logits = clean_out.logits[torch.arange(clean_out.logits.shape[0], device=clean_out.logits.device), pos_gpu]
        store: dict[str, torch.Tensor] = {}
        handles = []

        def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]):
            x = inputs[0]
            out = x.clone()
            pb = pre_b.to(out.device)
            coeff_rows = []
            for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                vecs = out[bi, pos, :].float()
                proj = (vecs @ pb.T) @ pb
                out[bi, pos, :] = out[bi, pos, :] - proj.to(out.dtype)
                coeff_rows.append(vecs.mean(dim=0) @ pb.T)
            coeff = torch.stack(coeff_rows, dim=0)
            store["support_add"] = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
            return (out,) + inputs[1:]

        handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        def apply_site(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            bidx = torch.arange(out.shape[0], device=out.device)
            apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
            out[bidx, apos, :] = out[bidx, apos, :] + support_scale * store["support_add"].to(out.device, dtype=out.dtype)
            if lm_b is not None and lm_scale != 0:
                out[bidx, apos, :] = out[bidx, apos, :] + lm_scale * lm_b.to(out.device, dtype=out.dtype)
            return out

        if site_kind == "pre":
            def site_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                return (apply_site(inputs[0]),) + inputs[1:]
            handles.append(site_module.register_forward_pre_hook(site_pre_hook))
        else:
            def site_post_hook(_module: Any, _inputs: Any, output: Any):
                out = apply_site(tensor_from_output(output))
                if isinstance(output, tuple):
                    return (out,) + output[1:]
                return out
            handles.append(site_module.register_forward_hook(site_post_hook))

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[layer_id]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, ans_basis))
        token_metrics.append(token_audit(logits, clean_logits, target_ids, tokenizer))
        del out, clean_out, batch
        torch.cuda.empty_cache()
    merged = {}
    if token_metrics and token_metrics[0]:
        for key in ["target_token_delta", "target_token_rank_mean", "target_token_rank_median", "target_token_argmax_rate"]:
            merged[key] = float(np.mean([m[key] for m in token_metrics]))
        merged["top_tokens"] = token_metrics[0]["top_tokens"]
    return {
        "scores": np.concatenate(scores, axis=0),
        "answer_proj": np.concatenate(answer_proj, axis=0),
        "token": merged,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    phase147_path = PHASE147_ROOT / f"phase147_{args.model}_train_router_format_token.json"
    if not phase147_path.exists():
        raise SystemExit(f"Missing Phase147 result: {phase147_path}")
    phase147 = json.loads(phase147_path.read_text(encoding="utf-8"))
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        allowed_formats = set(parse_str_list(args.formats))
        allowed_families = set(parse_str_list(args.template_families))
        allowed_splits = set(parse_str_list(args.splits))
        lm_scales = parse_float_list(args.lm_scales)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: feature audit cases from Phase147, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 148,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lm_scales": lm_scales,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        for key, prev in phase147["results"].items():
            split, family, fmt, cat = key.split(":")
            if cat not in test_categories or fmt not in allowed_formats or family not in allowed_families or split not in allowed_splits:
                continue
            best = prev["train_best"]
            layer_id = int(best["layer_id"])
            site = best["site"]
            support_scale = float(best["scale"])
            train_idx, test_idx = split_indices(split, int(args.train_objects), int(args.test_objects))
            train_all = []
            for c in categories:
                train_all.extend(build_items(c, family, train_tpl, train_idx, fmt, options))
            train_cat = build_items(cat, family, train_tpl, train_idx, fmt, options)
            held_cat = build_items(cat, family, heldout_tpl, test_idx, fmt, options)
            train_records_all = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
            train_records_cat = [r for r in train_records_all if r["cat"] == cat]
            held_records_cat = capture_records(model, tokenizer, device, layers, held_cat, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
            train_pre_centers = centers_from_records(train_records_all, categories, "pre_vec", len(train_tpl))
            train_ans_centers = centers_from_records(train_records_all, categories, "answer_vec", len(train_tpl))
            # Heldout centers are approximated from all categories with heldout template/object.
            held_all = []
            for c in categories:
                held_all.extend(build_items(c, family, heldout_tpl, test_idx, fmt, options))
            held_records_all = capture_records(model, tokenizer, device, layers, held_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
            held_pre_centers = centers_from_records(held_records_all, categories, "pre_vec", 1)
            held_ans_centers = centers_from_records(held_records_all, categories, "answer_vec", 1)
            train_pre_basis, _ = svd_basis(build_category_contrast_matrix(train_pre_centers, categories, cat), args.rank)
            train_ans_basis, _ = svd_basis(build_category_contrast_matrix(train_ans_centers, categories, cat), args.rank)
            held_pre_basis, _ = svd_basis(build_category_contrast_matrix(held_pre_centers, categories, cat), args.rank)
            held_ans_basis, _ = svd_basis(build_category_contrast_matrix(held_ans_centers, categories, cat), args.rank)
            x_train = project_np(np.stack([r["pre_vec"] for r in train_records_cat]), train_pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in train_records_cat]), train_ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            x_held = project_np(np.stack([r["pre_vec"] for r in held_records_cat]), train_pre_basis)
            y_held = project_np(np.stack([r["answer_vec"] for r in held_records_cat]), train_ans_basis)
            pred_held = x_held @ transfer
            target_ids = target_token_ids(tokenizer, cat)
            lm_dir = lm_head_direction(model, target_ids)
            support_dir = compute_support_direction(held_records_cat, train_pre_basis, train_ans_basis, transfer)
            clean = clean_baseline(held_records_cat, train_ans_basis, device)
            target_idx = categories.index(cat)
            # Use Phase147 heldout remove/clean metrics as the recovery denominator.
            remove_target_delta = prev["heldout"]["target_delta"] - prev["heldout"]["recovery_ratio"] * abs(prev["heldout"]["target_delta"] - prev["heldout"]["target_delta"] + 1e-8)
            # Safer: recompute remove with support scale 0 at input_answer.
            remove = run_condition_lmsteer(model, tokenizer, device, layers, held_cat, cat_local_ids, categories, args.batch_size, args.max_length, layer_id, "input_answer", train_pre_basis, train_ans_basis, transfer, 0.0, None, 0.0, target_ids)
            remove_summary = summarize_delta(remove["scores"] - clean["scores"], target_idx, categories)
            steering_rows = []
            for lm_scale in lm_scales:
                patched = run_condition_lmsteer(model, tokenizer, device, layers, held_cat, cat_local_ids, categories, args.batch_size, args.max_length, layer_id, site, train_pre_basis, train_ans_basis, transfer, support_scale, lm_dir, lm_scale, target_ids)
                row = score_row(patched, clean, remove_summary, target_idx, categories, args.release_threshold)
                row.update({"lm_scale": lm_scale})
                steering_rows.append(row)
            best_steer = min(steering_rows, key=lambda r: (r.get("token", {}).get("target_token_rank_mean", 1e9), r["max_other_delta"]))
            result["results"][key] = {
                "path": {"layer_id": layer_id, "site": site, "scale": support_scale},
                "phase147_train_clean": prev["train_best"]["is_constrained_clean"],
                "phase147_held_clean": prev["heldout"]["is_constrained_clean"],
                "pre_basis_overlap": subspace_overlap(train_pre_basis, held_pre_basis),
                "ans_basis_overlap": subspace_overlap(train_ans_basis, held_ans_basis),
                "heldout_transfer_r2": r2_score(y_held, pred_held),
                "answer_norm_train": float(np.linalg.norm(np.stack([r["answer_vec"] for r in train_records_cat]), axis=1).mean()),
                "answer_norm_heldout": float(np.linalg.norm(np.stack([r["answer_vec"] for r in held_records_cat]), axis=1).mean()),
                "support_lm_cosine": float(np.dot(support_dir, lm_dir)),
                "steering_rows": steering_rows,
                "best_steering": best_steer,
            }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 148 Router Feature LM-Head Alignment: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | prev clean | overlap pre/ans | R2 held | cos support-lm | best lm | rank | argmax | clean |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, item in result["results"].items():
        b = item["best_steering"]
        tok = b.get("token", {})
        lines.append(
            f"| {key} | {item['phase147_held_clean']} | {item['pre_basis_overlap']:.2f}/{item['ans_basis_overlap']:.2f} | "
            f"{item['heldout_transfer_r2']:+.2f} | {item['support_lm_cosine']:+.2f} | {b['lm_scale']} | "
            f"{tok.get('target_token_rank_mean', 0):.1f} | {tok.get('target_token_argmax_rate', 0):.2f} | {b['is_constrained_clean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word")
    parser.add_argument("--lm-scales", default="0.0,0.05,0.1,0.2,0.5")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase148_{args.model}_router_feature_lmhead_alignment.json"
    md_path = out_dir / f"phase148_{args.model}_router_feature_lmhead_alignment.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
