#!/usr/bin/env python3
"""
Phase 147: train-time router generalization and format-gated token closure.

Select layer/site/scale on train templates/objects, then evaluate that selected
router on heldout templates/objects under several prompt-format tails.
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
from phase107_causal_boundary_removal_cuda import summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import TEMPLATE_FAMILIES, split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import (  # noqa: E402
    capture_records,
    centers_from_records,
    layer_from_offset,
    run_condition,
    target_token_ids,
)


OUT_ROOT = Path("results/gpt5_phase147_train_router_format_token")
TEST_CATEGORIES = ["plant", "time", "container", "number"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def format_prompt(base: str, fmt: str, options: list[str]) -> str:
    if fmt == "plain":
        return base
    if fmt == "label_colon":
        return base.rstrip(" :") + " Category:"
    if fmt == "answer_one_word":
        return base.rstrip(" :") + " Answer with one category word:"
    if fmt == "multiple_choice":
        return base.rstrip(" :") + " Options: " + ", ".join(options) + ". Answer:"
    raise ValueError(fmt)


def build_items(cat: str, family: str, template_ids: list[int], object_indices: list[int], fmt: str, options: list[str]) -> list[dict[str, Any]]:
    items = []
    templates = TEMPLATE_FAMILIES[family]
    objects = CATEGORY_OBJECTS[cat]
    for local_ti, ti in enumerate(template_ids):
        tpl = templates[ti % len(templates)]
        for oi in object_indices:
            obj = objects[oi % len(objects)]
            base = tpl["prefix"] + obj + tpl["relation"] + tpl["bridge"] + tpl["tail"]
            items.append({
                "ti": local_ti,
                "cat": cat,
                "obj": obj,
                "prompt": format_prompt(base, fmt, options),
                "template": tpl,
                "format": fmt,
            })
    return items


def add_metrics(row: dict[str, Any], remove_target_delta: float, release_threshold: float) -> None:
    recovery = (row["target_delta"] - remove_target_delta) / (abs(remove_target_delta) + 1e-8)
    row["recovery_ratio"] = float(recovery)
    row["is_constrained_clean"] = bool(recovery >= 0.5 and row["max_other_delta"] <= release_threshold)


def clean_baseline(records: list[dict[str, Any]], ans_basis: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    return {
        "scores": np.stack([r["scores"] for r in records]),
        "answer_proj": projection_values(torch.tensor(np.stack([r["answer_vec"] for r in records]), device=device), ans_basis),
    }


def score_row(patched: dict[str, Any], clean: dict[str, np.ndarray], remove_summary: dict[str, Any], target_idx: int, categories: list[str], release_threshold: float) -> dict[str, Any]:
    row = {
        **summarize_delta(patched["scores"] - clean["scores"], target_idx, categories),
        "answer_proj_delta": float((patched["answer_proj"] - clean["answer_proj"]).mean()),
        "token": patched.get("token", {}),
    }
    add_metrics(row, remove_summary["target_delta"], release_threshold)
    return row


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories) or TEST_CATEGORIES
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        offsets = [int(x) for x in parse_str_list(args.layer_offsets)]
        sites = parse_str_list(args.sites)
        scales = parse_float_list(args.scales)
        options = test_categories
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: L{last_layer}, cats={test_categories}, formats={formats}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 147,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_last_layer": last_layer,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "formats": formats,
            "layer_offsets": offsets,
            "sites": sites,
            "scales": scales,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                for fmt in formats:
                    log(f"Router {split}/{family}/{fmt}")
                    train_items_all = []
                    heldout_items_by_cat = {}
                    train_items_by_cat = {}
                    for cat in categories:
                        train_items = build_items(cat, family, train_tpl, train_idx, fmt, options)
                        train_items_all.extend(train_items)
                        if cat in test_categories:
                            train_items_by_cat[cat] = train_items
                            heldout_items_by_cat[cat] = build_items(cat, family, heldout_tpl, test_idx, fmt, options)
                    layer_cache: dict[int, dict[str, Any]] = {}
                    for offset in offsets:
                        layer_id = layer_from_offset(last_layer, offset)
                        recs = capture_records(model, tokenizer, device, layers, train_items_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                        pre_centers = centers_from_records(recs, categories, "pre_vec", len(train_tpl))
                        ans_centers = centers_from_records(recs, categories, "answer_vec", len(train_tpl))
                        layer_cache[layer_id] = {"basis": {}, "train_records": recs}
                        for cat in test_categories:
                            pre_basis, _ = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
                            ans_basis, _ = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
                            cat_train = [r for r in recs if r["cat"] == cat]
                            x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
                            y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
                            layer_cache[layer_id]["basis"][cat] = {
                                "pre_basis": pre_basis,
                                "ans_basis": ans_basis,
                                "transfer": ridge_map(x_train, y_train, args.ridge),
                            }
                    for cat in test_categories:
                        target_idx = categories.index(cat)
                        target_ids = target_token_ids(tokenizer, cat)
                        best_train = None
                        train_rows = []
                        for offset in offsets:
                            layer_id = layer_from_offset(last_layer, offset)
                            basis = layer_cache[layer_id]["basis"][cat]
                            train_records = capture_records(model, tokenizer, device, layers, train_items_by_cat[cat], cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                            train_clean = clean_baseline(train_records, basis["ans_basis"], device)
                            train_remove = run_condition(model, tokenizer, device, layers, train_items_by_cat[cat], cat_local_ids, categories, args.batch_size, args.max_length, layer_id, "input_answer", basis["pre_basis"], basis["ans_basis"], basis["transfer"], 0.0, target_ids, mode="remove")
                            train_remove_summary = summarize_delta(train_remove["scores"] - train_clean["scores"], target_idx, categories)
                            for site in sites:
                                for scale in scales:
                                    patched = run_condition(model, tokenizer, device, layers, train_items_by_cat[cat], cat_local_ids, categories, args.batch_size, args.max_length, layer_id, site, basis["pre_basis"], basis["ans_basis"], basis["transfer"], scale, target_ids)
                                    row = score_row(patched, train_clean, train_remove_summary, target_idx, categories, args.release_threshold)
                                    row.update({"layer_id": layer_id, "layer_offset": offset, "site": site, "scale": scale})
                                    train_rows.append(row)
                        best_train = max(train_rows, key=lambda r: (r["is_constrained_clean"], r["recovery_ratio"], -r["max_other_delta"]))
                        basis = layer_cache[best_train["layer_id"]]["basis"][cat]
                        held_items = heldout_items_by_cat[cat]
                        held_records = capture_records(model, tokenizer, device, layers, held_items, cat_local_ids, categories, args.batch_size, args.max_length, best_train["layer_id"])
                        held_clean = clean_baseline(held_records, basis["ans_basis"], device)
                        held_remove = run_condition(model, tokenizer, device, layers, held_items, cat_local_ids, categories, args.batch_size, args.max_length, best_train["layer_id"], "input_answer", basis["pre_basis"], basis["ans_basis"], basis["transfer"], 0.0, target_ids, mode="remove")
                        held_remove_summary = summarize_delta(held_remove["scores"] - held_clean["scores"], target_idx, categories)
                        held_patch = run_condition(model, tokenizer, device, layers, held_items, cat_local_ids, categories, args.batch_size, args.max_length, best_train["layer_id"], best_train["site"], basis["pre_basis"], basis["ans_basis"], basis["transfer"], float(best_train["scale"]), target_ids)
                        held_row = score_row(held_patch, held_clean, held_remove_summary, target_idx, categories, args.release_threshold)
                        held_row.update({"layer_id": best_train["layer_id"], "site": best_train["site"], "scale": best_train["scale"]})
                        key = f"{split}:{family}:{fmt}:{cat}"
                        result["results"][key] = {
                            "train_best": best_train,
                            "heldout": held_row,
                            "n_train_candidates": len(train_rows),
                            "n_heldout_prompts": len(held_items),
                        }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 147 Train Router Format Token: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | train path | train clean | held T | held R | held rec | held clean | token_rank | token_argmax |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, item in result["results"].items():
        tr = item["train_best"]
        h = item["heldout"]
        tok = h.get("token", {})
        lines.append(
            f"| {key} | L{tr['layer_id']} {tr['site']} s{tr['scale']} | {tr['is_constrained_clean']} | "
            f"{h['target_delta']:+.2f} | {h['max_other_delta']:+.2f} | {h['recovery_ratio']:+.2f} | "
            f"{h['is_constrained_clean']} | {tok.get('target_token_rank_mean', 0):.1f} | {tok.get('target_token_argmax_rate', 0):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
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
    parser.add_argument("--formats", default="plain,label_colon,answer_one_word,multiple_choice")
    parser.add_argument("--layer-offsets", default="0,-1")
    parser.add_argument("--sites", default="input_answer,attention_output,mlp_input")
    parser.add_argument("--scales", default="0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase147_{args.model}_train_router_format_token.json"
    md_path = out_dir / f"phase147_{args.model}_train_router_format_token.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
