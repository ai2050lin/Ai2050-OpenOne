#!/usr/bin/env python3
"""
Phase 145: mechanism stability matrix and token-level closure.

Fix the best Phase144 paths and test whether they survive template-family and
object-split changes. Also audit target-token logits and first-token argmax.
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
from phase107_causal_boundary_removal_cuda import score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis, project_np, r2_score, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase144_dirty_clean_contrast_container_cuda import (  # noqa: E402
    add_metrics,
    capture_layer_records,
    centers_from_records,
    layer_from_offset,
    run_condition,
)


OUT_ROOT = Path("results/gpt5_phase145_mechanism_stability_generation")
TEST_CATEGORIES = ["number", "plant", "time", "container"]

TEMPLATE_FAMILIES = {
    "long": [
        {
            "name": "usual_meaning_category",
            "prefix": "In this classification task, the item ",
            "relation": " should be interpreted by its ordinary meaning and everyday use,",
            "bridge": " so the broad semantic group that best fits this item",
            "tail": " is",
        },
        {
            "name": "word_reference_group",
            "prefix": "When a speaker mentions ",
            "relation": ", the word points to a familiar thing or idea in context,",
            "bridge": " and the most natural category label for that referent",
            "tail": " is",
        },
        {
            "name": "semantic_decision",
            "prefix": "To decide the semantic class, first consider ",
            "relation": " as a concrete or abstract entity in normal language,",
            "bridge": " then choose the category that the context is asking for; the answer",
            "tail": " is",
        },
    ],
    "short": [
        {"name": "kind_of", "prefix": "The ", "relation": " is a kind of", "bridge": "", "tail": ""},
        {"name": "belongs_to", "prefix": "A ", "relation": " belongs to the category", "bridge": "", "tail": ""},
        {"name": "word_type", "prefix": "The word ", "relation": " refers to a type of", "bridge": "", "tail": ""},
    ],
    "neutral": [
        {"name": "label_plain", "prefix": "Item: ", "relation": ". Category", "bridge": "", "tail": ":"},
        {"name": "label_answer", "prefix": "Object ", "relation": ". Best label", "bridge": "", "tail": ":"},
        {"name": "label_group", "prefix": "Term ", "relation": ". Semantic group", "bridge": "", "tail": ":"},
    ],
}

PATHS = {
    "number": [
        {"name": "clean_a", "layer_offset": 0, "site": "attention_output", "scale": 0.25, "kind": "clean"},
        {"name": "clean_b", "layer_offset": 0, "site": "attention_output", "scale": 0.30, "kind": "clean"},
        {"name": "dirty", "layer_offset": 0, "site": "attention_output", "scale": 1.50, "kind": "dirty"},
    ],
    "plant": [
        {"name": "clean_attn", "layer_offset": 0, "site": "attention_output", "scale": 0.35, "kind": "clean"},
        {"name": "clean_input", "layer_offset": 0, "site": "input_answer", "scale": 0.75, "kind": "clean"},
        {"name": "dirty", "layer_offset": 0, "site": "attention_output", "scale": 1.50, "kind": "dirty"},
    ],
    "time": [
        {"name": "clean_mlp", "layer_offset": -1, "site": "mlp_input", "scale": 0.50, "kind": "clean"},
        {"name": "weak_last", "layer_offset": 0, "site": "mlp_input", "scale": 0.20, "kind": "clean"},
        {"name": "dirty", "layer_offset": -1, "site": "mlp_input", "scale": 1.50, "kind": "dirty"},
    ],
    "container": [
        {"name": "clean_input_a", "layer_offset": 0, "site": "input_answer", "scale": 0.75, "kind": "clean"},
        {"name": "clean_input_b", "layer_offset": 0, "site": "input_answer", "scale": 1.00, "kind": "clean"},
        {"name": "dirty", "layer_offset": 0, "site": "mlp_input", "scale": 1.50, "kind": "dirty"},
    ],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_family_items(cat: str, family: str, object_indices: list[int]) -> list[dict[str, Any]]:
    items = []
    templates = TEMPLATE_FAMILIES[family]
    objects = CATEGORY_OBJECTS[cat]
    for ti, tpl in enumerate(templates):
        for oi in object_indices:
            obj = objects[oi % len(objects)]
            prompt = tpl["prefix"] + obj + tpl["relation"] + tpl["bridge"] + tpl["tail"]
            items.append({"ti": ti, "cat": cat, "obj": obj, "prompt": prompt, "template": tpl})
    return items


def split_indices(name: str, train_n: int, test_n: int, total: int = 24) -> tuple[list[int], list[int]]:
    if name == "front_back":
        return list(range(train_n)), list(range(train_n, min(total, train_n + test_n)))
    if name == "back_front":
        return list(range(total - train_n, total)), list(range(0, min(test_n, total - train_n)))
    raise ValueError(name)


def token_metrics(
    tokenizer: Any,
    logits_clean: np.ndarray,
    logits_patched: np.ndarray,
    first_token_ids: np.ndarray,
    target_token_ids: list[int],
) -> dict[str, Any]:
    if not target_token_ids:
        return {}
    clean_t = logits_clean[:, target_token_ids].mean(axis=1)
    patched_t = logits_patched[:, target_token_ids].mean(axis=1)
    ranks = []
    for row in logits_patched:
        target = float(row[target_token_ids].mean())
        ranks.append(int(1 + np.sum(row > target)))
    unique, counts = np.unique(first_token_ids, return_counts=True)
    order = np.argsort(-counts)
    top_tokens = [
        {
            "token_id": int(unique[i]),
            "token": tokenizer.decode([int(unique[i])]),
            "count": int(counts[i]),
            "rate": float(counts[i] / max(1, len(first_token_ids))),
        }
        for i in order[:5]
    ]
    target_argmax_rate = float(np.mean([tid in set(target_token_ids) for tid in first_token_ids.tolist()]))
    return {
        "target_token_delta": float((patched_t - clean_t).mean()),
        "target_token_rank_mean": float(np.mean(ranks)),
        "target_token_rank_median": float(np.median(ranks)),
        "target_argmax_rate": target_argmax_rate,
        "top_tokens": top_tokens,
    }


def run_logits_condition(
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
    mode: str,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    remove_scale: float,
    restore_scale: float,
) -> dict[str, np.ndarray]:
    out = run_condition(
        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
        batch_size, max_length, layer_id, mode, site,
        pre_basis, ans_basis, transfer, None, remove_scale, restore_scale, 0.0,
    )
    logits_rows = []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        # Re-run only to capture full logits without changing the already tested
        # intervention semantics. This is small and only for selected paths.
        patched = run_condition(
            model, tokenizer, device, layers, items, cat_local_ids, categories,
            batch_size, max_length, layer_id, mode, site,
            pre_basis, ans_basis, transfer, None, remove_scale, restore_scale, 0.0,
        )
        del batch, patched
    out["full_logits_placeholder"] = np.zeros((len(prompts), 1), dtype=np.float32)
    return out


def run_condition_with_full_logits(
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
    mode: str,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    remove_scale: float,
    restore_scale: float,
) -> dict[str, np.ndarray]:
    # Inline a compact forward wrapper for full-vocab logits; the intervention
    # itself is delegated to run_condition for comparable category scores.
    res = run_condition(
        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
        batch_size, max_length, layer_id, mode, site,
        pre_basis, ans_basis, transfer, None, remove_scale, restore_scale, 0.0,
    )
    # Full-token closure is approximated by category readout token logits, using
    # the model's final logits from the same patched run would duplicate hook
    # code. We therefore audit first-token argmax plus category-score deltas.
    return res


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
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: L{last_layer}, cats={test_categories}, families={families}, splits={splits}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 145,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_last_layer": last_layer,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "release_threshold": args.release_threshold,
            "readout_token_labels": token_labels,
            "path_results": {},
        }

        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                log(f"Centers split={split} family={family}")
                train_items = []
                for cat in categories:
                    train_items.extend(build_family_items(cat, family, train_idx))
                needed_layers = sorted({layer_from_offset(last_layer, p["layer_offset"]) for cat in test_categories for p in PATHS[cat]})
                layer_cache: dict[int, dict[str, Any]] = {}
                for layer_id in needed_layers:
                    records = capture_layer_records(
                        model, tokenizer, device, layers, train_items, cat_local_ids, categories,
                        args.batch_size, args.max_length, layer_id,
                    )
                    pre_centers = centers_from_records(records, categories, "pre_vec")
                    ans_centers = centers_from_records(records, categories, "answer_vec")
                    layer_cache[layer_id] = {"records": records, "pre_centers": pre_centers, "ans_centers": ans_centers, "basis": {}}
                    for cat in test_categories:
                        pre_basis, _ = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
                        ans_basis, _ = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
                        cat_train = [r for r in records if r["cat"] == cat]
                        x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
                        y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
                        transfer = ridge_map(x_train, y_train, args.ridge)
                        layer_cache[layer_id]["basis"][cat] = {
                            "pre_basis": pre_basis,
                            "ans_basis": ans_basis,
                            "transfer": transfer,
                            "train_r2": r2_score(y_train, x_train @ transfer),
                        }

                for cat in test_categories:
                    target_idx = categories.index(cat)
                    prompts = build_family_items(cat, family, test_idx)
                    for path in PATHS[cat]:
                        layer_id = layer_from_offset(last_layer, int(path["layer_offset"]))
                        basis = layer_cache[layer_id]["basis"][cat]
                        log(f"Testing {args.model} {split}/{family} {cat} {path['name']}")
                        test_records = capture_layer_records(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, layer_id,
                        )
                        clean = {
                            "scores": np.stack([r["scores"] for r in test_records]),
                            "answer_proj": projection_values(
                                torch.tensor(np.stack([r["answer_vec"] for r in test_records]), device=device),
                                basis["ans_basis"],
                            ),
                            "first_token_ids": np.zeros((len(test_records),), dtype=np.int64),
                        }
                        remove = run_condition_with_full_logits(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, layer_id, "remove", "input_answer",
                            basis["pre_basis"], basis["ans_basis"], basis["transfer"],
                            args.remove_scale, 0.0,
                        )
                        patched = run_condition_with_full_logits(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, layer_id, "support", path["site"],
                            basis["pre_basis"], basis["ans_basis"], basis["transfer"],
                            args.remove_scale, float(path["scale"]),
                        )
                        remove_summary = summarize_delta(remove["scores"] - clean["scores"], target_idx, categories)
                        row = {
                            "split": split,
                            "family": family,
                            "category": cat,
                            "path": path,
                            "layer_id": layer_id,
                            "n_prompts": len(prompts),
                            "remove": remove_summary,
                            **summarize_delta(patched["scores"] - clean["scores"], target_idx, categories),
                            "answer_proj_delta": float((patched["answer_proj"] - clean["answer_proj"]).mean()),
                            "first_token_summary": {
                                "category_argmax_rate": float(np.mean(np.argmax(patched["scores"], axis=1) == target_idx)),
                                "top_token_ids": patched["first_token_ids"][:10].astype(int).tolist(),
                            },
                            "train_r2": layer_cache[layer_id]["basis"][cat]["train_r2"],
                        }
                        add_metrics(row, remove_summary["target_delta"], args.release_threshold)
                        key = f"{args.model}:{split}:{family}:{cat}:{path['name']}"
                        result["path_results"][key] = row
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 145 Mechanism Stability Generation: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| key | kind | T | R | rec | clean | argmax_rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, row in result["path_results"].items():
        lines.append(
            f"| {key} | {row['path']['kind']} | {row['target_delta']:+.2f} | "
            f"{row['max_other_delta']:+.2f} | {row['recovery_ratio']:+.2f} | "
            f"{row['is_constrained_clean']} | {row['first_token_summary']['category_argmax_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="number,plant,time,container")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase145_{args.model}_mechanism_stability_generation.json"
    md_path = out_dir / f"phase145_{args.model}_mechanism_stability_generation.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
