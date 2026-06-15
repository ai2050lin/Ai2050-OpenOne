#!/usr/bin/env python3
"""
Phase 142: support/suppressor split and time alternative path.

Test whether adding a competitor-suppressor removal improves clean restore, and
whether time can be restored from earlier layer interfaces.
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
from phase135_long_template_source_field_cuda import LONG_TEMPLATES, batch_context, build_long_prompts  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import (  # noqa: E402
    build_items,
    cosine_mean,
    normalize_basis,
    project_np,
    r2_score,
    ridge_map,
)
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase142_support_suppressor_timepath")
TEST_CATEGORIES = ["number", "container", "plant", "time"]
DEFAULT_COMPETITORS = {
    "number": "animal",
    "container": "machine",
    "plant": "tool",
    "time": "clothing",
    "clothing": "furniture",
    "furniture": "clothing",
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_summary(tokenizer: Any, token_ids: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    counts = Counter(int(x) for x in token_ids.tolist())
    total = max(1, len(token_ids))
    return [
        {"token_id": tok, "token": tokenizer.decode([tok]), "count": int(count), "rate": float(count / total)}
        for tok, count in counts.most_common(limit)
    ]


def layer_from_offset(last_layer: int, offset: int) -> int:
    return max(1, min(last_layer, last_layer + offset))


def get_site_module(layers: list[Any], layer_id: int, site: str) -> tuple[Any, str]:
    layer = layers[layer_id - 1]
    if site == "input_answer":
        return layer, "pre"
    if site == "attention_output":
        return get_attention_module(layer), "post"
    if site == "mlp_input":
        return get_mlp_module(layer), "pre"
    if site == "mlp_output":
        return get_mlp_module(layer), "post"
    if site == "block_output":
        return layer, "post"
    raise ValueError(site)


def centers_from_records(records: list[dict[str, Any]], categories: list[str], key: str) -> np.ndarray:
    d_model = int(records[0][key].shape[0])
    centers = np.zeros((len(LONG_TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(LONG_TEMPLATES), len(categories)), dtype=np.int64)
    cat_index = {cat: i for i, cat in enumerate(categories)}
    for rec in records:
        centers[int(rec["ti"]), cat_index[rec["cat"]]] += rec[key]
        counts[int(rec["ti"]), cat_index[rec["cat"]]] += 1
    return (centers / np.maximum(counts[:, :, None], 1)).astype(np.float32)


def capture_layer_records(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    layer_id: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    module = layers[layer_id - 1]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = batch_context(tokenizer, batch, batch_items)
            store: dict[str, torch.Tensor] = {}

            def pre_hook(_module: Any, inputs: tuple[Any, ...]):
                store["pre"] = inputs[0].detach()

            def post_hook(_module: Any, _inputs: Any, output: Any):
                store["answer"] = tensor_from_output(output).detach()

            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            out = model(**batch, use_cache=False)
            h1.remove()
            h2.remove()
            pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
            scores = score_logits(logits, cat_local_ids, categories)
            for bi, item in enumerate(batch_items):
                src = ctx["source_groups"]["all_pre_answer"][bi]
                pos = torch.tensor(src, device=store["pre"].device, dtype=torch.long)
                pre_vec = store["pre"][bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                ans_vec = store["answer"][bi, ctx["last_pos"][bi], :].float().detach().cpu().numpy()
                records.append({
                    "ti": item["ti"],
                    "cat": item["cat"],
                    "obj": item["obj"],
                    "prompt": item["prompt"],
                    "pre_vec": pre_vec.astype(np.float32),
                    "answer_vec": ans_vec.astype(np.float32),
                    "scores": scores[bi].astype(np.float32),
                })
            del out, batch
            torch.cuda.empty_cache()
    return records


def summarize_condition(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def add_metrics(row: dict[str, Any], remove_target_delta: float, release_threshold: float) -> None:
    recovery = (row["target_delta"] - remove_target_delta) / (abs(remove_target_delta) + 1e-8)
    row["recovery_ratio"] = float(recovery)
    row["is_constrained_clean"] = bool(recovery >= 0.5 and row["max_other_delta"] <= release_threshold)
    row["suppressor_component"] = float(row["target_delta"] - row["max_other_delta"])


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
    layer_id: int,
    mode: str,
    site: str,
    target_pre_basis: np.ndarray,
    target_ans_basis: np.ndarray,
    transfer: np.ndarray,
    competitor_ans_basis: np.ndarray | None,
    remove_scale: float,
    restore_scale: float,
    suppress_scale: float,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    first_token_ids = []
    layer = layers[layer_id - 1]
    site_module, site_kind = get_site_module(layers, layer_id, site)
    pre_b = torch.tensor(normalize_basis(target_pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(target_ans_basis), dtype=torch.float32)
    comp_b = None if competitor_ans_basis is None else torch.tensor(normalize_basis(competitor_ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
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
                out[bi, pos, :] = out[bi, pos, :] - remove_scale * proj.to(out.dtype)
                coeff_rows.append(vecs.mean(dim=0) @ pb.T)
            if mode in {"support", "joint"}:
                coeff = torch.stack(coeff_rows, dim=0)
                store["support_add"] = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
            return (out,) + inputs[1:]

        handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        if mode in {"support", "suppressor", "joint"}:
            def apply_site(x: torch.Tensor) -> torch.Tensor:
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                if mode in {"support", "joint"}:
                    out[bidx, apos, :] = out[bidx, apos, :] + restore_scale * store["support_add"].to(out.device, dtype=out.dtype)
                if mode in {"suppressor", "joint"} and comp_b is not None:
                    cb = comp_b.to(out.device)
                    vecs = out[bidx, apos, :].float()
                    proj = (vecs @ cb.T) @ cb
                    out[bidx, apos, :] = out[bidx, apos, :] - suppress_scale * proj.to(out.dtype)
                return out

            if site_kind == "pre":
                def site_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                    return (apply_site(inputs[0]),) + inputs[1:]
                # input_answer is already handled by layer pre hook ordering; add second hook for same module is OK.
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
        pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        first_token_ids.append(logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64))
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[layer_id]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, target_ans_basis))
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
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories) or TEST_CATEGORIES
        layer_offsets = [int(x) for x in parse_str_list(args.layer_offsets)]
        sites = parse_str_list(args.restore_sites)
        scales = parse_float_list(args.restore_scales)
        modes = parse_str_list(args.modes)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: true_last=L{last_layer}, train/test={args.train_objects}/{args.test_objects}, "
            f"cats={test_categories}, offsets={layer_offsets}, sites={sites}, vram={alloc:.2f}/{reserved:.2f}GB"
        )

        result: dict[str, Any] = {
            "phase": 142,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_last_layer": last_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "layer_offsets": layer_offsets,
            "restore_sites": sites,
            "restore_scales": scales,
            "modes": modes,
            "rank": args.rank,
            "release_threshold": args.release_threshold,
            "competitors": DEFAULT_COMPETITORS,
            "templates": [x["name"] for x in LONG_TEMPLATES],
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for offset in layer_offsets:
            layer_id = layer_from_offset(last_layer, offset)
            log(f"Capturing train centers for L{layer_id}")
            train_items = build_items(categories, args.train_objects, args.test_objects, "train")
            train_records = capture_layer_records(
                model, tokenizer, device, layers, train_items, cat_local_ids, categories,
                args.batch_size, args.max_length, layer_id,
            )
            pre_centers = centers_from_records(train_records, categories, "pre_vec")
            ans_centers = centers_from_records(train_records, categories, "answer_vec")
            cache: dict[str, Any] = {}
            for cat in test_categories:
                pre_basis, pre_sv = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
                ans_basis, ans_sv = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
                comp = DEFAULT_COMPETITORS.get(cat, "animal")
                comp_basis, _comp_sv = svd_basis(build_category_contrast_matrix(ans_centers, categories, comp), args.rank)
                cat_train = [r for r in train_records if r["cat"] == cat]
                x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
                y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
                cache[cat] = {
                    "pre_basis": pre_basis,
                    "ans_basis": ans_basis,
                    "comp_basis": comp_basis,
                    "transfer": ridge_map(x_train, y_train, args.ridge),
                    "pre_sv": pre_sv,
                    "ans_sv": ans_sv,
                }

            for ci, cat in enumerate(test_categories, 1):
                log(f"Testing {args.model} L{layer_id} {ci}/{len(test_categories)} {cat}")
                target_idx = categories.index(cat)
                prompts = build_long_prompts(cat, args.train_objects, args.test_objects)
                records = capture_layer_records(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, layer_id,
                )
                cc = cache[cat]
                x_test = project_np(np.stack([r["pre_vec"] for r in records]), cc["pre_basis"])
                y_test = project_np(np.stack([r["answer_vec"] for r in records]), cc["ans_basis"])
                pred_test = x_test @ cc["transfer"]
                clean = {
                    "scores": np.stack([r["scores"] for r in records]),
                    "answer_proj": projection_values(torch.tensor(np.stack([r["answer_vec"] for r in records]), device=device), cc["ans_basis"]),
                    "first_token_ids": np.zeros((len(records),), dtype=np.int64),
                }
                remove = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, layer_id, "remove", "input_answer",
                    cc["pre_basis"], cc["ans_basis"], cc["transfer"], cc["comp_basis"],
                    args.remove_scale, 0.0, args.suppress_scale,
                )
                remove_summary = summarize_condition(remove, clean, target_idx, categories)
                rows = []
                for site in sites:
                    for scale in scales:
                        for mode in modes:
                            patched = run_condition(
                                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                                args.batch_size, args.max_length, layer_id, mode, site,
                                cc["pre_basis"], cc["ans_basis"], cc["transfer"], cc["comp_basis"],
                                args.remove_scale, scale, args.suppress_scale,
                            )
                            row = {
                                "layer_id": layer_id,
                                "layer_offset": offset,
                                "site": site,
                                "scale": scale,
                                "mode": mode,
                                **summarize_condition(patched, clean, target_idx, categories),
                                "token_audit": token_summary(tokenizer, patched["first_token_ids"]),
                            }
                            add_metrics(row, remove_summary["target_delta"], args.release_threshold)
                            row["top_competitor"] = max(
                                ((k, v) for k, v in row["mean_delta_by_category"].items() if k != cat),
                                key=lambda x: x[1],
                            )
                            rows.append(row)
                constrained = [r for r in rows if r["is_constrained_clean"]]
                best_constrained = max(constrained, key=lambda x: (x["recovery_ratio"], x["target_delta"])) if constrained else None
                best_support = max([r for r in rows if r["mode"] == "support"], key=lambda x: (x["recovery_ratio"], -x["max_other_delta"]))
                best_joint = max([r for r in rows if r["mode"] == "joint"], key=lambda x: (x["recovery_ratio"], -x["max_other_delta"])) if "joint" in modes else None
                key = f"{cat}@L{layer_id}"
                result["category_results"][key] = {
                    "category": cat,
                    "layer_id": layer_id,
                    "layer_offset": offset,
                    "competitor": DEFAULT_COMPETITORS.get(cat, "animal"),
                    "transfer_r2": r2_score(y_test, pred_test),
                    "transfer_cosine": cosine_mean(y_test, pred_test),
                    "remove": remove_summary,
                    "rows": rows,
                    "constrained_clean_count": len(constrained),
                    "best_constrained_clean": best_constrained,
                    "best_support": best_support,
                    "best_joint": best_joint,
                }
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NONE"
    return (
        f"L{row['layer_id']} {row['mode']} {row['site']} s{row['scale']} "
        f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} rec{row['recovery_ratio']:+.2f} "
        f"clean={row['is_constrained_clean']} comp={row['top_competitor'][0]}:{row['top_competitor'][1]:+.2f}"
    )


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 142 Support/Suppressor Timepath: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"True last layer: L{result['true_last_layer']}; train/test: {result['train_objects_per_category']}/{result['test_objects_per_category']}")
    lines.append("")
    lines.append("| category@layer | transfer | remove | clean count | best clean | best support | best joint |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, item in result["category_results"].items():
        transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
        rem = f"T{item['remove']['target_delta']:+.2f} R{item['remove']['max_other_delta']:+.2f}"
        lines.append(
            f"| {key} | {transfer} | {rem} | {item['constrained_clean_count']} | "
            f"{_fmt(item['best_constrained_clean'])} | {_fmt(item['best_support'])} | {_fmt(item['best_joint'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=10)
    parser.add_argument("--test-objects", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--suppress-scale", type=float, default=1.0)
    parser.add_argument("--restore-scales", default="0.25,0.3,0.35,0.4,0.5")
    parser.add_argument("--restore-sites", default="attention_output,mlp_input")
    parser.add_argument("--modes", default="support,joint")
    parser.add_argument("--layer-offsets", default="0")
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase142_{args.model}_support_suppressor_timepath.json"
    md_path = out_dir / f"phase142_{args.model}_support_suppressor_timepath.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
