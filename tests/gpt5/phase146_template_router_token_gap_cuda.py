#!/usr/bin/env python3
"""
Phase 146: template-conditioned router and token selection gap.

For each template family, sweep layer/site/scale candidates and report the best
readout-clean path plus token-level gap metrics.
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
from phase105_global_category_atlas_cuda import (  # noqa: E402
    CATEGORY_OBJECTS,
    CATEGORY_READOUT_WORDS,
    collect_readout_rows,
    find_token_id,
)
from phase107_causal_boundary_removal_cuda import score_logits, summarize_delta  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis, project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import TEMPLATE_FAMILIES, build_family_items, split_indices  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase146_template_router_token_gap")
TEST_CATEGORIES = ["number", "plant", "time", "container"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    raise ValueError(site)


def centers_from_records(records: list[dict[str, Any]], categories: list[str], key: str, n_templates: int) -> np.ndarray:
    d_model = int(records[0][key].shape[0])
    centers = np.zeros((n_templates, len(categories), d_model), dtype=np.float64)
    counts = np.zeros((n_templates, len(categories)), dtype=np.int64)
    cat_index = {cat: i for i, cat in enumerate(categories)}
    for rec in records:
        centers[int(rec["ti"]), cat_index[rec["cat"]]] += rec[key]
        counts[int(rec["ti"]), cat_index[rec["cat"]]] += 1
    return (centers / np.maximum(counts[:, :, None], 1)).astype(np.float32)


def capture_records(
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
                records.append({
                    "ti": item["ti"],
                    "cat": item["cat"],
                    "obj": item["obj"],
                    "prompt": item["prompt"],
                    "pre_vec": store["pre"][bi, pos, :].float().mean(dim=0).detach().cpu().numpy().astype(np.float32),
                    "answer_vec": store["answer"][bi, ctx["last_pos"][bi], :].float().detach().cpu().numpy().astype(np.float32),
                    "scores": scores[bi].astype(np.float32),
                })
            del out, batch
            torch.cuda.empty_cache()
    return records


def target_token_ids(tokenizer: Any, cat: str) -> list[int]:
    ids = []
    for word in CATEGORY_READOUT_WORDS[cat]:
        tid = find_token_id(tokenizer, word)
        if tid is not None:
            ids.append(int(tid))
    return sorted(set(ids))


def token_audit(logits: torch.Tensor, clean_logits: torch.Tensor, target_ids: list[int], tokenizer: Any) -> dict[str, Any]:
    if not target_ids:
        return {}
    ids = torch.tensor(target_ids, device=logits.device, dtype=torch.long)
    target_mean = logits[:, ids].float().mean(dim=1)
    clean_mean = clean_logits[:, ids].float().mean(dim=1)
    target_max = logits[:, ids].float().max(dim=1).values
    ranks = (logits.float() > target_max[:, None]).sum(dim=1).float() + 1
    argmax = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    counts = Counter(int(x) for x in argmax.tolist())
    top = [
        {"token_id": tok, "token": tokenizer.decode([tok]), "count": int(cnt), "rate": float(cnt / max(1, len(argmax)))}
        for tok, cnt in counts.most_common(5)
    ]
    return {
        "target_token_delta": float((target_mean - clean_mean).mean().detach().cpu()),
        "target_token_rank_mean": float(ranks.mean().detach().cpu()),
        "target_token_rank_median": float(ranks.median().detach().cpu()),
        "target_token_argmax_rate": float(np.mean([x in set(target_ids) for x in argmax.tolist()])),
        "top_tokens": top,
    }


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
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    scale: float,
    target_ids: list[int],
    mode: str = "support",
) -> dict[str, Any]:
    scores = []
    answer_proj = []
    token_metrics = []
    layer = layers[layer_id - 1]
    site_module, site_kind = get_site_module(layers, layer_id, site)
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)

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
            if mode == "support":
                coeff = torch.stack(coeff_rows, dim=0)
                store["support_add"] = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
            return (out,) + inputs[1:]

        handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        if mode == "support":
            def apply_site(x: torch.Tensor) -> torch.Tensor:
                out = x.clone()
                bidx = torch.arange(out.shape[0], device=out.device)
                apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
                out[bidx, apos, :] = out[bidx, apos, :] + scale * store["support_add"].to(out.device, dtype=out.dtype)
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


def add_metrics(row: dict[str, Any], remove_target_delta: float, release_threshold: float) -> None:
    recovery = (row["target_delta"] - remove_target_delta) / (abs(remove_target_delta) + 1e-8)
    row["recovery_ratio"] = float(recovery)
    row["is_constrained_clean"] = bool(recovery >= 0.5 and row["max_other_delta"] <= release_threshold)


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
        offsets = [int(x) for x in parse_str_list(args.layer_offsets)]
        sites = parse_str_list(args.sites)
        scales = parse_float_list(args.scales)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: L{last_layer}, cats={test_categories}, families={families}, splits={splits}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 146,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_last_layer": last_layer,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "layer_offsets": offsets,
            "sites": sites,
            "scales": scales,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "readout_token_labels": token_labels,
            "results": {},
        }
        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                log(f"Centers {split}/{family}")
                train_items = []
                for cat in categories:
                    train_items.extend(build_family_items(cat, family, train_idx))
                layer_cache: dict[int, dict[str, Any]] = {}
                for offset in offsets:
                    layer_id = layer_from_offset(last_layer, offset)
                    records = capture_records(model, tokenizer, device, layers, train_items, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                    pre_centers = centers_from_records(records, categories, "pre_vec", len(TEMPLATE_FAMILIES[family]))
                    ans_centers = centers_from_records(records, categories, "answer_vec", len(TEMPLATE_FAMILIES[family]))
                    layer_cache[layer_id] = {"basis": {}}
                    for cat in test_categories:
                        pre_basis, _ = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
                        ans_basis, _ = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
                        cat_train = [r for r in records if r["cat"] == cat]
                        x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
                        y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
                        layer_cache[layer_id]["basis"][cat] = {
                            "pre_basis": pre_basis,
                            "ans_basis": ans_basis,
                            "transfer": ridge_map(x_train, y_train, args.ridge),
                        }
                for cat in test_categories:
                    prompts = build_family_items(cat, family, test_idx)
                    target_idx = categories.index(cat)
                    target_ids = target_token_ids(tokenizer, cat)
                    clean_records_by_layer: dict[int, dict[str, Any]] = {}
                    for offset in offsets:
                        layer_id = layer_from_offset(last_layer, offset)
                        test_records = capture_records(model, tokenizer, device, layers, prompts, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                        clean_records_by_layer[layer_id] = {
                            "scores": np.stack([r["scores"] for r in test_records]),
                            "answer_vec": np.stack([r["answer_vec"] for r in test_records]),
                        }
                    rows = []
                    for offset in offsets:
                        layer_id = layer_from_offset(last_layer, offset)
                        basis = layer_cache[layer_id]["basis"][cat]
                        clean = clean_records_by_layer[layer_id]
                        clean_base = {
                            "scores": clean["scores"],
                            "answer_proj": projection_values(torch.tensor(clean["answer_vec"], device=device), basis["ans_basis"]),
                        }
                        remove = run_condition(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, layer_id, "input_answer",
                            basis["pre_basis"], basis["ans_basis"], basis["transfer"], 0.0, target_ids, mode="remove",
                        )
                        remove_summary = summarize_delta(remove["scores"] - clean_base["scores"], target_idx, categories)
                        for site in sites:
                            for scale in scales:
                                patched = run_condition(
                                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                                    args.batch_size, args.max_length, layer_id, site,
                                    basis["pre_basis"], basis["ans_basis"], basis["transfer"], scale, target_ids,
                                )
                                row = {
                                    "split": split,
                                    "family": family,
                                    "category": cat,
                                    "layer_id": layer_id,
                                    "layer_offset": offset,
                                    "site": site,
                                    "scale": scale,
                                    **summarize_delta(patched["scores"] - clean_base["scores"], target_idx, categories),
                                    "answer_proj_delta": float((patched["answer_proj"] - clean_base["answer_proj"]).mean()),
                                    "token": patched["token"],
                                }
                                add_metrics(row, remove_summary["target_delta"], args.release_threshold)
                                rows.append(row)
                    best = max(rows, key=lambda r: (r["is_constrained_clean"], r["recovery_ratio"], -r["max_other_delta"]))
                    key = f"{split}:{family}:{cat}"
                    result["results"][key] = {"best": best, "rows": rows}
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 146 Template Router Token Gap: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| split/family/category | best path | T | R | rec | clean | token_rank | token_argmax |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, item in result["results"].items():
        b = item["best"]
        tok = b.get("token", {})
        lines.append(
            f"| {key} | L{b['layer_id']} {b['site']} s{b['scale']} | "
            f"{b['target_delta']:+.2f} | {b['max_other_delta']:+.2f} | {b['recovery_ratio']:+.2f} | "
            f"{b['is_constrained_clean']} | {tok.get('target_token_rank_mean', 0):.1f} | {tok.get('target_token_argmax_rate', 0):.2f} |"
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
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="number,plant,time,container")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--layer-offsets", default="0,-1")
    parser.add_argument("--sites", default="input_answer,attention_output,mlp_input")
    parser.add_argument("--scales", default="0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase146_{args.model}_template_router_token_gap.json"
    md_path = out_dir / f"phase146_{args.model}_template_router_token_gap.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
